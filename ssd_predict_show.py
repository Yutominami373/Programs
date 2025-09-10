"""
第2章SSDで予測結果を画像として描画するクラス

"""
import numpy as np
import matplotlib.pyplot as plt 
import cv2  # OpenCVライブラリ
import torch
import torch.nn.functional as F
from matplotlib import patches
import json
import os

from utils.config import SSD_DEFAULTS
from utils.ssd_model import DataTransform

# ---- Partial Missing (ABN) mode: single source of truth -----------------
PM_ALLOWED_MODES = {"default", "sensitive"}
DEFAULT_PM_MODE = "default"  # ★ 方針に合わせてここを決め打ち

def to_model_device(t, net):
    """tensor t を net と同じデバイスへ移す（non_blocking有効）。"""
    return t.to(next(net.parameters()).device, non_blocking=True)
    
def to_model_device_all(obj, net):
    """list/tuple/dictに入ったテンソルもまとめて移す（任意で利用）。"""
    dev = next(net.parameters()).device
    if torch.is_tensor(obj):
        return obj.to(dev, non_blocking=True)
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_model_device_all(o, net) for o in obj)
    if isinstance(obj, dict):
        return {k: to_model_device_all(v, net) for k, v in obj.items()}
    return obj

def _load_ae_stats(path, device="cpu"):
    with open(path, "r") as f:
        stat = json.load(f)
    for d in stat['layers']:
        # ここでは計算時に device へ載せ替えるので tensor 化は後段で可
        pass
    return stat

def _residual_heatmap(rec, stats, input_hw, device):
    """多層Z正規化→入力解像度へアップサンプル→平均融合。戻り [N,H,W]"""
    H_in, W_in = input_hw
    zs = []
    for li,(fh,fr) in enumerate(zip(rec['feat_ref'], rec['feat_hat'])):
        fh = fh.to(device); fr = fr.to(device)
        e = (fh - fr).abs().mean(dim=1, keepdim=True)  # [N,1,Hl,Wl]
        mu = torch.tensor(stats['layers'][li]['mu'], dtype=torch.float32, device=device)
        sg = torch.tensor(stats['layers'][li]['sigma'], dtype=torch.float32, device=device)
        z  = (e - mu) / (sg + 1e-6)
        z  = F.interpolate(z, size=(H_in, W_in), mode='bilinear', align_corners=False)
        zs.append(z)
    H = torch.stack(zs, dim=0).mean(dim=0).squeeze(1)  # [N,H,W]
    # 軽い平滑化（3x3 Gaussian）
    k = torch.tensor([[1,2,1],[2,4,2],[1,2,1]], dtype=torch.float32, device=device)
    k = (k/k.sum()).view(1,1,3,3)
    H = F.conv2d(H.unsqueeze(1), k, padding=1).squeeze(1)
    return H  # [N,H,W]

def _inner_roi(x0,y0,x1,y1, shrink, H, W):
    cx, cy = 0.5*(x0+x1), 0.5*(y0+y1)
    w, h   = (x1-x0)*shrink, (y1-y0)*shrink
    x0s, x1s = max(0.0, cx-0.5*w), min(1.0, cx+0.5*w)
    y0s, y1s = max(0.0, cy-0.5*h), min(1.0, cy+0.5*h)
    ix0, ix1 = int(x0s*W), int(x1s*W)
    iy0, iy1 = int(y0s*H), int(y1s*H)
    return ix0, iy0, ix1, iy1

# 置き場所：partial_missing_from_roi_px の定義を丸ごとこれに差し替え
def partial_missing_from_roi_px(Hmap, box_xyxy_px,
                                tau_px, tau_reg, r_min,
                                shrink=1.00,
                                mode=DEFAULT_PM_MODE,   # ★ デフォルトを定数に
                                morph=1,
                                min_pixels=2):
    # ★ 不正値と将来のタイプミスを早期検知
    if mode not in PM_ALLOWED_MODES:
        raise ValueError(f"partial_missing_from_roi_px: invalid mode={mode}")
    import torch, numpy as np, torch.nn.functional as F
    if isinstance(Hmap, np.ndarray):
        Hmap = torch.from_numpy(Hmap)
    if Hmap.dim() == 3:
        Hmap = Hmap.squeeze(0)
    Hmap = Hmap.float()

    H, W = int(Hmap.shape[-2]), int(Hmap.shape[-1])
    x1,y1,x2,y2 = [float(v) for v in box_xyxy_px]
    cx,cy = 0.5*(x1+x2), 0.5*(y1+y2)
    w,h   = (x2-x1)*shrink, (y2-y1)*shrink
    xi1 = max(0, min(W-1, int(np.floor(cx-0.5*w))))
    yi1 = max(0, min(H-1, int(np.floor(cy-0.5*h))))
    xi2 = max(0, min(W,   int(np.ceil (cx+0.5*w))))
    yi2 = max(0, min(H,   int(np.ceil (cy+0.5*h))))
    if xi2 <= xi1 or yi2 <= yi1:
        return False, 0.0, 0.0, 0.0

    roi = Hmap[yi1:yi2, xi1:xi2]
    if roi.numel() == 0:
        return False, 0.0, 0.0, 0.0

    if mode == "sensitive":
        # 1) ROI内の局所基準を使う：tau を ROI の中央値に（負側でも拾える）
        rmed = torch.nanmedian(roi)
        mask = roi > (rmed - 1e-9)  # ほぼ全正側を採用
        if morph and mask.numel() > 0:
            t = mask.float().unsqueeze(0).unsqueeze(0)
            t = F.max_pool2d(t, kernel_size=1+2*morph, stride=1, padding=morph)
            mask = (t[0,0] > 0.5)
        c = int(mask.sum().item())
        r = float(c / max(1, roi.numel()))
        m = float(roi[mask].mean().item()) if c > 0 else float(roi.mean().item())
        ok = (c >= min_pixels)  # 1pxでもOKに近い判定
        score = m if c > 0 else float(roi.max().item())
        return bool(ok), float(score), r, m

    # 既定（従来ロジック）
    mask = roi > float(tau_px)
    r = float(mask.float().mean().item())
    if r == 0.0:
        return False, 0.0, 0.0, 0.0
    m = float(roi[mask].mean().item())
    ok = (r >= float(r_min)) and (m >= float(tau_reg))
    score = r * m
    return bool(ok), float(score), r, m

def _auto_layer_weights_from_rec(rec, drop_first=0, gamma=0.2):
    """
    段（層）ごとの自動重みを作る。小さい解像度（深層）ほど重くする。
    - drop_first: 浅い層を前から K 個ゼロ重みで無効化（例: 1）
    - gamma: 逆面積^gamma を重み（gamma=0.5 で緩やかに深層寄り）
    """
    refs = rec['feat_ref']
    L = len(refs)
    import math
    ws = []
    for li in range(L):
        H, W = refs[li].shape[-2], refs[li].shape[-1]
        area = max(1, H * W)
        w = (1.0 / float(area)) ** float(gamma)
        if li < int(drop_first):
            w = 0.0
        ws.append(w)

    s = sum(ws)
    if s <= 0:
        # すべてゼロになった場合は最深層に全重みを置く
        ws = [0.0] * (L - 1) + [1.0]
    else:
        ws = [w / s for w in ws]
    return ws  # list[float] 長さ L

def _make_anomaly_map_from_feats(net, size_hw, fuse="wmean", use_l2=False, make_vis=False,
                                 layer_weights=None, drop_first=1, gamma=0.5):
    """
    net.last_rec から異常ヒートマップを作る。
    重要: スコア計算は“層ごとの μ/σ 標準化のみ”（画像内0-1正規化はしない）。
          表示用にだけ 0-1 正規化版を作る（make_vis=True のとき）。

    引数:
      - fuse: "wmean"(推奨) | "mean" | "max"
      - layer_weights: 層融合の重み（長さLのlist）。None のときは自動計算（深層寄り）
      - drop_first: 浅い層を前からK層 無効化（デフォルト=1）
      - gamma: 自動重みの深層寄り度合い（逆面積^gamma）
    戻り値:
      - make_vis=False:   hm_std [N,H,W]（標準化のみ／スコア用）
      - make_vis=True :  (hm_std [N,H,W], hm_vis [N,H,W])（hm_visは表示専用）
    """
    rec = getattr(net, 'last_rec', None)
    if rec is None:
        raise RuntimeError("last_rec is None. 先に net(x) を呼んでください。")

    calib_mu  = getattr(net, 'calib_mu',  None)
    calib_std = getattr(net, 'calib_std', None)

    if (calib_mu is not None) and (calib_std is not None):
        L_rec = len(rec['feat_hat'])
        L_mu  = len(calib_mu)
        L_sd  = len(calib_std)
        if (L_mu != L_rec) or (L_sd != L_rec):
            raise RuntimeError(
                f"[ABORT] AE calib length mismatch: last_rec={L_rec}, mu={L_mu}, sigma={L_sd}. "
                f"Recompute calib μ/σ with the current model and feature layout."
            )

    errs = []
    for li, (fhat, f) in enumerate(zip(rec['feat_hat'], rec['feat_ref'])):
        e = (fhat - f)
        e = e.pow(2) if use_l2 else e.abs()
        # [N,C,H,W] -> [N,1,H,W]
        e = e.mean(dim=1, keepdim=True)

        # 層ごとの μ/σ 標準化（冒頭で長さを検証済み）
        if (calib_mu is not None) and (calib_std is not None):
            mu  = float(calib_mu[li])
            std = float(calib_std[li])
            e = (e - mu) / (std + 1e-6)

        # 入力画像サイズへ拡大
        if size_hw is not None:
            e = F.interpolate(e, size=size_hw, mode="bilinear", align_corners=False)

        errs.append(e)

    # [L,N,1,H,W]
    stack = torch.stack(errs, dim=0)

    # 層融合
    if fuse == "max":
        hm_std = stack.max(dim=0).values.squeeze(1)  # [N,H,W]
    else:
        if layer_weights is None:
            # 自動重み（深層寄り）。浅層を drop_first 層だけゼロ化
            wlist = _auto_layer_weights_from_rec(rec, drop_first=drop_first, gamma=gamma)
        else:
            wlist = list(layer_weights)

        L = stack.size(0)
        if len(wlist) != L:
            # 長さ不一致は“自動重み”に切り替え
            wlist = _auto_layer_weights_from_rec(rec, drop_first=drop_first, gamma=gamma)

        w = torch.tensor(wlist, dtype=stack.dtype, device=stack.device).view(L, 1, 1, 1, 1)
        hm_std = (stack * w).sum(dim=0).squeeze(1)  # [N,H,W]

    if not make_vis:
        return hm_std

    # 表示専用の 0–1 正規化（スコアには不使用）
    p05 = torch.quantile(hm_std.flatten(1), 0.05, dim=1).view(-1, 1, 1)
    p95 = torch.quantile(hm_std.flatten(1), 0.95, dim=1).view(-1, 1, 1)
    hm_vis = (hm_std - p05) / (p95 - p05 + 1e-6)
    hm_vis = hm_vis.clamp(0.0, 1.0)

    return hm_std, hm_vis


def _anomaly_score_for_boxes(hm, boxes, ring=10, inner_shrink=0.85):
    """
    hm: 標準化後の“生”マップ [H,W]（0–1 ではない）
    boxes: [[x1,y1,x2,y2], ...]
    内側 ROI を収縮（inner_shrink）して平均を取り、周囲リング平均を引いて背景補正。
    """
    scores = []
    H, W = hm.shape
    for (x1, y1, x2, y2) in boxes:
        x1 = int(max(0, min(W - 1, x1))); x2 = int(max(0, min(W, x2)))
        y1 = int(max(0, min(H - 1, y1))); y2 = int(max(0, min(H, y2)))
        if x2 <= x1 or y2 <= y1:
            scores.append(0.0); continue

        # 内側 ROI を収縮（背景の巻き込みを軽減）
        cx = 0.5 * (x1 + x2); cy = 0.5 * (y1 + y2)
        w  = (x2 - x1) * inner_shrink; h = (y2 - y1) * inner_shrink
        xi1 = int(max(0, min(W - 1, cx - 0.5 * w))); xi2 = int(max(0, min(W, cx + 0.5 * w)))
        yi1 = int(max(0, min(H - 1, cy - 0.5 * h))); yi2 = int(max(0, min(H, cy + 0.5 * h)))

        roi = hm[yi1:yi2, xi1:xi2]
        m_in = float(roi.mean().item()) if roi.numel() else 0.0

        # 周囲リングの平均（背景）
        xr1 = max(0, x1 - ring); yr1 = max(0, y1 - ring)
        xr2 = min(W, x2 + ring); yr2 = min(H, y2 + ring)
        if (xr2 - xr1) <= 0 or (yr2 - yr1) <= 0:
            scores.append(max(0.0, m_in)); continue

        ring_patch = hm[yr1:yr2, xr1:xr2].clone()                 # 周囲パッチを複製（in-place無効化のため）
        ring_patch[(y1 - yr1):(y2 - yr1), (x1 - xr1):(x2 - xr1)] = float('nan')  # 内側を NaN にしてリングだけ残す
        if torch.isnan(ring_patch).any():                         # NaN が含まれる場合（通常は必ず含まれる）
            m_bg = float(torch.nanmean(ring_patch).item())        # NaN を無視した平均を背景平均とする
        else:
            m_bg = float(ring_patch.mean().item())                # 念のためのフォールバック（NaNが無いケース）

        scores.append(max(0.0, m_in - m_bg)) 
    return scores

class SSDPredictShow():
    """SSDでの予測と画像の表示をまとめて行うクラス"""

    def __init__(self, eval_categories, net, cfg=None,
                 *, stats_policy="external_overrides", tau_scale=None, **kwargs):
        self.eval_categories = eval_categories
        self.net = net
        self._cfg = cfg or SSD_DEFAULTS

        color_mean = tuple(self._cfg.get("preprocess",{}).get("color_mean",(104,117,123)))

        size_from_net = int(getattr(net, "input_size", 0) or self._cfg.get("input_size", 300))
        if not isinstance(size_from_net, int) or size_from_net <= 0:
            size_from_net = 300
        self._input_size = size_from_net

        # DataTransform も cfg から
        self.transform = DataTransform(self._input_size, color_mean, aug_cfg=self._cfg.get("preprocess"))

        # τ/AE 関連の既定
        ae = self._cfg.get("ae", {})
        self._tau_px  = float(ae.get("tau_px", 1.0))
        self._tau_reg = float(ae.get("tau_reg", 1.5))
        self._r_min   = float(ae.get("r_min", 0.05))
        self._stats_path = ae.get("stats_path", "ae_stats.json")
        self._ae_metric = None
        self._ae_num_layers = None
        self._stats_source = "defaults"

        # ★ pm_mode（部分欠損の動作モード）: 既定は DEFAULT_PM_MODE、JSONで上書き可
        self._pm_mode = DEFAULT_PM_MODE

        # === AE/部分欠損のしきい値と層別 μ/σ をロード ===
        stats_path = "ae_stats.json"
        st = None
        if os.path.exists(stats_path):
            try:
                with open(stats_path, "r") as f:
                    st = json.load(f)
            except Exception as e:
                print(f"[warn] AE stats load failed: {e}")
                st = None

        if st is not None:
            # τ群（存在しなければ既定）
            if "tau_px"  in st: self._tau_px  = float(st["tau_px"])
            if "tau_reg" in st: self._tau_reg = float(st["tau_reg"])
            if "r_min"   in st: self._r_min   = float(st["r_min"])

            # pm_mode の既定と上書き（許可セット内のみ採用）
            pm_mode = st.get("pm_mode", None)
            if isinstance(pm_mode, str) and pm_mode in PM_ALLOWED_MODES:
                self._pm_mode = pm_mode

            # metric（L1/L2）と後方互換（use_l2）
            metric = st.get("metric", None)
            if metric is None:
                metric = "L2" if bool(st.get("use_l2", False)) else "L1"
            self._ae_metric = str(metric).upper()  # "L1" or "L2"

            # 任意：重みメタがあれば一致チェックのみ（停止はしない）
            json_md5 = st.get("weight_md5", None)
            ckpt_md5 = getattr(self.net, "weight_md5", None)
            if json_md5 and ckpt_md5 and (json_md5 != ckpt_md5):
                print(f"[warn] weight_md5 mismatch: stats={json_md5} vs net={ckpt_md5}")

            # μ/σの適用方針
            layers = st.get("layers", None)  # 期待: [{"mu":..., "sigma":...}, ...]
            if isinstance(layers, list) and len(layers) > 0:
                mu_list  = [float(x.get("mu", 0.0))    for x in layers]
                std_list = [float(x.get("sigma", 1.0)) for x in layers]

                if stats_policy == "external_overrides":
                    self.net.calib_mu  = mu_list
                    self.net.calib_std = std_list
                    self._stats_source = "json:override"
                elif stats_policy == "external_if_empty":
                    if getattr(self.net, "calib_mu", None) is None or \
                       getattr(self.net, "calib_std", None) is None:
                        self.net.calib_mu  = mu_list
                        self.net.calib_std = std_list
                        self._stats_source = "json:filled"
                    else:
                        self._stats_source = "model:keep"
                elif stats_policy == "ignore_external":
                    self._stats_source = "model:keep"
                else:
                    print(f"[warn] unknown stats_policy='{stats_policy}', fallback to 'external_overrides'")
                    self.net.calib_mu  = mu_list
                    self.net.calib_std = std_list
                    self._stats_source = "json:override"

                self._ae_num_layers = len(layers)
            else:
                # μ/σ未記載 → τだけ反映
                self._stats_source = "json:taus_only"
        else:
            # JSONがない → 既定値のまま
            self._stats_source = "defaults"

        # τスケールは明示時のみ適用（勝手な0.9等の縮小はしない）
        if tau_scale is not None:
            try:
                s_px, s_reg, s_r = map(float, tau_scale)
                self._tau_px  *= s_px
                self._tau_reg *= s_reg
                self._r_min   *= s_r
                self._stats_source += "+tau_scaled"
            except Exception as e:
                print(f"[warn] invalid tau_scale={tau_scale}: {e}")

        #参考ログ（必要なら）
        print(f"[info] stats_policy={stats_policy}, source={self._stats_source}, "
        f"tau=({self._tau_px:.3f},{self._tau_reg:.3f},{self._r_min:.3f}), "
        f"metric={self._ae_metric}, pm_mode={self._pm_mode}")
    
    def _warn_if_hardcoded_mode_mismatch(self):
        """pm_mode が既定(DEFAULT_PM_MODE)から上書きされている場合に一度だけ警告。"""
        if not hasattr(self, "_pm_warned"):
            self._pm_warned = False
        # 既定と違っていて、まだ警告していなければ出す
        if (getattr(self, "_pm_mode", DEFAULT_PM_MODE) != DEFAULT_PM_MODE) and (not self._pm_warned):
            print(f"[warn] pm_mode is overridden by config: {self._pm_mode} (default is {DEFAULT_PM_MODE})")
            self._pm_warned = True

        # ---- AE 互換性チェック（L1/L2・層数） -------------------------------
    def _ae_preflight_ok(self, use_l2: bool) -> bool:
        """
        ABN（二段閾）を実行してよいかの互換性チェック。
        - L1/L2 の不一致
        - calib_mu/std 不在
        - 層数不一致
        の場合は False を返す。
        """
        # L1/L2 チェック
        requested = "L2" if use_l2 else "L1"
        metric = getattr(self, "_ae_metric", None)
        if metric is not None and metric.upper() != requested:
            print(f"[warn] AE metric mismatch: ae_stats metric={metric} vs requested={requested}. ABN disabled.")
            return False

        # μ/σの存在チェック
        if getattr(self.net, "calib_mu", None) is None or getattr(self.net, "calib_std", None) is None:
            print("[warn] calib_mu/std not loaded. ABN disabled.")
            return False

        # 層数の事前チェック（last_recが未確定な場合は保留）
        # 最終的な層数チェックは forward後（last_rec確定後）にも行う。
        if getattr(self, "_ae_num_layers", None) is not None:
            # ここではメタの存在のみを確認（厳密な一致は後段で）
            pass

        return True

    def _ae_postforward_layers_ok(self):
        rec = getattr(self.net, "last_rec", None)
        if (rec is None) or (not isinstance(rec, dict)):
            print("[warn] AE: last_rec is not available.")
            return False

        def _layers_len(x):
            # None → 0
            if x is None:
                return 0
            # list/tuple → len
            if isinstance(x, (list, tuple)):
                return len(x)
            # Tensor → 0次元なら1、1次元以上なら先頭次元
            try:
                import torch
                if torch.is_tensor(x):
                    return int(x.shape[0]) if x.ndim >= 1 else 1
            except Exception:
                pass
            # その他 → 0（未知型は非対応として扱う）
            return 0

        L_rec = _layers_len(rec.get("feat_hat", []))
        mu_attr = getattr(self.net, "calib_mu", None)
        sd_attr = getattr(self.net, "calib_std", None)
        L_mu = _layers_len(mu_attr)
        L_sd = _layers_len(sd_attr)

        if (L_rec == 0) or (L_mu == 0) or (L_sd == 0):
            print(f"[warn] AE: layer empty. rec={L_rec} mu={L_mu} std={L_sd}")
            return False

        if not (L_rec == L_mu == L_sd):
            print(f"[warn] AE: layer count mismatch. rec={L_rec} mu={L_mu} std={L_sd}")
            return False

        return True


    def ssd_predict_with_anomaly(self, image_file_path, data_confidence_level=None,
                             fuse=None, use_l2=None, inner_shrink=None, ring=None,
                             layer_weights=None, drop_first=None, gamma=None):
        ae = self._cfg.get("ae", {})
        det = self._cfg.get("detect", {})
        data_confidence_level = det.get("conf_thresh", 0.4) if data_confidence_level is None else data_confidence_level
        fuse         = ae.get("fuse","wmean") if fuse is None else fuse
        use_l2       = ae.get("use_l2",False) if use_l2 is None else use_l2
        inner_shrink = ae.get("inner_shrink",1.0) if inner_shrink is None else inner_shrink
        ring         = ae.get("ring",8) if ring is None else ring
        drop_first   = ae.get("drop_first",1) if drop_first is None else drop_first
        gamma        = ae.get("gamma",0.8) if gamma is None else gamma
        """
        既存のSSD予測に、AEヒートマップ由来の異常スコア＆部分欠損判定を付与して返す。
        戻り値:
          (rgb_img, boxes_px, labels, conf_scores,
           anomaly_scores, pm_flags, pm_scores, pm_area, pm_mean, hm_vis_np)
        """
        # --- 検出（この forward で net.last_rec が更新される） ---
        rgb_img, predict_bbox, pre_dict_label_index, scores = self.ssd_predict(
            image_file_path, data_confidence_level)

        H, W = rgb_img.shape[:2]

        # --- AEヒートマップ（標準化マップと可視化用） ---
        self.net.eval()
        with torch.no_grad():
            hm_std_b, hm_vis_b = _make_anomaly_map_from_feats(
                self.net, size_hw=(H, W), fuse=fuse, use_l2=use_l2, make_vis=True,
                layer_weights=layer_weights, drop_first=drop_first, gamma=gamma
            )
        hm_std = hm_std_b[0]; hm_vis = hm_vis_b[0]
        hm_vis_np = hm_vis.cpu().numpy()

        # === 互換性チェック（L1/L2 & μ/σ & 層数） ===
        abn_ok = self._ae_preflight_ok(use_l2) and self._ae_postforward_layers_ok()

        # --- 先に正規化座標 → ピクセル座標へ変換（常に作る） ---
        boxes_px = []
        for i in range(predict_bbox.shape[0]):
            x1 = int(max(0, min(W - 1, predict_bbox[i, 0] * W)))
            y1 = int(max(0, min(H - 1, predict_bbox[i, 1] * H)))
            x2 = int(max(0, min(W,     predict_bbox[i, 2] * W)))
            y2 = int(max(0, min(H,     predict_bbox[i, 3] * H)))
            boxes_px.append((x1, y1, x2, y2))

        # --- anomaly（内側-周囲リング）は相対差のため常に算出 ---
        #     ※ 関数名は _anomaly_score_for_boxes（定義名と合わせる）
        anomaly_scores = _anomaly_score_for_boxes(
            hm_std, boxes_px, ring=ring, inner_shrink=inner_shrink
        )

        # --- ABN（二段閾）: 互換性NGなら停止（安全側） ---
        pm_flags, pm_scores, pm_area, pm_mean = [], [], [], []
        if abn_ok:
            # --- 部分欠損の二段閾（pm_mode は self._pm_mode を使用） ---
            for (x1, y1, x2, y2) in boxes_px:
                ok, sc, r, m = partial_missing_from_roi_px(
                    hm_std, (x1, y1, x2, y2),
                    tau_px = getattr(self, "_tau_px", 0.7),
                    tau_reg = getattr(self, "_tau_reg", 1.2),
                    r_min = max(0.02, getattr(self, "_r_min", 0.02)),
                    shrink = inner_shrink,
                    mode = self._pm_mode,
                    morph = 0,
                    min_pixels = 8
                )
                pm_flags.append(bool(ok))
                pm_scores.append(float(sc))
                pm_area.append(float(r))
                pm_mean.append(float(m))
        else:
            n = predict_bbox.shape[0]
            pm_flags  = [False] * n
            pm_scores = [0.0]   * n
            pm_area   = [0.0]   * n
            pm_mean   = [0.0]   * n

        return (rgb_img, predict_bbox, pre_dict_label_index, scores,
                anomaly_scores, pm_flags, pm_scores, pm_area, pm_mean, hm_vis_np)


    def show(self, image_file_path, data_confidence_level):
        """
        物体検出の予測結果を表示をする関数。

        Parameters
        ----------
        image_file_path:  str
            画像のファイルパス
        data_confidence_level: float
            予測で発見とする確信度の閾値

        Returns
        -------
        なし。rgb_imgに物体検出結果が加わった画像が表示される。
        """
        rgb_img, predict_bbox, pre_dict_label_index, scores = self.ssd_predict(
            image_file_path, data_confidence_level)

        self.vis_bbox(rgb_img, bbox=predict_bbox, label_index=pre_dict_label_index,
              conf_scores=scores, label_names=self.eval_categories)


    def ssd_predict(self, image_file_path, data_confidence_level=0.5):
        """
        1枚の画像パスを受け取り、検出結果を返す。
        戻り値:
          rgb_img: np.uint8, (H,W,3), RGB
          predict_bbox: np.float32, (N,4), 0..1 正規化座標 [x1,y1,x2,y2]
          pre_dict_label_index: list[int], 長さ N
          scores: np.float32, (N,)
        """
        # --- 画像読み込み（BGR）と RGB の保持 -------------------------------
        bgr = cv2.imread(image_file_path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"image not found: {image_file_path}")
        rgb_img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)  # ← 最終返却用（可視化用）

        # --- 前処理（DataTransformはBGR前提が多い） -----------------------
        img_transformed, _, _ = self.transform(
            bgr, phase="val", boxes=None, labels=None
        )

        # --- HWC/CHW の両系に対応 → BCHW tensor 化 ------------------------
        if isinstance(img_transformed, np.ndarray):
            # ndarray: HWC(BGR or RGB?) を想定 → RGB化してCHWへ
            if img_transformed.ndim != 3 or img_transformed.shape[-1] != 3:
                raise ValueError(f"unexpected transform output shape: {img_transformed.shape}")
            # BGR想定で RGB 並びに変換してから CHW
            img_t = torch.from_numpy(img_transformed[:, :, (2, 1, 0)]).permute(2, 0, 1).contiguous()
        elif isinstance(img_transformed, torch.Tensor):
            t = img_transformed
            if t.dim() == 3 and t.shape[0] == 3:
                # 既に CHW
                img_t = t.contiguous()
            elif t.dim() == 3 and t.shape[-1] == 3:
                # HWC → CHW（色順はここでの変換不要＝transform側が責務を負っているなら）
                img_t = t.permute(2, 0, 1).contiguous()
            else:
                raise ValueError(f"unexpected tensor shape from transform: {t.shape}")
        else:
            raise TypeError(f"unexpected type from transform: {type(img_transformed)}")

        x = img_t.unsqueeze(0).to(next(self.net.parameters()).device)

        # --- 推論（forward 戻り値のバリエーションを吸収） -------------------
        self.net.eval()
        with torch.no_grad():
            out = self.net(x)

        loc = conf = dbox_list = None
        final_preds = None  # Detect 後の最終予測（任意）

        if isinstance(out, tuple):
            if len(out) == 3:
                loc, conf, dbox_list = out
            elif len(out) == 2:
                loc, conf = out
                dbox_list = getattr(self.net, "dbox_list", None)
                if dbox_list is None:
                    from utils.ssd_model import DBox
                    cfg = getattr(self.net, "cfg", None)
                    if cfg is None:
                        raise ValueError("SSD.forward returned 2-tuple but net.cfg is None")
                    dbox_list = DBox(cfg).make_dbox_list()
            else:
                raise ValueError(f"Unexpected tuple length from SSD.forward(): {len(out)}")

        elif isinstance(out, dict):
            loc = out.get("loc", None)
            conf = out.get("conf", None)
            dbox_list = out.get("dbox_list", getattr(self.net, "dbox_list", None))
            final_preds = out.get("final", None)
            if (loc is None or conf is None or dbox_list is None) and final_preds is None:
                raise ValueError("SSD.forward dict missing keys: need ('loc','conf','dbox_list') or 'final'")

        elif isinstance(out, torch.Tensor):
            # 単一Tensor返し → Detect後の最終予測とみなす
            final_preds = out

        else:
            raise ValueError(
                f"SSD.forward returned unsupported type: {type(out)}. "
                "Expected (loc, conf[, dbox_list]) or dict or Tensor."
            )

        # --- Detect 未実行ならここで適用 -----------------------------------

                # --- priors の“別レシピ”を封じ、順序ズレも検知（Detect 実行有無に関わらず先に実施） ---
        # loc/conf から P を取得し、期待 priors（net 既存優先→無ければ cfg 生成）に強制統一。
        try:
            import numpy as np, hashlib, torch
            try:
                from utils.ssd_model import DBox
            except Exception:
                from ssd_model import DBox
            cfg = getattr(self.net, "cfg", None)
            if cfg is not None:
                dev = loc.device if torch.is_tensor(loc) else next(self.net.parameters()).device
                P = int(loc.shape[1])
                # 1) 期待 priors を決定（学習由来を優先／無ければ cfg 生成）
                dbox_expected = getattr(self.net, "dbox_list", None)
                if not (torch.is_tensor(dbox_expected) and dbox_expected.dim() == 2):
                    dbox_expected = DBox(cfg).make_dbox_list().to(dev, dtype=torch.float32).contiguous()
                    self.net.dbox_list = dbox_expected  # 推論以後は固定（dtype/レイアウトを正規化）
                # 2) 形状 P と一致させる（本数不一致は即エラー）
                if int(dbox_expected.shape[0]) != P:
                    raise RuntimeError(
                        f"[FATAL] priors count mismatch: expected {int(dbox_expected.shape[0])} vs predictions {P}. "
                        "cfg（min_sizes/max_sizes/feature_maps/steps/aspect_ratios）やヘッド順序を確認してください。"
                    )
                # 3) モデル出力に dbox が同梱されている場合は“中身”を厳密検証（採用はしない）
                def _md5_tensor_canon(t: torch.Tensor) -> str:
                    a = t.detach().to(dtype=torch.float32, memory_format=torch.contiguous_format).cpu().numpy()
                    a = np.ascontiguousarray(a)
                    return hashlib.md5(a.tobytes()).hexdigest()
                if 'dbox_list' in locals() and torch.is_tensor(dbox_list):
                    if (tuple(dbox_list.shape) != tuple(dbox_expected.shape)) or \
                       (_md5_tensor_canon(dbox_list) != _md5_tensor_canon(dbox_expected)):
                        raise RuntimeError(
                            f"[FATAL] priors content mismatch in raw outputs: "
                            f"supplied.shape={tuple(dbox_list.shape)} expected.shape={tuple(dbox_expected.shape)} "
                            f"supplied.md5={_md5_tensor_canon(dbox_list)} expected.md5={_md5_tensor_canon(dbox_expected)}. "
                            "cfg やヘッド順序の不整合を確認してください。"
                        )
                # 4) 以後は期待 priors だけを採用（“別レシピ”の混入を排除）
                dbox_list = dbox_expected
        except Exception:
            raise

        # --- Detect 未実行ならここで適用 -----------------------------------
        if final_preds is None:

            if not hasattr(self.net, "detect"):
                raise RuntimeError("net.detect is not available; cannot decode loc/conf.")
            # Detect 側で温度スケーリング込みの softmax を実施するため、
            # ここでは logits（未softmax）をそのまま渡す。
            final_preds = self.net.detect(loc, conf, dbox_list)
        else:
                        # --- final 経路でも priors の“中身”を厳密検証（本数＋md5） ---
            try:
                try:
                    from utils.ssd_model import DBox
                except Exception:
                    from ssd_model import DBox
                import numpy as np, hashlib, torch
                cfg = getattr(self.net, "cfg", None)
                if cfg is not None:
                    dev = next(self.net.parameters()).device
                    dbox_cfg = DBox(cfg).make_dbox_list().to(dev, dtype=torch.float32).contiguous()
                    dbox_net = getattr(self.net, "dbox_list", None)
                    def _md5_tensor_canon(t: torch.Tensor) -> str:
                        a = t.detach().to(dtype=torch.float32, memory_format=torch.contiguous_format).cpu().numpy()
                        a = np.ascontiguousarray(a)
                        return hashlib.md5(a.tobytes()).hexdigest()
                    if torch.is_tensor(dbox_net):
                        if (dbox_net.shape != dbox_cfg.shape) or (_md5_tensor_canon(dbox_net) != _md5_tensor_canon(dbox_cfg)):
                            raise RuntimeError(
                                f"[FATAL] priors content mismatch (final path): "
                                f"net={_md5_tensor_canon(dbox_net)} cfg={_md5_tensor_canon(dbox_cfg)}. "
                                "cfg（input_size/feature_maps/steps/min_sizes/max_sizes/aspect_ratios）やヘッド順序の不一致を確認してください。"
                            )
                    else:
                        # net が priors を保持していなければ cfg 生成を採用
                        self.net.dbox_list = dbox_cfg
            except Exception:
                # 不一致は安全側で停止（運用で警告にしたい場合はここを print に変更可）
                raise
            # 検証済みの final_preds をそのまま使用

        # --- 最終予測テンソルの形状差を吸収 --------------------------------
        if not isinstance(final_preds, torch.Tensor):
            raise TypeError(f"final prediction must be Tensor, got {type(final_preds)}")

        # 2D → [1,N,C] に正規化
        if final_preds.dim() == 2:
            final_preds = final_preds.unsqueeze(0)

        # case1: [1, N, C]（各行が [x1,y1,x2,y2,score,label] 等）
        if final_preds.dim() == 3:
            B, N, C = final_preds.shape
            if B != 1:
                raise ValueError(f"Batch size must be 1, got {B}")
            pred = final_preds[0]  # [N,C]

            # 列レイアウトを推定（候補）
            idx_layout_candidates = [
                (0, 1, 2, 3, 4, 5),  # x1,y1,x2,y2,score,label
                (1, 2, 3, 4, 0, 5),  # score が先頭
                (0, 1, 2, 3, 5, 4),  # label が最後-1
                (2, 3, 4, 5, 1, 0),  # label,score が先頭
            ]

            def is_valid_layout(p, idxs):
                xi1, yi1, xi2, yi2, si, li = idxs
                if C <= max(idxs):
                    return False
                xy = p[:, [xi1, yi1, xi2, yi2]]
                if not torch.isfinite(xy).all():
                    return False
                x1, y1, x2, y2 = xy[:, 0], xy[:, 1], xy[:, 2], xy[:, 3]
                ok_range = ((x1 >= -0.05) & (x1 <= 1.05) &
                            (y1 >= -0.05) & (y1 <= 1.05) &
                            (x2 >= -0.05) & (x2 <= 1.05) &
                            (y2 >= -0.05) & (y2 <= 1.05)).float().mean() > 0.7
                ok_order = ((x2 - x1 > 0) & (y2 - y1 > 0)).float().mean() > 0.7
                return bool(ok_range and ok_order)

            chosen = None
            for cand in idx_layout_candidates:
                if is_valid_layout(pred, cand):
                    chosen = cand
                    break
            if chosen is None:
                raise ValueError(
                    f"Cannot infer layout of final prediction tensor with shape {final_preds.shape}. "
                    "Please standardize Detect to output a known layout."
                )
            xi1, yi1, xi2, yi2, si, li = chosen
            boxes  = pred[:, [xi1, yi1, xi2, yi2]]
            scores = pred[:, si]
            labels = pred[:, li].to(torch.int64)

        # case2: [1, C, K, 5] or [1, K, C, 5]（クラス別 top_k）
        elif final_preds.dim() == 4:
            B, A, K, D = final_preds.shape  # A=クラス or K と入れ替わっている可能性あり
            if B != 1:
                raise ValueError(f"Batch size must be 1, got {B}")

            num_classes = int(getattr(self.net, "num_classes", A))
            preds = final_preds

            # [1, K, C, 5] → [1, C, K, 5] に揃える
            if A != num_classes and final_preds.shape[2] == num_classes:
                preds = final_preds.permute(0, 2, 1, 3).contiguous()
                _, A, K, D = preds.shape

            if A != num_classes or D != 5:
                raise ValueError(
                    f"Unsupported 4D layout: got shape {final_preds.shape}, expected [1, num_classes, top_k, 5]"
                )

            # 末尾5要素の並びを推定: (x1,y1,x2,y2,score) または (score,x1,y1,x2,y2)
            def split_xy_score(t):  # t: [K,5]
                xy1, sc1 = t[:, :4], t[:, 4]
                xy2, sc2 = t[:, 1:], t[:, 0]
                def score_ok(sc):
                    return float(((sc >= -0.01) & (sc <= 1.01)).float().mean())
                def xy_ok(xy):
                    x1,y1,x2,y2 = xy[:,0],xy[:,1],xy[:,2],xy[:,3]
                    ok_range = ((x1 >= -0.05) & (x1 <= 1.05) &
                                (y1 >= -0.05) & (y1 <= 1.05) &
                                (x2 >= -0.05) & (x2 <= 1.05) &
                                (y2 >= -0.05) & (y2 <= 1.05)).float().mean().item()
                    ok_order = ((x2 - x1 > 0) & (y2 - y1 > 0)).float().mean().item()
                    return ok_range * ok_order
                score1, score2 = score_ok(sc1), score_ok(sc2)
                xyv1, xyv2 = xy_ok(xy1), xy_ok(xy2)
                return (xy1, sc1) if (xyv1 * score1 >= xyv2 * score2) else (xy2, sc2)

            boxes_list, scores_list, labels_list = [], [], []
            # 背景(0)は除外
            for cls in range(1, num_classes):
                t = preds[0, cls]  # [K,5]
                xy, sc = split_xy_score(t)
                keep = sc >= data_confidence_level
                if keep.any():
                    boxes_list.append(xy[keep])
                    scores_list.append(sc[keep])
                    labels_list.append(torch.full((int(keep.sum()),), cls, dtype=torch.int64, device=xy.device))

            if len(boxes_list) == 0:
                boxes  = torch.empty((0,4), dtype=torch.float32, device=preds.device)
                scores = torch.empty((0,),   dtype=torch.float32, device=preds.device)
                labels = torch.empty((0,),   dtype=torch.int64,  device=preds.device)
            else:
                boxes  = torch.cat(boxes_list,  dim=0)
                scores = torch.cat(scores_list, dim=0)
                labels = torch.cat(labels_list, dim=0)

        else:
            raise ValueError(f"Unexpected final preds shape: {final_preds.shape}")

        # --- ラベル正規化（Detectの1..K → 0..K-1） --------------------------
        labels0 = labels.to(dtype=torch.long) - 1
        if labels0.numel() > 0:
            # 背景は除外済みのため負値は出ない想定（出たら異常）
            assert labels0.min().item() >= 0, f"pred label < 0 detected: {labels0.min().item()}"

        # --- numpy へ変換＆返却 ---------------------------------------------
        predict_bbox = boxes.detach().cpu().numpy().astype(np.float32)          # (N,4)
        pre_dict_label_index = labels0.detach().cpu().numpy().astype(np.int64).tolist()
        scores_np = scores.detach().cpu().numpy().astype(np.float32)

        return rgb_img, predict_bbox, pre_dict_label_index, scores_np

    def vis_bbox(self, rgb_img, bbox, label_index, conf_scores, label_names,
             anomaly_scores=None, pm_flags=None, pm_area=None, pm_mean=None,
             custom_names=None):
      import matplotlib.pyplot as plt
      from matplotlib import patches
      fig, ax = plt.subplots(1, figsize=(10, 10))
      ax.imshow(rgb_img)

      for i, (bb, lb) in enumerate(zip(bbox, label_index)):
          x1, y1, x2, y2 = bb

          # 欠損なら赤、通常はライム
          is_pm = bool(pm_flags[i]) if (pm_flags is not None and i < len(pm_flags)) else False
          edgecolor = 'red' if is_pm else 'lime'

          rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                  fill=False, linewidth=2, edgecolor=edgecolor)
          ax.add_patch(rect)

          # ラベル名の上書き（custom_names があれば優先）
          name = (custom_names[i] if (custom_names is not None and i < len(custom_names))
                  else label_names[lb])

          conf = conf_scores[i] if conf_scores is not None else None
          anom = anomaly_scores[i] if (anomaly_scores is not None and i < len(anomaly_scores)) else None

          txt = name
          if conf is not None:
              txt += f" conf:{conf:.2f}"
          if anom is not None:
              txt += f" anom:{anom:.2f}"
          if is_pm and pm_area is not None and pm_mean is not None and i < len(pm_area) and i < len(pm_mean):
              txt += f" r:{pm_area[i]:.2f} m:{pm_mean[i]:.2f}"

          ax.text(x1, y1, txt, fontsize=10, bbox=dict(facecolor='yellow', alpha=0.5))

      plt.axis('off')
      plt.show()

    def ssd_predict_batch_with_anomaly(self, image_file_paths, data_confidence_level=0.5,
                                   fuse="wmean", use_l2=False,
                                   inner_shrink=0.85, ring=10,
                                   layer_weights=None, drop_first=1, gamma=0.5):
      """
      単画像メソッド ssd_predict_with_anomaly を複数ファイルに回すだけの
      シンプルなバッチ版（I/O互換を優先）。
      """
      results = []
      for p in image_file_paths:
          (rgb_img, predict_bbox_px, pre_dict_label_index,
          conf_scores, anomaly_scores,
          pm_flags, pm_scores, pm_area, pm_mean,
          hm_vis_np) = self.ssd_predict_with_anomaly(
              p, data_confidence_level, fuse=fuse, use_l2=use_l2,
              inner_shrink=inner_shrink, ring=ring,
              layer_weights=layer_weights, drop_first=drop_first, gamma=gamma
          )
          results.append(dict(
              path=p,
              rgb_img=rgb_img,
              boxes=predict_bbox_px,
              labels=pre_dict_label_index,
              conf=conf_scores,
              anom=anomaly_scores,
              pm_flag=pm_flags,
              pm_score=pm_scores,
              pm_area=pm_area,
              pm_mean=pm_mean,
              hm_vis=hm_vis_np,
          ))
      return results
    def make_display_names(self, label_index, pm_flags, mode="replace", abnormal_label="Abnormal"):
      """
      mode="replace": 欠損はラベル名を Abnormal に置換
      mode="append" : 欠損は "Crab1(Abnormal)" のように括弧追記
      """
      out = []
      for i, lb in enumerate(label_index):
          base = self.eval_categories[lb]
          is_pm = bool(pm_flags[i]) if (pm_flags is not None and i < len(pm_flags)) else False
          if is_pm:
              if mode == "replace":
                  out.append(abnormal_label)
              else:  # "append"
                  out.append(f"{base}({abnormal_label})")
          else:
              out.append(base)
      return out