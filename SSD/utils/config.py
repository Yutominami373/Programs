# utils/config.py
from copy import deepcopy

SSD_DEFAULTS = {
    # モデル入出力
    "input_size": 300,
    "num_classes": None,             # 学習時に上書き（len(voc_classes)+1）

    # Anchors / DBox
    "feature_maps": [38, 19, 10, 5, 3, 1],
    "steps": [8, 16, 32, 64, 100, 300],
    "min_sizes": [21, 45, 99, 153, 207, 261],
    "max_sizes": [45, 99, 153, 207, 261, 315],
    "aspect_ratios": [[2], [2,3], [2,3], [2,3], [2], [2]],
    "bbox_aspect_num": [4, 6, 6, 6, 4, 4],

    # 変換スケール（学習encode/推論decodeで共通使用）
    "variance": (0.1, 0.2),

    # Detect（推論）
    "detect": {
        "conf_thresh": 0.4,
        "nms_thresh": 0.55,
        "top_k": 400,
        "keep_top_k": 200,
        "cross_class_nms_iou": 0.70,
        "temperature": 1.0,
    },

    # NMSユーティリティ既定（必要なら上書き）
    "nms": {
        "overlap": 0.45,
        "top_k": 200,
    },

    # 損失
    "loss": {
        "jaccard_thresh": 0.5,
        "neg_pos": 3,
        "label_smoothing": 0.05,
        "max_neg_per_img": 64,
    },

    # 前処理/水増し
    "preprocess": {
        "color_mean": (104,117,123),                  # BGR
        "aug_prob_photometric": 0.5,
        "aug_prob_hflip": 0.5,
        "aug_random_crop": False,
        "enable_expand": True,
        "photometric": {   # PhotometricDistort 内訳
            "brightness_delta": 32,
            "contrast": (0.5, 1.5),
            "saturation": (0.5, 1.5),
            "hue_delta": 18
        },
        "random_sample_crop": {
            "sample_options": [(0.1,None),(0.3,None),(0.7,None),(0.9,None),(None,None)],
            "max_trials": 50
        }
    },

    # 異常/部分欠損（AE）
    "ae": {
        "stats_path": "ae_stats.json",
        "tau_px": 1.0,
        "tau_reg": 1.5,
        "r_min": 0.05,
        "fuse": "wmean",
        "use_l2": False,
        "inner_shrink": 1.0,
        "ring": 8,
        "drop_first": 1,
        "gamma": 0.8,
    },
}

def build_cfg(overrides: dict | None = None):
    cfg = deepcopy(SSD_DEFAULTS)
    if overrides:
        # かんたん深いマージ
        def merge(d, u):
            for k,v in u.items():
                if isinstance(v, dict) and isinstance(d.get(k), dict):
                    merge(d[k], v)
                else:
                    d[k] = v
        merge(cfg, overrides)
    return cfg
