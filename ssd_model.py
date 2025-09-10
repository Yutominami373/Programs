"""
第2章SSDで実装した内容をまとめたファイル
"""

# パッケージのimport
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Function
import torch.utils.data as data
import torch
import cv2
import numpy as np
import os.path as osp
from itertools import product as product
from math import sqrt as sqrt
# --- ランダム適用の汎用ラッパ ---
import random
from utils.data_augmentation import (  # ← 1回に統一
    Compose, ConvertFromInts, ToAbsoluteCoords, PhotometricDistort,
    Expand, RandomSampleCrop, RandomMirror, ToPercentCoords,
    Resize, SubtractMeans
)

class RandomApply(object):
    """
    任意の transform を確率 p で実行するラッパ。
    transform が (img, boxes, labels) を取る/取らないの両方に対応。
    """
    def __init__(self, op, p=0.5):
        self.op = op
        self.p = float(p)

    def __call__(self, img, boxes=None, labels=None):
        if random.random() < self.p:
            # 3引数を受ける transform か、imgのみかを動的に判定
            try:
                out = self.op(img, boxes, labels)
            except TypeError:
                img = self.op(img)
                return img, boxes, labels
            else:
                return out
        # スキップ時は入力をそのまま返す（Compose 互換）
        return img, boxes, labels

class FeatDecoder(nn.Module):
    """同解像度・同チャネルでの軽量再構成デコーダ（段ごとに使う）"""
    def __init__(self, in_ch: int):
        super().__init__()
        hidden = max(in_ch // 2, 32)  # 最低32chは確保
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, in_ch, kernel_size=3, padding=1)
        )
    def forward(self, x):
        return self.body(x)

# XMLをファイルやテキストから読み込んだり、加工したり、保存したりするためのライブラリ
import xml.etree.ElementTree as ET

# フォルダ「utils」にある関数matchを記述したmatch.pyからimport
from utils.match import match


# 学習、検証の画像データとアノテーションデータへのファイルパスリストを作成する

def make_datapath_list(rootpath):
    """
    データへのパスを格納したリストを作成する。

    Parameters
    ----------
    rootpath : str
        データフォルダへのパス

    Returns
    -------
    ret : train_img_list, train_anno_list, val_img_list, val_anno_list
        データへのパスを格納したリスト
    """

    # 画像ファイルとアノテーションファイルへのパスのテンプレートを作成
    imgpath_template = osp.join(rootpath, 'JPEGImages', '%s.jpg')
    annopath_template = osp.join(rootpath, 'Annotations', '%s.xml')

    # 訓練と検証、それぞれのファイルのID（ファイル名）を取得する
    # ★ 連結ミス修正：osp.join に分割引数で渡す（rootpath が末尾スラッシュでなくても安全）
    train_id_names = osp.join(rootpath, 'ImageSets', 'Main', 'train.txt')
    val_id_names   = osp.join(rootpath, 'ImageSets', 'Main', 'val.txt')

    # 訓練データの画像ファイルとアノテーションファイルへのパスリストを作成
    train_img_list = []
    train_anno_list = []

    for line in open(train_id_names):
        file_id = line.strip()  # 空白スペースと改行を除去
        img_path = (imgpath_template % file_id)  # 画像のパス
        anno_path = (annopath_template % file_id)  # アノテーションのパス
        train_img_list.append(img_path)  # リストに追加
        train_anno_list.append(anno_path)  # リストに追加

    # 検証データの画像ファイルとアノテーションファイルへのパスリストを作成
    val_img_list = []
    val_anno_list = []

    for line in open(val_id_names):
        file_id = line.strip()  # 空白スペースと改行を除去
        img_path = (imgpath_template % file_id)  # 画像のパス
        anno_path = (annopath_template % file_id)  # アノテーションのパス
        val_img_list.append(img_path)  # リストに追加
        val_anno_list.append(anno_path)  # リストに追加

    return train_img_list, train_anno_list, val_img_list, val_anno_list


# 「XML形式のアノテーション」を、リスト形式に変換するクラス


class Anno_xml2list(object):
    """
    1枚の画像に対する「XML形式のアノテーションデータ」を、画像サイズで規格化してからリスト形式に変換する。

    Attributes
    ----------
    classes : リスト
        VOCのクラス名を格納したリスト
    """

    def __init__(self, classes):

        self.classes = classes

    def __call__(self, xml_path, width, height):
        """
        1枚の画像に対する「XML形式のアノテーションデータ」を、画像サイズで規格化してからリスト形式に変換する。

        Parameters
        ----------
        xml_path : str
            xmlファイルへのパス。
        width : int
            対象画像の幅。
        height : int
            対象画像の高さ。

        Returns
        -------
        ret : [[xmin, ymin, xmax, ymax, label_ind], ... ]
            物体のアノテーションデータを格納したリスト。画像内に存在する物体数分のだけ要素を持つ。
        """

        # 画像内の全ての物体のアノテーションをこのリストに格納します
        ret = []

        # xmlファイルを読み込む
        xml = ET.parse(xml_path).getroot()

        # 画像内にある物体（object）の数だけループする
        for obj in xml.iter('object'):

            # アノテーションで検知がdifficultに設定されているものは除外
            difficult = int(obj.find('difficult').text)
            if difficult == 1:
                continue

            # 1つの物体に対するアノテーションを格納するリスト
            bndbox = []

            name = obj.find('name').text.lower().strip()  # 物体名
            if name not in self.classes:
              # 未知ラベルは完全に除外（検出/AEの両方から外す）
              # ここでログを出すなら print や logger を使用
              continue
            bbox = obj.find('bndbox')  # バウンディングボックスの情報

            # アノテーションの xmin, ymin, xmax, ymaxを取得し、0～1に規格化
            pts = ['xmin', 'ymin', 'xmax', 'ymax']

            for pt in (pts):
                # VOCは原点が(1,1)なので1を引き算して（0, 0）に
                cur_pixel = int(bbox.find(pt).text) - 1

                # 幅、高さで規格化
                if pt == 'xmin' or pt == 'xmax':  # x方向のときは幅で割算
                    cur_pixel /= width
                else:  # y方向のときは高さで割算
                    cur_pixel /= height

                bndbox.append(cur_pixel)

            # アノテーションのクラス名のindexを取得して追加
            label_idx = self.classes.index(name)
            bndbox.append(label_idx)

            # resに[xmin, ymin, xmax, ymax, class]を足す
            ret += [bndbox]

        # ★ 空でも形状を(0,5)に固定（後段で :4 / 4 のスライスが安全に通る）
        if len(ret) == 0:
            return np.zeros((0, 5), dtype=np.float32)
        return np.array(ret, dtype=np.float32)  # [[xmin, ymin, xmax, ymax, label_ind], ... ]

# 入力画像の前処理をするクラス

class DataTransform(object):
    def __init__(self, input_size, color_mean, aug_cfg=None):
        self.input_size = input_size
        self.color_mean = color_mean
        a = aug_cfg or {}
        p_photo = a.get("aug_prob_photometric", 0.5)
        p_hflip = a.get("aug_prob_hflip", 0.5)
        use_crop = a.get("aug_random_crop", False)
        use_expand = a.get("enable_expand", True)
        photo = a.get("photometric", {})
        crop  = a.get("random_sample_crop", {})

        train_ops = [ConvertFromInts(), ToAbsoluteCoords()]
        if p_photo and p_photo > 0:
            train_ops += [RandomApply(PhotometricDistort(
                brightness_delta=photo.get("brightness_delta", 32),
                contrast=tuple(photo.get("contrast", (0.5,1.5))),
                saturation=tuple(photo.get("saturation", (0.5,1.5))),
                hue_delta=photo.get("hue_delta", 18)
            ), p=p_photo)]
        if use_expand:
            train_ops += [Expand(self.color_mean)]
        if use_crop:
            train_ops += [RandomSampleCrop(
                sample_options=crop.get("sample_options"),
                max_trials=crop.get("max_trials", 50)
            )]
        if p_hflip and p_hflip > 0:
            train_ops += [RandomApply(RandomMirror(), p=p_hflip)]
        train_ops += [ToPercentCoords(), Resize(self.input_size), SubtractMeans(self.color_mean)]

        val_ops = [ConvertFromInts(), ToAbsoluteCoords(),
                   ToPercentCoords(), Resize(self.input_size), SubtractMeans(self.color_mean)]
        self.data_transform = {'train': Compose(train_ops), 'val': Compose(val_ops)}

    def __call__(self, img, phase, boxes, labels):
        # VOCDataset.pull_item の呼び出し順に合わせる（img, phase, boxes, labels）
        return self.data_transform['train' if phase == 'train' else 'val'](img, boxes, labels)




class VOCDataset(data.Dataset):
    """
    VOC2012のDatasetを作成するクラス。PyTorchのDatasetクラスを継承。

    Attributes
    ----------
    img_list : リスト
        画像のパスを格納したリスト
    anno_list : リスト
        アノテーションへのパスを格納したリスト
    phase : 'train' or 'test'
        学習か訓練かを設定する。
    transform : object
        前処理クラスのインスタンス
    transform_anno : object
        xmlのアノテーションをリストに変換するインスタンス
    """

    def __init__(self, img_list, anno_list, phase, transform, transform_anno):
        self.img_list = img_list
        self.anno_list = anno_list
        self.phase = phase  # train もしくは valを指定
        self.transform = transform  # 画像の変形
        self.transform_anno = transform_anno  # アノテーションデータをxmlからリストへ

    def __len__(self):
        '''画像の枚数を返す'''
        return len(self.img_list)

    def __getitem__(self, index):
        '''
        前処理をした画像のテンソル形式のデータとアノテーションを取得
        '''
        im, gt, h, w = self.pull_item(index)
        return im, gt

    def pull_item(self, index):
        '''前処理をした画像テンソル、アノテーション、画像の高さ、幅を取得する'''

        # 1. 画像読み込み
        image_file_path = self.img_list[index]
        img = cv2.imread(image_file_path)  # [H][W][BGR]
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {image_file_path}")
        height, width, channels = img.shape

        anno_file_path = self.anno_list[index]
        anno_list = self.transform_anno(anno_file_path, width, height)  # (N,5) or (0,5)

        # ★ 空の時も2次元を前提に安全に扱う
        if anno_list.shape[0] == 0:
            boxes  = np.zeros((0, 4), dtype=np.float32)
            labels = np.zeros((0,),   dtype=np.int64)
        else:
            boxes  = anno_list[:, :4].astype(np.float32)
            labels = anno_list[:, 4].astype(np.int64)

        # ★ 念のため2次元保証
        if anno_list.ndim == 1:
            anno_list = np.zeros((0, 5), dtype=np.float32)

        # 3. 前処理（空でも :4 / 4 が安全に通る）
        img, boxes, labels = self.transform(
            img, self.phase, anno_list[:, :4], anno_list[:, 4])
      
        # BGR→RGB, [H,W,C]→[C,H,W]（※ ToTensor をCompose内に入れていない前提）
        img = torch.from_numpy(img[:, :, (2, 1, 0)]).permute(2, 0, 1).contiguous()

        # gt = [xmin, ymin, xmax, ymax, class]
        if boxes.size == 0:
            gt = np.zeros((0, 5), dtype=np.float32)
        else:
            gt = np.hstack((boxes, np.expand_dims(labels, axis=1)))

        return img, gt, height, width


def od_collate_fn(batch):
    """
    Datasetから取り出すアノテーションデータのサイズが画像ごとに異なります。
    画像内の物体数が2個であれば(2, 5)というサイズですが、3個であれば（3, 5）など変化します。
    この変化に対応したDataLoaderを作成するために、
    カスタイマイズした、collate_fnを作成します。
    collate_fnは、PyTorchでリストからmini-batchを作成する関数です。
    ミニバッチ分の画像が並んでいるリスト変数batchに、
    ミニバッチ番号を指定する次元を先頭に1つ追加して、リストの形を変形します。
    """

    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])  # sample[0] は画像imgです
        targets.append(torch.FloatTensor(sample[1]))  # sample[1] はアノテーションgtです

    # imgsはミニバッチサイズのリストになっています
    # リストの要素はtorch.Size([3, 300, 300])です。
    # このリストをtorch.Size([batch_num, 3, 300, 300])のテンソルに変換します
    imgs = torch.stack(imgs, dim=0)

    # targetsはアノテーションデータの正解であるgtのリストです。
    # リストのサイズはミニバッチサイズです。
    # リストtargetsの要素は [n, 5] となっています。
    # nは画像ごとに異なり、画像内にある物体の数となります。
    # 5は [xmin, ymin, xmax, ymax, class_index] です

    return imgs, targets


# 35層にわたる、vggモジュールを作成
def make_vgg():
    layers = []
    in_channels = 3  # 色チャネル数

    # vggモジュールで使用する畳み込み層やマックスプーリングのチャネル数
    cfg = [64, 64, 'M', 128, 128, 'M', 256, 256,
           256, 'MC', 512, 512, 512, 'M', 512, 512, 512]

    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'MC':
            # ceilは出力サイズを、計算結果（float）に対して、切り上げで整数にするモード
            # デフォルトでは出力サイズを計算結果（float）に対して、切り下げで整数にするfloorモード
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v

    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return nn.ModuleList(layers)


# 8層にわたる、extrasモジュールを作成
def make_extras():
    layers = []
    in_channels = 1024  # vggモジュールから出力された、extraに入力される画像チャネル数

    # extraモジュールの畳み込み層のチャネル数を設定するコンフィギュレーション
    cfg = [256, 512, 128, 256, 128, 256, 128, 256]

    layers += [nn.Conv2d(in_channels, cfg[0], kernel_size=(1))]
    layers += [nn.Conv2d(cfg[0], cfg[1], kernel_size=(3), stride=2, padding=1)]
    layers += [nn.Conv2d(cfg[1], cfg[2], kernel_size=(1))]
    layers += [nn.Conv2d(cfg[2], cfg[3], kernel_size=(3), stride=2, padding=1)]
    layers += [nn.Conv2d(cfg[3], cfg[4], kernel_size=(1))]
    layers += [nn.Conv2d(cfg[4], cfg[5], kernel_size=(3))]
    layers += [nn.Conv2d(cfg[5], cfg[6], kernel_size=(1))]
    layers += [nn.Conv2d(cfg[6], cfg[7], kernel_size=(3))]

    return nn.ModuleList(layers)


# デフォルトボックスのオフセットを出力するloc_layers、
# デフォルトボックスに対する各クラスの確率を出力するconf_layersを作成


def make_loc_conf(num_classes=21, bbox_aspect_num=[4, 6, 6, 6, 4, 4]):

    loc_layers = []
    conf_layers = []

    # VGGの22層目、conv4_3（source1）に対する畳み込み層
    loc_layers += [nn.Conv2d(512, bbox_aspect_num[0]
                             * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(512, bbox_aspect_num[0]
                              * num_classes, kernel_size=3, padding=1)]

    # VGGの最終層（source2）に対する畳み込み層
    loc_layers += [nn.Conv2d(1024, bbox_aspect_num[1]
                             * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(1024, bbox_aspect_num[1]
                              * num_classes, kernel_size=3, padding=1)]

    # extraの（source3）に対する畳み込み層
    loc_layers += [nn.Conv2d(512, bbox_aspect_num[2]
                             * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(512, bbox_aspect_num[2]
                              * num_classes, kernel_size=3, padding=1)]

    # extraの（source4）に対する畳み込み層
    loc_layers += [nn.Conv2d(256, bbox_aspect_num[3]
                             * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(256, bbox_aspect_num[3]
                              * num_classes, kernel_size=3, padding=1)]

    # extraの（source5）に対する畳み込み層
    loc_layers += [nn.Conv2d(256, bbox_aspect_num[4]
                             * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(256, bbox_aspect_num[4]
                              * num_classes, kernel_size=3, padding=1)]

    # extraの（source6）に対する畳み込み層
    loc_layers += [nn.Conv2d(256, bbox_aspect_num[5]
                             * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(256, bbox_aspect_num[5]
                              * num_classes, kernel_size=3, padding=1)]

    return nn.ModuleList(loc_layers), nn.ModuleList(conf_layers)


# convC4_3からの出力をscale=20のL2Normで正規化する層
class L2Norm(nn.Module):
    def __init__(self, input_channels=512, scale=20):
        super(L2Norm, self).__init__()  # 親クラスのコンストラクタ実行
        self.weight = nn.Parameter(torch.Tensor(input_channels))
        self.scale = scale  # 係数weightの初期値として設定する値
        self.reset_parameters()  # パラメータの初期化
        self.eps = 1e-10

    def reset_parameters(self):
        '''結合パラメータを大きさscaleの値にする初期化を実行'''
        init.constant_(self.weight, self.scale)  # weightの値がすべてscale（=20）になる

    def forward(self, x):
        '''38×38の特徴量に対して、512チャネルにわたって2乗和のルートを求めた
        38×38個の値を使用し、各特徴量を正規化してから係数をかけ算する層'''

        # 各チャネルにおける38×38個の特徴量のチャネル方向の2乗和を計算し、
        # さらにルートを求め、割り算して正規化する
        # normのテンソルサイズはtorch.Size([batch_num, 1, 38, 38])になります
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()+self.eps
        x = torch.div(x, norm)

        # 係数をかける。係数はチャネルごとに1つで、512個の係数を持つ
        # self.weightのテンソルサイズはtorch.Size([512])なので
        # torch.Size([batch_num, 512, 38, 38])まで変形します
        weights = self.weight.unsqueeze(
            0).unsqueeze(2).unsqueeze(3).expand_as(x)
        out = weights * x

        return out


# デフォルトボックスを出力するクラス
class DBox(object):
    def __init__(self, cfg):
        super(DBox, self).__init__()

        # 初期設定
        self.image_size = cfg['input_size']  # 画像サイズの300
        # [38, 19, …] 各sourceの特徴量マップのサイズ
        self.feature_maps = cfg['feature_maps']
        self.num_priors = len(cfg["feature_maps"])  # sourceの個数=6
        self.steps = cfg['steps']  # [8, 16, …] DBoxのピクセルサイズ

        self.min_sizes = cfg['min_sizes']
        # [30, 60, …] 小さい正方形のDBoxのピクセルサイズ(正確には面積)

        self.max_sizes = cfg['max_sizes']
        # [60, 111, …] 大きい正方形のDBoxのピクセルサイズ(正確には面積)

        self.aspect_ratios = cfg['aspect_ratios']  # 長方形のDBoxのアスペクト比

    def make_dbox_list(self):
        '''DBoxを作成する'''
        mean = []
        # 'feature_maps': [38, 19, 10, 5, 3, 1]
        for k, f in enumerate(self.feature_maps):
            for i, j in product(range(f), repeat=2):  # fまでの数で2ペアの組み合わせを作る　f_P_2 個
                # 特徴量の画像サイズ
                # 300 / 'steps': [8, 16, 32, 64, 100, 300],
                f_k = self.image_size / self.steps[k]

                # DBoxの中心座標 x,y　ただし、0～1で規格化している
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k

                # アスペクト比1の小さいDBox [cx,cy, width, height]
                # 'min_sizes': [30, 60, 111, 162, 213, 264]
                s_k = self.min_sizes[k]/self.image_size
                mean += [cx, cy, s_k, s_k]

                # アスペクト比1の大きいDBox [cx,cy, width, height]
                # 'max_sizes': [60, 111, 162, 213, 264, 315],
                s_k_prime = sqrt(s_k * (self.max_sizes[k]/self.image_size))
                mean += [cx, cy, s_k_prime, s_k_prime]

                # その他のアスペクト比のdefBox [cx,cy, width, height]
                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)]
                    mean += [cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)]

        # DBoxをテンソルに変換 torch.Size([8732, 4])
        output = torch.Tensor(mean).view(-1, 4)

        # DBoxの大きさが1を超えている場合は1にする
        output.clamp_(max=1, min=0)

        return output


# オフセット情報を使い、DBoxをBBoxに変換する関数
def decode(loc, dbox_list, variances=(0.1, 0.2)):
    """
    loc: [P,4], dbox_list: [P,4], variances: (vcxcy, vwh)
    """
    v0, v1 = variances
    boxes = torch.cat((
        dbox_list[:, :2] + loc[:, :2] * v0 * dbox_list[:, 2:],
        dbox_list[:, 2:] * torch.exp(loc[:, 2:] * v1)
    ), dim=1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes
# Non-Maximum Suppressionを行う関数


def nm_suppression(boxes, scores, overlap=0.45, top_k=200):
    """
    Non-Maximum Suppressionを行う関数。
    boxesのうち被り過ぎ（overlap以上）のBBoxを削除する。

    Parameters
    ----------
    boxes : [確信度閾値（0.01）を超えたBBox数,4]
        BBox情報。
    scores :[確信度閾値（0.01）を超えたBBox数]
        confの情報

    Returns
    -------
    keep : リスト
        confの降順にnmsを通過したindexが格納
    count：int
        nmsを通過したBBoxの数
    """

    # returnのひな形を作成
    count = 0
    keep = scores.new(scores.size(0)).zero_().long()
    # keep：torch.Size([確信度閾値を超えたBBox数])、要素は全部0

    # 各BBoxの面積areaを計算
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)

    # socreを昇順に並び変える
    v, idx = scores.sort(0)
    idx = idx[-top_k:]

    while idx.numel() > 0:
        i = idx[-1]
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]

        # ★ out= を使わない安全な選択と計算
        x1_sel = torch.index_select(x1, 0, idx)
        y1_sel = torch.index_select(y1, 0, idx)
        x2_sel = torch.index_select(x2, 0, idx)
        y2_sel = torch.index_select(y2, 0, idx)

        xx1 = torch.clamp(x1_sel, min=x1[i])
        yy1 = torch.clamp(y1_sel, min=y1[i])
        xx2 = torch.clamp(x2_sel, max=x2[i])
        yy2 = torch.clamp(y2_sel, max=y2[i])

        w = torch.clamp(xx2 - xx1, min=0.0)
        h = torch.clamp(yy2 - yy1, min=0.0)
        inter = w * h

        rem_areas = torch.index_select(area, 0, idx)
        union = (rem_areas - inter) + area[i]
        IoU = inter / union

        idx = idx[IoU.le(overlap)]
    # whileのループが抜けたら終了

    return keep, count


# SSDの推論時にconfとlocの出力から、被りを除去したBBoxを出力する


class Detect(nn.Module):
    def __init__(self,
                 conf_thresh=0.01,           # ← スコアの下限（小さめで良い）
                 top_k=200,                  # ← クラス内で残す最大数
                 nms_thresh=0.45,            # ← IoU のしきい値（NMS用）
                 keep_top_k=200,             # ← 画像あたり最終的に残す総数（0で無効）
                 cross_class_nms_iou=0.0,    # ← >0なら全クラス横断で追加NMS
                 temperature=1.0,            # ← softmax の温度（>=1を推奨）
                 variances=(0.1, 0.2)):      # ← デコード時のバラつき（学習と揃える）
        super(Detect, self).__init__()       # ← nn.Module の初期化
        self.conf_thresh = float(conf_thresh)            # ← 閾値を保存
        self.top_k = int(top_k)                          # ← クラス内の上限
        self.nms_thresh = float(nms_thresh)              # ← NMS 閾値
        self.keep_top_k = int(keep_top_k)                # ← 画像あたり総上限
        self.cross_class_nms_iou = float(cross_class_nms_iou)  # ← 追加NMSのIoU
        self.temperature = float(max(1.0, temperature))  # ← 温度は 1.0 未満にならないよう保護
        self.variances = tuple(variances)                # ← decode で使用

    @torch.no_grad()                      # ← 推論専用（勾配不要）
    def forward(self, loc_data, conf_data, dbox_list):
        # loc_data : [B, N, 4]        ← 回帰結果（dx,dy,dw,dh）
        # conf_data: [B, N, C]        ← 各アンカのクラスごとのロジット
        # dbox_list: [N, 4]           ← 事前定義ボックス（cx,cy,w,h か xyxy 実装依存）

        B = int(loc_data.size(0))                            # ← バッチ数
        N = int(dbox_list.size(0))                           # ← アンカ総数
        C = int(conf_data.size(2))                           # ← クラス数（背景含む）

        # dbox を conf/loc と同じデバイスに載せ替え（CPU/GPUズレを防止）
        device = conf_data.device                            # ← デバイス取得
        if dbox_list.device != device:                       # ← デバイスが違えば
            dbox_list = dbox_list.to(device)                 # ← 同期させる

        # 返り値テンソルを事前に用意（[B, C, K, 5] 形式、5は [score, x1, y1, x2, y2]）
        K = int(self.keep_top_k if self.keep_top_k > 0 else self.top_k)  # ← 最終保持数
        output = loc_data.new_zeros((B, C, K, 5))            # ← ゼロ初期化

        # 画像ごとに処理
        for b in range(B):                                   # ← バッチループ
            # 1) box をデコード（中心幅高 or xyxy → xyxy で返る decode を想定）
            boxes = decode(loc_data[b], dbox_list, self.variances)  # ← variances を明示
            boxes = boxes.clamp_(min=0.0, max=1.0)           # ← 座標を 0..1 にクリップ

            # 2) クラス確率（温度スケーリングつき softmax）
            probs = F.softmax(conf_data[b] / self.temperature, dim=1)  # ← 各アンカ×各クラス

            # クラスごとの検出結果を一時保持
            all_boxes = []    # ← すべてのクラスのボックス（後で topK 絞りに使う）
            all_scores = []   # ← すべてのクラスのスコア
            all_labels = []   # ← すべてのクラスのラベル

            # 背景（0）を除く 1..C-1 を走査
            for cl in range(1, C):                           # ← 各クラスごと
                cls_scores = probs[:, cl]                    # ← N 個のスコア
                mask = cls_scores > self.conf_thresh         # ← スコア閾値で間引き
                if mask.sum().item() == 0:                   # ← 1つも無ければスキップ
                    continue

                b_keep = boxes[mask]                         # ← 閾値を超えた box
                s_keep = cls_scores[mask]                    # ← 同スコア

                # 3) クラス内 NMS で冗長候補を除去
                ids, count = nm_suppression(                 # ← 互換NMS関数を利用
                    b_keep, s_keep, overlap=self.nms_thresh, top_k=self.top_k
                )
                if count == 0:                               # ← 残らなければスキップ
                    continue
                keep = ids[:count]                           # ← 有効インデックス

                b_sel = b_keep[keep]                         # ← NMS後の box
                s_sel = s_keep[keep]                         # ← NMS後の score
                l_sel = torch.full(                          # ← 対応ラベル（cl）
                    (count,), cl, dtype=torch.long, device=device
                )

                # 一時配列に追加（後で cross/topK に使用）
                all_boxes.append(b_sel)
                all_scores.append(s_sel)
                all_labels.append(l_sel)

            # 4) 何も残らなかった場合は次へ（ゼロ行はそのまま）
            if len(all_boxes) == 0:
                continue

            # 5) クラス横断の結合 → （任意）cross-class NMS → （任意）全体 topK
            cat_boxes  = torch.cat(all_boxes, dim=0)         # ← すべて結合（M,4）
            cat_scores = torch.cat(all_scores, dim=0)        # ← （M,）
            cat_labels = torch.cat(all_labels, dim=0)        # ← （M,）

            # （任意）クロスクラスNMS：IoU が高すぎる重複をまとめて抑制
            if self.cross_class_nms_iou > 0.0 and cat_boxes.size(0) > 1:
                keep_idx, kept = nm_suppression(             # ← ラベル無視で一括NMS
                    cat_boxes, cat_scores,
                    overlap=self.cross_class_nms_iou,
                    top_k=cat_boxes.size(0)
                )
                if kept > 0:
                    keep_idx = keep_idx[:kept]
                    cat_boxes  = cat_boxes[keep_idx]
                    cat_scores = cat_scores[keep_idx]
                    cat_labels = cat_labels[keep_idx]

            # （任意）全体 topK：スコア上位のみ残す（keep_top_k>0 のとき）
            if self.keep_top_k > 0 and cat_scores.numel() > self.keep_top_k:
                v, i = torch.topk(cat_scores, self.keep_top_k, dim=0, largest=True, sorted=True)
                cat_boxes, cat_scores, cat_labels = cat_boxes[i], v, cat_labels[i]

            # 6) 形式を [C, K, 5] に展開して output[b] に書き込む
            #    クラスごとの上位から K（足りなければ残りはゼロ）だけ詰める
            for cl in range(1, C):
                cls_mask = (cat_labels == cl)                # ← このクラスの要素
                if cls_mask.sum().item() == 0:               # ← 無ければスキップ
                    continue
                bx = cat_boxes[cls_mask]                     # ← 取り出し box
                sc = cat_scores[cls_mask]                    # ← 取り出し score

                # K を超える分は切り捨て（出力形状を一定に保つ）
                k = min(K, bx.size(0))                       # ← 実際に書く個数
                output[b, cl, :k, 0] = sc[:k]                # ← スコア
                output[b, cl, :k, 1:] = bx[:k]               # ← [x1,y1,x2,y2]

        return output




# SSDクラスを作成する


class SSD(nn.Module):
    def __init__(self, phase, cfg):
        super(SSD, self).__init__()

        self.phase = phase  # "train" or "inference"
        self.num_classes = cfg["num_classes"]
        # 推論・前処理の入力サイズを一元化（学習cfgと同一）
        self.input_size = int(cfg.get('input_size', 300))


        # --- Backbone & heads ---
        self.vgg = make_vgg()
        self.extras = make_extras()
        self.L2Norm = L2Norm()
        self.loc, self.conf = make_loc_conf(cfg["num_classes"], cfg["bbox_aspect_num"])

        # --- AE（軽量デコーダ）---
        # sources のチャンネルは順に [512, 1024, 512, 256, 256, 256]
        in_chs = [512, 1024, 512, 256, 256, 256]
        self.feat_decoders = nn.ModuleList([FeatDecoder(c) for c in in_chs])

        # ★ 浅層AE（conv2_2 / conv3_3 / conv4_3(L2後)）を “追加” で保持
        #   - conv2_2: (B,128,150,150)
        #   - conv3_3: (B,256, 75, 75)
        #   - conv4_3(L2後): (B,512, 38, 38) → 1x1で256へ投影
        self.ae_proj2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1), nn.ReLU(inplace=True)
        )
        self.ae_proj3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1), nn.ReLU(inplace=True)
        )
        self.ae_proj4 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1), nn.ReLU(inplace=True)
        )
        self.ae_dec2 = FeatDecoder(128)
        self.ae_dec3 = FeatDecoder(256)
        self.ae_dec4 = FeatDecoder(256)

        self.last_rec = None  # forward後に {'feat_hat': [...], 'feat_ref': [...]} を格納


        # DBox作成
        self.cfg = cfg
        dbox = DBox(self.cfg)
        self.dbox_list = dbox.make_dbox_list()

        # 推論時はクラス「Detect」を用意します
        if phase == 'inference':
          d = cfg.get("detect", {})
          self.detect = Detect(
              conf_thresh=d.get("conf_thresh", 0.4),
              top_k=d.get("top_k", 400),
              nms_thresh=d.get("nms_thresh", 0.55),
              keep_top_k=d.get("keep_top_k", 200),
              cross_class_nms_iou=d.get("cross_class_nms_iou", 0.70),
              temperature=d.get("temperature", 1.0),
              variances=tuple(cfg.get("variance", (0.1, 0.2))),
          )
        # デバッグ用：最初の1回だけ形状を出すフラグ
        self._printed_src_shapes = False
        self._printed_vgg_shapes = False

    def forward(self, x):
        sources = []  # 6段分の特徴
        loc = []
        conf = []

        # --------------------------
        # 1) VGG conv4_3 まで（512ch, 38x38）＋ 浅層 taps
        # --------------------------
        f2 = None  # conv2_2 の出力（ReLU後）
        f3 = None  # conv3_3 の出力（ReLU後）
        for k in range(23):
            x = self.vgg[k](x)
            if k == 8:   # conv2_2 の ReLU 直後（インデックスは make_vgg の並びに準拠）
                f2 = x
            if k == 15:  # conv3_3 の ReLU 直後
                f3 = x

        # conv4_3 を L2Norm して source1
        source1 = self.L2Norm(x)  # (B,512,38,38)
        sources.append(source1)

        # --------------------------
        # 2) VGG の残り → (B,1024,19,19) を source2
        # --------------------------
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
        x_vgg = x
        sources.append(x_vgg)

        # デバッグ（初回だけ）
        if not self._printed_vgg_shapes:
            try:
                print("[SSD] vgg shapes: source1 =", tuple(source1.shape), " vgg_end =", tuple(x_vgg.shape))
            except Exception:
                pass
            self._printed_vgg_shapes = True

        assert x_vgg.size(1) == 1024, f"extras 直前のチャネルが {x_vgg.size(1)}ch。1024ch のはず。"

        # --------------------------
        # 3) Extras で source3〜6 を作る
        # --------------------------
        x_extra = x_vgg
        for k, v in enumerate(self.extras):
            x_extra = F.relu(v(x_extra), inplace=True)
            if k % 2 == 1:
                sources.append(x_extra)

        if not self._printed_src_shapes:
            try:
                print("[SSD] sources shapes:", [tuple(s.shape) for s in sources])
            except Exception:
                pass
            self._printed_src_shapes = True

        # --------------------------
        # 4) AE（再構成）: 6段 + 浅層（合計9本）を last_rec に格納
        # --------------------------
        # 6段（検出の各sourceに1:1）
        feat_hat_6 = [dec(f) for dec, f in zip(self.feat_decoders, sources)]
        feat_ref_6 = list(sources)

        # 浅層（conv2_2, conv3_3, conv4_3）
        shallow_pairs = []
        if f2 is not None:
            f2p = self.ae_proj2(f2)
            shallow_pairs.append((self.ae_dec2(f2p), f2p))
        if f3 is not None:
            f3p = self.ae_proj3(f3)
            shallow_pairs.append((self.ae_dec3(f3p), f3p))
        f4p = self.ae_proj4(source1)  # L2後の512→256
        shallow_pairs.append((self.ae_dec4(f4p), f4p))

        feat_hat_all = feat_hat_6 + [h for (h, r) in shallow_pairs]
        feat_ref_all = feat_ref_6 + [r for (h, r) in shallow_pairs]
        self.last_rec = {'feat_hat': feat_hat_all, 'feat_ref': feat_ref_all}

        # --------------------------
        # 5) 検出ヘッド（loc/conf）
        # --------------------------

        # --- feature_maps 整合チェック（順序・解像度） ---
        # cfg['feature_maps'] が [int,...] でも [(H,W), ...] でも対応。
        if isinstance(getattr(self, "cfg", None), dict) and 'feature_maps' in self.cfg:
            exp = list(self.cfg.get('feature_maps', []))
            try:
                got_pairs = [(int(f.shape[-2]), int(f.shape[-1])) for f in sources]
                if all(isinstance(e, int) for e in exp):
                    got = [w for (_, w) in got_pairs]
                elif all((isinstance(e, (tuple, list)) and len(e) == 2) for e in exp):
                    got = [(int(h), int(w)) for (h, w) in got_pairs]
                else:
                    got = None
            except Exception:
                got = None
            assert got == exp, f"[FATAL] feature_maps mismatch: cfg={exp} vs forward={got}"

        for (feat, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(feat).permute(0, 2, 3, 1).contiguous())
            conf.append(c(feat).permute(0, 2, 3, 1).contiguous())

        # [B, N, 4] / [B, N, num_classes] に整形
        loc  = torch.cat([o.view(o.size(0), -1) for o in loc], dim=1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], dim=1)
        loc  = loc.view(loc.size(0), -1, 4)
        conf = conf.view(conf.size(0), -1, self.num_classes)

        output = (loc, conf, self.dbox_list)

        # dbox_list を必ず用意（念のための保険込み）
        dbox_list = getattr(self, "dbox_list", None)
        if dbox_list is None:
            # 自己回復（通常は __init__ で self.dbox_list 済み）
            dbox_list = DBox(getattr(self, "cfg", {})).make_dbox_list()
            self.dbox_list = dbox_list

        if self.phase == 'inference':
            # Detect は内部で実行するが、結果も dict で返却して二度実行を防ぐ。
            # softmax は Detect 側（温度スケーリング込み）で実施するため logits を渡す。
            final = self.detect(loc, conf, dbox_list)  # 実装依存の形状

            # 推論フェーズの戻り値は dict で統一し、呼び出し側が 'final' を受け取れるようにする
            return {
                "loc": loc,
                "conf": conf,
                "dbox_list": dbox_list,
                "final": final,
            }

        # 学習時（または共通パス）は従来どおり三つ組で返す（学習ループ互換）
        return loc, conf, dbox_list

class MultiBoxLoss(nn.Module):
    """SSDの損失関数のクラスです。"""

    
    def __init__(self, jaccard_thresh=0.5, neg_pos=3, device='cpu',
                 variances=(0.1,0.2), label_smoothing=0.05, max_neg_per_img=64):
        super().__init__()
        self.jaccard_thresh = jaccard_thresh
        self.negpos_ratio = neg_pos
        self.device = device
        self.variances = tuple(variances)
        self.label_smoothing = float(label_smoothing)
        self.max_neg_per_img = int(max_neg_per_img)
    
    def softmax_focal_loss(logits, target, gamma=2.0):
      # logits: [N, C], target: [N]
      logp = F.log_softmax(logits, dim=-1)
      p = logp.exp()
      pt = p.gather(1, target.view(-1,1)).squeeze(1).clamp_min(1e-6)
      loss = - ((1-pt)**gamma) * logp.gather(1, target.view(-1,1)).squeeze(1)
      return loss

    def forward(self, predictions, targets):
        """
        損失関数の計算。

        Parameters
        ----------
        predictions : SSD netの訓練時の出力(tuple)
            (loc=torch.Size([num_batch, 8732, 4]), conf=torch.Size([num_batch, 8732, 21]), dbox_list=torch.Size [8732,4])。

        targets : [num_batch, num_objs, 5]
            5は正解のアノテーション情報[xmin, ymin, xmax, ymax, label_ind]を示す

        Returns
        -------
        loss_l : テンソル
            locの損失の値
        loss_c : テンソル
            confの損失の値

        """

        # SSDモデルの出力がタプルになっているので、個々にばらす
        loc_data, conf_data, dbox_list = predictions

        # 要素数を把握
        num_batch = loc_data.size(0)  # ミニバッチのサイズ
        num_dbox = loc_data.size(1)  # DBoxの数 = 8732
        num_classes = conf_data.size(2)  # クラス数 = 21

        # 損失の計算に使用するものを格納する変数を作成
        # conf_t_label：各DBoxに一番近い正解のBBoxのラベルを格納させる
        # loc_t:各DBoxに一番近い正解のBBoxの位置情報を格納させる
        conf_t_label = torch.zeros(num_batch, num_dbox,     dtype=torch.long,   device=self.device)
        loc_t        = torch.zeros(num_batch, num_dbox, 4,  dtype=torch.float32, device=self.device)

        # loc_tとconf_t_labelに、
        # DBoxと正解アノテーションtargetsをmatchさせた結果を上書きする
        for idx in range(num_batch):  # ミニバッチでループ
            # 現在のミニバッチの正解アノテーション
            if targets[idx] is None:
                # 念のため None ガード
                conf_t_label[idx].zero_()
                loc_t[idx].zero_()
                continue

            if targets[idx].numel() == 0:
                # ★ 空GT（背景のみ画像）：match をスキップして背景ラベル固定
                conf_t_label[idx].zero_()
                loc_t[idx].zero_()
                continue

            truths = targets[idx][:, :-1].to(self.device)
            labels_raw = targets[idx][:, -1].to(self.device).long()
            dbox = dbox_list.to(self.device)

            # --- ラベルを 0..K-1 に正規化（match() 内の +1 と二重適用を防ぐ） ---
            num_classes = conf_data.size(2)   # 背景込みの総クラス数 = K+1
            K = num_classes - 1               # 既知クラス数（背景を除く）

            # 1..K → 0..K-1（-1） / 0..K-1 はそのまま / 範囲外は除外
            labels_0 = labels_raw.clone()
            # 1..K を検出して -1
            mask_1toK = (labels_0 >= 1) & (labels_0 <= K)
            labels_0[mask_1toK] = labels_0[mask_1toK] - 1

            # 0..K-1 に入っていないものは無効（truths と同じ行を落とす）
            valid = (labels_0 >= 0) & (labels_0 <= (K - 1))
            if not bool(valid.all()):
                truths = truths[valid]
                labels_0 = labels_0[valid]
                # すべて落ちたら背景のみ扱いにする
                if labels_0.numel() == 0:
                    conf_t_label[idx].zero_()
                    loc_t[idx].zero_()
                    continue

            variance = self.variances
            match(self.jaccard_thresh, truths, dbox, variance, labels_0, loc_t, conf_t_label, idx)

            # （任意の安全確認：背景0/前景1..K に収まっているか）
            assert conf_t_label[idx].min().item() >= 0 and conf_t_label[idx].max().item() <= K

        # ---------- 位置の損失 ----------
        pos_mask = conf_t_label > 0
        pos_idx = pos_mask.unsqueeze(pos_mask.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum') if loc_p.numel() else loc_p.sum()

        # ---------- クラス損失（標準手順 + optional label smoothing） ----------
        num_classes = conf_data.size(2)
        batch_conf = conf_data.view(-1, num_classes)

        # 全アンカーに対するCE（あとでHNM用に使う）※reduction=none
        loss_c = F.cross_entropy(
            batch_conf, conf_t_label.view(-1), reduction='none',
            label_smoothing=self.label_smoothing
        ).view(num_batch, -1)

        # Positive を 0 にして Negative のランキングを作る
        loss_c[pos_mask] = 0

        # Hard Negative Mining（負例の取りすぎを抑制）
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)

        num_pos = pos_mask.long().sum(1, keepdim=True)      # [B,1]
        num_dbox = conf_data.size(1)

        # 基本は 3:1
        num_neg = num_pos * self.negpos_ratio               # [B,1]

        # 1) 正例があるバッチ：上限を「1画像あたり max_neg_per_img」に制限
        max_neg_per_img = self.max_neg_per_img                                # ★推奨 32～128 で調整
        num_neg = torch.minimum(num_neg, torch.full_like(num_neg, max_neg_per_img))

        # 2) 正例がゼロの画像には、少数の負例だけを使う（例: 32/画像）
        zero_pos_mask = (num_pos == 0)
        if zero_pos_mask.any():
            num_neg = torch.where(zero_pos_mask, torch.full_like(num_neg, 32), num_neg)

        # 絶対上限：全アンカー数
        num_neg = torch.clamp(num_neg, max=num_dbox)

        neg_mask = idx_rank < num_neg.expand_as(idx_rank)

        # pos/neg 抽出
        pos_idx_mask = pos_mask.unsqueeze(2).expand_as(conf_data)
        neg_idx_mask = neg_mask.unsqueeze(2).expand_as(conf_data)

        conf_hnm = conf_data[(pos_idx_mask + neg_idx_mask).gt(0)].view(-1, num_classes)
        conf_t_label_hnm = conf_t_label[(pos_mask + neg_mask).gt(0)]

        # CE（選抜された分だけ）
        if conf_hnm.numel() == 0:
            loss_c_final = torch.tensor(0., device=self.device)
        else:
            loss_c_final = F.cross_entropy(conf_hnm, conf_t_label_hnm,
                                          reduction='sum', label_smoothing=0.05)

        # ===== 正規化 =====
        # 位置損失は従来通り「正例数」で割る
        N_pos = torch.clamp(num_pos.sum(), min=1).float()
        loss_l = loss_l / N_pos

        # 分母を状況に応じて変更：
        # - 正例があれば、典型SSD同様に N_pos（または N_pos* (1+negpos_ratio) でもOK）
        # - 正例ゼロのバッチは「選抜された負例数の合計」で割る（スケール破綻を防ぐ）
        selected_neg = neg_mask.long().sum().clamp(min=1).float()
        if (num_pos.sum() > 0):
            loss_c = loss_c_final / N_pos
        else:
            loss_c = loss_c_final / selected_neg

        return loss_l, loss_c   # ★ これを追加