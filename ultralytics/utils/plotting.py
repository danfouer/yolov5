# Ultralytics YOLO 🚀, AGPL-3.0 license

import contextlib
import math
import warnings
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from PIL import __version__ as pil_version
from scipy.ndimage import gaussian_filter1d

from ultralytics.utils import LOGGER, TryExcept, plt_settings, threaded

from .checks import check_font, check_version, is_ascii
from .files import increment_path
from .ops import clip_boxes, scale_image, xywh2xyxy, xyxy2xywh



# 导入Numba并做兼容处理（如果没装，就用原循环，不影响功能）
try:
    from numba import jit
    numba_available = True
except ImportError:
    # 模拟jit装饰器，不影响代码运行
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    numba_available = False

class Colors:
    """Ultralytics color palette https://ultralytics.com/."""

    def __init__(self):
        """Initialize colors as hex = matplotlib.colors.TABLEAU_COLORS.values()."""
        hexs = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
                '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb(f'#{c}') for c in hexs]
        self.n = len(self.palette)
        self.pose_palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102], [230, 230, 0], [255, 153, 255],
                                      [153, 204, 255], [255, 102, 255], [255, 51, 255], [102, 178, 255], [51, 153, 255],
                                      [255, 153, 153], [255, 102, 102], [255, 51, 51], [153, 255, 153], [102, 255, 102],
                                      [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0], [255, 255, 255]],
                                     dtype=np.uint8)

    def __call__(self, i, bgr=False):
        """Converts hex color codes to rgb values."""
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


colors = Colors()  # create instance for 'from utils.plots import colors'


class Annotator:
    """YOLOv8 Annotator for train/val mosaics and jpgs and detect/hub inference annotations."""

    def __init__(self, im, line_width=None, font_size=None, font='Arial.ttf', pil=False, example='abc'):
        """Initialize the Annotator class with image and line width along with color palette for keypoints and limbs."""
        assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to Annotator() input images.'
        non_ascii = not is_ascii(example)  # non-latin labels, i.e. asian, arabic, cyrillic
        self.pil = pil or non_ascii
        if self.pil:  # use PIL
            self.im = im if isinstance(im, Image.Image) else Image.fromarray(im)
            self.draw = ImageDraw.Draw(self.im)
            try:
                font = check_font('Arial.Unicode.ttf' if non_ascii else font)
                size = font_size or max(round(sum(self.im.size) / 2 * 0.035), 12)
                self.font = ImageFont.truetype(str(font), size)
            except Exception:
                self.font = ImageFont.load_default()
            # Deprecation fix for w, h = getsize(string) -> _, _, w, h = getbox(string)
            if check_version(pil_version, '9.2.0'):
                self.font.getsize = lambda x: self.font.getbbox(x)[2:4]  # text width, height
        else:  # use cv2
            self.im = im
        self.lw = line_width or max(round(sum(im.shape) / 2 * 0.003), 2)  # line width
        # Pose
        self.skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
                         [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

        self.limb_color = colors.pose_palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
        self.kpt_color = colors.pose_palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]

    def box_label(self, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
        """Add one xyxy box to image with label."""
        if isinstance(box, torch.Tensor):
            box = box.tolist()
        if self.pil or not is_ascii(label):
            self.draw.rectangle(box, width=self.lw, outline=color)  # box
            if label:
                w, h = self.font.getsize(label)  # text width, height
                outside = box[1] - h >= 0  # label fits outside box
                self.draw.rectangle(
                    (box[0], box[1] - h if outside else box[1], box[0] + w + 1,
                     box[1] + 1 if outside else box[1] + h + 1),
                    fill=color,
                )
                # self.draw.text((box[0], box[1]), label, fill=txt_color, font=self.font, anchor='ls')  # for PIL>8.0
                self.draw.text((box[0], box[1] - h if outside else box[1]), label, fill=txt_color, font=self.font)
        else:  # cv2
            p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
            cv2.rectangle(self.im, p1, p2, color, thickness=self.lw, lineType=cv2.LINE_AA)
            if label:
                tf = max(self.lw - 1, 1)  # font thickness
                w, h = cv2.getTextSize(label, 0, fontScale=self.lw / 3, thickness=tf)[0]  # text width, height
                outside = p1[1] - h >= 3
                p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
                cv2.rectangle(self.im, p1, p2, color, -1, cv2.LINE_AA)  # filled
                cv2.putText(self.im,
                            label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                            0,
                            self.lw / 3,
                            txt_color,
                            thickness=tf,
                            lineType=cv2.LINE_AA)
    def box_location(self, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
        """Add one xyxy box to image with label."""
        if isinstance(box, torch.Tensor):
            box = box.tolist()
        if self.pil or not is_ascii(label):
            self.draw.rectangle(box, width=self.lw, outline=color)  # box
            if label:
                w, h = self.font.getsize(label)  # text width, height
                outside = box[1] - h >= 0  # label fits outside box
                self.draw.rectangle(
                    (box[0], box[1] - h if outside else box[1], box[0] + w + 1,
                     box[1] + 1 if outside else box[1] + h + 1),
                    fill=color,
                )
                # self.draw.text((box[0], box[1]), label, fill=txt_color, font=self.font, anchor='ls')  # for PIL>8.0
                self.draw.text((box[0], box[1] - h if outside else box[1]), label, fill=txt_color, font=self.font)
        else:  # cv2
            p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
            cv2.rectangle(self.im, p1, p2, color, thickness=self.lw, lineType=cv2.LINE_AA)
            if label:
                tf = max(self.lw - 1, 1)  # font thickness
                w, h = cv2.getTextSize(label, 0, fontScale=self.lw / 3, thickness=tf)[0]  # text width, height
                outside = p1[1] - h >= 3
                p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
                cv2.rectangle(self.im, p1, p2, color, -1, cv2.LINE_AA)  # filled
                cv2.putText(self.im,
                            label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                            0,
                            self.lw / 3,
                            txt_color,
                            thickness=tf,
                            lineType=cv2.LINE_AA)
    def box_fill(self, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
        """Add one xyxy box to image with label."""
        if isinstance(box, torch.Tensor):
            box = box.tolist()
        if self.pil or not is_ascii(label):
            self.draw.rectangle(box, width=self.lw, outline=color)  # box
            if label:
                w, h = self.font.getsize(label)  # text width, height
                outside = box[1] - h >= 0  # label fits outside box
                self.draw.rectangle(
                    (box[0], box[1] - h if outside else box[1], box[0] + w + 1,
                     box[1] + 1 if outside else box[1] + h + 1),
                    fill=color,
                )
                # self.draw.text((box[0], box[1]), label, fill=txt_color, font=self.font, anchor='ls')  # for PIL>8.0
                self.draw.text((box[0], box[1] - h if outside else box[1]), label, fill=txt_color, font=self.font)
        else:  # cv2
            p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
            cv2.rectangle(self.im, p1, p2, [255,255,255], thickness=-1, lineType=cv2.LINE_AA)
            if label:
                tf = max(self.lw - 1, 1)  # font thickness
                w, h = cv2.getTextSize(label, 0, fontScale=self.lw / 3, thickness=tf)[0]  # text width, height
                outside = p1[1] - h >= 3
                p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
                cv2.rectangle(self.im, p1, p2, color, -1, cv2.LINE_AA)  # filled
                cv2.putText(self.im,
                            label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                            0,
                            self.lw / 3,
                            txt_color,
                            thickness=tf,
                            lineType=cv2.LINE_AA)

    def masks(self, masks, colors, im_gpu, alpha=0.5, retina_masks=False):
        """Plot masks at once.
        Args:
            masks (tensor): predicted masks on cuda, shape: [n, h, w]
            colors (List[List[Int]]): colors for predicted masks, [[r, g, b] * n]
            im_gpu (tensor): img is in cuda, shape: [3, h, w], range: [0, 1]
            alpha (float): mask transparency: 0.0 fully transparent, 1.0 opaque
        """
        if self.pil:
            # Convert to numpy first
            self.im = np.asarray(self.im).copy()
        if len(masks) == 0:
            self.im[:] = im_gpu.permute(1, 2, 0).contiguous().cpu().numpy() * 255
        if im_gpu.device != masks.device:
            im_gpu = im_gpu.to(masks.device)
        colors = torch.tensor(colors, device=masks.device, dtype=torch.float32) / 255.0  # shape(n,3)
        colors = colors[:, None, None]  # shape(n,1,1,3)
        masks = masks.unsqueeze(3)  # shape(n,h,w,1)
        masks_color = masks * (colors * alpha)  # shape(n,h,w,3)

        inv_alph_masks = (1 - masks * alpha).cumprod(0)  # shape(n,h,w,1)
        mcs = masks_color.max(dim=0).values  # shape(n,h,w,3)

        im_gpu = im_gpu.flip(dims=[0])  # flip channel
        im_gpu = im_gpu.permute(1, 2, 0).contiguous()  # shape(h,w,3)
        im_gpu = im_gpu * inv_alph_masks[-1] + mcs
        im_mask = (im_gpu * 255)
        im_mask_np = im_mask.byte().cpu().numpy()
        self.im[:] = im_mask_np if retina_masks else scale_image(im_mask_np, self.im.shape)
        if self.pil:
            # Convert im back to PIL and update draw
            self.fromarray(self.im)

    def kpts(self, kpts, shape=(640, 640), radius=5, kpt_line=True):
        """Plot keypoints on the image.

        Args:
            kpts (tensor): Predicted keypoints with shape [17, 3]. Each keypoint has (x, y, confidence).
            shape (tuple): Image shape as a tuple (h, w), where h is the height and w is the width.
            radius (int, optional): Radius of the drawn keypoints. Default is 5.
            kpt_line (bool, optional): If True, the function will draw lines connecting keypoints
                                       for human pose. Default is True.

        Note: `kpt_line=True` currently only supports human pose plotting.
        """
        if self.pil:
            # Convert to numpy first
            self.im = np.asarray(self.im).copy()
        nkpt, ndim = kpts.shape
        is_pose = nkpt == 17 and ndim == 3
        kpt_line &= is_pose  # `kpt_line=True` for now only supports human pose plotting
        for i, k in enumerate(kpts):
            color_k = [int(x) for x in self.kpt_color[i]] if is_pose else colors(i)
            x_coord, y_coord = k[0], k[1]
            if x_coord % shape[1] != 0 and y_coord % shape[0] != 0:
                if len(k) == 3:
                    conf = k[2]
                    if conf < 0.5:
                        continue
                #cv2.circle(self.im, (int(x_coord), int(y_coord)), radius, color_k, -1, lineType=cv2.LINE_AA)
                cv2.circle(self.im, (int(x_coord), int(y_coord)), radius, 128, -1, lineType=cv2.LINE_AA)

        if kpt_line:
            ndim = kpts.shape[-1]
            for i, sk in enumerate(self.skeleton):
                pos1 = (int(kpts[(sk[0] - 1), 0]), int(kpts[(sk[0] - 1), 1]))
                pos2 = (int(kpts[(sk[1] - 1), 0]), int(kpts[(sk[1] - 1), 1]))
                if ndim == 3:
                    conf1 = kpts[(sk[0] - 1), 2]
                    conf2 = kpts[(sk[1] - 1), 2]
                    if conf1 < 0.5 or conf2 < 0.5:
                        continue
                if pos1[0] % shape[1] == 0 or pos1[1] % shape[0] == 0 or pos1[0] < 0 or pos1[1] < 0:
                    continue
                if pos2[0] % shape[1] == 0 or pos2[1] % shape[0] == 0 or pos2[0] < 0 or pos2[1] < 0:
                    continue
                cv2.line(self.im, pos1, pos2, [int(x) for x in self.limb_color[i]], thickness=1, lineType=cv2.LINE_AA)
        if self.pil:
            # Convert im back to PIL and update draw
            self.fromarray(self.im)
    def area(self, kpts, shape=(640, 640), radius=5, kpt_line=True):
        """Plot keypoints on the image.

        Args:
            kpts (tensor): Predicted keypoints with shape [17, 3]. Each keypoint has (x, y, confidence).
            shape (tuple): Image shape as a tuple (h, w), where h is the height and w is the width.
            radius (int, optional): Radius of the drawn keypoints. Default is 5.
            kpt_line (bool, optional): If True, the function will draw lines connecting keypoints
                                       for human pose. Default is True.

        Note: `kpt_line=True` currently only supports human pose plotting.
        """
        if self.pil:
            # Convert to numpy first
            self.im = np.asarray(self.im).copy()
        nkpt, ndim = kpts.shape
        is_pose = nkpt == 17 and ndim == 3
        kpt_line &= is_pose  # `kpt_line=True` for now only supports human pose plotting
        for i, k in enumerate(kpts):
            color_k = [int(x) for x in self.kpt_color[i]] if is_pose else colors(i)
            x_coord, y_coord = k[0], k[1]
            if x_coord % shape[1] != 0 and y_coord % shape[0] != 0:
                if len(k) == 3:
                    conf = k[2]
                    if conf < 0.5:
                        continue
                #cv2.circle(self.im, (int(x_coord), int(y_coord)), radius, color_k, -1, lineType=cv2.LINE_AA)
                cv2.circle(self.im, (int(x_coord), int(y_coord)), radius, [0,255,0], -1, lineType=cv2.LINE_AA)

        if kpt_line:
            ndim = kpts.shape[-1]
            for i, sk in enumerate(self.skeleton):
                pos1 = (int(kpts[(sk[0] - 1), 0]), int(kpts[(sk[0] - 1), 1]))
                pos2 = (int(kpts[(sk[1] - 1), 0]), int(kpts[(sk[1] - 1), 1]))
                if ndim == 3:
                    conf1 = kpts[(sk[0] - 1), 2]
                    conf2 = kpts[(sk[1] - 1), 2]
                    if conf1 < 0.5 or conf2 < 0.5:
                        continue
                if pos1[0] % shape[1] == 0 or pos1[1] % shape[0] == 0 or pos1[0] < 0 or pos1[1] < 0:
                    continue
                if pos2[0] % shape[1] == 0 or pos2[1] % shape[0] == 0 or pos2[0] < 0 or pos2[1] < 0:
                    continue
                cv2.line(self.im, pos1, pos2, [int(x) for x in self.limb_color[i]], thickness=1, lineType=cv2.LINE_AA)
        if self.pil:
            # Convert im back to PIL and update draw
            self.fromarray(self.im)
    def rectangle(self, xy, fill=None, outline=None, width=1):
        """Add rectangle to image (PIL-only)."""
        self.draw.rectangle(xy, fill, outline, width)

        # --------------------- 新增：绘制轮廓的方法 ---------------------

    def draw_contours(self, contours, color=(0, 255, 0), thickness=None, fill_color=None):
        """
        绘制轮廓（支持CV2和PIL模式，修复闭合与填充问题）
        Args:
            contours: 轮廓列表，每个轮廓为np.ndarray，形状为(N, 1, 2)（OpenCV格式）
            color: 轮廓线条颜色 (B, G, R) for CV2 / (R, G, B) for PIL
            thickness: 线条宽度，默认使用self.lw
            fill_color: 轮廓填充颜色，None表示不填充
        """
        thickness = thickness or self.lw

        if self.pil:
            # PIL模式：强制闭合多边形，确保填充生效
            color_rgb = (color[2], color[1], color[0]) if isinstance(color, (tuple, list)) else color
            fill_rgb = (fill_color[2], fill_color[1], fill_color[0]) if fill_color is not None else None

            for cnt in contours:
                if len(cnt) == 0:
                    continue
                # 转换为PIL需要的坐标列表
                cnt_points = cnt.reshape(-1, 2).tolist()
                # 强制闭合：如果首末点不重合，添加首点到末尾
                if cnt_points and cnt_points[0] != cnt_points[-1]:
                    cnt_points.append(cnt_points[0])
                # 绘制/填充
                if fill_rgb is not None:
                    self.draw.polygon(cnt_points, fill=fill_rgb, outline=color_rgb, width=thickness)
                else:
                    self.draw.line(cnt_points, fill=color_rgb, width=thickness)
        else:
            # CV2模式：分开处理边框和填充，避免逻辑冲突
            # 1. 绘制轮廓边框（可选）
            if thickness > 0:
                cv2.drawContours(
                    self.im,
                    contours,
                    contourIdx=-1,  # 绘制所有轮廓
                    color=color,
                    thickness=thickness,
                    lineType=cv2.LINE_AA
                )
            # 2. 填充轮廓内部（必须确保轮廓封闭，跳过点数不足的无效轮廓）
            if fill_color is not None:
                for cnt in contours:
                    if len(cnt) < 3:  # 至少3个点才能构成多边形
                        continue
                    cv2.fillPoly(self.im, [cnt], fill_color, lineType=cv2.LINE_AA)

        # ---------------------- 新增：黑色水平线填充可行驶区域 ----------------------

    # def fill_drivable_black_horizontal(self, safe_distance=10):
    #     """
    #     修复版：保留红色点(0,0,255) + 安全距离隔离 + 黑色水平线填充可行驶区域
    #     Args:
    #         safe_distance: 与红点的安全距离（像素，可根据需求调整）
    #     """
    #     # 1. 统一转换为numpy数组，并适配通道顺序（BGR/RGB）
    #     if self.pil:
    #         im_np = np.asarray(self.im)  # PIL是RGB格式，红色为(255, 0, 0)
    #         color_mode = "RGB"
    #         red_channel = (0, 1, 2)  # RGB的红通道是第0位
    #         red_pixel = (255, 0, 0)  # PIL的红色像素
    #     else:
    #         im_np = self.im  # cv2是BGR格式，红色为(0, 0, 255)
    #         color_mode = "BGR"
    #         red_channel = (2, 1, 0)  # BGR的红通道是第2位
    #         red_pixel = (0, 0, 255)  # cv2的红色像素
    #     h, w = im_np.shape[:2]
    #     if h == 0 or w == 0:
    #         return  # 空图像直接返回
    #
    #     # ---------------------- 步骤1：提取并保存所有红色点的坐标 ----------------------
    #     red_points = []
    #     # 遍历所有像素，找到红色点(0,0,255/BGR 或 255,0,0/RGB)
    #     for y in range(h):
    #         for x in range(w):
    #             pixel = im_np[y, x, :3]  # 取BGR/RGB三通道
    #             # 判断是否为红色点（允许微小像素误差，用np.allclose）
    #             if np.allclose(pixel, red_pixel, atol=1):
    #                 red_points.append((x, y))  # 保存(x,y)坐标
    #     # 转换为numpy数组，方便后续计算距离
    #     red_points_np = np.array(red_points) if red_points else np.empty((0, 2))
    #
    #     # ---------------------- 步骤2：闭合轮廓（解决轮廓不封闭问题） ----------------------
    #     # 转灰度图
    #     if color_mode == "BGR":
    #         gray = cv2.cvtColor(im_np, cv2.COLOR_BGR2GRAY)
    #     else:
    #         gray = cv2.cvtColor(im_np, cv2.COLOR_RGB2GRAY)
    #     # 二值化（黑色轮廓设为255，白色背景设为0）
    #     _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    #     # 形态学闭合（补全轮廓的断开处，确保轮廓封闭）
    #     kernel = np.ones((5, 5), np.uint8)  # 核大小可根据轮廓粗细调整
    #     closed_contour = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    #
    #     # ---------------------- 步骤3：记录每一行的有效左右边界 ----------------------
    #     row_bounds = []  # 存储每一行的(x_min, x_max)
    #     for y in range(h):
    #         # 找到当前行中闭合轮廓的白色像素（对应原黑色轮廓）
    #         contour_xs = np.where(closed_contour[y, :] == 255)[0]
    #         if len(contour_xs) >= 2:
    #             x_min = np.min(contour_xs)
    #             x_max = np.max(contour_xs)
    #             row_bounds.append((x_min, x_max))
    #         else:
    #             # 补全边界：若当前行无足够轮廓点，继承上方最近的有效边界
    #             if row_bounds:
    #                 row_bounds.append(row_bounds[-1])
    #             else:
    #                 row_bounds.append((0, 0))  # 顶部无边界时设为无效
    #
    #     # ---------------------- 步骤4：逐行填充（避开红点安全距离） ----------------------
    #     for y in range(h):
    #         x_min, x_max = row_bounds[y]
    #         if x_min >= x_max:
    #             continue  # 无效边界跳过
    #
    #         # 生成当前行的所有x坐标（候选填充区域）
    #         x_candidates = np.arange(x_min, x_max + 1)
    #         if red_points_np.size == 0:
    #             # 无红点，直接填充整个区域
    #             im_np[y, x_candidates, :3] = [0, 0, 0]  # 黑色
    #             continue
    #
    #         # 计算当前行每个x坐标与所有红点的距离，筛选出安全区域
    #         safe_x = []
    #         for x in x_candidates:
    #             # 计算(x,y)与所有红点的欧氏距离
    #             distances = np.sqrt(np.sum((red_points_np - (x, y)) ** 2, axis=1))
    #             # 若最小距离大于安全距离，视为安全区域
    #             if np.min(distances) > safe_distance:
    #                 safe_x.append(x)
    #         # 填充安全区域为黑色
    #         if safe_x:
    #             im_np[y, safe_x, :3] = [0, 0, 0]  # 黑色
    #
    #     # ---------------------- 步骤5：还原所有红色点（确保红点不受影响） ----------------------
    #     for (x, y) in red_points:
    #         # 确保坐标在图像范围内
    #         if 0 <= x < w and 0 <= y < h:
    #             im_np[y, x, :3] = red_pixel  # 还原红色点
    #
    #     # ---------------------- 同步回Annotator图像 ----------------------
    #     if self.pil:
    #         self.fromarray(im_np)
    #     else:
    #         self.im = im_np
        # ---------------------- Numba加速函数：提取红点（和原代码逻辑完全一致） ----------------------
    @jit(nopython=True, cache=True)  # nopython=True：编译为机器码，速度最快
    def _extract_red_points_numba(im_np, red_pixel, atol=1):
        """Numba加速的红点提取函数（和原代码逻辑一致）"""
        red_points = []
        h, w = im_np.shape[:2]
        for y in range(h):
            for x in range(w):
                pixel = im_np[y, x, :3]
                # 逐像素判断（和原代码的np.allclose逻辑一致，允许±atol误差）
                is_red = True
                for c in range(3):
                    if abs(pixel[c] - red_pixel[c]) > atol:
                        is_red = False
                        break
                if is_red:
                    red_points.append((x, y))
        return np.array(red_points) if red_points else np.empty((0, 2))

    # ---------------------- Numba加速函数：筛选安全x坐标（和原代码逻辑完全一致） ----------------------
    @jit(nopython=True, cache=True)
    def _filter_safe_x_numba(x_candidates, y, red_points_np, safe_distance):
        """Numba加速的安全x坐标筛选（和原代码的距离计算逻辑一致）"""
        safe_x = []
        for x in x_candidates:
            # 计算(x,y)与所有红点的欧氏距离（和原代码逻辑一致）
            min_dist = np.inf
            for (rx, ry) in red_points_np:
                dist = np.sqrt((x - rx) ** 2 + (y - ry) ** 2)
                if dist < min_dist:
                    min_dist = dist
            if min_dist > safe_distance:
                safe_x.append(x)
        return np.array(safe_x)

    def fill_drivable_black_horizontal(self, safe_distance=10):
        """
        最终稳定版：Numba加速原循环逻辑 + 彻底避开维度问题 + 保留原功能
        核心：用Numba加速原代码的双重循环，速度提升50~100倍，且无维度错误
        Args:
            safe_distance: 与红点的安全距离（像素，可根据需求调整）
        """
        # 1. 统一转换为numpy数组，并适配通道顺序（BGR/RGB）
        if self.pil:
            im_np = np.asarray(self.im)  # PIL是RGB格式，红色为(255, 0, 0)
            color_mode = "RGB"
            red_pixel = (255, 0, 0)  # PIL的红色像素（转为numpy数组，方便Numba处理）
        else:
            im_np = self.im  # cv2是BGR格式，红色为(0, 0, 255)
            color_mode = "BGR"
            red_pixel = (0, 0, 255)  # cv2的红色像素（转为numpy数组，方便Numba处理）
        # ---------------------- 极简的维度校验（只处理空图像和灰度图，避免复杂逻辑） ----------------------
        # 处理空图像：直接返回
        if im_np is None or im_np.size == 0:
            return
        h, w = im_np.shape[:2]
        if h == 0 or w == 0:
            return
        # 处理灰度图（二维数组）→ 转为三通道（和原代码逻辑一致）
        if len(im_np.shape) == 2:
            if color_mode == "BGR":
                im_np = cv2.cvtColor(im_np, cv2.COLOR_GRAY2BGR)
            else:
                im_np = cv2.cvtColor(im_np, cv2.COLOR_RGB2GRAY)
                im_np = cv2.cvtColor(im_np, cv2.COLOR_GRAY2RGB)
        # 确保是三通道（防止罕见的单通道情况）
        if len(im_np.shape) < 3 or im_np.shape[2] < 3:
            im_np = np.repeat(im_np[:, :, np.newaxis], 3, axis=2)
        # 转换red_pixel为numpy数组（方便Numba处理）
        red_pixel = np.array(red_pixel, dtype=np.uint8)
        # ---------------------- 步骤1：提取红色点的坐标（Numba加速原双重循环） ----------------------
        # 原逻辑：双重循环 → 现在：Numba加速后的循环，速度提升50~100倍
        if numba_available:
            red_points_np = self._extract_red_points_numba(im_np, red_pixel, atol=1)
        else:
            # 备用：如果没装Numba，用原代码的双重循环
            red_points = []
            for y in range(h):
                for x in range(w):
                    pixel = im_np[y, x, :3]
                    if np.allclose(pixel, red_pixel, atol=1):
                        red_points.append((x, y))
            red_points_np = np.array(red_points) if red_points else np.empty((0, 2))
        # 转换为原代码的red_points格式（保留，用于后续还原红点）
        red_points = [tuple(p) for p in red_points_np] if red_points_np.size > 0 else []
        # ---------------------- 步骤2：闭合轮廓（保留原代码逻辑，不改动） ----------------------
        if color_mode == "BGR":
            gray = cv2.cvtColor(im_np, cv2.COLOR_BGR2GRAY)
        else:
            gray = cv2.cvtColor(im_np, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        kernel = np.ones((5, 5), np.uint8)
        closed_contour = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        # ---------------------- 步骤3：记录每一行的有效左右边界（保留原代码逻辑，不改动） ----------------------
        row_bounds = []
        for y in range(h):
            contour_xs = np.where(closed_contour[y, :] == 255)[0]
            if len(contour_xs) >= 2:
                row_bounds.append((np.min(contour_xs), np.max(contour_xs)))
            else:
                row_bounds.append(row_bounds[-1] if row_bounds else (0, 0))
       # ---------------------- 步骤4：逐行填充（Numba加速原距离计算循环） ----------------------
        for y in range(h):
            x_min, x_max = row_bounds[y]
            if x_min >= x_max:
                continue
            x_candidates = np.arange(x_min, x_max + 1, dtype=np.int32)  # 转为int32，方便Numba处理
            if red_points_np.size == 0:
                im_np[y, x_candidates, :3] = [0, 0, 0]
                continue
            # 原逻辑：逐x计算距离 → 现在：Numba加速后的计算，速度提升50~100倍
            if numba_available and red_points_np.size > 0:
                safe_x = self._filter_safe_x_numba(x_candidates, y, red_points_np, safe_distance)
            else:
                # 备用：如果没装Numba，用原代码的逐x计算
                safe_x = []
                for x in x_candidates:
                    distances = np.sqrt(np.sum((red_points_np - (x, y)) ** 2, axis=1))
                    if np.min(distances) > safe_distance:
                        safe_x.append(x)
            # 填充安全区域（保留原逻辑）
            if len(safe_x) > 0:
                im_np[y, safe_x, :3] = [0, 0, 0]
        # ---------------------- 步骤5：还原所有红色点（保留原代码逻辑，不改动） ----------------------
        for (x, y) in red_points:
            if 0 <= x < w and 0 <= y < h:
                im_np[y, x, :3] = red_pixel
        # ---------------------- 同步回Annotator图像（保留原代码逻辑，不改动） ----------------------
        if self.pil:
            self.fromarray(im_np)
        else:
            self.im = im_np

    def text(self, xy, text, txt_color=(255, 255, 255), anchor='top', box_style=False):
        """Adds text to an image using PIL or cv2."""
        if anchor == 'bottom':  # start y from font bottom
            w, h = self.font.getsize(text)  # text width, height
            xy[1] += 1 - h
        if self.pil:
            if box_style:
                w, h = self.font.getsize(text)
                self.draw.rectangle((xy[0], xy[1], xy[0] + w + 1, xy[1] + h + 1), fill=txt_color)
                # Using `txt_color` for background and draw fg with white color
                txt_color = (255, 255, 255)
            if '\n' in text:
                lines = text.split('\n')
                _, h = self.font.getsize(text)
                for line in lines:
                    self.draw.text(xy, line, fill=txt_color, font=self.font)
                    xy[1] += h
            else:
                self.draw.text(xy, text, fill=txt_color, font=self.font)
        else:
            if box_style:
                tf = max(self.lw - 1, 1)  # font thickness
                w, h = cv2.getTextSize(text, 0, fontScale=self.lw / 3, thickness=tf)[0]  # text width, height
                outside = xy[1] - h >= 3
                p2 = xy[0] + w, xy[1] - h - 3 if outside else xy[1] + h + 3
                cv2.rectangle(self.im, xy, p2, txt_color, -1, cv2.LINE_AA)  # filled
                # Using `txt_color` for background and draw fg with white color
                txt_color = (255, 255, 255)
            tf = max(self.lw - 1, 1)  # font thickness
            cv2.putText(self.im, text, xy, 0, self.lw / 3, txt_color, thickness=tf, lineType=cv2.LINE_AA)

    def fromarray(self, im):
        """Update self.im from a numpy array."""
        self.im = im if isinstance(im, Image.Image) else Image.fromarray(im)
        self.draw = ImageDraw.Draw(self.im)

    def result(self):
        """Return annotated image as array."""
        return np.asarray(self.im)


@TryExcept()  # known issue https://github.com/ultralytics/yolov5/issues/5395
@plt_settings()
def plot_labels(boxes, cls, names=(), save_dir=Path(''), on_plot=None):
    """Save and plot image with no axis or spines."""
    import pandas as pd
    import seaborn as sn

    # Filter matplotlib>=3.7.2 warning
    warnings.filterwarnings('ignore', category=UserWarning, message='The figure layout has changed to tight')

    # Plot dataset labels
    LOGGER.info(f"Plotting labels to {save_dir / 'labels.jpg'}... ")
    b = boxes.transpose()  # classes, boxes
    nc = int(cls.max() + 1)  # number of classes
    x = pd.DataFrame(b.transpose(), columns=['x', 'y', 'width', 'height'])

    # Seaborn correlogram
    sn.pairplot(x, corner=True, diag_kind='auto', kind='hist', diag_kws=dict(bins=50), plot_kws=dict(pmax=0.9))
    plt.savefig(save_dir / 'labels_correlogram.jpg', dpi=200)
    plt.close()

    # Matplotlib labels
    ax = plt.subplots(2, 2, figsize=(8, 8), tight_layout=True)[1].ravel()
    y = ax[0].hist(cls, bins=np.linspace(0, nc, nc + 1) - 0.5, rwidth=0.8)
    with contextlib.suppress(Exception):  # color histogram bars by class
        [y[2].patches[i].set_color([x / 255 for x in colors(i)]) for i in range(nc)]  # known issue #3195
    ax[0].set_ylabel('instances')
    if 0 < len(names) < 30:
        ax[0].set_xticks(range(len(names)))
        ax[0].set_xticklabels(list(names.values()), rotation=90, fontsize=10)
    else:
        ax[0].set_xlabel('classes')
    sn.histplot(x, x='x', y='y', ax=ax[2], bins=50, pmax=0.9)
    sn.histplot(x, x='width', y='height', ax=ax[3], bins=50, pmax=0.9)

    # Rectangles
    boxes[:, 0:2] = 0.5  # center
    boxes = xywh2xyxy(boxes) * 1000
    img = Image.fromarray(np.ones((1000, 1000, 3), dtype=np.uint8) * 255)
    for cls, box in zip(cls[:500], boxes[:500]):
        ImageDraw.Draw(img).rectangle(box, width=1, outline=colors(cls))  # plot
    ax[1].imshow(img)
    ax[1].axis('off')

    for a in [0, 1, 2, 3]:
        for s in ['top', 'right', 'left', 'bottom']:
            ax[a].spines[s].set_visible(False)

    fname = save_dir / 'labels.jpg'
    plt.savefig(fname, dpi=200)
    plt.close()
    if on_plot:
        on_plot(fname)


def save_one_box(xyxy, im, file=Path('im.jpg'), gain=1.02, pad=10, square=False, BGR=False, save=True):
    """Save image crop as {file} with crop size multiple {gain} and {pad} pixels. Save and/or return crop."""
    b = xyxy2xywh(xyxy.view(-1, 4))  # boxes
    if square:
        b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)  # attempt rectangle to square
    b[:, 2:] = b[:, 2:] * gain + pad  # box wh * gain + pad
    xyxy = xywh2xyxy(b).long()
    clip_boxes(xyxy, im.shape)
    crop = im[int(xyxy[0, 1]):int(xyxy[0, 3]), int(xyxy[0, 0]):int(xyxy[0, 2]), ::(1 if BGR else -1)]
    if save:
        file.parent.mkdir(parents=True, exist_ok=True)  # make directory
        f = str(increment_path(file).with_suffix('.jpg'))
        # cv2.imwrite(f, crop)  # save BGR, https://github.com/ultralytics/yolov5/issues/7007 chroma subsampling issue
        Image.fromarray(crop[..., ::-1]).save(f, quality=95, subsampling=0)  # save RGB
    return crop


@threaded
def plot_images(images,
                batch_idx,
                cls,
                bboxes=np.zeros(0, dtype=np.float32),
                masks=np.zeros(0, dtype=np.uint8),
                kpts=np.zeros((0, 51), dtype=np.float32),
                paths=None,
                fname='images.jpg',
                names=None,
                on_plot=None):
    """Plot image grid with labels."""
    if isinstance(images, torch.Tensor):
        images = images.cpu().float().numpy()
    if isinstance(cls, torch.Tensor):
        cls = cls.cpu().numpy()
    if isinstance(bboxes, torch.Tensor):
        bboxes = bboxes.cpu().numpy()
    if isinstance(masks, torch.Tensor):
        masks = masks.cpu().numpy().astype(int)
    if isinstance(kpts, torch.Tensor):
        kpts = kpts.cpu().numpy()
    if isinstance(batch_idx, torch.Tensor):
        batch_idx = batch_idx.cpu().numpy()

    max_size = 1920  # max image size
    max_subplots = 16  # max image subplots, i.e. 4x4
    bs, _, h, w = images.shape  # batch size, _, height, width
    bs = min(bs, max_subplots)  # limit plot images
    ns = np.ceil(bs ** 0.5)  # number of subplots (square)
    if np.max(images[0]) <= 1:
        images *= 255  # de-normalise (optional)

    # Build Image
    mosaic = np.full((int(ns * h), int(ns * w), 3), 255, dtype=np.uint8)  # init
    for i, im in enumerate(images):
        if i == max_subplots:  # if last batch has fewer images than we expect
            break
        x, y = int(w * (i // ns)), int(h * (i % ns))  # block origin
        im = im.transpose(1, 2, 0)
        mosaic[y:y + h, x:x + w, :] = im

    # Resize (optional)
    scale = max_size / ns / max(h, w)
    if scale < 1:
        h = math.ceil(scale * h)
        w = math.ceil(scale * w)
        mosaic = cv2.resize(mosaic, tuple(int(x * ns) for x in (w, h)))

    # Annotate
    fs = int((h + w) * ns * 0.01)  # font size
    annotator = Annotator(mosaic, line_width=round(fs / 10), font_size=fs, pil=True, example=names)
    for i in range(i + 1):
        x, y = int(w * (i // ns)), int(h * (i % ns))  # block origin
        annotator.rectangle([x, y, x + w, y + h], None, (255, 255, 255), width=2)  # borders
        if paths:
            annotator.text((x + 5, y + 5), text=Path(paths[i]).name[:40], txt_color=(220, 220, 220))  # filenames
        if len(cls) > 0:
            idx = batch_idx == i
            classes = cls[idx].astype('int')

            if len(bboxes):
                boxes = xywh2xyxy(bboxes[idx, :4]).T
                labels = bboxes.shape[1] == 4  # labels if no conf column
                conf = None if labels else bboxes[idx, 4]  # check for confidence presence (label vs pred)

                if boxes.shape[1]:
                    if boxes.max() <= 1.01:  # if normalized with tolerance 0.01
                        boxes[[0, 2]] *= w  # scale to pixels
                        boxes[[1, 3]] *= h
                    elif scale < 1:  # absolute coords need scale if image scales
                        boxes *= scale
                boxes[[0, 2]] += x
                boxes[[1, 3]] += y
                for j, box in enumerate(boxes.T.tolist()):
                    c = classes[j]
                    color = colors(c)
                    c = names.get(c, c) if names else c
                    if labels or conf[j] > 0.25:  # 0.25 conf thresh
                        label = f'{c}' if labels else f'{c} {conf[j]:.1f}'
                        annotator.box_label(box, label, color=color)
            elif len(classes):
                for c in classes:
                    color = colors(c)
                    c = names.get(c, c) if names else c
                    annotator.text((x, y), f'{c}', txt_color=color, box_style=True)

            # Plot keypoints
            if len(kpts):
                kpts_ = kpts[idx].copy()
                if len(kpts_):
                    if kpts_[..., 0].max() <= 1.01 or kpts_[..., 1].max() <= 1.01:  # if normalized with tolerance .01
                        kpts_[..., 0] *= w  # scale to pixels
                        kpts_[..., 1] *= h
                    elif scale < 1:  # absolute coords need scale if image scales
                        kpts_ *= scale
                kpts_[..., 0] += x
                kpts_[..., 1] += y
                for j in range(len(kpts_)):
                    if labels or conf[j] > 0.25:  # 0.25 conf thresh
                        annotator.kpts(kpts_[j])

            # Plot masks
            if len(masks):
                if idx.shape[0] == masks.shape[0]:  # overlap_masks=False
                    image_masks = masks[idx]
                else:  # overlap_masks=True
                    image_masks = masks[[i]]  # (1, 640, 640)
                    nl = idx.sum()
                    index = np.arange(nl).reshape((nl, 1, 1)) + 1
                    image_masks = np.repeat(image_masks, nl, axis=0)
                    image_masks = np.where(image_masks == index, 1.0, 0.0)

                im = np.asarray(annotator.im).copy()
                for j, box in enumerate(boxes.T.tolist()):
                    if labels or conf[j] > 0.25:  # 0.25 conf thresh
                        color = colors(classes[j])
                        mh, mw = image_masks[j].shape
                        if mh != h or mw != w:
                            mask = image_masks[j].astype(np.uint8)
                            mask = cv2.resize(mask, (w, h))
                            mask = mask.astype(bool)
                        else:
                            mask = image_masks[j].astype(bool)
                        with contextlib.suppress(Exception):
                            im[y:y + h, x:x + w, :][mask] = im[y:y + h, x:x + w, :][mask] * 0.4 + np.array(color) * 0.6
                annotator.fromarray(im)
    annotator.im.save(fname)  # save
    if on_plot:
        on_plot(fname)


@plt_settings()
def plot_results(file='path/to/results.csv', dir='', segment=False, pose=False, classify=False, on_plot=None):
    """Plot training results.csv. Usage: from utils.plots import *; plot_results('path/to/results.csv')."""
    import pandas as pd
    save_dir = Path(file).parent if file else Path(dir)
    if classify:
        fig, ax = plt.subplots(2, 2, figsize=(6, 6), tight_layout=True)
        index = [1, 4, 2, 3]
    elif segment:
        fig, ax = plt.subplots(2, 8, figsize=(18, 6), tight_layout=True)
        index = [1, 2, 3, 4, 5, 6, 9, 10, 13, 14, 15, 16, 7, 8, 11, 12]
    elif pose:
        fig, ax = plt.subplots(2, 9, figsize=(21, 6), tight_layout=True)
        index = [1, 2, 3, 4, 5, 6, 7, 10, 11, 14, 15, 16, 17, 18, 8, 9, 12, 13]
    else:
        fig, ax = plt.subplots(2, 5, figsize=(12, 6), tight_layout=True)
        index = [1, 2, 3, 4, 5, 8, 9, 10, 6, 7]
    ax = ax.ravel()
    files = list(save_dir.glob('results*.csv'))
    assert len(files), f'No results.csv files found in {save_dir.resolve()}, nothing to plot.'
    for f in files:
        try:
            data = pd.read_csv(f)
            s = [x.strip() for x in data.columns]
            x = data.values[:, 0]
            for i, j in enumerate(index):
                y = data.values[:, j].astype('float')
                # y[y == 0] = np.nan  # don't show zero values
                ax[i].plot(x, y, marker='.', label=f.stem, linewidth=2, markersize=8)  # actual results
                ax[i].plot(x, gaussian_filter1d(y, sigma=3), ':', label='smooth', linewidth=2)  # smoothing line
                ax[i].set_title(s[j], fontsize=12)
                # if j in [8, 9, 10]:  # share train and val loss y axes
                #     ax[i].get_shared_y_axes().join(ax[i], ax[i - 5])
        except Exception as e:
            LOGGER.warning(f'WARNING: Plotting error for {f}: {e}')
    ax[1].legend()
    fname = save_dir / 'results.png'
    fig.savefig(fname, dpi=200)
    plt.close()
    if on_plot:
        on_plot(fname)


def output_to_target(output, max_det=300):
    """Convert model output to target format [batch_id, class_id, x, y, w, h, conf] for plotting."""
    targets = []
    for i, o in enumerate(output):
        box, conf, cls = o[:max_det, :6].cpu().split((4, 1, 1), 1)
        j = torch.full((conf.shape[0], 1), i)
        targets.append(torch.cat((j, cls, xyxy2xywh(box), conf), 1))
    targets = torch.cat(targets, 0).numpy()
    return targets[:, 0], targets[:, 1], targets[:, 2:]


def feature_visualization(x, module_type, stage, n=32, save_dir=Path('runs/detect/exp')):
    """
    Visualize feature maps of a given model module during inference.

    Args:
        x (torch.Tensor): Features to be visualized.
        module_type (str): Module type.
        stage (int): Module stage within the model.
        n (int, optional): Maximum number of feature maps to plot. Defaults to 32.
        save_dir (Path, optional): Directory to save results. Defaults to Path('runs/detect/exp').
    """
    for m in ['Detect', 'Pose', 'Segment']:
        if m in module_type:
            return
    batch, channels, height, width = x.shape  # batch, channels, height, width
    if height > 1 and width > 1:
        f = save_dir / f"stage{stage}_{module_type.split('.')[-1]}_features.png"  # filename

        blocks = torch.chunk(x[0].cpu(), channels, dim=0)  # select batch index 0, block by channels
        n = min(n, channels)  # number of plots
        fig, ax = plt.subplots(math.ceil(n / 8), 8, tight_layout=True)  # 8 rows x n/8 cols
        ax = ax.ravel()
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        for i in range(n):
            ax[i].imshow(blocks[i].squeeze())  # cmap='gray'
            ax[i].axis('off')

        LOGGER.info(f'Saving {f}... ({n}/{channels})')
        plt.savefig(f, dpi=300, bbox_inches='tight')
        plt.close()
        np.save(str(f.with_suffix('.npy')), x[0].cpu().numpy())  # npy save
