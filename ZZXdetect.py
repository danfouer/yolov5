# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.
修改后支持：
1. 批量处理指定路径下的所有视频/图像文件
2. 可通过--scale-ratio配置loc窗口的显示缩放比例（默认0.5即50%）
3. JSON文件加载逻辑：优先源文件同名JSON，不存在则用--jsonfile指定的
4. BEV二值图像保存到视频所在目录的同名文件夹，命名为视频名_帧数.png
5. 仅保存BEV二值图片，不保存标记后的视频/图像
6. 计算并输出原图绿色轮廓和BEV黑色轮廓的安全区域面积（优化输出逻辑，确保日志显示）
7. 将每帧面积数据、检测框坐标、检测类别、原图轮廓、BEV轮廓分别保存到视频所在目录同名文件夹下的视频名.json文件中
8. BEV图像可上下翻转（默认开启），使图像最下方延伸区域对应车辆坐标系原点，符合视觉习惯
"""

import tempfile
import glob
import shutil
import argparse
import csv
import os
import platform
import sys
from pathlib import Path
import numpy as np
import torch
import json  # 确保导入json库
import warnings

warnings.filterwarnings('ignore')

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.utils.plotting import Annotator, colors, save_one_box

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.torch_utils import select_device, smart_inference_mode
# 注意：请确保utilsbev.py存在并包含所需函数（如create_birdimage、compute_uv2xy_projection等）
from utils.utilsbev import *

# ====================== 新增：轮廓数据序列化辅助函数 ====================== #
def contour_to_list(contour):
    """
    将OpenCV的轮廓（numpy.ndarray）转换为嵌套列表，支持JSON序列化
    :param contour: OpenCV轮廓（单个轮廓，numpy数组）
    :return: 轮廓的列表形式
    """
    # 去除多余的维度并转换为列表
    contour_squeezed = contour.squeeze()
    # 处理单个点的情况（确保返回二维列表）
    if len(contour_squeezed.shape) == 1:
        return [contour_squeezed.tolist()]
    return [point.tolist() for point in contour_squeezed]

def contours_to_list(contours):
    """
    将多个OpenCV轮廓转换为嵌套列表的列表
    :param contours: OpenCV轮廓列表（cv2.findContours返回的结果）
    :return: 轮廓列表的序列化形式
    """
    return [contour_to_list(cont) for cont in contours]

# ====================== 新增：保存面积数据及检测框/类别/轮廓到JSON的函数 ====================== #
def save_area_data(video_path, data):
    video_path = Path(video_path)
    video_dir = video_path.parent  # 视频所在目录
    video_name = video_path.stem  # 视频文件名（不含扩展名）
    save_bev_dir = video_dir / video_name  # 视频同名文件夹
    save_bev_dir.mkdir(parents=True, exist_ok=True)

    # JSON文件路径：视频名.json
    json_path = save_bev_dir / f"{video_name}.json"

    # 读取已有数据（如果存在）
    existing_data = {}
    if json_path.exists():
        try:
            with open(json_path, 'r') as f:
                existing_data = json.load(f)
        except json.JSONDecodeError:
            # 如果文件损坏，重新初始化
            existing_data = {}
            LOGGER.warning(f"JSON文件{json_path}损坏，将重新创建")

    # 添加当前视频数据（video_1, video_2...）
    video_key = f"video_{len(existing_data) + 1}"
    existing_data[video_key] = data

    # 保存更新后的数据
    with open(json_path, 'w') as f:
        json.dump(existing_data, f, indent=2)
    LOGGER.info(f"面积数据、检测框、类别及轮廓信息已保存到：{json_path}")

# ============================================================================== #

@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s.pt',  # model path or triton URL
        jsonfile=ROOT / 'Trans_Mat_05_highway_lanechange_25s.json',  # json file path
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        view_bev=True,  # show bird of view results
        view_loc=True,  # show location results
        save_txt=False,  # save results to *.txt
        save_csv=False,  # save results in CSV format
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos（现在仅影响txt/csv/crop，不影响BEV图片）
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=1,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
        scale_ratio=0.5,  # 新增：loc窗口显示的图像缩放比例（默认0.5即50%）
        flip_bev=True,  # 新增：是否上下翻转BEV图像（默认开启，符合视觉习惯）
):
    # ====================== 初始化数据存储（检测框、类别、轮廓分开） ====================== #
    area_data = {}  # 格式: {视频路径: {'video_id': ..., 'loc_area': [], 'bev_area': [], 'bboxes': [], 'classes': [], 'original_contours': [], 'bev_contours': []}}
    current_video_path = None  # 跟踪当前处理的视频路径
    # 每帧临时存储变量
    current_frame_bboxes = []  # 检测框坐标，格式: [[x1,y1,x2,y2], ...]
    current_frame_classes = []  # 检测类别，格式: [cls1, cls2, ...]（与bboxes一一对应）
    current_frame_original_contours = []  # 原图安全区域轮廓，格式: [[[x1,y1], [x2,y2], ...], ...]
    current_frame_bev_contours = []  # BEV安全区域轮廓，格式: [[[x1,y1], [x2,y2], ...], ...]
    # ============================================================================== #

    source = str(source)
    # 原save_img逻辑保留，但后续不再使用它来控制BEV保存，且强制关闭标记后视频保存
    save_img = not nosave and not source.endswith('.txt')  # 仅用于兼容原有txt/csv/crop逻辑
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download
        print(source)

    # ================ 新增：批量处理目录下的所有视频/图像文件 ================ #
    if os.path.isdir(source):
        LOGGER.info(f"递归查找目录中的图像和视频: {source}")

        # 递归查找所有视频文件（注释掉的是包含图片的逻辑）
        media_files = []
        valid_suffixes = [ext.lower() for ext in VID_FORMATS]  # 仅处理视频

        for root, _, files in os.walk(source):
            for file in files:
                if Path(file).suffix[1:].lower() in valid_suffixes:
                    media_files.append(os.path.join(root, file))

        if not media_files:
            LOGGER.warning(f"目录中没有找到视频文件: {source}")
            return

        # 创建临时文件保存路径列表
        temp_list = tempfile.NamedTemporaryFile(mode='w+', suffix='.txt', delete=False)
        with open(temp_list.name, 'w') as f:
            f.write('\n'.join(media_files))

        LOGGER.info(f"创建包含 {len(media_files)} 个视频文件的临时列表: {temp_list.name}")
        source = temp_list.name
        is_file = False  # 现在源是文件列表

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    print(names)
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # ================ 新增：JSON文件加载逻辑优化 ================ #
    if view_bev:
        # 构建源文件对应的JSON路径（使用Path处理更规范，避免重复后缀问题）
        source_path = Path(source)
        source_json_path = source_path.with_suffix('.json')  # 替代source+'.json'

        if is_file:
            # 优先检查源文件同名JSON是否存在
            if source_json_path.exists():
                LOGGER.info(f"加载源文件同名JSON: {source_json_path}")
                with open(source_json_path, 'r') as f:
                    Trans_Mat = json.load(f)
            else:
                # 源文件同名JSON不存在，使用--jsonfile指定的文件
                LOGGER.warning(f"源文件同名JSON不存在: {source_json_path}，将使用指定的JSON文件: {jsonfile}")
                # 处理jsonfile可能为列表的情况
                used_json = jsonfile[0] if (isinstance(jsonfile, list) and jsonfile) else str(jsonfile)
                # 检查指定的JSON文件是否存在
                if not Path(used_json).exists():
                    LOGGER.error(f"指定的JSON文件不存在: {used_json}")
                    raise FileNotFoundError(f"JSON file not found: {used_json}")
                with open(used_json, 'r') as f:
                    Trans_Mat = json.load(f)
        else:
            # 非单个文件时，使用指定的jsonfile
            used_json = jsonfile[0] if (isinstance(jsonfile, list) and jsonfile) else str(jsonfile)
            if not Path(used_json).exists():
                LOGGER.error(f"指定的JSON文件不存在: {used_json}")
                raise FileNotFoundError(f"JSON file not found: {used_json}")
            LOGGER.info(f"使用指定的JSON文件: {used_json}")
            with open(used_json, 'r') as f:
                Trans_Mat = json.load(f)
        # 读取转换矩阵等参数
        BevSize = np.array(Trans_Mat['BevSize'])
        srcXIntrinsic = np.array(Trans_Mat['srcXIntrinsic'])
        srcYIntrinsic = np.array(Trans_Mat['srcYIntrinsic'])
        V2I_Mat_T = np.array(Trans_Mat['V2I_Mat_T'])
        I2V_Mat_T = np.array(Trans_Mat['I2V_Mat_T'])
        V2B_Mat_T = np.array(Trans_Mat['V2B_Mat_T'])
        B2V_Mat_T = np.array(Trans_Mat['B2V_Mat_T'])
        I2B_Mat_T = np.array(Trans_Mat['I2B_Mat_T'])
        B2I_Mat_T = np.array(Trans_Mat['B2I_Mat_T'])

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Define the path for the CSV file
        csv_path = save_dir / 'predictions.csv'

        # Create or append to the CSV file
        def write_to_csv(image_name, prediction, confidence):
            data = {'Image Name': image_name, 'Prediction': prediction, 'Confidence': confidence}
            with open(csv_path, mode='a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=data.keys())
                if not csv_path.is_file():
                    writer.writeheader()
                writer.writerow(data)

        # ====================== 重置每帧的临时存储变量 ====================== #
        current_frame_bboxes = []
        current_frame_classes = []
        current_frame_original_contours = []
        current_frame_bev_contours = []
        # ====================================================================== #

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, imc, frame = path, im0s.copy(), im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg（后续不再使用）
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            mask = np.ones((im0.shape[0], im0.shape[1]), dtype=np.uint8) * 255
            obstacle_mask = np.zeros((im0.shape[0], im0.shape[1]), dtype=np.uint8)  # 障碍物掩码
            has_class10 = False  # 标记是否存在关键目标
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            # 初始化面积变量，防止未定义
            total_original_area = 0.0
            total_bev_area = 0.0
            # 初始化轮廓变量，防止后续未定义报错
            contours = []
            contoursBevLoc = []
            if view_bev:
                # 初始化BEV图像
                IhsvMat = cv2.cvtColor(imc, cv2.COLOR_BGR2HSV)
                Ihsv = IhsvMat[:, :, ::-1]  # transform image to hsv
                V = Ihsv[:, :, 0]
                BirdImage_V = create_birdimage(V, srcXIntrinsic, srcYIntrinsic)
                BirdImage_VMat = np2cv(BirdImage_V)
                BirdImage_VMat = np.ones((BirdImage_VMat.shape[0], BirdImage_VMat.shape[1], 1), dtype=np.uint8) * 255
                Bird_annotator = Annotator(BirdImage_VMat, line_width=line_thickness, example=str(names))

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = names[c] if hide_conf else f'{names[c]}'
                    confidence = float(conf)
                    confidence_str = f'{confidence:.2f}'

                    if save_csv:
                        write_to_csv(p.name, label, confidence_str)

                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

                    if view_loc:  # Add bbox to image
                        c = int(cls)
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()
                        if c in [0, 1, 2, 3, 4, 5, 6, 7]:
                            # 计算车辆坐标系位置
                            xyImageLoc = np.array([[xywh[0], xywh[0] - xywh[2] / 2, xywh[0] + xywh[2] / 2],
                                                   [xywh[1] + xywh[3] / 2, xywh[1] + xywh[3] / 2,
                                                    xywh[1] + xywh[3] / 2]])
                            xyVehicleLoc = compute_uv2xy_projection(xyImageLoc, I2V_Mat_T)
                            objVehicleLoc = '(%.1fm,%.1fm)' % (xyVehicleLoc[0, 0], xyVehicleLoc[1, 0])
                            annotator.box_location(xyxy, objVehicleLoc, color=colors(c, True))
                            # 获取检测框坐标（整数类型）
                            x1, y1, x2, y2 = map(int, xyxy)
                            # 绘制障碍物掩码
                            cv2.rectangle(obstacle_mask, (x1, y1), (x2, y2), 255, -1)
                            # 计算BEV坐标系位置
                            xyBevLoc = compute_uv2xy_projection(xyImageLoc, I2B_Mat_T)
                            Bird_annotator.kpts(xyBevLoc.T, BevSize, radius=3)
                            # 存储检测框和类别（一一对应）
                            current_frame_bboxes.append([x1, y1, x2, y2])
                            current_frame_classes.append(c)
                        elif c in [10]:
                            # 标记存在class10目标（安全区域）
                            has_class10 = True
                            # 获取class10目标的坐标并绘制掩码
                            x1, y1, x2, y2 = map(int, xyxy)
                            cv2.rectangle(mask, (x1, y1), (x2, y2), color=0, thickness=-1)

            # Stream results（原始图像显示，无实际作用可忽略）
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow('img', im0)
                cv2.waitKey(1)  # 1 millisecond

            # ================ 处理原图安全区域轮廓（loc窗口） ================ #
            if view_loc:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])

                # 只有检测到class10时，才处理掩码和轮廓提取
                if has_class10:
                    # Step 1: 反转掩码 - 矩形区域变白(255)，背景变黑(0)
                    mask_inv = 255 - mask
                    # 对障碍物掩码进行膨胀处理（扩大障碍物区域）
                    kernel_size = 5
                    iterations = 5
                    obstacle_dilate_kernel = np.ones((kernel_size, kernel_size), np.uint8)
                    dilated_obstacle_mask = cv2.dilate(obstacle_mask, obstacle_dilate_kernel, iterations=iterations)
                    mask_inv[dilated_obstacle_mask > 0] = 0  # 障碍物区域置为0

                    # 形态学膨胀（合并相邻的安全区域）
                    kernel_size = 3
                    iterations = 3
                    kernel = np.ones((kernel_size, kernel_size), np.uint8)
                    dilated_mask = cv2.dilate(mask_inv, kernel, iterations=iterations)

                    # Canny边缘检测（提取轮廓边缘）
                    edges = cv2.Canny(dilated_mask, threshold1=0, threshold2=100, apertureSize=3)

                    # 形态学闭合（修复断开的边缘）
                    close_kernel = np.ones((3, 3), np.uint8)
                    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, close_kernel, iterations=1)

                    # 查找轮廓（仅外部轮廓）
                    contours, _ = cv2.findContours(
                        edges,
                        mode=cv2.RETR_EXTERNAL,
                        method=cv2.CHAIN_APPROX_SIMPLE
                    )

                    # 在原图上绘制绿色轮廓
                    cv2.drawContours(im0, contours, -1, (0, 255, 0), 2)

                    # 转换原图轮廓为列表（支持JSON序列化）并存储
                    current_frame_original_contours = contours_to_list(contours)

                    # 计算原图轮廓面积（像素）并转换为实际面积（平方米）
                    if contours:
                        total_original_area = sum(cv2.contourArea(c) for c in contours)
                        total_original_area *= 0.01  # 像素到平方米的转换比例（可根据实际场景调整）
                        LOGGER.info(f"Frame {frame} - 原图安全区域面积: {total_original_area:.2f} 平方米")

                # 缩放图像以适应显示（scale_ratio控制缩放比例）
                scaled_im0 = cv2.resize(im0, None, fx=scale_ratio, fy=scale_ratio)
                cv2.imshow('loc', scaled_im0)
                cv2.waitKey(1)

            # ================ 处理BEV安全区域轮廓及保存 ================ #
            if view_bev and has_class10 and contours:
                # 计算BEV轮廓（将原图轮廓点转换到BEV坐标系）
                contoursBevLoc = []
                for contour in contours:
                    contour_points = contour.squeeze()
                    # 处理单个点的情况（扩展维度）
                    if len(contour_points.shape) == 1:
                        contour_points = np.expand_dims(contour_points, axis=0)
                    # 转换每个轮廓点到BEV坐标系
                    bev_points = []
                    for point in contour_points:
                        uv = np.array([[point[0]], [point[1]]])
                        bev_xy = compute_uv2xy_projection(uv, I2B_Mat_T)
                        bev_points.append([bev_xy[0, 0], bev_xy[1, 0]])
                    contoursBevLoc.append(np.array(bev_points, dtype=np.int32))

                # 绘制BEV轮廓（黑色填充）
                for bev_contour in contoursBevLoc:
                    if len(bev_contour) >= 3:  # 确保轮廓点足够（至少3个点构成封闭区域）
                        cv2.drawContours(BirdImage_VMat, [bev_contour], -1, (0, 0, 0), 1)

                # 转换BEV轮廓为列表（支持JSON序列化）并存储
                current_frame_bev_contours = contours_to_list(contoursBevLoc)

                # 计算BEV轮廓面积（像素）并转换为实际面积（平方米）
                if contoursBevLoc:
                    total_bev_area = sum(cv2.contourArea(c) for c in contoursBevLoc)
                    total_bev_area *= 0.01  # 像素到平方米的转换比例（可根据实际场景调整）
                    LOGGER.info(f"Frame {frame} - BEV安全区域面积: {total_bev_area:.2f} 平方米")

                # 上下翻转BEV图像（符合视觉习惯：下方对应车辆坐标系原点）
                if flip_bev:
                    BirdImage_VMat = cv2.flip(BirdImage_VMat, 0)
                    BirdImage_VMat = cv2.flip(BirdImage_VMat, 1)

                # 保存BEV二值图像到视频同名文件夹
                video_name = p.stem
                video_dir = p.parent
                save_bev_dir = video_dir / video_name
                save_bev_dir.mkdir(parents=True, exist_ok=True)
                bev_save_path = save_bev_dir / f"{video_name}_{frame}.png"
                cv2.imwrite(str(bev_save_path), BirdImage_VMat)
                LOGGER.info(f"BEV图像已保存到: {bev_save_path}")

                # 显示BEV图像
                if view_bev:
                    cv2.imshow('bev', BirdImage_VMat)
                    cv2.waitKey(1)

            # ================ 收集视频帧的所有数据到字典 ================ #
            if dataset.mode == 'video':
                # 提取video_id（从视频名“1_002.mp4”中取“1_002”）
                video_name = p.stem
                video_id = '_'.join(video_name.split('_')[:2])  # 取前两部分作为video_id

                # 初始化当前视频的数据存储
                if str(p) not in area_data:
                    # 如果是新视频，先保存上一个视频的数据（如果存在）
                    if current_video_path and current_video_path in area_data:
                        save_area_data(current_video_path, area_data[current_video_path])
                    # 初始化视频数据结构（新增original_contours和bev_contours字段）
                    area_data[str(p)] = {
                        'video_id': video_id,
                        'loc_area': [],
                        'bev_area': [],
                        'bboxes': [],  # 每帧检测框坐标
                        'classes': [],  # 每帧检测类别
                        'original_contours': [],  # 每帧原图轮廓
                        'bev_contours': []  # 每帧BEV轮廓
                    }
                    current_video_path = str(p)

                # 添加当前帧的所有数据
                area_data[str(p)]['loc_area'].append(round(total_original_area, 2))
                area_data[str(p)]['bev_area'].append(round(total_bev_area, 2))
                area_data[str(p)]['bboxes'].append(current_frame_bboxes)
                area_data[str(p)]['classes'].append(current_frame_classes)
                area_data[str(p)]['original_contours'].append(current_frame_original_contours)
                area_data[str(p)]['bev_contours'].append(current_frame_bev_contours)
            # ====================================================================== #

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # 处理最后一个视频的数据（确保数据被保存）
    if current_video_path and current_video_path in area_data:
        save_area_data(current_video_path, area_data[current_video_path])

    # Print results（输出推理速度等信息）
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)

def parse_opt():
    """解析命令行参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path or triton URL')
    parser.add_argument('--jsonfile', type=str, default=ROOT / 'Trans_Mat_05_highway_lanechange_25s.json', help='json file path')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--view-bev', action='store_true', default=True, help='show bird of view results')
    parser.add_argument('--view-loc', action='store_true', default=True, help='show location results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-csv', action='store_true', help='save results in CSV format')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=1, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    parser.add_argument('--scale-ratio', type=float, default=0.5, help='loc window display scale ratio')
    parser.add_argument('--flip-bev', type=bool, default=True, help='whether to flip BEV image vertically')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # 扩展为(height, width)（如[640]→[640,640]）
    print_args(vars(opt))
    return opt

def main(opt):
    """主函数：检查依赖并运行推理"""
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))

if __name__ == "__main__":
    """程序入口"""
    opt = parse_opt()
    main(opt)