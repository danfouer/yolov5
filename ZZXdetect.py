# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.
修改后支持：
1. 批量处理指定路径下的所有视频/图像文件
2. 可通过--scale-ratio配置loc窗口的显示缩放比例（默认0.5即50%）
3. JSON文件加载逻辑：优先源文件同名JSON，不存在则用--jsonfile指定的
4. BEV二值图像保存到视频所在目录的同名文件夹，命名为视频名_帧数.png
5. 仅保存BEV二值图片，不保存标记后的视频/图像
6. 计算并输出原图绿色轮廓和BEV黑色轮廓的安全区域面积
7. 将每帧面积数据保存到视频所在目录同名文件夹下的视频名.json文件中
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


# ====================== 新增：保存面积数据到JSON的函数 ====================== #
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
    LOGGER.info(f"面积数据已保存到：{json_path}")


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
):
    # ====================== 新增：初始化面积数据存储 ====================== #
    area_data = {}  # 格式: {视频路径: {'video_id': ..., 'loc_area': [], 'bev_area': []}}
    current_video_path = None  # 跟踪当前处理的视频路径
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

        # 递归查找所有图片和视频文件
        media_files = []
        # 合并图像和视频格式
        valid_suffixes = [ext.lower() for ext in IMG_FORMATS + VID_FORMATS]

        for root, _, files in os.walk(source):
            for file in files:
                if Path(file).suffix[1:].lower() in valid_suffixes:
                    media_files.append(os.path.join(root, file))

        if not media_files:
            LOGGER.warning(f"目录中没有找到图像或视频文件: {source}")
            return

        # 创建临时文件保存路径列表
        temp_list = tempfile.NamedTemporaryFile(mode='w+', suffix='.txt', delete=False)
        with open(temp_list.name, 'w') as f:
            f.write('\n'.join(media_files))

        LOGGER.info(f"创建包含 {len(media_files)} 个媒体文件的临时列表: {temp_list.name}")
        source = temp_list.name
        print(source)
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
                if isinstance(jsonfile, list) and jsonfile:
                    used_json = jsonfile[0]
                else:
                    used_json = str(jsonfile)
                # 检查指定的JSON文件是否存在
                if not Path(used_json).exists():
                    LOGGER.error(f"指定的JSON文件不存在: {used_json}")
                    raise FileNotFoundError(f"JSON file not found: {used_json}")
                with open(used_json, 'r') as f:
                    Trans_Mat = json.load(f)
        else:
            # 非单个文件时，使用指定的jsonfile
            if isinstance(jsonfile, list) and jsonfile:
                used_json = jsonfile[0]
            else:
                used_json = str(jsonfile)
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
            if view_bev:
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
                            xyImageLoc = np.array([[xywh[0], xywh[0] - xywh[2] / 2, xywh[0] + xywh[2] / 2],
                                                   [xywh[1] + xywh[3] / 2, xywh[1] + xywh[3] / 2,
                                                    xywh[1] + xywh[3] / 2]])
                            xyVehicleLoc = compute_uv2xy_projection(xyImageLoc, I2V_Mat_T)
                            objVehicleLoc = '(%.1fm,%.1fm)' % (xyVehicleLoc[0, 0], xyVehicleLoc[1, 0])
                            annotator.box_location(xyxy, objVehicleLoc, color=colors(c, True))
                            # 获取矩形坐标（整数类型）
                            x1, y1, x2, y2 = map(int, xyxy)
                            cv2.rectangle(obstacle_mask, (x1, y1), (x2, y2), 255, -1)
                            # ====================== 修复：定义xyBevLoc变量（关键修改） ====================== #
                            xyBevLoc = compute_uv2xy_projection(xyImageLoc, I2B_Mat_T)  # 计算BEV坐标
                            # ============================================================================== #
                            Bird_annotator.kpts(xyBevLoc.T, BevSize, radius=3)
                        elif c in [10]:
                            has_class10 = True
                            # 获取矩形坐标（整数类型）
                            x1, y1, x2, y2 = map(int, xyxy)
                            # 在掩码上绘制实心黑色矩形（填充色=0）
                            cv2.rectangle(mask, (x1, y1), (x2, y2), color=0, thickness=-1)

            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow('img', im0)
                cv2.waitKey(1)  # 1 millisecond

            # ================ 修改：loc窗口按scale_ratio缩放显示 ================ #
            if view_loc:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                if view_loc and has_class10:
                    # Step 1: 反转掩码 - 矩形区域变白(255)，背景变黑(0)
                    mask_inv = 255 - mask
                    # 对障碍物掩码进行膨胀处理
                    kernel_size = 5
                    iterations = 5
                    obstacle_dilate_kernel = np.ones((kernel_size, kernel_size), np.uint8)
                    dilated_obstacle_mask = cv2.dilate(obstacle_mask, obstacle_dilate_kernel, iterations=iterations)
                    mask_inv[dilated_obstacle_mask > 0] = 0

                    # 形态学膨胀（合并相邻矩形）
                    kernel_size = 3
                    iterations = 3
                    kernel = np.ones((kernel_size, kernel_size), np.uint8)
                    dilated_mask = cv2.dilate(mask_inv, kernel, iterations=iterations)

                    # Canny边缘检测
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
                    cv2.drawContours(
                        image=im0,
                        contours=contours,
                        contourIdx=-1,  # 修复：绘制所有轮廓（原0只绘制第一个，改为-1）
                        color=(0, 255, 0),  # BGR绿色
                        thickness=1,
                        lineType=cv2.LINE_AA
                    )

                    # ====================== 计算并输出原图绿色轮廓（安全区域）的总面积 ====================== #
                    if contours:
                        # 初始化总面积
                        total_original_area = 0.0
                        for cnt in contours:
                            # 计算单个轮廓的面积并累加（cv2.contourArea返回浮点数）
                            cnt_area = cv2.contourArea(cnt)
                            if cnt_area > 0:  # 过滤无效的轮廓面积
                                total_original_area += cnt_area
                        # 输出面积（保留2位小数，单位：像素²）
                        LOGGER.info(f"【{p.name}_帧{frame}】原图安全区域（绿色轮廓）总面积：{total_original_area:.2f} 像素²")
                    else:
                        LOGGER.info(f"【{p.name}_帧{frame}】原图未检测到有效轮廓，安全区域面积为0")
                    # =========================================================================================== #

                # 核心：根据scale_ratio计算新尺寸并缩放图像
                h, w = im0.shape[:2]
                new_w = int(w * scale_ratio)
                new_h = int(h * scale_ratio)
                im0_resized = cv2.resize(im0, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                # 显示缩放后的图像
                cv2.imshow('loc', im0_resized)
                cv2.waitKey(1)  # 1 millisecond

                # 处理轮廓并转换为投影后的轮廓（contoursBevLoc）
                contoursBevLoc = []  # 存储每个轮廓投影后的结果
                # ====================== 修复：判断contours是否存在（避免未定义错误） ====================== #
                if 'contours' not in locals():
                    contours = []  # 初始化空轮廓，防止后续循环报错
                # ============================================================================== #
                for cnt in contours:
                    if len(cnt) == 0:
                        continue  # 跳过空轮廓

                    # 步骤1：将单个轮廓从 (N, 1, 2) 重塑为 (2, N)
                    cnt_points = cnt.reshape(-1, 2).T  # 结果：(2, N)，对应u/v坐标

                    # 步骤2：调用投影函数
                    cnt_bev = compute_uv2xy_projection(cnt_points, I2B_Mat_T)  # I2B_Mat_T是你的变换矩阵

                    # 步骤3：强制闭合并转换格式
                    cnt_bev_2d = cnt_bev.T  # 转成 (N, 2) 格式（浮点数）
                    # 强制闭合：兼容浮点数微小误差
                    if cnt_bev_2d.shape[0] > 0 and not np.allclose(cnt_bev_2d[0], cnt_bev_2d[-1]):
                        cnt_bev_2d = np.vstack([cnt_bev_2d, cnt_bev_2d[0]])  # 拼接首点，强制闭合
                    # 转成OpenCV要求的格式
                    cnt_bev_reshaped = cnt_bev_2d.reshape(-1, 1, 2).astype(np.int32)
                    contoursBevLoc.append(cnt_bev_reshaped)

                Bird_annotator.draw_contours(
                    contours=contoursBevLoc,
                    color=0,  # BGR黑色
                    thickness=1,
                )

                # ====================== 计算并输出BEV黑色轮廓（安全区域）的总面积 ====================== #
                if contoursBevLoc:
                    # 初始化总面积
                    total_bev_area = 0.0
                    for cnt in contoursBevLoc:
                        # 计算单个轮廓的面积并累加
                        cnt_area = cv2.contourArea(cnt)
                        if cnt_area > 0:  # 过滤无效的轮廓面积
                            total_bev_area += cnt_area
                    # 输出面积（保留2位小数，单位：像素²；若有实际物理尺度，可在此处添加转换逻辑）
                    LOGGER.info(f"【{p.name}_帧{frame}】BEV安全区域（黑色轮廓）总面积：{total_bev_area:.2f} 像素²")
                else:
                    LOGGER.info(f"【{p.name}_帧{frame}】BEV未检测到有效轮廓，安全区域面积为0")
                # =========================================================================================== #

                # 二值化处理BEV图像
                thresh = 64
                maxval = 255
                ret, BirdEdge_VMat = cv2.threshold(BirdImage_VMat, thresh, maxval, cv2.THRESH_BINARY)

            # ================ 新增：保存BEV二值图像（修改：移除save_img依赖，只保留视频模式判断） ================ #
            if view_bev:
                BirdImage_VMat = Bird_annotator.result()
                cv2.imshow('bev', BirdEdge_VMat)
                # 保存二值图片到视频所在目录的同名文件夹（仅当处理视频时保存，不再依赖save_img）
                if dataset.mode == 'video':  # 关键修改：移除save_img，只判断是否是视频模式
                    video_path = Path(p)
                    video_dir = video_path.parent  # 视频所在目录
                    video_name = video_path.stem  # 视频文件名（不含扩展名）
                    # 创建保存图片的文件夹（视频名命名）
                    save_bev_dir = video_dir / video_name
                    save_bev_dir.mkdir(parents=True, exist_ok=True)
                    # 生成图片文件名（视频名_帧数.png）
                    img_filename = f"{video_name}_{frame}.png"
                    img_save_path = save_bev_dir / img_filename
                    # 保存二值图片
                    cv2.imwrite(str(img_save_path), BirdEdge_VMat)
                cv2.waitKey(1)  # 1 millisecond

            # ====================== 新增：收集视频帧的面积数据到字典 ====================== #
            if dataset.mode == 'video':
                # 提取video_id（从视频名“1_002_0_149.mp4”中取“1_002”）
                video_name = p.stem
                video_id_parts = video_name.split('_')[:2]  # 取前两部分
                video_id = '_'.join(video_id_parts)  # 如：1_002

                # 初始化当前视频的数据存储
                if str(p) not in area_data:
                    # 如果是新视频，先保存上一个视频的数据（如果存在）
                    if current_video_path and current_video_path in area_data:
                        save_area_data(current_video_path, area_data[current_video_path])
                    area_data[str(p)] = {
                        'video_id': video_id,
                        'loc_area': [],
                        'bev_area': []
                    }
                    current_video_path = str(p)

                # 添加当前帧的面积数据（保留2位小数）
                area_data[str(p)]['loc_area'].append(round(total_original_area, 2))
                area_data[str(p)]['bev_area'].append(round(total_bev_area, 2))
            # ============================================================================== #

            # ================ 关键修改：强制关闭标记后的视频/图像保存（注释或添加False条件） ================ #
            # 原save_img逻辑被注释，彻底禁用标记后的视频/图像保存
            # if save_img:
            #     if dataset.mode == 'image':
            #         cv2.imwrite(save_path, im0)
            #     else:  # 'video' or 'stream'
            #         if vid_path[i] != save_path:  # new video
            #             vid_path[i] = save_path
            #             if isinstance(vid_writer[i], cv2.VideoWriter):
            #                 vid_writer[i].release()  # release previous video writer
            #             if vid_cap:  # video
            #                 fps = vid_cap.get(cv2.CAP_PROP_FPS)
            #                 w = int(vid_cap.get(cv2.CAP_PROP_FPS))
            #                 h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            #             else:  # stream
            #                 fps, w, h = 30, im0.shape[1], im0.shape[0]
            #             save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix
            #             vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            #         vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # ====================== 新增：保存最后一个视频的面积数据 ====================== #
    if dataset.mode == 'video' and current_video_path and current_video_path in area_data:
        save_area_data(current_video_path, area_data[current_video_path])
    # ============================================================================== #

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path or triton URL')
    parser.add_argument('--jsonfile', nargs='+', type=str, default=ROOT / 'Trans_Mat_05_highway_lanechange_25s.json',
                        help='json file path')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--view-bev', action='store_true', help='show bird of view results')
    parser.add_argument('--view-loc', action='store_true', help='show location results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-csv', action='store_true', help='save results in CSV format')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos（不影响BEV图片）')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=1, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=True, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=True, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    # 新增：scale-ratio参数，用于配置loc窗口的缩放比例
    parser.add_argument('--scale-ratio', type=float, default=0.5,
                        help='loc窗口显示的图像缩放比例（如0.5表示50%，1.0表示100%）')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)