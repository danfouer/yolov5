# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/LNwODJXcvt4'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
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
import json
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
from utils.utilsbev import *


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
        view_img=True,  # show results
        view_bev=False,  # show brid of view results
        view_loc=False,  # show location results
        save_txt=False,  # save results to *.txt
        save_csv=False,  # save results in CSV format
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
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
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download
        print(source)

    # ================ 新增代码开始 ================ #
    # 如果源是目录，递归查找所有图像文件
    if os.path.isdir(source):
        LOGGER.info(f"递归查找目录中的图像: {source}")

        # 递归查找所有图片文件
        img_files = []
        valid_suffixes = [ext.lower() for ext in IMG_FORMATS]

        for root, _, files in os.walk(source):
            for file in files:
                if Path(file).suffix[1:].lower() in valid_suffixes:
                    img_files.append(os.path.join(root, file))

        if not img_files:
            LOGGER.warning(f"目录中没有找到图像文件: {source}")
            return

        # 创建临时文件保存路径列表
        temp_list = tempfile.NamedTemporaryFile(mode='w+', suffix='.txt', delete=False)
        with open(temp_list.name, 'w') as f:
            f.write('\n'.join(img_files))

        LOGGER.info(f"创建包含 {len(img_files)} 个图像的临时列表: {temp_list.name}")
        source = temp_list.name
        print(source)
        is_file = False  # 现在源是文件列表
    # ================ 新增代码结束 ================ #


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
 #############################################################################
    if view_bev:
        if is_file:
            with open(source+'.json', 'r') as f:
                Trans_Mat = json.load(f)
        else:
            #jsonfile = str(jsonfile)
            #json_path = Path(jsonfile)
            if isinstance(jsonfile, list) and jsonfile:
                jsonfile = jsonfile[0]  # 取列表中的第一个元素
            jsonfile = str(jsonfile)  # 确保是字符串类型
            print(jsonfile)
            with open(jsonfile, 'r') as f:
                Trans_Mat = json.load(f)
    # if view_bev:
    #     json_path = None
    #
    #     # 如果是单个文件，尝试使用与图像同名的JSON文件
    #     if is_file:
    #         json_path = Path(source).with_suffix('.json')
    #         if not json_path.exists():
    #             LOGGER.warning(f"Image-specific JSON not found: {json_path}")
    #             json_path = None
    #
    #     # 如果未找到图像特定JSON，使用指定的jsonfile
    #     if not json_path:
    #         json_path = Path(jsonfile)
    #         LOGGER.info(f"Using specified JSON file: {json_path}")
    #
    #     # 检查并加载JSON文件
    #     if json_path.exists():
    #         with open(json_path, 'r') as f:
    #             Trans_Mat = json.load(f)
    #     else:
    #         LOGGER.error(f"JSON file not found: {json_path}")
    #         continue
        # 读取前视图I-车辆坐标系V-鸟瞰图相互转换矩阵
        # V2I_Mat = np.array(Trans_Mat['V2I_Mat'])
        # I2V_Mat = np.array(Trans_Mat['I2V_Mat'])
        BevSize = np.array(Trans_Mat['BevSize'])
        # V2B_Mat = np.array(Trans_Mat['V2B_Mat'])
        # B2V_Mat = np.array(Trans_Mat['B2V_Mat'])
        srcXIntrinsic = np.array(Trans_Mat['srcXIntrinsic'])
        srcYIntrinsic = np.array(Trans_Mat['srcYIntrinsic'])
        # I2B_Mat = np.array(Trans_Mat['I2B_Mat'])
        # B2I_Mat = np.array(Trans_Mat['B2I_Mat'])
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
            #print(im.shape)
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

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

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
                p, im0, imc,frame = path, im0s.copy(),im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            #imc = im0.copy() if save_crop else im0  # for save_crop
            mask = np.ones((im0.shape[0], im0.shape[1]), dtype=np.uint8) * 255
            obstacle_mask = np.zeros((im0.shape[0], im0.shape[1]), dtype=np.uint8)  # 障碍物掩码
            has_class10 = False  # 标记是否存在关键目标
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            #maskannotator = Annotator(mask, line_width=line_thickness, example=str(names))
            if view_bev:
                IhsvMat = cv2.cvtColor(imc, cv2.COLOR_BGR2HSV)
                Ihsv = IhsvMat[:, :, ::-1]  # transform image to hsv
                V = Ihsv[:, :, 0]
                #V = cv2.normalize(V, None, 100, 255, cv2.NORM_MINMAX, cv2.CV_8U) / 255.0
                BirdImage_V = create_birdimage(V, srcXIntrinsic, srcYIntrinsic)
                BirdImage_VMat = np2cv(BirdImage_V)
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
                        #xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        #label = None if hide_labels else (f'{c:d}' if hide_conf else f'{c:d} {conf:.2f}')

                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

                    if view_loc:  # Add bbox to image
                        c = int(cls)  # integer class
                        #label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        #label = None if hide_labels else (f'{c:d}' if hide_conf else f'{c:d} {conf:.2f}')
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()
                        if c in [0,1,2,3,4,5,6,7]:
                            xyImageLoc = np.array([[xywh[0],xywh[0]-xywh[2]/2,xywh[0]+xywh[2]/2],
                                                  [xywh[1]+xywh[3]/2,xywh[1]+xywh[3]/2,xywh[1]+xywh[3]/2]])
                            xyVehicleLoc = compute_uv2xy_projection(xyImageLoc, I2V_Mat_T)
                            #print(xyVehicleLoc)
                            xyBevLoc = compute_uv2xy_projection(xyImageLoc, I2B_Mat_T)
                            objVehicleLoc = '(%.1fm,%.1fm)' % (xyVehicleLoc[0,0], xyVehicleLoc[1,0])
                            annotator.box_location(xyxy, objVehicleLoc, color=colors(c, True))
                            # 获取矩形坐标（整数类型）
                            x1, y1, x2, y2 = map(int, xyxy)
                            cv2.rectangle(obstacle_mask, (x1, y1), (x2, y2), 255, -1)
                            #annotator.kpts(xyImageLoc.T)
                            Bird_annotator.kpts(xyBevLoc.T,BevSize,radius=3)
                        elif c in [10]:
                            # label = None if hide_labels else (f'{c:d}' if hide_conf else f'{c:d} {conf:.2f}')
                            # # #annotator.box_label(xyxy, label, color=colors(c, True))
                            # maskannotator.box_fill(xyxy, label, color=colors(0, True))
                            # x1, y1, x2, y2 = map(int, xyxy)
                            # points = np.array([[x1, y1], [x1, y2], [x2, y2], [x2, y1]], dtype=np.int32)
                            #
                            # # 如果是第一个类别10的点，初始化列表；否则追加
                            # if 'class10_points' not in locals():
                            #     class10_points = []
                            # class10_points.append(points)
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
                #cv2.imshow('im0', im0)
                #cv2.imshow('imc', imc)
                cv2.waitKey(1)  # 1 millisecond
            if view_loc:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                if view_loc and has_class10:
                    # Step 1: 反转掩码 - 矩形区域变白(255)，背景变黑(0)
                    mask_inv = 255 - mask
                    # Step 1.5: 对障碍物掩码进行膨胀处理（关键修改）
                    # Step 2: 形态学膨胀（合并相邻矩形）
                    kernel_size = 5  # 核大小（可调整）
                    iterations = 5  # 膨胀迭代次数（可调整）
                    obstacle_dilate_kernel = np.ones((kernel_size, kernel_size), np.uint8)  # 使用5x5的膨胀核
                    dilated_obstacle_mask = cv2.dilate(obstacle_mask, obstacle_dilate_kernel, iterations=iterations)
                    mask_inv[dilated_obstacle_mask > 0] = 0

                    # Step 2: 形态学膨胀（合并相邻矩形）
                    kernel_size = 3  # 核大小（可调整）
                    iterations = 3  # 膨胀迭代次数（可调整）
                    kernel = np.ones((kernel_size, kernel_size), np.uint8)
                    dilated_mask = cv2.dilate(mask_inv, kernel, iterations=iterations)
                    #dilated_mask=dilated_mask-obstacle_mask

                    # Step 3: Canny边缘检测
                    edges = cv2.Canny(dilated_mask, threshold1=0, threshold2=100, apertureSize=3)

                    # Step 4: 查找轮廓（仅外部轮廓）
                    contours, _ = cv2.findContours(
                        edges,
                        mode=cv2.RETR_EXTERNAL,
                        method=cv2.CHAIN_APPROX_SIMPLE
                    )

                    # Step 5: 在原图上绘制蓝色轮廓（厚度=5像素）
                    cv2.drawContours(
                        image=im0,
                        contours=contours,
                        contourIdx=-1,  # 绘制所有轮廓
                        color=(0, 255, 0),  # BGR绿色 (0,255,0)
                        thickness=1,
                        lineType=cv2.LINE_AA
                    )
                cv2.imshow('loc', im0)
                #cv2.imshow('im0', im0)
                #cv2.imshow('imc', imc)
                cv2.waitKey(1)  # 1 millisecond
            if view_bev:
                BirdImage_VMat =Bird_annotator.result()

                #ret, BirdImage_VMat = cv2.threshold(BirdImage_VMat, 0, 255, cv2.THRESH_BINARY)
                cv2.imshow('bev', BirdImage_VMat)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

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
    parser.add_argument('--jsonfile', nargs='+', type=str, default=ROOT / 'Trans_Mat_05_highway_lanechange_25s.json', help='json file path')
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
    parser.add_argument('--hide-labels', default=True, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=True, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
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
