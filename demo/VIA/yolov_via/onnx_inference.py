#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os

import cv2
import numpy as np

import onnxruntime
import torch

from yolox.data.data_augment import preproc as preprocess
# from yolox.data.datasets import COCO_CLASSES
# from exps.yolov.yolov_base import pr
from yolox.data.datasets import VIA_classes
from yolox.data.data_augment import ValTransform,Vid_Val_Transform
from yolox.utils import mkdir, multiclass_nms, demo_postprocess, vis


def make_parser():
    parser = argparse.ArgumentParser("onnxruntime inference sample")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="yolox.onnx",
        help="Input your onnx model.",
    )
    parser.add_argument(
        "-i",
        "--image_path",
        type=str,
        default='test_image.png',
        help="Path to your input image.",
    )
    parser.add_argument(
        "-v",
        "--video_path",
        type=str,
        default='test_video.mp4',
        help="Path to your input video.",
    )

    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default='demo_output',
        help="Path to your output directory.",
    )
    parser.add_argument(
        "-s",
        "--score_thr",
        type=float,
        default=0.3,
        help="Score threshould to filter the result.",
    )
    parser.add_argument(
        "--test_shape",
        type=str,
        default="640,640",
        help="Specify an input shape for inference.",
    )
    parser.add_argument(
        "--with_p6",
        action="store_true",
        help="Whether your model uses p6 in FPN/PAN.",
    )
    return parser


def VIA_preprocess():
    cap = cv2.VideoCapture(args.video_path)
    print("File Exists:", os.path.exists(args.video_path))
    # print(args.video_path)

    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    # print('video height, width, fps:',height, width, fps)

    input_shape = tuple(map(int, args.test_shape.split(',')))
    ratio = min(input_shape[0]/ height, input_shape[1]/ width)
    print(ratio)

    # 切分图片
    vid_writer = cv2.VideoWriter(
    args.output_dir, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    )

    frames = []
    outputs = []
    ori_frames = []

    val_transform = ValTransform()
    while True:
        ret_val, frame = cap.read()
        if ret_val:
            ori_frames.append(frame)
            frame, _ = val_transform(frame, None, input_shape)
            frames.append(torch.tensor(frame))
        else:
            break

    res = []
    frame_len = len(frames)
    index_list = list(range(frame_len))
    print(frames[0].shape)
    return  frames

if __name__ == '__main__':
    args = make_parser().parse_args()

    # custom
    # preprocess
    # frames = preprocess()
    #
    # # inference
    # session = onnxruntime.InferenceSession(args.model)
    # #
    # ort_inputs = {session.get_inputs()[0].name: img[:, :, :]}
    # # print(ort_inputs['images'].shape)
    # output = session.run(None, frames)
    # print(len(output),output[0].shape,output[1].shape)


    # # yolox
    # input_shape = tuple(map(int, args.test_shape.split(',')))
    # origin_img = cv2.imread(args.image_path)
    # print(origin_img.shape)
    # img, ratio = preprocess(origin_img, input_shape)
    # print(img.shape, type(img), ratio)
    #
    # img = torch.rand(size=(1,3,288, 512)).numpy()
    #
    # session = onnxruntime.InferenceSession(args.model)
    #
    # ort_inputs = {session.get_inputs()[0].name: img[:]}
    # print(ort_inputs['images'].shape)
    #
    # output = session.run(None, ort_inputs)
    # print(len(output),output[0].shape,output[1].shape)
    #
    # # yolov postprocess
    #
    # predictions = demo_postprocess(output[0], input_shape, p6=args.with_p6)[0]
    #
    # boxes = predictions[:, :4]
    # scores = predictions[:, 4:5] * predictions[:, 5:]
    #
    # boxes_xyxy = np.ones_like(boxes)
    # boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
    # boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
    # boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
    # boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.
    # boxes_xyxy /= ratio
    # dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.1)
    # if dets is not None:
    #     final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
    #     origin_img = vis(origin_img, final_boxes, final_scores, final_cls_inds,
    #                      conf=args.score_thr, class_names=VIA_classes)
    #
    # mkdir(args.output_dir)
    # output_path = os.path.join(args.output_dir, os.path.basename(args.image_path))
    # cv2.imwrite(output_path, origin_img)
