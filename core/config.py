#! /usr/bin/env python
# coding=utf-8
from easydict import EasyDict as edict


__C                           = edict()
# Consumers can get config by: from config import cfg

cfg                           = __C

# YOLO options
__C.YOLO                      = edict()

__C.YOLO.NUM_YOLOLAYERS       = 1
__C.YOLO.NUM_CLASSES          = 3
__C.YOLO.CLASSES              = r".\scripts\yolo_custom\obj.names"
# __C.YOLO.CLASSES              = r"C:\Users\PrecisionT3500\PycharmProjects\pythonProject\venv\tensorflow-yolov4-tflite-master\scripts\yolo_custom\obj.names"
__C.YOLO.NUM_ANCHORS          = 3
__C.YOLO.ANCHORS              = [33, 55,  56, 35,  47, 58,  60, 51,  56, 61,  52, 76,  76, 52,  80, 74,  79, 85]
__C.YOLO.ANCHORS_V3           = [10,13, 16,30, 33,23, 30,61, 62,45, 59,119, 116,90, 156,198, 373,326]
__C.YOLO.ANCHORS_TINY         = [23,27, 37,58, 81,82, 81,82, 135,169, 344,319]
__C.YOLO.ANCHORS_V4           = [23,27, 37,58, 81,82]
__C.YOLO.ANCHORS_CUSTOM       = [90,90,102,168,135,168]
__C.YOLO.STRIDES              = [8, 16, 32]
__C.YOLO.STRIDES_TINY         = [16, 32]
__C.YOLO.STRIDES_CUSTOM       = [32]  # output size 40x40으로하려면 16
__C.YOLO.XYSCALE              = [1.05, 1.05, 1.05]
__C.YOLO.XYSCALE_TINY         = [1.05, 1.05]
__C.YOLO.XYSCALE_CUSTOM       = [1.05]
__C.YOLO.ANCHOR_PER_SCALE     = 3
__C.YOLO.IOU_LOSS_THRESH      = 0.5


# Train options
__C.TRAIN                     = edict()

__C.TRAIN.ANNOT_PATH          = r".\scripts\yolo_custom\train.txt"   # 학습시킬 train 이미지 경로
__C.TRAIN.BATCH_SIZE          = 32
__C.TRAIN.INPUT_SIZE          = 640
__C.TRAIN.DATA_AUG            = True
__C.TRAIN.LR_INIT             = 0.00261
__C.TRAIN.LR_END              = 1e-4   #1e-6
__C.TRAIN.WARMUP_EPOCHS       = 40   # 40
__C.TRAIN.FISRT_STAGE_EPOCHS    = 1200
__C.TRAIN.SECOND_STAGE_EPOCHS   = 1600 #400
# __C.TRAIN.MOSAIC_AUG          = False


# TEST options
__C.TEST                      = edict()

__C.TEST.ANNOT_PATH           = r"./scripts/yolo_custom/test.txt"  # 학습 중 test 이미지 경로
__C.TEST.BATCH_SIZE           = 2
__C.TEST.INPUT_SIZE           = 320
__C.TEST.DATA_AUG             = False
# __C.TEST.DECTECTED_IMAGE_PATH = r"D:\ckeckpoint\result"
__C.TEST.CONF_THRESHOLD       = 0.45
__C.TEST.SCORE_THRESHOLD      = 0.25
__C.TEST.IOU_THRESHOLD        = 0.5


