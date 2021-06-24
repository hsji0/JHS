#! /usr/bin/env python
# coding=utf-8

import os
import cv2
import random
import numpy as np
import tensorflow as tf
import core.utils as utils
from core.config import cfg




class Dataset(object):
    """implement Dataset here"""

    def __init__(self, FLAGS, input_channel, is_training: bool, dataset_type: str = "yolo"):

        self.input_channel = input_channel
        self.strides, self.anchors, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
        self.dataset_type = dataset_type

        self.annot_path = (
            cfg.TRAIN.ANNOT_PATH if is_training else cfg.TEST.ANNOT_PATH
        )
        self.input_sizes = (
            cfg.TRAIN.INPUT_SIZE if is_training else cfg.TEST.INPUT_SIZE
        )
        self.batch_size = (
            cfg.TRAIN.BATCH_SIZE if is_training else cfg.TEST.BATCH_SIZE
        )
        self.data_aug = cfg.TRAIN.DATA_AUG if is_training else cfg.TEST.DATA_AUG

        self.train_input_sizes = cfg.TRAIN.INPUT_SIZE
        self.classes = utils.read_class_names(cfg.YOLO.CLASSES)
        self.num_classes = len(self.classes)
        self.anchor_per_scale = cfg.YOLO.ANCHOR_PER_SCALE
        self.max_bbox_per_scale = 150

        self.annotations = self.load_annotations()
        self.num_samples = len(self.annotations)
        self.num_batchs = int(np.ceil(self.num_samples / self.batch_size))
        self.batch_count = 0

        # self.mosaic = True

    # def mosaic_aug(self, image, bboxes, xrange = [0.3, 0.7] , yrange = [0.3, 0.7] ):
    #     image_size = cfg.TRAIN.INPUT_SIZE
    #     output_image = np.zeros((image_size, image_size, 3), dtype=np.float32)
    #     xratio = xrange[0] + (xrange[1] - xrange[0]) * random.random()
    #     neww = int(image_size * xratio)
    #     yratio = yrange[0] + (yrange[1] - yrange[0]) * random.random()
    #     newh = int(image_size * yratio)
    #
    #     if len(image) < 4:
    #         return image, bboxes
    #
    #     mosaic_bboxes = []
    #     for img_idx, bboxes_per_img in enumerate(bboxes):
    #         mosaic_bboxes_per_img = []
    #         for bbox_idx, bbox in enumerate(bboxes_per_img):
    #             # print("bbox:{}".format(bbox))
    #             bbox = np.array(bbox).astype(np.float32)
    #             if img_idx == 0:  # top left
    #                 output_image[:newh, :neww, :] = cv2.resize(image[0], (neww, newh))
    #                 bbox[...,0] *= xratio
    #                 bbox[...,1] *= yratio
    #                 bbox[...,2] *= xratio
    #                 bbox[...,3] *= yratio
    #             elif img_idx == 1:  # top right
    #                 output_image[:newh, neww:, :] = cv2.resize(image[1], (image_size - neww, newh))
    #                 bbox[..., 0] = bbox[..., 2] + bbox[..., 0] * (1-xratio)
    #                 bbox[..., 1] *= yratio
    #                 bbox[..., 2] *= (1-xratio)
    #                 bbox[..., 3] *= yratio
    #             elif img_idx == 2:  # bottom left
    #                 output_image[newh:, :neww, :] = cv2.resize(image[2], (neww, image_size - newh))
    #                 bbox[..., 0] *= xratio
    #                 bbox[..., 1] *= bbox[..., 3] + bbox[..., 1] * (1-yratio)
    #                 bbox[..., 2] *= xratio
    #                 bbox[..., 3] *= (1-yratio)
    #             else:
    #                 output_image[newh:, neww:, :] = cv2.resize(image[3], (image_size - neww, image_size - newh))
    #                 bbox[..., 0] = bbox[..., 2] + bbox[..., 0] * (1 - xratio)
    #                 bbox[..., 1] = bbox[..., 3] + bbox[..., 1] * (1 - yratio)
    #                 bbox[..., 2] = bbox[..., 2] * (1 - xratio)
    #                 bbox[..., 3] = bbox[..., 3] * (1 - yratio)
    #             mosaic_bboxes_per_img.append(bbox)
    #         mosaic_bboxes.append(mosaic_bboxes_per_img)

        # cv2.imshow("MOSAIC AUG", output_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # return output_image, mosaic_bboxes

    def load_annotations(self):
        with open(self.annot_path, "r") as f:
            txt = f.readlines()
            if self.dataset_type == "converted_coco":
                annotations = [
                    line.strip()
                    for line in txt
                    if len(line.strip().split()[1:]) != 0
                ]
            elif self.dataset_type == "yolo":
                annotations = []
                for line in txt:
                    image_path = line.strip()
                    root, _ = os.path.splitext(image_path)

                    with open(root + ".txt") as fd:
                        boxes = fd.readlines()
                        string = ""
                        for box in boxes:
                            box = box.strip()
                            box = box.split()

                            if len(box) > 4:
                                class_num = int(box[0])
                                center_x = float(box[1])
                                center_y = float(box[2])
                                half_width = float(box[3]) / 2
                                half_height = float(box[4]) / 2
                                string += " {},{},{},{},{}".format(
                                    center_x - half_width,
                                    center_y - half_height,
                                    center_x + half_width,
                                    center_y + half_height,
                                    class_num,
                                )
                        annotations.append(image_path + string)

        np.random.shuffle(annotations)
        return annotations

    def __iter__(self):
        return self

    def __next__(self):
        with tf.device("/gpu:0"):
            # self.train_input_size = random.choice(self.train_input_sizes)
            self.num_detection_layers = cfg.YOLO.NUM_YOLOLAYERS
            self.train_input_size = cfg.TRAIN.INPUT_SIZE
            self.train_output_sizes = self.train_input_size // self.strides  # big -> small

            batch_image = np.zeros(
                (
                    self.batch_size,
                    self.train_input_size,
                    self.train_input_size,
                    self.input_channel,
                ),
                dtype=np.float32,
            )

            batch_label_sbbox = 0
            batch_label_mbbox = 0
            batch_label_lbbox = 0

            self.bboxes_list = []

            if self.num_detection_layers == 3:
                batch_label_lbbox = \
                    np.zeros(
                        (self.batch_size,
                         self.train_output_sizes[0],
                         self.train_output_sizes[0],
                         self.anchor_per_scale,
                         5 + self.num_classes
                         ), dtype=np.float32)

                batch_label_mbbox = \
                    np.zeros(
                        (self.batch_size,
                         self.train_output_sizes[1],
                         self.train_output_sizes[1],
                         self.anchor_per_scale,
                         5 + self.num_classes
                         ), dtype=np.float32)

                batch_label_sbbox = \
                    np.zeros(
                        (self.batch_size,
                         self.train_output_sizes[2],
                         self.train_output_sizes[2],
                         self.anchor_per_scale,
                         5 + self.num_classes
                         ), dtype=np.float32)

            elif self.num_detection_layers == 2:
                batch_label_mbbox = \
                    np.zeros(
                        (self.batch_size,
                         self.train_output_sizes[0],
                         self.train_output_sizes[0],
                         self.anchor_per_scale,
                         5 + self.num_classes
                         ), dtype=np.float32)

                batch_label_sbbox = \
                    np.zeros(
                        (self.batch_size,
                         self.train_output_sizes[1],
                         self.train_output_sizes[1],
                         self.anchor_per_scale,
                         5 + self.num_classes
                         ), dtype=np.float32)
            else:
                batch_label_mbbox = \
                    np.zeros(
                        (self.batch_size,
                         self.train_output_sizes[0],
                         self.train_output_sizes[0],
                         self.anchor_per_scale,
                         5 + self.num_classes
                         ), dtype=np.float32)

            batch_sbboxes = np.zeros(
                (self.batch_size, self.max_bbox_per_scale, 4), dtype=np.float32
            )
            batch_mbboxes = np.zeros(
                (self.batch_size, self.max_bbox_per_scale, 4), dtype=np.float32
            )
            batch_lbboxes = np.zeros(
                (self.batch_size, self.max_bbox_per_scale, 4), dtype=np.float32
            )

            num = 0
            if self.batch_count < self.num_batchs:
                while num < self.batch_size:
                    index = self.batch_count * self.batch_size + num
                    if index >= self.num_samples:
                        index -= self.num_samples
                    annotation = self.annotations[index]

                    if cfg.YOLO.NUM_YOLOLAYERS == 3:
                        image, bboxes = self.parse_annotation(annotation)
                        (
                            label_sbbox,
                            label_mbbox,
                            label_lbbox,
                            sbboxes,
                            mbboxes,
                            lbboxes,
                        ) = self.preprocess_true_boxes(bboxes, self.num_detection_layers)

                        batch_image[num, :, :, :] = image
                        batch_label_sbbox[num, :, :, :, :] = label_sbbox
                        batch_label_mbbox[num, :, :, :, :] = label_mbbox
                        batch_label_lbbox[num, :, :, :, :] = label_lbbox
                        batch_sbboxes[num, :, :] = sbboxes
                        batch_mbboxes[num, :, :] = mbboxes
                        batch_lbboxes[num, :, :] = lbboxes
                        num += 1
                    elif cfg.YOLO.NUM_YOLOLAYERS == 2:
                        image, bboxes = self.parse_annotation(annotation)
                        (
                            label_mbbox,
                            label_lbbox,
                            mbboxes,
                            lbboxes,
                        ) = self.preprocess_true_boxes(bboxes, self.num_detection_layers)
                        batch_image[num, :, :, :] = image
                        batch_label_mbbox[num, :, :, :, :] = label_mbbox
                        batch_label_lbbox[num, :, :, :, :] = label_lbbox
                        batch_mbboxes[num, :, :] = mbboxes
                        batch_lbboxes[num, :, :] = lbboxes
                        num += 1
                    else:
                        image, bboxes = self.parse_annotation(annotation)

                        (
                            label_mbbox,
                            mbboxes,
                        ) = self.preprocess_true_boxes(bboxes, self.num_detection_layers)

                        batch_image[num:num+1, :, :, :] = image
                        batch_label_mbbox[num:num+1, :, :, :, :] = label_mbbox
                        batch_mbboxes[num:num+1, :, :] = mbboxes
                        num += 1



                        # cv2.imshow(image, "image_222")
                        # cv2.waitKey(0)
                        # cv2.destroyAllWindows()

                        # if self.mosaic == True and max(1, num) % 4 == 0:
                        #     if random.randint(0,1) == 0:  # 50% 확률
                        #         mosaic_imgs = []
                        #         mosaic_bboxes = []
                        #         picked_indexes = random.sample(range(num-4, num), 4)
                        #         # print("picked indexes shape:{}".format(picked_indexes))
                        #         # print("batch_image shape:{}".format(np.array(batch_image).shape))
                        #
                        #         for picked_idx in picked_indexes:
                        #             mosaic_imgs.append(batch_image[picked_idx:picked_idx+1,...])
                        #             mosaic_bboxes.append(np.array(self.bboxes_list)[picked_idx:picked_idx+1,...])
                        #         # print("mosaic_bboxes shape:{}".format(np.array(mosaic_bboxes).shape))
                        #         image, bboxes = self.mosaic_aug(np.concatenate(mosaic_imgs, axis=0), np.concatenate(mosaic_bboxes, axis=0))
                self.batch_count += 1


                if cfg.YOLO.NUM_YOLOLAYERS == 3:  # yolov4
                    batch_smaller_target = batch_label_sbbox, batch_sbboxes
                    batch_medium_target = batch_label_mbbox, batch_mbboxes
                    batch_larger_target = batch_label_lbbox, batch_lbboxes
                    return (
                        batch_image,
                        (
                            batch_smaller_target,
                            batch_medium_target,
                            batch_larger_target,
                        ),
                    )
                elif cfg.YOLO.NUM_YOLOLAYERS == 2: # tiny yolov4
                    batch_medium_target = batch_label_mbbox, batch_mbboxes
                    batch_larger_target = batch_label_lbbox, batch_lbboxes
                    return (
                        batch_image,
                        (
                            batch_medium_target,
                            batch_larger_target,
                        ),
                    )
                elif cfg.YOLO.NUM_YOLOLAYERS == 1:
                    batch_target = batch_label_mbbox, batch_mbboxes

                    return (
                        batch_image,
                        (
                            batch_target,
                        ),
                    )

                # batch_smaller_target = batch_label_sbbox, batch_sbboxes
                # batch_medium_target = batch_label_mbbox, batch_mbboxes
                # batch_larger_target = batch_label_lbbox, batch_lbboxes

                # return (
                #     batch_image,
                #     (
                #         batch_smaller_target,
                #         batch_medium_target,
                #         batch_larger_target,
                #     ),
                # )
            else:
                self.batch_count = 0
                np.random.shuffle(self.annotations)
                raise StopIteration

    def random_horizontal_flip(self, image, bboxes):
        if random.random() < 0.5:
            _, w, _ = image.shape
            image = image[:, ::-1, :]
            bboxes[:, [0, 2]] = w - bboxes[:, [2, 0]]
        return image, bboxes

    def random_crop(self, image, bboxes):
        if random.random() < 0.5:
            h, w, _ = image.shape
            max_bbox = np.concatenate(
                [
                    np.min(bboxes[:, 0:2], axis=0),
                    np.max(bboxes[:, 2:4], axis=0),
                ],
                axis=-1,
            )

            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w - max_bbox[2]
            max_d_trans = h - max_bbox[3]

            crop_xmin = max(
                0, int(max_bbox[0] - random.uniform(0, max_l_trans))
            )
            crop_ymin = max(
                0, int(max_bbox[1] - random.uniform(0, max_u_trans))
            )
            crop_xmax = max(
                w, int(max_bbox[2] + random.uniform(0, max_r_trans))
            )
            crop_ymax = max(
                h, int(max_bbox[3] + random.uniform(0, max_d_trans))
            )

            image = image[crop_ymin:crop_ymax, crop_xmin:crop_xmax]

            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] - crop_xmin
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] - crop_ymin

        return image, bboxes

    def random_translate(self, image, bboxes):
        if random.random() < 0.5:
            h, w, _ = image.shape
            max_bbox = np.concatenate(
                [
                    np.min(bboxes[:, 0:2], axis=0),
                    np.max(bboxes[:, 2:4], axis=0),
                ],
                axis=-1,
            )

            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w - max_bbox[2]
            max_d_trans = h - max_bbox[3]

            tx = random.uniform(-(max_l_trans - 1), (max_r_trans - 1))
            ty = random.uniform(-(max_u_trans - 1), (max_d_trans - 1))

            M = np.array([[1, 0, tx], [0, 1, ty]])
            image = cv2.warpAffine(image, M, (w, h))

            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] + tx
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] + ty

        return image, bboxes

    def random_color(self, image, factor=0.5, srange=(-30, 30), vrange=(-30,30)):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        factor = np.random.uniform(0.5, 1.5)
        image[:,:,2] = np.clip(image[:,:,2] * factor, 0, 255)
        image= cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

        # cv2.imshow("COLOR AUG", image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return image

    def parse_annotation(self, annotation):
        line = annotation.split()
        image_path = line[0]

        if not os.path.exists(image_path):
            raise KeyError("%s does not exist ... " % image_path)
        image = cv2.imread(image_path)
        if self.dataset_type == "converted_coco":
            bboxes = np.array(
                [list(map(int, box.split(","))) for box in line[1:]]
            )
        elif self.dataset_type == "yolo":
            height, width, _ = image.shape
            bboxes = np.array(
                [list(map(float, box.split(","))) for box in line[1:]]
            )
            bboxes = bboxes * np.array([width, height, width, height, 1])  # <x_center> <y_center> <width> <height>
            bboxes = bboxes.astype(np.int64)

        if self.data_aug:
            image, bboxes = self.random_horizontal_flip(
                np.copy(image), np.copy(bboxes)
            )
            image, bboxes = self.random_crop(
                np.copy(image), np.copy(bboxes)
            )
            image, bboxes = self.random_translate(
                np.copy(image), np.copy(bboxes)
            )

            image = self.random_color(
                np.copy(image)
            )

            # if self.mosaic == True:
            #     self.bboxes_list.append(bboxes)

        image, bboxes = utils.image_preprocess(
            np.copy(image),
            [self.train_input_size, self.train_input_size],
            np.copy(bboxes),
        )
        return image, bboxes

    def preprocess_true_boxes(self, bboxes, num_detection_layers):
        label = [
            np.zeros(
                (
                    self.train_output_sizes[i],
                    self.train_output_sizes[i],
                    self.anchor_per_scale,
                    5 + self.num_classes,
                )
            )
            for i in range(num_detection_layers)
        ]

        count_overlap = [
            np.zeros(
                (
                    self.train_output_sizes[i],
                    self.train_output_sizes[i],
                )
                , dtype=np.int32
            )
            for i in range(num_detection_layers)
        ]

        bboxes_xywh = [np.zeros((self.max_bbox_per_scale, 4)) for _ in range(num_detection_layers)]
        bbox_count = np.zeros((num_detection_layers,))

        for bbox in bboxes:
            bbox_coor = bbox[:4]
            bbox_class_ind = bbox[4]

            onehot = np.zeros(self.num_classes, dtype=np.float)
            onehot[bbox_class_ind] = 1.0
            uniform_distribution = np.full(
                self.num_classes, 1.0 / self.num_classes
            )
            deta = 0.01
            smooth_onehot = onehot * (1 - deta) + deta * uniform_distribution

            bbox_xywh = np.concatenate(
                [
                    (bbox_coor[2:] + bbox_coor[:2]) * 0.5,
                    bbox_coor[2:] - bbox_coor[:2],
                ],
                axis=-1,
            )


            bbox_xywh_scaled = (
                1.0 * bbox_xywh[np.newaxis, :] / self.strides[:, np.newaxis]
            )

            iou = []
            exist_positive = False
            for i in range(num_detection_layers):
                anchors_xywh = np.zeros((self.anchor_per_scale, 4))
                anchors_xywh[:, 0:2] = (
                    np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32) + 0.5
                )
                # anchors_xywh[:, 2:4] = self.anchors[i]
                anchors_xywh[:, 2:4] = self.anchors[i] / cfg.YOLO.STRIDES_CUSTOM[i]

                iou_scale = utils.bbox_iou(
                    bbox_xywh_scaled[i][np.newaxis, :], anchors_xywh
                )
                iou.append(iou_scale)
                iou_mask = iou_scale > 0.3


                if np.any(iou_mask):
                    xind, yind = np.floor(bbox_xywh_scaled[i, 0:2]).astype(
                        np.int32
                    )


                    label_idx = count_overlap[i][yind, xind]
                    # label[i][yind, xind, iou_mask, :] = 0
                    # label[i][yind, xind, iou_mask, 0:4] = bbox_xywh
                    # label[i][yind, xind, iou_mask, 4:5] = 1.0
                    # label[i][yind, xind, iou_mask, 5:] = smooth_onehot
                    label[i][yind, xind, label_idx:label_idx+1, :] = 0
                    label[i][yind, xind, label_idx:label_idx+1, 0:4] = bbox_xywh
                    label[i][yind, xind, label_idx:label_idx+1, 4:5] = 1.0
                    label[i][yind, xind, label_idx:label_idx+1, 5:] = smooth_onehot
                    count_overlap[i][yind, xind] += 1   # 한 grid 에 중심점 3개 초과시 에러날 것

                    bbox_ind = int(bbox_count[i] % self.max_bbox_per_scale)
                    bboxes_xywh[i][bbox_ind, :4] = bbox_xywh
                    bbox_count[i] += 1

                    exist_positive = True

            if not exist_positive:
                best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1)

                best_detect = int(best_anchor_ind / self.anchor_per_scale)

                best_anchor = int(best_anchor_ind % self.anchor_per_scale)
                xind, yind = np.floor(
                    bbox_xywh_scaled[best_detect, 0:2]
                ).astype(np.int32)

                label_idx = count_overlap[best_detect][yind, xind]
                label[best_detect][yind, xind, label_idx:label_idx+1, :] = 0
                label[best_detect][yind, xind, label_idx:label_idx+1, 0:4] = bbox_xywh
                label[best_detect][yind, xind, label_idx:label_idx+1, 4:5] = 1.0
                label[best_detect][yind, xind, label_idx:label_idx+1, 5:] = smooth_onehot
                count_overlap[best_detect][yind, xind] += 1  # 한 grid 에 중심점 3개 초과시 에러날 것

                ## 원래
                # label[best_detect][yind, xind, best_anchor, :] = 0
                # label[best_detect][yind, xind, best_anchor, 0:4] = bbox_xywh
                # label[best_detect][yind, xind, best_anchor, 4:5] = 1.0
                # label[best_detect][yind, xind, best_anchor, 5:] = smooth_onehot

                bbox_ind = int(
                    bbox_count[best_detect] % self.max_bbox_per_scale
                )
                bboxes_xywh[best_detect][bbox_ind, :4] = bbox_xywh
                bbox_count[best_detect] += 1
        if cfg.YOLO.NUM_YOLOLAYERS == 3:
            label_sbbox, label_mbbox, label_lbbox = label
            sbboxes, mbboxes, lbboxes = bboxes_xywh
            return label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes
        elif cfg.YOLO.NUM_YOLOLAYERS == 2:
            label_mbbox, label_lbbox = label
            mbboxes, lbboxes = bboxes_xywh
            return label_mbbox, label_lbbox, mbboxes, lbboxes
        elif cfg.YOLO.NUM_YOLOLAYERS == 1:
            label_mbbox = label
            mbboxes = bboxes_xywh
            return label_mbbox, mbboxes
        # return label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes

    def __len__(self):
        return self.num_batchs
