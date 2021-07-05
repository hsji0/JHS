#! /usr/bin/env python
# coding=utf-8

import os
import cv2
import random
import numpy as np
import tensorflow as tf
from tensorflow.python.data.ops.dataset_ops import AUTOTUNE
import core.utils as utils
from core.config import cfg
import tensorflow_addons as tfa

class ImageShape:
    def __init__(self, input_shape):
        assert len(input_shape) == 3
        self._h, self._w, self._c = input_shape

    def __eq__(self, o: object) -> bool:
        return self._h == o._h and self._w == o._w and self._c == o._c

    @property
    def height(self):
        return self._h

    @property
    def width(self):
        return self._w

    @property
    def channel(self):
        return self._c

    @property
    def h(self):
        return self._h

    @property
    def w(self):
        return self._w

    @property
    def c(self):
        return self._c

    @property
    def shape(self):
        return (self._h, self._w, self._c)


class DatasetTF(object):
    """implement Dataset here"""

    def __init__(self, FLAGS, input_channel, is_training: bool, dataset_type: str = "yolo"):
        self.input_channel = input_channel
        self.strides, self.anchors, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
        print(f"self.strides: {self.strides}, self.anchors: {self.anchors}, NUM_CLASS: {NUM_CLASS}, XYSCALE: {XYSCALE}")
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

        self._input = ImageShape((cfg.TRAIN.INPUT_SIZE, cfg.TRAIN.INPUT_SIZE, input_channel))  if is_training \
            else ImageShape((cfg.TEST.INPUT_SIZE, cfg.TEST.INPUT_SIZE, input_channel))  # (y, x, c)
        output_shape = None
        self._output = ImageShape(output_shape) if output_shape is not None else self._input
        self.num_detection_layers = cfg.YOLO.NUM_YOLOLAYERS

        self.classes = utils.read_class_names(cfg.YOLO.CLASSES)
        self.num_classes = len(self.classes)
        self.anchor_per_scale = cfg.YOLO.ANCHOR_PER_SCALE
        self.num_detection_layers = cfg.YOLO.NUM_YOLOLAYERS
        self.max_bbox_per_scale = 150

        self.train_input_sizes = cfg.TRAIN.INPUT_SIZE if is_training else cfg.TEST.INPUT_SIZE
        self.train_input_size = cfg.TRAIN.INPUT_SIZE if is_training else cfg.TEST.INPUT_SIZE
        self.train_output_sizes = self.train_input_size // self.strides  # big -> small
        self._train_images = []
        self._train_labels = []

        self.load_train_label()
        self.steps_for_train = int(len(self._train_images) / self.batch_size)
        # self.annotations = self.load_annotations()
        self.num_samples = len(self._train_images)
        self.num_batchs = int(np.ceil(self.num_samples / self.batch_size))
        self.batch_count = 0
        # self.mosaic = cfg.TRAIN.MOSAIC_AUG


    # def mosaic_aug(self, image, bboxes, xrange=[0.4, 0.6], yrange=[0.4, 0.6]):
    #     image_size = cfg.TRAIN.INPUT_SIZE
    #     output_image = np.zeros((image_size, image_size, 3), dtype=np.float32)
    #     xratio = xrange[0] + (xrange[1] - xrange[0]) * random.random()
    #     neww = int(image_size * xratio)
    #     yratio = yrange[0] + (yrange[1] - yrange[0]) * random.random()
    #     newh = int(image_size * yratio)
    #
    #     mosaic_bboxes = []
    #     for img_idx, bboxes_per_img in enumerate(bboxes):
    #         mosaic_bboxes_per_img = []
    #         for bbox_idx, bbox in enumerate(bboxes_per_img):
    #             bbox = np.array(bbox).astype(np.float32)
    #             if img_idx == 0:
    #                 output_image[:newh, :neww, :] = cv2.resize(image[0], (neww, newh))
    #                 bbox[..., 0] *= xratio
    #                 bbox[..., 1] *= yratio
    #                 bbox[..., 2] *= xratio
    #                 bbox[..., 3] *= yratio
    #             elif img_idx == 1:
    #                 output_image[:newh, neww:, :] = cv2.resize(image[1], (image_size - neww, newh))
    #                 bbox[..., 0] = bbox[..., 2] + bbox[..., 0] * (1 - xratio)
    #                 bbox[..., 1] *= yratio
    #                 bbox[..., 2] *= (1 - xratio)
    #                 bbox[..., 3] *= yratio
    #             elif img_idx == 2:
    #                 output_image[newh:, :neww, :] = cv2.resize(image[2], (neww, image_size - newh))
    #                 bbox[..., 0] *= xratio
    #                 bbox[..., 1] *= bbox[..., 3] + bbox[..., 1] * (1 - yratio)
    #                 bbox[..., 2] *= xratio
    #                 bbox[..., 3] *= (1 - yratio)
    #             else:
    #                 output_image[newh:, neww:, :] = cv2.resize(image[3], (image_size - neww, image_size - newh))
    #                 bbox[..., 0] = bbox[..., 2] + bbox[..., 0] * (1 - xratio)
    #                 bbox[..., 1] = bbox[..., 3] + bbox[..., 1] * (1 - yratio)
    #                 bbox[..., 2] = bbox[..., 2] * (1 - xratio)
    #                 bbox[..., 3] = bbox[..., 3] * (1 - yratio)
    #             mosaic_bboxes_per_img.append(bbox)
    #         mosaic_bboxes.append(mosaic_bboxes_per_img)
    #
    #     return output_image, mosaic_bboxes

    def load_train_label(self):
        with open(self.annot_path, "r") as f:
            txt = f.readlines()
            annotations = []
            for line in txt:
                image_path = line.strip()
                label_path = os.path.splitext(image_path)[0] + ".txt"
                self._train_images.append(image_path)
                self._train_labels.append(label_path)
        # self._train_images = self._train_images[5000:]
        # self._train_labels= self._train_labels[5000:]

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

    @tf.function
    def _load_images(self, image_fullpath):
        img = tf.io.read_file(image_fullpath)
        img = tf.io.decode_image(img)  #, channels=0)
        img = tf.cond(tf.shape(img)[2] == 1 and self._input.c == 3, lambda: tf.image.grayscale_to_rgb(img), lambda: img)
        img = tf.cond(tf.shape(img)[2] == 3 and self._input.c == 1, lambda: tf.image.rgb_to_grayscale(img), lambda: img)
        img.set_shape(self._input.shape)  # 실제 이미지 크기랑 상관 없음
        return img

    @tf.function
    def _load_labels(self, label_fullpath):
        label = tf.io.read_file(label_fullpath)
        if label == '':
            bboxes = tf.constant([[0,0,0,0,-1]], dtype=tf.float32)
        else:
            boxes = tf.strings.to_number(tf.strings.split(label))
            boxes = tf.reshape(boxes, (len(boxes) // 5, 5))
            # class_num, center_x, center_y, half_w, half_h = tf.split(boxes, 5, 1)
            class_num, center_x, center_y, width_, height_ = tf.split(boxes, 5, 1)

            half_w, half_h = width_ / 2, height_ / 2
            bboxes = tf.concat([center_x - half_w, center_y - half_h, center_x + half_w, center_y + half_h, class_num], 1)
            bboxes = tf.multiply(bboxes, tf.constant([self._input.w, self._input.h, self._input.w, self._input.h, 1],
                                                     dtype=tf.float32))
        return bboxes

    @tf.function
    def _preprocess_images(self, img):
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, (self._input.h, self._input.w))

        ## add more preprocessing here
        # img /= 255.0
        ## ex) img = tf.image.per_image_standarization(img)
        return img

    @tf.function
    def augmentation(self, img, lbl):
        img, lbl = self.random_crop(img, lbl)
        img, lbl = self.random_horizontal_flip(img, lbl)
        img      = self.random_color(img)
        # img, lbl = self.random_translate(img, lbl)
        return (img, lbl)

    # def pad_image(self, img_dataset):
    #     height, width, channels = tf.keras.backend.int_shape(img_dataset)
    #     paddings = tf.constant([[0, self.input_sizes - height], [0, self.input_sizes - width]])
    #     img = tf.pad(img_dataset, paddings, mode='CONSTANT', constant_values=0)
    #
    #     return img

    @tf.function
    def _adjust_shape(self, img_dataset, preprocessed_lbl, lbl_dataset):
        img = img_dataset
        batch_target = preprocessed_lbl, lbl_dataset
        return (
            img,
            (
                batch_target,
            ),
        )

    def get_dataset(self):
        img_dataset = tf.data.Dataset.from_tensor_slices(self._train_images)
        img_dataset = img_dataset.map(map_func=self._load_images,
                                      num_parallel_calls=AUTOTUNE)
        img_dataset = img_dataset.map(map_func=self._preprocess_images,
                                      num_parallel_calls=AUTOTUNE)
        lbl_dataset = tf.data.Dataset.from_tensor_slices(self._train_labels)
        lbl_dataset = lbl_dataset.map(map_func=self._load_labels,
                                      num_parallel_calls=AUTOTUNE)

        dataset = tf.data.Dataset.zip((img_dataset, lbl_dataset))

        # dataset = dataset.map(
        #     lambda img_dataset, lbl_dataset: tf.py_function(self.augmentation, [img_dataset, lbl_dataset], [tf.float32, tf.float32])
        # )

        if self.data_aug:
            dataset = dataset.map(
                lambda img_dataset, lbl_dataset: self.augmentation(img_dataset, lbl_dataset), num_parallel_calls=AUTOTUNE)

        dataset = dataset.map(
            lambda img_dataset, lbl_dataset: tf.py_function(self.preprocess_true_boxes, [img_dataset, lbl_dataset], [tf.float32, tf.float32, tf.float32]),
                                                            num_parallel_calls=AUTOTUNE)
        # dataset = dataset.map(
        #     lambda img_dataset, lbl_dataset: tf.numpy_function(self.preprocess_true_boxes, [img_dataset, lbl_dataset],
        #                                                     [tf.float32, tf.float32, tf.float32]),
        #     num_parallel_calls=AUTOTUNE)

        dataset = dataset.map(map_func=self._adjust_shape,  num_parallel_calls=AUTOTUNE)

        dataset = dataset.cache("")
        dataset = dataset.shuffle(16000, reshuffle_each_iteration=True)
        dataset = dataset.repeat()
        dataset = dataset.batch(self.batch_size).prefetch(AUTOTUNE)  # https://github.com/tensorflow/tensorflow/issues/32376
        return dataset

    def __iter__(self):
        return self

    @tf.function
    def random_horizontal_flip(self, image, bboxes):
        random_num = tf.random.uniform([1, ], 0, 4)
        if random_num[0] < 2:
            w = self.input_sizes
            image = image[:, ::-1, :]
            new_right = w - bboxes[:,0:1]
            new_left = w - bboxes[:,2:3]

            bboxes = tf.concat([new_left, bboxes[:, 1:2], new_right, bboxes[:, 3:5]], axis=1)
        return image, bboxes

    # @tf.function
    # def random_vertical_flip(self, image, bboxes):
    #     if random.random() < 0.5:
    #         h = self.input_sizes
    #         image = image[:, :, ::-1]
    #           new_up = h - bboxes[:,1:2]
    #           new_down = h - bboxes[:,3:4]
    #           bboxes = tf.concat([bboxes[:,0:1], new_up, bboxes[:,2:3],new_down, bboxes[:,4:5]] axis=1)
    #     return image, bboxes

    @tf.function
    def random_crop(self, image, bboxes):  # 일반적으로 사람, 물건 등 detection 문제에서는 성능 높을지 몰라도 거의 똑같이 생긴 웨이퍼 등에도 적용해야 할지?
        random_num = tf.random.uniform([1, ], 0, 4)
        if random_num[0] < 1:
            h, w, ch = tf.keras.backend.int_shape(image)
            max_bbox = tf.concat(
                [
                tf.reduce_min(bboxes[:,0:2], axis=0),
                tf.reduce_max(bboxes[:,2:4], axis=0),
                ],
                axis=-1,
            )

            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w - max_bbox[2]
            max_d_trans = h - max_bbox[3]

            crop_xmin = tf.maximum(
                0, tf.cast(max_bbox[0] - tf.random.uniform([], minval=0, maxval=max_l_trans), dtype=tf.int32)
            )
            crop_ymin = tf.maximum(
                0, tf.cast(max_bbox[1] - tf.random.uniform([], minval=0, maxval=max_u_trans), dtype=tf.int32)
            )
            crop_xmax = tf.minimum(
                w, tf.cast(max_bbox[2] + tf.random.uniform([], minval=0, maxval=max_r_trans), dtype=tf.int32)
            )
            crop_ymax = tf.minimum(
                h, tf.cast(max_bbox[3] + tf.random.uniform([], minval=0, maxval=max_d_trans), dtype=tf.int32)
            )

            image = image[crop_ymin:crop_ymax, crop_xmin:crop_xmax]

            bboxes_list = []

            bboxes_list.append(tf.subtract(bboxes[:, 0:1], tf.maximum(tf.cast(0., dtype=tf.float32), tf.subtract(tf.cast(crop_xmin, dtype=tf.float32), bboxes[:,0:1]))))
            bboxes_list.append(tf.subtract(bboxes[:, 1:2], tf.maximum(tf.cast(0., dtype=tf.float32), tf.subtract(tf.cast(crop_ymin, dtype=tf.float32), bboxes[:,1:2]))))
            bboxes_list.append(tf.subtract(bboxes[:, 2:3], tf.maximum(tf.cast(0., dtype=tf.float32), tf.subtract(bboxes[:,2:3], tf.cast(crop_xmax, dtype=tf.float32)))))
            bboxes_list.append(tf.subtract(bboxes[:, 3:4], tf.maximum(tf.cast(0., dtype=tf.float32), tf.subtract(bboxes[:,3:4], tf.cast(crop_ymax, dtype=tf.float32)))))
            bboxes_list.append(bboxes[:,4:5]) # class label

            bboxes = tf.concat(bboxes_list, axis=1)

            paddings =  [[crop_ymin, h - crop_ymax], [crop_xmin, w - crop_xmax], [0,0]]  #tf.constant()
            image = tf.pad(image, paddings, mode='CONSTANT', constant_values=0.5)

        return image, bboxes

    @tf.function
    def random_translate(self, image, bboxes):
        random_num = tf.random.uniform([1, ], 0, 4)
        if random_num[0] < 1:
            max_bbox = tf.concat(
                [
                tf.reduce_min(bboxes[:,0:2], axis=0),
                tf.reduce_max(bboxes[:,2:4], axis=0),
                ],
                axis=-1,
            )

            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w - max_bbox[2]
            max_d_trans = h - max_bbox[3]

            tx = tf.random.uniform([], -(max_l_trans - 1), (max_r_trans - 1))
            ty = tf.random.uniform([], -(max_u_trans - 1), (max_d_trans - 1))
            # crop 하며 bbox가 많이 손상되지 않도록 주의
            # 모든 데이터셋에 대해 과검나지 않고 돌도록 다시 체크
            image = tfa.image.transform(image, transforms=(1,0,ty,0,1,tx,0,0), interpolation= "nearest")
            bboxes = tf.concat([bboxes[:,0:1]+tx, bboxes[:,1:2]+ty, bboxes[:,2:3]+tx, bboxes[:,3:4]+ty, bboxes[:,4:5]], axis=1)

        return image, bboxes

    @tf.function
    def random_color(self, image):
        random_num = tf.random.uniform([1, ], 0, 4)
        if random_num[0] < 1:
            image = tf.image.random_brightness(image, 0.1)
        random_num = tf.random.uniform([1, ], 0, 4)
        if random_num[0] < 1:
            image = tf.image.random_saturation(image, 0, 0.1)
        image = tf.clip_by_value(image, 0, 1.0)
        return image

    def preprocess_true_boxes(self, image, bboxes):
        label = [
            np.zeros(
                (
                    self.train_output_sizes[i],
                    self.train_output_sizes[i],
                    self.anchor_per_scale,
                    5 + self.num_classes,
                )
            )
            for i in range(self.num_detection_layers)
        ]
        count_overlap = [
            np.zeros(
                (
                    self.train_output_sizes[i],
                    self.train_output_sizes[i],
                )
                , dtype=np.int32
            )
            for i in range(self.num_detection_layers)
        ]

        _, feath, featw, _, _ = np.array(label).shape

        bboxes_xywh = [np.zeros((self.max_bbox_per_scale, 4)) for _ in range(self.num_detection_layers)]
        bbox_count = np.zeros((self.num_detection_layers,))

        """
        negative sample 처리
        """
        if np.cast[np.int](bboxes[0][4]) == -1:  # bbox_class_ind = np.cast[np.int](bboxes[0][4])
            for i in range(cfg.YOLO.NUM_YOLOLAYERS):
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
                    label_mbbox = np.reshape(label_mbbox, [feath, featw, self.anchor_per_scale, 5 + self.num_classes])
                    mbboxes = bboxes_xywh
                    mbboxes = np.reshape(mbboxes, [-1, 4])
                    return (tf.cast(image, tf.float32), tf.cast(label_mbbox, tf.float32), tf.cast(mbboxes, tf.float32))

        for bbox in bboxes:
            bbox_coor = bbox[:4]
            bbox_class_ind = np.cast[np.int](bbox[4]) #tf.cast(bbox[4], tf.int32)
            # print("bbox_class_ind:{}".format(bbox_class_ind))
            # onehot = tf.one_hot(tf.cast(bbox[4],tf.int32), self.num_classes)
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
            bbox_xywh = np.minimum(bbox_xywh, self.train_input_size-1)
            bbox_xywh = np.maximum(bbox_xywh, 0)

            bbox_xywh_scaled = (
                    1.0 * bbox_xywh[np.newaxis, :] / self.strides[:, np.newaxis]
            )
            bbox_xywh_scaled = np.minimum(bbox_xywh_scaled, 19)
            bbox_xywh_scaled = np.maximum(bbox_xywh_scaled, 0)

            iou = []
            exist_positive = False
            i = 0
            for i in range(self.num_detection_layers):  ## self.num_detection_layers == 1
                anchors_xywh = np.zeros((self.anchor_per_scale, 4))  ## self.anchor_per_scale == 3
                anchors_xywh[:, 0:2] = (
                        np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32) + 0.5
                )

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

                    # label[i][yind, xind, iou_mask, :] = 0
                    # label[i][yind, xind, iou_mask, 0:4] = bbox_xywh
                    # label[i][yind, xind, iou_mask, 4:5] = 1.0
                    # label[i][yind, xind, iou_mask, 5:] = smooth_onehot
                    label_idx = count_overlap[i][yind, xind]
                    label[i][yind, xind, label_idx:label_idx + 1, :] = 0
                    label[i][yind, xind, label_idx:label_idx + 1, 0:4] = bbox_xywh
                    label[i][yind, xind, label_idx:label_idx + 1, 4:5] = 1.0
                    label[i][yind, xind, label_idx:label_idx + 1, 5:] = smooth_onehot
                    count_overlap[i][yind, xind] += 1  # 한 grid 에 중심점 3개 초과시 에러날 것

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
                label[best_detect][yind, xind, label_idx:label_idx + 1, :] = 0
                label[best_detect][yind, xind, label_idx:label_idx + 1, 0:4] = bbox_xywh
                label[best_detect][yind, xind, label_idx:label_idx + 1, 4:5] = 1.0
                label[best_detect][yind, xind, label_idx:label_idx + 1, 5:] = smooth_onehot
                count_overlap[best_detect][yind, xind] += 1  # 한 grid 에 중심점 3개 초과시 에러날 것

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
            label_mbbox = np.reshape(label_mbbox, [feath, featw, self.anchor_per_scale, 5 + self.num_classes])
            mbboxes = bboxes_xywh
            mbboxes = np.reshape(mbboxes, [-1, 4])
            return (tf.cast(image, tf.float32), tf.cast(label_mbbox, tf.float32), tf.cast(mbboxes, tf.float32))

    def __len__(self):
        return self.num_batchs


# if __name__ == "__main__":
#     from absl import app, flags, logging
#     from absl.flags import FLAGS
#     # from core.dataset_tf import DatasetTF
#     # from core.dataset import Dataset as DTN
#     import numpy as np
#     from ati.general_function import dump_pickle, get_pickle
#     import tensorflow as tf
#     from tensorflow.python.data.ops.dataset_ops import AUTOTUNE
#
#     flags.DEFINE_string('model', 'yolov4', 'yolov4, yolov3')
#     flags.DEFINE_string('weights', './scripts/yolov4.weights', 'pretrained weights')
#     flags.DEFINE_integer('num_detection_layer', 1, '3: yolov4 2:yolov4-tiny  3:custom model')
#
#     input_channel = 3
#
#     self = DatasetTF(FLAGS, input_channel, is_training=True)
#
#     dataset = self.get_dataset()
#     it = iter(dataset)
#     data = it.get_next()
#
#     image_data = test[0]
#
#     for i in range(32):
#         check = np.array(image_data[i]).reshape(640,640,3)
#         cv2.imshow("img", check)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
#
#     # train_step(data[0], data[1])
#
#     # ref = get_pickle(r"D:\ATI_CODE\0.ALGO_CODE\TF2.0U\tensorflow-yolov4-tflite-master\scripts\yolo_custom\test.pkl")
#     #
#     # test[0].shape
#     # test[1][0][0].shape
#     # test[1][0][1].shape
#     #
#     # test_imgs = test[0]
#     # test_lbl1 = test[1][0][0]
#     # test_lbl2 = test[1][0][1]
#     # test_lbl2.shape
#     #
#     # dataset_normal = DTN(FLAGS, input_channel, is_training=True)
#     # ref = self.__next__()
#     # ref_imgs = ref[0]
#     # ref_lbl1 = ref[1][0][0]
#     # ref_lbl2 = ref[1][0][1]
#     # ref_bbox = self.temp_bbox_list
#     #
#     # for idx, (image_data, target) in enumerate(dataset_normal):
#     #     print("uidx : ", idx)
#     #     print("image_data.shape ", image_data.shape)
#     #     print("target[0][0].shape ", target[0][0].shape)
#     #     print("target[0][1].shape ", target[0][1].shape)
#     #
#     # ######
#     # tf.reduce_all(tf.equal(test_imgs[0], ref_imgs[2]))
#     # ref_bbox[0].shape #9_h3436w3500
#     # ref_bbox[1].shape #15
#     # ref_bbox[2].shape #19
#     # ref_bbox[3].shape #9_h3436w2860
#     # dd = {}
#     # dd['19']={"img":ref_imgs[2], "label_mbbox":ref_lbl1[2], "mbboxes":ref_lbl2[2], "bbox":ref_bbox[2]}
#     # dd['15']={"img":ref_imgs[1], "label_mbbox":ref_lbl1[1], "mbboxes":ref_lbl2[1], "bbox":ref_bbox[1]}
#     # dd['9_h3436w3500']={"img":ref_imgs[0], "label_mbbox":ref_lbl1[0], "mbboxes":ref_lbl2[0], "bbox":ref_bbox[0]}
#     # dd['9_h3436w2860']={"img":ref_imgs[3], "label_mbbox":ref_lbl1[3], "mbboxes":ref_lbl2[3], "bbox":ref_bbox[3]}
#     # dump_pickle(dd, r"D:\ATI_CODE\0.ALGO_CODE\TF2.0U\tensorflow-yolov4-tflite-master\scripts\yolo_custom/test.pkl")
