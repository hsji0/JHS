import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from core.config import cfg
from core.utils import load_multiple_img
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from core.utils import nms
import os

"""
+multiple result save
+pretrained weight use
+early stopping
+negative sample training 
+nms for other shape of yolo && for batch 
+preprocess해서 model 결과 본 걸 다시 원래 이미지처럼 바꾸는 postprocess (resize, pad 등)
? classweight
"""

# 고정
flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt)')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_float('iou', 0.5, 'iou threshold')
flags.DEFINE_float('score', 0.25, 'score threshold')
# flags.DEFINE_string('weights', r'D:\ckeckpoint\210413_custom','path to weights file')

# detection시 수정
flags.DEFINE_string('weights', r"D:/ckeckpoint",'path to weights file')
flags.DEFINE_string('save_result_img', r"./data/detection/nmserr", 'folder to save result')
flags.DEFINE_boolean('use_trainres', True, "false for detect using converted weight(weights->pb) / True for training from scratch")
flags.DEFINE_integer('size', 640, 'resize images to')
flags.DEFINE_integer('num_detection_layer', 1, "3:yolov4 2:yolo-tiny 1:custom model")  # 디폴트 custom model
flags.DEFINE_string('image', r"./data/detection/nmserr", 'path to input image (.bmp면 하나만 detect ,폴더명이면 해당 폴더 이미지 전부')
flags.DEFINE_boolean('save_multiple', True, 'detection 결과 한꺼번에 여러개 보려면 True')
# flags.DEFINE_string('output', r"...\test_result", 'path to output image')


def nms_convertedw(boxes, num_classes, iou_threshold=0.5, score_threshold=0.25,max_boxes_per_class=50):
    batch, num_bbox, xywhcp = tf.keras.backend.int_shape(boxes)

    boxes = np.array(boxes).reshape(1, -1, xywhcp)  # (batch, -1, xywhcp)
    boxes_conf = boxes[..., 4:]
    boxes_conf = np.max(boxes_conf, axis=-1, keepdims=True)
    conf_mask = boxes_conf >= score_threshold
    boxes_conf = boxes_conf[conf_mask]
    boxes_conf = np.reshape(boxes_conf, newshape=(1, boxes_conf.shape[0], 1))

    conf_mask = np.tile(conf_mask, (1, 1, 4 + num_classes))
    boxes = boxes[conf_mask]
    boxes = np.reshape(boxes, newshape=(1, -1, 4 + num_classes))

    boxes_classes = np.reshape(np.argmax(boxes[..., 4:], axis=-1), newshape=(1, -1, 1))
    boxes_coord = boxes[..., :4]

    bboxes_coord = []
    bboxes_scores = []
    bboxes_classes = []
    for class_ind in range(num_classes):
        mask_class = boxes_classes[..., 0:1] == class_ind
        boxes_class = boxes_classes[mask_class]
        boxes_conf_class = boxes_conf[mask_class]

        mask_class = np.tile(mask_class, (1, 1, 4))
        boxes_coord_class = boxes_coord[mask_class]
        boxes_coord_class = np.reshape(boxes_coord_class, (1, -1, 4))

        # conf 내림차순 정렬
        sorted_idx = np.argsort(boxes_conf_class)
        sorted_idx = sorted_idx[::-1]

        boxes_class = np.reshape(boxes_class, newshape=(len(sorted_idx), 1))
        boxes_class = boxes_class[sorted_idx]
        # boxes_class = np.expand_dims(boxes_class, axis=0)

        boxes_conf_class = np.reshape(boxes_conf_class, newshape=(len(sorted_idx), 1))
        boxes_conf_class = boxes_conf_class[sorted_idx]
        # boxes_conf_class = np.expand_dims(boxes_conf_class, axis=0)

        boxes_coord_class = np.reshape(boxes_coord_class, newshape=(len(sorted_idx), 4))
        boxes_coord_class = boxes_coord_class[sorted_idx]
        # boxes_coord_class = np.expand_dims(boxes_coord_class, axis=0)

        best_conf_ind = 0
        num_process = boxes_class.shape[0]
        while best_conf_ind + 1 < num_process:
            iou_scores = utils.bbox_iou(boxes_coord_class[best_conf_ind:best_conf_ind + 1, :],
                                  boxes_coord_class[best_conf_ind + 1:, :])
            iou_mask = iou_scores < iou_threshold
            iou_mask = np.reshape(iou_mask, newshape=(-1, 1))

            boxes_class = np.vstack([boxes_class[:best_conf_ind, :],
                                     np.expand_dims(boxes_class[best_conf_ind + 1:, :][iou_mask], axis=-1)])
            boxes_conf_class = np.vstack([boxes_conf_class[:best_conf_ind, :],
                                          np.expand_dims(boxes_conf_class[best_conf_ind + 1:, :][iou_mask], axis=-1)])

            iou_mask = np.tile(iou_mask, (1, 4))
            boxes_coord_class = np.vstack([boxes_coord_class[:best_conf_ind, :],
                                           np.reshape(boxes_coord_class[best_conf_ind + 1:, :][iou_mask],
                                                      newshape=(-1, 4))])

            best_conf_ind += 1
            num_process, _ = np.array(boxes_coord_class).shape

        bboxes_coord.append(boxes_coord_class)
        bboxes_scores.append(boxes_conf_class)
        bboxes_classes.append(boxes_class)

    bboxes_coord = np.vstack(bboxes_coord)
    bboxes_scores = np.vstack(bboxes_scores)
    bboxes_classes = np.vstack(bboxes_classes)
    return bboxes_coord, bboxes_scores, bboxes_classes

def nms(boxes, num_classes, iou_threshold=0.5, score_threshold=0.25,max_boxes_per_class=50):
    # batch, feath, featw, num_anchors, xywhcp = tf.keras.backend.int_shape(boxes)  # for train from scratch
    batch, feath, featw, num_anchors, xywhcp = tf.keras.backend.int_shape(boxes) # for yolov4 pretrained weights

    # print("tf.keras.backend.int_shape(boxes) :{}".format(tf.keras.backend.int_shape(boxes) ))
    boxes = np.array(boxes).reshape(1, -1, xywhcp)  # (batch, -1, xywhcp)
    boxes_conf = boxes[..., 4:5]
    boxes_classprob = boxes[..., 5:]

    boxes_conf = boxes_conf * boxes_classprob
    boxes_conf = np.max(boxes_conf, axis=-1, keepdims=True)
    conf_mask = boxes_conf >= score_threshold
    boxes_conf = boxes_conf[conf_mask]
    boxes_conf = np.reshape(boxes_conf, newshape=(1, boxes_conf.shape[0], 1))
    # print("boxes_conf:{}".format(boxes_conf))

    conf_mask = np.tile(conf_mask, (1, 1, 5 + num_classes))
    boxes = boxes[conf_mask]
    boxes = np.reshape(boxes, newshape=(1, len(boxes) // (5 + num_classes), 5 + num_classes))

    boxes_classes = np.reshape(np.argmax(boxes[..., 5:], axis=-1), newshape=(1, -1, 1))
    boxes_coord = boxes[..., :4]

    bboxes_coord = []
    bboxes_scores = []
    bboxes_classes = []
    for class_ind in range(num_classes):
        mask_class = boxes_classes[..., 0:1] == class_ind
        boxes_class = boxes_classes[mask_class]
        boxes_conf_class = boxes_conf[mask_class]

        mask_class = np.tile(mask_class, (1, 1, 4))
        boxes_coord_class = boxes_coord[mask_class]
        boxes_coord_class = np.reshape(boxes_coord_class, (1, -1, 4))

        # conf 내림차순 정렬
        sorted_idx = np.argsort(-boxes_conf_class)
        # sorted_idx = sorted_idx[::-1]

        boxes_class = np.reshape(boxes_class, newshape=(len(sorted_idx), 1))
        boxes_class = boxes_class[sorted_idx]
        # boxes_class = np.expand_dims(boxes_class, axis=0)

        boxes_conf_class = np.reshape(boxes_conf_class, newshape=(len(sorted_idx), 1))
        boxes_conf_class = boxes_conf_class[sorted_idx]
        # boxes_conf_class = np.expand_dims(boxes_conf_class, axis=0)

        boxes_coord_class = np.reshape(boxes_coord_class, newshape=(len(sorted_idx), 4))
        boxes_coord_class = boxes_coord_class[sorted_idx]
        # boxes_coord_class = np.expand_dims(boxes_coord_class, axis=0)

        best_conf_ind = 0
        num_process = boxes_class.shape[0]
        while best_conf_ind+1 < num_process:
            iou_scores = utils.bbox_iou(boxes_coord_class[best_conf_ind:best_conf_ind + 1, :],
                                  boxes_coord_class[best_conf_ind + 1:, :])
            iou_mask = iou_scores < iou_threshold
            iou_mask = np.reshape(iou_mask, newshape=(-1, 1))

            boxes_class = np.vstack([boxes_class[:best_conf_ind + 1, :],
                                     np.expand_dims(boxes_class[best_conf_ind + 1:, :][iou_mask], axis=-1)])
            boxes_conf_class = np.vstack([boxes_conf_class[:best_conf_ind + 1, :],
                                          np.expand_dims(boxes_conf_class[best_conf_ind + 1:, :][iou_mask], axis=-1)])

            iou_mask = np.tile(iou_mask, (1, 4))
            boxes_coord_class = np.vstack([boxes_coord_class[:best_conf_ind + 1, :],
                                           np.reshape(boxes_coord_class[best_conf_ind + 1:, :][iou_mask],
                                                      newshape=(-1, 4))])

            best_conf_ind += 1
            num_process, _ = np.array(boxes_coord_class).shape

        bboxes_coord.append(boxes_coord_class)
        bboxes_scores.append(boxes_conf_class)
        bboxes_classes.append(boxes_class)

    # max_bbox = max_boxes_per_class * num_classes
    bboxes_coord = np.vstack(bboxes_coord)
    bboxes_scores = np.vstack(bboxes_scores)
    bboxes_classes = np.vstack(bboxes_classes)
    return bboxes_coord, bboxes_scores, bboxes_classes

def main(_argv):
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    image_path = FLAGS.image

    if not os.path.exists(FLAGS.save_result_img):
        os.makedirs(FLAGS.save_result_img)

    images_path, images_data = utils.load_multiple_img(image_path, input_size, FLAGS.save_multiple)

    if FLAGS.framework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        interpreter.set_tensor(input_details[0]['index'], images_data)
        interpreter.invoke()
        pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]

        if FLAGS.model == 'yolov3' and FLAGS.tiny == True:
            boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25, input_shape=tf.constant([input_size, input_size]))
        else:
            boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25, input_shape=tf.constant([input_size, input_size]))
    else:
        saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
        print("saved_model_loaded : {}".format(FLAGS.weights))

        infer = saved_model_loaded.signatures['serving_default']
        batch_data = tf.constant(images_data)
        pred_bbox = infer(batch_data)

        image_size = 640
        for key, value in pred_bbox.items():
            # for custom model
            if FLAGS.use_trainres == True and key == "tf_op_layer_concat_10":   # key는 model.summary의 마지막 layer
                if FLAGS.save_multiple == True:
                    for idx in range(len(images_data)):
                        bboxes_coord, bboxes_scores, bboxes_classes = nms(tf.expand_dims(value[idx], axis=0), num_classes=cfg.YOLO.NUM_CLASSES)
                        thres_score = 0.6

                        image = utils.draw_bbox_trainw(images_data[idx], image_size, bboxes_coord, bboxes_classes, bboxes_scores, thres_score)

                        # check result image and save
                        cv2.imshow("result", image)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
                        cv2.imwrite(os.path.join(FLAGS.save_result_img, "{}".format(os.path.basename(images_path[idx]))), image * 255.0)
                else:
                    bboxes_coord, bboxes_scores, bboxes_classes = nms(value, num_classes=cfg.YOLO.NUM_CLASSES)
                    image = utils.draw_bbox_trainw(images_data, image_size, bboxes_coord , bboxes_classes,  bboxes_scores)
                    image = np.reshape(image, (input_size, input_size, 3))

                    # check result image and save
                    cv2.imshow("result", image)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                    cv2.imwrite(os.path.join(FLAGS.save_result_img, os.path.basename(images_path)), image)

            elif FLAGS.use_trainres == False:  # for pretrained weights (.weights -> .pb by saved_model.py) 그 결과
                bboxes_coord, bboxes_scores, bboxes_classes = nms_convertedw(value, num_classes=cfg.YOLO.NUM_CLASSES)
                image = utils.draw_bbox_convertedw(images_data, image_size, bboxes_coord, bboxes_classes, bboxes_scores)
                image = np.reshape(image, (input_size, input_size, 3))

                # check result image and save
                cv2.imshow("result", image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                cv2.imwrite(os.path.join(FLAGS.save_result_img, os.path.basename(images_path)), image)

                """
                ## 기존 - nms 이상하게 나옴 (stitch pretrained yolov4)
                
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

                boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression \
                        (
                        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
                        scores=tf.reshape(
                            pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
                        max_output_size_per_class=50,
                        max_total_size=50,
                        iou_threshold=FLAGS.iou,
                        score_threshold=FLAGS.score
                    )
                pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
                image = utils.draw_bbox(images_data[0], pred_bbox)
                cv2.imshow("result", image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            print("valid_detections:{}".format(valid_detections))
            pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
            print("pred_bbox:{}".format(pred_bbox))
            image = utils.draw_bbox(original_image, pred_bbox)
            # cv2.imshow("result", image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            image = Image.fromarray(image.astype(np.uint8))
            image.show()
            image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
            cv2.imwrite(FLAGS.output, image)
            """

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
