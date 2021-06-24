import cv2
import random
import colorsys
import numpy as np
import tensorflow as tf
from core.config import cfg
import tensorflow.keras.backend as K
import os

def load_freeze_layer(model='yolov4', num_detection_layers=1):
    if num_detection_layers == 3:  # for yolo
        if model == 'yolov3':
            freeze_layouts = ['conv2d_58', 'conv2d_66', 'conv2d_74']
        else:
            freeze_layouts = ['conv2d_93', 'conv2d_101', 'conv2d_109']
    elif num_detection_layers == 2: # yolo-tiny
        if model == 'yolov3':
            freeze_layouts = ['conv2d_9', 'conv2d_12']
        else:
            freeze_layouts = ['conv2d_17', 'conv2d_20']
    elif num_detection_layers == 1: # yolo-custom-tiny
        # custom yolov4
        freeze_layouts = ['conv2d_17']
    return freeze_layouts

def load_weights(model, weights_file, model_name='yolov4', num_detection_layer=1):
    if num_detection_layer == 3:
        if model_name == 'yolov3':
            layer_size = 75
            output_pos = [58, 66, 74]
        else:
            layer_size = 110
            output_pos = [93, 101, 109]
    elif num_detection_layer == 2:   # yolo-tiny ver
        if model_name == 'yolov3':
            layer_size = 13
            output_pos = [9, 12]
        else:
            layer_size = 21
            output_pos = [17, 20]
    elif num_detection_layer == 1:   # custom model (detection 층 1개)
        if model_name == 'yolov3':  # 얘 안 씀
            layer_size = 13
            output_pos = [12]
        else:
            layer_size = 18
            output_pos = [17]

    wf = open(weights_file, 'rb')
    major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)

    j = 0
    for i in range(layer_size):
        conv_layer_name = 'conv2d_%d' %i if i > 0 else 'conv2d'
        bn_layer_name = 'batch_normalization_%d' %j if j > 0 else 'batch_normalization'

        conv_layer = model.get_layer(conv_layer_name)
        filters = conv_layer.filters
        k_size = conv_layer.kernel_size[0]
        in_dim = conv_layer.input_shape[-1]

        if i not in output_pos:
            # darknet weights: [beta, gamma, mean, variance]
            bn_weights = np.fromfile(wf, dtype=np.float32, count=4 * filters)
            # tf weights: [gamma, beta, mean, variance]
            bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]
            bn_layer = model.get_layer(bn_layer_name)
            j += 1
        else:
            conv_bias = np.fromfile(wf, dtype=np.float32, count=filters)

        # darknet shape (out_dim, in_dim, height, width)
        conv_shape = (filters, in_dim, k_size, k_size)
        conv_weights = np.fromfile(wf, dtype=np.float32, count=np.product(conv_shape))
        # tf shape (height, width, in_dim, out_dim)
        conv_weights = conv_weights.reshape(conv_shape).transpose([2, 3, 1, 0])

        if i not in output_pos:
            conv_layer.set_weights([conv_weights])
            bn_layer.set_weights(bn_weights)
        else:
            conv_layer.set_weights([conv_weights, conv_bias])

    # assert len(wf.read()) == 0, 'failed to read all data'
    wf.close()


def read_class_names(class_file_name):
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names

def load_config(FLAGS):
    if cfg.YOLO.NUM_YOLOLAYERS == 3:    # 3 for yolov4
        STRIDES = np.array(cfg.YOLO.STRIDES)
        if FLAGS.model == 'yolov4':
            ANCHORS = get_anchors(cfg.YOLO.ANCHORS)
        elif FLAGS.model == 'yolov3':
            ANCHORS = get_anchors(cfg.YOLO.ANCHORS_V3)

        XYSCALE = cfg.YOLO.XYSCALE if FLAGS.model == 'yolov4' else [1, 1, 1]
    elif cfg.YOLO.NUM_YOLOLAYERS == 2:    # 2 for tiny-yolov4
        STRIDES = np.array(cfg.YOLO.STRIDES_TINY)
        ANCHORS = get_anchors(cfg.YOLO.ANCHORS_TINY)
        XYSCALE = cfg.YOLO.XYSCALE_TINY if FLAGS.model == 'yolov4' else [1, 1]
    else:
        STRIDES = np.array(cfg.YOLO.STRIDES_CUSTOM)
        ANCHORS = get_anchors(cfg.YOLO.ANCHORS_CUSTOM)
        XYSCALE = cfg.YOLO.XYSCALE_CUSTOM if FLAGS.model == 'yolov4' else [1]
    NUM_CLASS = len(read_class_names(cfg.YOLO.CLASSES))

    return STRIDES, ANCHORS, NUM_CLASS, XYSCALE

def get_anchors(anchors_path):  # 디폴트 ver=2  (custom model)
    anchors = np.array(anchors_path)
    if cfg.YOLO.NUM_YOLOLAYERS == 3:
        return anchors.reshape(3, 3, 2)
    elif cfg.YOLO.NUM_YOLOLAYERS == 2:
        return anchors.reshape(2, 3, 2)
    else:
        return anchors.reshape(1, 3, 2)


def image_preprocess_convertedw(image, target_size, gt_boxes=None):
    ih, iw = target_size
    h, w, ch = image.shape
    if ch == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    scale = min(iw / w, ih / h)
    nw, nh = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))

    image_paded = np.full(shape=[ih, iw, 3], fill_value=128.0)
    dw, dh = (iw - nw) // 2, (ih - nh) // 2

    image_paded[dh:nh + dh, dw:nw + dw, :] = image_resized
    image_paded = image_paded / 255.

    if gt_boxes is None:
        return image_paded
    else:
        gt_boxes[:, [0,]] = gt_boxes[:, [0,]] * scale + dw
        gt_boxes[:, [2,]] = gt_boxes[:, [2,]] * scale
        gt_boxes[:, [1,]] = gt_boxes[:, [1,]] * scale + dh
        gt_boxes[:, [3,]] = gt_boxes[:, [3,]] * scale
        # gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
        # gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
        return image_paded, gt_boxes

def image_preprocess(image, target_size, gt_boxes=None):

    ih, iw    = target_size
    h,  w, _  = image.shape

    scale = min(iw/w, ih/h)
    nw, nh  = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))

    image_paded = np.full(shape=[ih, iw, 3], fill_value=128.0)
    dw, dh = (iw - nw) // 2, (ih-nh) // 2
    image_paded[dh:nh+dh, dw:nw+dw, :] = image_resized

    image_paded = image_paded / 255.

    if gt_boxes is None:
        return image_paded

    else:
        gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
        gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
        return image_paded, gt_boxes

def load_multiple_img(image_path, input_size, is_multiple):
    extensions = ['bmp', 'jpg', 'jpeg', 'png']
    isfolder = os.path.basename(image_path)
    if is_multiple:
        image_path = [os.path.join(image_path, path) for path in os.listdir(image_path) if any(path.endswith(ext) for ext in extensions)]
        images_data = []
        for idx, path in enumerate(image_path):
            original_image = cv2.imread(path)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            original_image = np.array(original_image)
            ih, iw, channels = original_image.shape

            if channels == 1:
                original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)

            image_data = image_preprocess(np.copy(original_image), [input_size, input_size])
            images_data.append(image_data)
        images_data = np.asarray(images_data).astype(np.float32)
    else:
        original_image = cv2.imread(image_path)

        image_data = image_preprocess(np.copy(original_image), [input_size, input_size])
        images_data = [image_data]
        images_data = np.asarray(images_data).astype(np.float32)
    return image_path, images_data

# from PIL import Image
# def load_multiple_img_converted(image_path, input_size, is_multiple):
#     extensions = ['bmp', 'jpg', 'jpeg', 'png']
#     isfolder = os.path.basename(image_path)
#     if is_multiple:
#         image_path = [os.path.join(image_path, path) for path in os.listdir(image_path) if any(path.endswith(ext) for ext in extensions)]
#         images_data = []
#         for idx, path in enumerate(image_path):
#             original_image = Image.open(path)
#             original_image = np.array(original_image)
#             _, _, ch = original_image.shape
#             if ch == 1:
#                 cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
#             # original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
#             image_data = image_preprocess(np.copy(original_image), [input_size, input_size])
#             images_data.append(image_data)
#         images_data = np.asarray(images_data).astype(np.float32)
#     else:
#         original_image = cv2.imread(image_path)
#         image_data = image_preprocess(np.copy(original_image), [input_size, input_size])
#         images_data = [image_data]
#         images_data = np.asarray(images_data).astype(np.float32)
#     return image_path, images_data

# def compute_map(pred, target_bboxes, num_classes):
#     """
#         >> gt_bboxes preprocess된 거에서 conf=1인 것들 숫자 세서 그게 진짜 gt개수와 같은지 확인
#
#         0. 배치 단위로 batch x ~ 함, class별로 나눔
#         1. 배치 단위로 batch x bboxes num x (5+num_classes) (xywh, conf, classprob) gt,pred 가져와 iou계산
#         2. iou 0.5 미만인 경우 일괄 fp처리
#         3. iou 0.5 이상인 경우 highest iou=>tp
#         4. 3 을 gt마다 매칭되는 pred들로 묶음(iou 0.5이상) 가장 confidence높은 애가 tp, 나머지 fp처리. tp,fp 확정되면 process되었음
#         5. 각 pred box마다 recall, precision 구함(해당되는 gt없는 경우는? 모두 fp이므로 tp=0으로 계산)
#         5. recall을 0.1단위로 나누어 0~1 사이 11개 구간에서 해당 구간(0.1) 내에서 가장 높은 precision을 구함
#         6. 위에서 계산한 11개의 max precision들의 평균 구함. 걔가 ap
#         7. 각 클래스 별로 구한 ap들의 평균을 구함
#     """
#     aps = np.zeros(num_classes, dtype=np.float32)
#
#     # for class_idx in range(num_classes):
#     #     class_true_bboxes = target_bboxes[]
#     pred_infos = {0: {"detection_id": [0],
#                       "confidence": [0], "iou": [0], "processed": [False],
#                       "acc_tp": [0], "acc_fp": [0]}}  # 0 for image_id
#     # pred
#     batch_, _, _, _, channels_ = tf.keras.backend.int_shape(pred)
#     pred = tf.reshape(pred, (batch_, -1, channels_))
#
#     # gt
#     _, gt_bboxes, _ = target_bboxes.shape
#
#     true_bboxes = []
#     for batch_idx in range(batch_):
#         gt_count = 0
#         for bbox_idx in range(gt_bboxes):
#             if np.sum(target_bboxes[batch_idx, bbox_idx, :]) > 0:
#                 gt_count += 1
#         true_bbox = target_bboxes[batch_idx, :gt_count, :]
#         true_bboxes.append(true_bbox)
#
#     _, max_bboxes_, _ = tf.keras.backend.int_shape(pred)
#     for batch_idx in range(batch_):
#
#         gt_count = len(true_bboxes[batch_idx])
#         for bbox_idx in range(max_bboxes_):
#             if pred[batch_idx, bbox_idx, 4] > 0:
#                 image_id = batch_idx
#                 pred_infos[image_id]['detection_id'].append(bbox_idx)
#                 pred_infos[image_id]['confidence'].append(pred[batch_idx, bbox_idx, 4])
#
#                 iou = bbox_iou(pred[batch_idx, bbox_idx:bbox_idx + 1, :4], true_bboxes[batch_idx][:])
#                 iou = np.array(iou)
#                 iou_idx = np.argmax(iou)
#
#                 iou = iou[iou_idx]
#                 pred_infos[image_id]['iou'].append(iou)
#                 if iou >= 0.5:
#                     try:
#                         pred_infos[image_id]['acc_tp'].append(pred_infos[image_id]['acc_tp'][-1] + 1)
#                     except Exception:
#                         pred_infos[image_id]['acc_tp'][-1] += 1
#                 else:
#                     try:
#                         pred_infos[image_id]['acc_fp'].append(pred_infos[image_id]['acc_fp'][-1] + 1)
#                     except Exception:
#                         pred_infos[image_id]['acc_fp'][-1] += 1
#
#     rec_list = []
#     prec_list = []
#     # recall, precision 으로 ap 계산
#     for idx in range(len(pred_infos)):
#         rec = pred_infos['recall'][idx + 1] - pred_infos['recall'][idx]
#         prec = pred_infos['precision'][idx + 1] - pred_infos['precision'][idx]
#         rec_list.append(rec)
#         prec_list.append(prec)
#
#     rec, prec = zip(*sorted(zip(rec_list, prec_list)))
#     # get highest pred from rec range
#     rec_range = np.arange(0, 1, 0.1)
#     # print("rec_range: {}".format(rec_range))
#
#
# def calc_map(pred, target_bboxes, target_infos, num_classes):
#     print("### clac_map inside")
#     print("target_bboxes:{}".format(tf.keras.backend.int_shape(target_bboxes)))
#     print("target_infos:{}".format(tf.keras.backend.int_shape(target_infos)))
#     print("num_classes:{}".format(tf.keras.backend.int_shape(num_classes)))
#     """
#         >> gt_bboxes preprocess된 거에서 conf=1인 것들 숫자 세서 그게 진짜 gt개수와 같은지 확인
#
#         0. 배치 단위로 batch x ~ 함, class별로 나눔
#         1. 배치 단위로 batch x bboxes num x (5+num_classes) (xywh, conf, classprob) gt,pred 가져와 iou계산
#         2. iou 0.5 미만인 경우 일괄 fp처리
#         3. iou 0.5 이상인 경우 highest iou=>tp
#         4. 3 을 gt마다 매칭되는 pred들로 묶음(iou 0.5이상) 가장 confidence높은 애가 tp, 나머지 fp처리. tp,fp 확정되면 process되었음
#         5. 각 pred box마다 recall, precision 구함(해당되는 gt없는 경우는? 모두 fp이므로 tp=0으로 계산)
#         5. recall을 0.1단위로 나누어 0~1 사이 11개 구간에서 해당 구간(0.1) 내에서 가장 높은 precision을 구함
#         6. 위에서 계산한 11개의 max precision들의 평균 구함. 걔가 ap
#         7. 각 클래스 별로 구한 ap들의 평균을 구함
#     """
#     aps = np.zeros(num_classes, dtype=np.float32)
#
#     pred_infos = {"image_id":[], "detection":[],
#                   "confidence":[], "iou":[], "processed":[],
#                   "acc_tp":[], "acc_fp":[]}
#     # pred
#     batch_, bboxes_, ch_ = tf.keras.backend.int_shape(pred)
#
#     # gt
#     _, gt_bboxes, _ = target_bboxes.shape
#
#     true_bboxes = []
#     for batch_idx in range(batch_):
#         gt_count = 0
#         for bbox_idx in range(gt_bboxes):
#             if np.sum(target_bboxes[batch_idx, bbox_idx, :]) > 0:
#                 gt_count +=1
#         true_bbox = target_bboxes[batch_idx,:gt_count,:]
#         true_bboxes.append(true_bbox)
#
#     for batch_idx in range(batch_):
#
#         gt_count = len(true_bboxes[batch_idx])
#         for bbox_idx in range(bboxes_):
#             if pred[batch_idx, bbox_idx, 4] > 0:
#
#                 pred_infos['image_id'].append(batch_idx)
#                 pred_infos['detection'].append(bbox_idx)
#                 pred_infos['confidence'].append(pred[batch_idx, bbox_idx, 4])
#
#                 # print("true_bboxes[{}][{}]: {}".format(batch_idx, bbox_idx, true_bboxes[batch_idx][bbox_idx]))
#                 iou = bbox_iou(pred[batch_idx, bbox_idx, :4], true_bboxes[batch_idx][bbox_idx])
#                 iou = np.array(iou)
#                 print("iou: {}".format(iou))
#
#                 iou_idx = np.argmax(iou)
#
#                 iou = iou[iou_idx]
#                 if iou >= 0.5:
#                     pred_infos['iou'].append(iou)
#                     pred_infos['acc_tp'] += 1
#                 else:
#                     pred_infos['iou'].append(fp)
#                     pred_infos['acc_fp'] += 1
#
#     print("len(pred_infos):{}".format(len(pred_infos)))
#
#     rec_list = []
#     prec_list = []
#     # recall, precision 으로 ap 계산
#     for idx in range(len(pred_infos)):
#         rec = pred_infos['recall'][idx+1] - pred_infos['recall'][idx]
#         prec = pred_infos['precision'][idx+1] - pred_infos['precision'][idx]
#         rec_list.append(red)
#         prec_list.append(pred)
#
#     # get highest pred from rec range
#     rec_range = np.arange(0,1,0.1)
#     print("rec_range: {}".format(rec_range))
#
#     # return np.mean(aps)
#
# def group_by_key(gt, key="image_id"):
#     for idx in range(len(gt)):
#         print("gt[idx]:{}".format(gt[idx]))
#         image_gts = {key:idx, boxes:gt[idx]}
#
# # gt == annotation
# def recall_precision(preds, gt_bboxes):
#     image_gts = group_by_key(gt, "image_id")
#
#     image_gt_boxes = {
#         img_id: np.array([[float(z) for z in b["bbox"]] for b in boxes]) for img_id, boxes in image_gts.items()
#     }
#     image_gt_checked = {img_id: np.zeros(len(boxes)) for img_id, boxes in image_gts.items()}
#
#     predictions = sorted(predictions, key=lambda x: x["score"], reverse=True)
#
#     # go down dets and mark TPs and FPs
#     num_predictions = len(predictions)
#     tp = np.zeros(num_predictions)
#     fp = np.zeros(num_predictions)
#
#     for prediction_index, prediction in enumerate(predictions):
#         box = prediction["bbox"]
#
#         max_overlap = -np.inf
#         jmax = -1
#
#         try:
#             gt_boxes = image_gt_boxes[prediction["image_id"]]  # gt_boxes per image
#             gt_checked = image_gt_checked[prediction["image_id"]]  # gt flags per image
#         except KeyError:
#             gt_boxes = []
#             gt_checked = None
#
#         if len(gt_boxes) > 0:
#             overlaps = get_overlaps(gt_boxes, box)
#
#             max_overlap = np.max(overlaps)
#             jmax = np.argmax(overlaps)
#
#         if max_overlap >= iou_threshold:
#             if gt_checked[jmax] == 0:
#                 tp[prediction_index] = 1.0
#                 gt_checked[jmax] = 1
#             else:
#                 fp[prediction_index] = 1.0
#         else:
#             fp[prediction_index] = 1.0
#
#     # compute precision recall
#     fp = np.cumsum(fp, axis=0)
#     tp = np.cumsum(tp, axis=0)
#
#     recalls = tp / float(num_gts)
#
#     # avoid divide by zero in case the first detection matches a difficult ground truth
#     precisions = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
#
#     ap = get_ap(recalls, precisions)
#
#     return recalls, precisions, ap
#
# def get_envelope(precisions: np.ndarray) -> np.array:
#     for i in range(precisions.size - 1, 0, -1):
#         precisions[i - 1] = np.maximum(precisions[i - 1], precisions[i])
#     return precisions
#
# def get_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
#     # correct AP calculation
#     # first append sentinel values at the end
#     recalls = np.concatenate(([0.0], recalls, [1.0]))
#     precisions = np.concatenate(([0.0], precisions, [0.0]))
#
#     precisions = get_envelope(precisions)
#
#     # to calculate area under PR curve, look for points where X axis (recall) changes value
#     i = np.where(recalls[1:] != recalls[:-1])[0]
#
#     # and sum (\Delta recall) * prec
#     ap = np.sum((recalls[i + 1] - recalls[i]) * precisions[i + 1])
#     return ap

def draw_bbox_trainw(original_image, image_size, pred_coord , pred_classes, pred_scores, thres_score=0.6):
    hsv_tuples = [(1.0 * x / cfg.YOLO.NUM_CLASSES, 1., 1.) for x in range(cfg.YOLO.NUM_CLASSES)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    original_image = np.reshape(original_image, (image_size, image_size, 3))

    for idx in range(len(pred_coord)):
        pred_class = int(pred_classes[idx][0])
        pred_score = pred_scores[idx][0]

        coord = pred_coord[idx]
        x_left = int((float(coord[0])-float(coord[2])/2))
        x_right = int((float(coord[0]) + float(coord[2])/2))
        y_top = int((float(coord[1] - float(coord[3])/2)))
        y_bottom = int((float(coord[1] + float(coord[3]) / 2)))

        color = colors[pred_class]
        cv2.rectangle(original_image, (x_left,y_top),(x_right,y_bottom), thickness=3, color=color)
        cv2.putText(original_image, text="{} score:{:.02f}".format(pred_class, pred_score), org=(x_left+40,y_bottom-40), thickness=2, color=color, fontFace=cv2.FONT_HERSHEY_SIMPLEX ,fontScale=0.7)

    return original_image

def draw_bbox_convertedw(original_image, image_size, pred_coord , pred_classes, pred_scores):
    hsv_tuples = [(1.0 * x / cfg.YOLO.NUM_CLASSES, 1., 1.) for x in range(cfg.YOLO.NUM_CLASSES)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    original_image = np.reshape(original_image, (image_size, image_size, 3))

    for idx in range(len(pred_coord)):
        pred_class = int(pred_classes[idx][0])
        pred_score = pred_scores[idx][0]
        coord = pred_coord[idx]

        y_top = int(float(coord[0]) * image_size)
        x_left = int(float(coord[1]) * image_size)
        y_bottom = int(float(coord[2]) * image_size)
        x_right = int(float(coord[3]) * image_size)

        color = colors[pred_class]
        cv2.rectangle(original_image, (x_left, y_top), (x_right, y_bottom), thickness=3, color=color)
        cv2.putText(original_image, text="{} score:{:.02f}".format(pred_class, pred_score),
                    org=(x_left + 30, y_bottom - 30), thickness=2, color=color, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5)
    return original_image

def draw_bbox(image, bboxes, classes=read_class_names(cfg.YOLO.CLASSES), show_label=True):
    num_classes = len(classes)
    image_h, image_w, _ = image.shape
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    random.seed(0)
    random.shuffle(colors)
    random.seed(None)

    out_boxes, out_scores, out_classes, num_boxes = bboxes

    for i in range(num_boxes[0]):
        if int(out_classes[0][i]) < 0 or int(out_classes[0][i]) > num_classes: continue
        coor = out_boxes[0][i]
        coor[0] = int(coor[0] * image_h)
        coor[2] = int(coor[2] * image_h)
        coor[1] = int(coor[1] * image_w)
        coor[3] = int(coor[3] * image_w)

        fontScale = 0.5
        score = out_scores[0][i]
        class_ind = int(out_classes[0][i])
        bbox_color = colors[class_ind]
        bbox_thick = int(0.6 * (image_h + image_w) / 600)
        c1, c2 = (coor[1], coor[0]), (coor[3], coor[2])
        cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)

        if show_label:
            bbox_mess = '%s: %.2f' % (classes[class_ind], score)
            t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick // 2)[0]
            c3 = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)
            cv2.rectangle(image, c1, (np.float32(c3[0]), np.float32(c3[1])), bbox_color, -1) #filled
            cv2.putText(image, bbox_mess, (c1[0], np.float32(c1[1] - 2)), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale, (0, 0, 0), bbox_thick // 2, lineType=cv2.LINE_AA)
    return image

def bbox_iou(bboxes1, bboxes2):
    """
    @param bboxes1: (a, b, ..., 4)
    @param bboxes2: (A, B, ..., 4)
        x:X is 1:n or n:n or n:1
    @return (max(a,A), max(b,B), ...)
    ex) (4,):(3,4) -> (3,)
        (2,1,4):(2,3,4) -> (2,3)
    """
    bboxes1_area = bboxes1[..., 2] * bboxes1[..., 3]
    bboxes2_area = bboxes2[..., 2] * bboxes2[..., 3]

    bboxes1_coor = tf.concat(
        [
            bboxes1[..., :2] - bboxes1[..., 2:] * 0.5,
            bboxes1[..., :2] + bboxes1[..., 2:] * 0.5,
        ],
        axis=-1,
    )
    bboxes2_coor = tf.concat(
        [
            bboxes2[..., :2] - bboxes2[..., 2:] * 0.5,
            bboxes2[..., :2] + bboxes2[..., 2:] * 0.5,
        ],
        axis=-1,
    )

    left_up = tf.maximum(bboxes1_coor[..., :2], bboxes2_coor[..., :2])
    right_down = tf.minimum(bboxes1_coor[..., 2:], bboxes2_coor[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]

    union_area = bboxes1_area + bboxes2_area - inter_area

    iou = tf.math.divide_no_nan(inter_area, union_area)

    return iou


def bbox_giou(bboxes1, bboxes2):
    """
    Generalized IoU
    @param bboxes1: (a, b, ..., 4)
    @param bboxes2: (A, B, ..., 4)
        x:X is 1:n or n:n or n:1
    @return (max(a,A), max(b,B), ...)
    ex) (4,):(3,4) -> (3,)
        (2,1,4):(2,3,4) -> (2,3)
    """
    bboxes1_area = bboxes1[..., 2] * bboxes1[..., 3]
    bboxes2_area = bboxes2[..., 2] * bboxes2[..., 3]

    bboxes1_coor = tf.concat(
        [
            bboxes1[..., :2] - bboxes1[..., 2:] * 0.5,
            bboxes1[..., :2] + bboxes1[..., 2:] * 0.5,
        ],
        axis=-1,
    )
    bboxes2_coor = tf.concat(
        [
            bboxes2[..., :2] - bboxes2[..., 2:] * 0.5,
            bboxes2[..., :2] + bboxes2[..., 2:] * 0.5,
        ],
        axis=-1,
    )

    left_up = tf.maximum(bboxes1_coor[..., :2], bboxes2_coor[..., :2])
    right_down = tf.minimum(bboxes1_coor[..., 2:], bboxes2_coor[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]

    union_area = bboxes1_area + bboxes2_area - inter_area

    iou = tf.math.divide_no_nan(inter_area, union_area)

    enclose_left_up = tf.minimum(bboxes1_coor[..., :2], bboxes2_coor[..., :2])
    enclose_right_down = tf.maximum(
        bboxes1_coor[..., 2:], bboxes2_coor[..., 2:]
    )

    enclose_section = enclose_right_down - enclose_left_up
    enclose_area = enclose_section[..., 0] * enclose_section[..., 1]

    giou = iou - tf.math.divide_no_nan(enclose_area - union_area, enclose_area)

    return giou


def bbox_ciou(bboxes1, bboxes2):
    """
    Complete IoU
    @param bboxes1: (a, b, ..., 4)
    @param bboxes2: (A, B, ..., 4)
        x:X is 1:n or n:n or n:1
    @return (max(a,A), max(b,B), ...)
    ex) (4,):(3,4) -> (3,)
        (2,1,4):(2,3,4) -> (2,3)
    """
    bboxes1_area = bboxes1[..., 2] * bboxes1[..., 3]
    bboxes2_area = bboxes2[..., 2] * bboxes2[..., 3]

    bboxes1_coor = tf.concat(
        [
            bboxes1[..., :2] - bboxes1[..., 2:] * 0.5,
            bboxes1[..., :2] + bboxes1[..., 2:] * 0.5,
        ],
        axis=-1,
    )
    bboxes2_coor = tf.concat(
        [
            bboxes2[..., :2] - bboxes2[..., 2:] * 0.5,
            bboxes2[..., :2] + bboxes2[..., 2:] * 0.5,
        ],
        axis=-1,
    )

    left_up = tf.maximum(bboxes1_coor[..., :2], bboxes2_coor[..., :2])
    right_down = tf.minimum(bboxes1_coor[..., 2:], bboxes2_coor[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]

    union_area = bboxes1_area + bboxes2_area - inter_area

    iou = tf.math.divide_no_nan(inter_area, union_area)

    enclose_left_up = tf.minimum(bboxes1_coor[..., :2], bboxes2_coor[..., :2])
    enclose_right_down = tf.maximum(
        bboxes1_coor[..., 2:], bboxes2_coor[..., 2:]
    )

    enclose_section = enclose_right_down - enclose_left_up

    c_2 = enclose_section[..., 0] ** 2 + enclose_section[..., 1] ** 2

    center_diagonal = bboxes2[..., :2] - bboxes1[..., :2]

    rho_2 = center_diagonal[..., 0] ** 2 + center_diagonal[..., 1] ** 2

    diou = iou - tf.math.divide_no_nan(rho_2, c_2)

    v = (
        (
            tf.math.atan(
                tf.math.divide_no_nan(bboxes1[..., 2], bboxes1[..., 3])
            )
            - tf.math.atan(
                tf.math.divide_no_nan(bboxes2[..., 2], bboxes2[..., 3])
            )
        )
        * 2
        / np.pi
    ) ** 2

    alpha = tf.math.divide_no_nan(v, 1 - iou + v)

    ciou = diou - alpha * v

    return ciou

def bbox_confidence(cls_ind, cls_bboxes, iou):
    first_bbox_classprob_ = pred_classprob[...,0, :]
    second_bbox_classprob_ = pred_classprob[...,1, :]
    third_bbox_classprob_ = pred_classprob[...,2, :]

    first_bbox_class_ = tf.expand_dims(tf.argmax(first_bbox_classprob_, axis=-1), axis=-1)
    second_bbox_class_ = tf.expand_dims(tf.argmax(second_bbox_classprob_, axis=-1), axis=-1)
    third_bbox_class_ = tf.expand_dims(tf.argmax(third_bbox_classprob_, axis=-1), axis=-1)

    pred_class = tf.cast(tf.concat([first_bbox_class_, second_bbox_class_, third_bbox_class_], axis=-1), dtype=tf.float32)

    if istrain == True:
        iou_score = bbox_iou([true_xy, true_wh, true_classprob], [pred_xy, pred_wh, pred_classprob])
        return tf.multiply(pred_objness, pred_class) * iou_score, iou_score
    return tf.multiply(pred_objness, pred_class)

def yolo_eval(yolo_outputs,
              anchors,
              num_classes,
              image_shape,
              max_boxes=20,
              score_threshold=.6,
              iou_threshold=.5):
    """Evaluate YOLO model on given input and return filtered boxes."""
    num_layers = len(yolo_outputs)
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [1,2,3]] # default setting
    input_shape = K.shape(yolo_outputs[0])[1:3] * 32
    boxes = []
    box_scores = []
    for l in range(num_layers):
        _boxes, _box_scores = yolo_boxes_and_scores(yolo_outputs[l],
            anchors[anchor_mask[l]], num_classes, input_shape, image_shape)
        boxes.append(_boxes)
        box_scores.append(_box_scores)
    boxes = K.concatenate(boxes, axis=0)
    box_scores = K.concatenate(box_scores, axis=0)

    mask = box_scores >= score_threshold
    max_boxes_tensor = K.constant(max_boxes, dtype='int32')
    boxes_ = []
    scores_ = []
    classes_ = []
    for c in range(num_classes):
        # TODO: use keras backend instead of tf.
        class_boxes = tf.boolean_mask(boxes, mask[:, c])
        class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
        nms_index = tf.image.non_max_suppression(
            class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=iou_threshold)
        class_boxes = K.gather(class_boxes, nms_index)
        class_box_scores = K.gather(class_box_scores, nms_index)
        classes = K.ones_like(class_box_scores, 'int32') * c
        boxes_.append(class_boxes)
        scores_.append(class_box_scores)
        classes_.append(classes)
    boxes_ = K.concatenate(boxes_, axis=0)
    scores_ = K.concatenate(scores_, axis=0)
    classes_ = K.concatenate(classes_, axis=0)

    return boxes_, scores_, classes_

def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape):
    '''Get corrected boxes'''
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = K.cast(input_shape, K.dtype(box_yx))
    image_shape = K.cast(image_shape, K.dtype(box_yx))
    new_shape = K.round(image_shape * K.min(input_shape/image_shape))
    offset = (input_shape-new_shape)/2./input_shape
    scale = input_shape/new_shape
    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes =  K.concatenate([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ])

    # Scale boxes back to original image shape.
    boxes *= K.concatenate([image_shape, image_shape])
    return boxes

def yolo_boxes_and_scores(box_xy, box_wh, box_confidence, box_class_probs,input_shape, image_shape,num_classes):
    '''Process Conv layer output'''
    boxes = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape)
    boxes = K.reshape(boxes, [-1, 4])
    box_scores = box_confidence * box_class_probs
    box_scores = K.reshape(box_scores, [-1, num_classes])
    return boxes, box_scores

def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape):
    '''Get corrected boxes'''
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = K.cast(input_shape, K.dtype(box_yx))
    image_shape = K.cast(image_shape, K.dtype(box_yx))
    new_shape = K.round(image_shape * K.min(input_shape/image_shape))
    # print("input_shape:{} image_shape:{} new_shape:{}".format(tf.keras.backend.int_shape(input_shape),
    #                                                           tf.keras.backend.int_shape(image_shape),
    #                                                           tf.keras.backend.int_shape(new_shape)))
    offset = (input_shape-new_shape)/2./input_shape
    scale = input_shape/new_shape
    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes =  K.concatenate([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ])

    # Scale boxes back to original image shape.
    boxes *= K.concatenate([image_shape, image_shape])
    return boxes

def freeze_all(model, frozen=True):
    model.trainable = not frozen
    if isinstance(model, tf.keras.Model):
        for l in model.layers:
            freeze_all(l, frozen)

def unfreeze_all(model, frozen=False):
    model.trainable = not frozen
    if isinstance(model, tf.keras.Model):
        for l in model.layers:
            unfreeze_all(l, frozen)

