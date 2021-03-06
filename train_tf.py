from absl import app, flags, logging
from absl.flags import FLAGS
import os

#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import shutil
import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
for physical_device in physical_devices:
    tf.config.experimental.set_memory_growth(physical_device, True)

from core.yolov4 import YOLO, YOLOv4_more_tiny, decode, compute_loss, decode_train
from core.dataset import Dataset
from core.dataset_tf import DatasetTF
from core.config import cfg
import numpy as np
from core import utils
from core.utils import freeze_all, unfreeze_all  # , eval_map
import time
import matplotlib.pyplot as plt

import cv2


flags.DEFINE_string('model', 'yolov4', 'yolov4, yolov3')
flags.DEFINE_string('weights', './scripts/yolov4.weights', 'pretrained weights')
flags.DEFINE_integer('num_detection_layer', 1, '3: yolov4 2:yolov4-tiny  3:custom model')

# 0 for YOLOv4, 1 for yolo-tiny 2 for custom yolo

def main(_argv):
    input_channel = 3
    patience = 30
    steps_in_epoch = 0
    epoch_loss = 0
    prev_minloss = np.inf

    trainset = DatasetTF(FLAGS, input_channel, is_training=True)
    testset = DatasetTF(FLAGS, input_channel, is_training=False)

    # testset = Dataset(FLAGS, input_channel, is_training=False)
    logdir = "./data/log_tf"
    isfreeze = False
    steps_per_epoch = len(trainset)
    ## for plot graph , early stopping
    loss_tracker = []

    # print("steps_per_epoch:{}".format(steps_per_epoch))

    first_stage_epochs = cfg.TRAIN.FISRT_STAGE_EPOCHS
    second_stage_epochs = cfg.TRAIN.SECOND_STAGE_EPOCHS
    global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)
    warmup_steps = cfg.TRAIN.WARMUP_EPOCHS * steps_per_epoch
    total_steps = (first_stage_epochs + second_stage_epochs) * steps_per_epoch
    # train_steps = (first_stage_epochs + second_stage_epochs) * steps_per_period

    input_layer = tf.keras.layers.Input([cfg.TRAIN.INPUT_SIZE, cfg.TRAIN.INPUT_SIZE, 3])
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    IOU_LOSS_THRESH = cfg.YOLO.IOU_LOSS_THRESH

    freeze_layers = utils.load_freeze_layer(FLAGS.model, FLAGS.num_detection_layer)
    # feature_maps = YOLO(input_layer, NUM_CLASS, FLAGS.model, FLAGS.tiny)
    feature_maps = YOLOv4_more_tiny(input_layer, NUM_CLASS)

    bbox_tensors = []
    for i, fm in enumerate(feature_maps):  # fm shape: (None, 40, 40, 21)
        bbox_tensor = decode_train(fm, cfg.TRAIN.INPUT_SIZE // 16, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
        bbox_tensors.append(fm)
        bbox_tensors.append(bbox_tensor)

    if cfg.YOLO.NUM_YOLOLAYERS == 3:  # yolov4
        bbox_tensors = []
        for i, fm in enumerate(feature_maps):
            if i == 0:
                bbox_tensor = decode_train(fm, cfg.TRAIN.INPUT_SIZE // 8, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
            elif i == 1:
                bbox_tensor = decode_train(fm, cfg.TRAIN.INPUT_SIZE // 16, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
            else:
                bbox_tensor = decode_train(fm, cfg.TRAIN.INPUT_SIZE // 32, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
            bbox_tensors.append(fm)
            bbox_tensors.append(bbox_tensor)
    elif cfg.YOLO.NUM_YOLOLAYERS == 2:  # yolo tiny
        bbox_tensors = []
        for i, fm in enumerate(feature_maps):
            if i == 0:
                bbox_tensor = decode_train(fm, cfg.TRAIN.INPUT_SIZE // 16, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
            else:
                bbox_tensor = decode_train(fm, cfg.TRAIN.INPUT_SIZE // 32, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
            bbox_tensors.append(fm)
            bbox_tensors.append(bbox_tensor)
    elif cfg.YOLO.NUM_YOLOLAYERS == 1:  # custom yolo
        bbox_tensors = []
        bbox_tensor = decode_train(feature_maps[0], cfg.TRAIN.INPUT_SIZE // cfg.YOLO.STRIDES_CUSTOM[0], NUM_CLASS,
                                   STRIDES, ANCHORS, 0, XYSCALE)
        bbox_tensors.append(feature_maps[0])
        bbox_tensors.append(bbox_tensor)

    model = tf.keras.Model(input_layer, bbox_tensors)
    model.summary()

    try:
        if FLAGS.weights.split(".")[len(FLAGS.weights.split(".")) - 1] == "weights":
            utils.load_weights(model, FLAGS.weights, FLAGS.model, FLAGS.num_detection_layer)
        else:
            model.load_weights(FLAGS.weights)
        print('Restoring weights from: %s ... ' % FLAGS.weights)
    except Exception:
        print("Training from scratch")
        FLAGS.weights = None

    optimizer = tf.keras.optimizers.Adam()

    if os.path.exists(logdir): shutil.rmtree(logdir)
    writer = tf.summary.create_file_writer(logdir)

    # define training step function
    @tf.function
    def train_step(image_data, target):
        with tf.GradientTape() as tape:
            pred_result = model(image_data, training=True)
            ciou_loss = conf_loss = prob_loss = 0

            # optimizing process
            for i in range(len(freeze_layers)):
                conv, pred = pred_result[i * 2], pred_result[i * 2 + 1]
                loss_items = compute_loss(pred, conv, target[i][0], target[i][1], STRIDES=STRIDES, NUM_CLASS=NUM_CLASS,     # label, bbox
                                          IOU_LOSS_THRESH=IOU_LOSS_THRESH, i=i)
                ciou_loss += loss_items[0]
                conf_loss += loss_items[1]
                prob_loss += loss_items[2]

            total_loss = ciou_loss + conf_loss + prob_loss

            gradients = tape.gradient(total_loss, model.trainable_variables)

            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            tf.print("=> STEP ", global_steps, "/", total_steps, "  lr: ", optimizer.lr, "   ciou_loss: ", ciou_loss,
                     "   conf_loss: ", conf_loss, "    prob_loss: ", prob_loss, "   total_loss: ", total_loss)
            # update learning rate
            global_steps.assign_add(1)
            if global_steps < warmup_steps:
                lr = global_steps / warmup_steps * cfg.TRAIN.LR_INIT
            else:
                lr = cfg.TRAIN.LR_END + 0.5 * (cfg.TRAIN.LR_INIT - cfg.TRAIN.LR_END) * (
                    (1 + tf.cos((global_steps - warmup_steps) / (total_steps - warmup_steps) * np.pi))
                )
            optimizer.lr.assign(tf.cast(lr, tf.float32))

            # writing summary data
            with writer.as_default():
                tf.summary.scalar("lr", optimizer.lr, step=global_steps)
                tf.summary.scalar("loss/total_loss", total_loss, step=global_steps)
                tf.summary.scalar("loss/ciou_loss", ciou_loss, step=global_steps)
                tf.summary.scalar("loss/conf_loss", conf_loss, step=global_steps)
                tf.summary.scalar("loss/prob_loss", prob_loss, step=global_steps)
            writer.flush()
            return total_loss

    @tf.function
    def test_step(image_data, target):
        with tf.GradientTape() as tape:
            pred_result = model(image_data, training=True)
            ciou_loss = conf_loss = prob_loss = 0

            # optimizing process
            for i in range(len(freeze_layers)):
                conv, pred = pred_result[i * 2], pred_result[i * 2 + 1]
                loss_items = compute_loss(pred, conv, target[i][0], target[i][1], STRIDES=STRIDES, NUM_CLASS=NUM_CLASS,
                                          IOU_LOSS_THRESH=IOU_LOSS_THRESH, i=i)
                # metric_items = compute_map(pred, conv, target[i][0], target[i][1], STRIDES=STRIDES, NUM_CLASS=NUM_CLASS, i=i)
                # loss_items = compute_loss(pred, conv, target[1][0], target[1][1], STRIDES=STRIDES, NUM_CLASS=NUM_CLASS, IOU_LOSS_THRESH=IOU_LOSS_THRESH, i=i)
                ciou_loss += loss_items[0]
                conf_loss += loss_items[1]
                prob_loss += loss_items[2]

            total_loss = ciou_loss + conf_loss + prob_loss

            tf.print( "==> TEST STEP: ", global_steps)
            tf.print(" ciou_loss: ", ciou_loss)
            tf.print(" conf_loss: ", conf_loss)
            tf.print(" prob_loss: ", prob_loss)
            tf.print("==> total_loss", total_loss)

            # tf.print("=> TEST STEP %4d   giou_loss: %4.2f   conf_loss: %4.2f   "
            #          "prob_loss: %4.2f   total_loss: %4.2f" % (global_steps, giou_loss, conf_loss,
            #                                                    prob_loss, total_loss))

    it = iter(trainset.get_dataset())
    it_test = iter(testset.get_dataset())
    start = time.time()
    for epoch in range(first_stage_epochs + second_stage_epochs):

        train_loss = 0.
        if epoch < first_stage_epochs:
            if not isfreeze:
                isfreeze = True
                for name in freeze_layers:
                    try:  # try?????? ??????
                        freeze = model.get_layer(name)
                        freeze_all(freeze)
                    except ValueError:
                        pass
        elif epoch >= first_stage_epochs:
            if isfreeze:
                isfreeze = False
                for name in freeze_layers:
                    try:  # try?????? ??????
                        freeze = model.get_layer(name)
                        freeze_all(freeze)
                    except ValueError:
                        pass
                    # freeze = model.get_layer(name)
                    # unfreeze_all(freeze)

        for i in range(trainset.steps_for_train):
            data = it.get_next()

            ### check input train images
            # image_data = data[0]
            # label_data = np.array(data[1])
            #
            # bboxes_list = np.array(label_data[0][1])
            # label_list = np.array(label_data[0][0])
            # for batch_idx, bboxes in enumerate(bboxes_list):
            #     class_inds = []
            #     check = np.array(image_data[batch_idx]).reshape(cfg.TRAIN.INPUT_SIZE, cfg.TRAIN.INPUT_SIZE, 3)
            #     # print(r"D:\tf_data_\preprocessed_{}.npy".format(batch_idx))
            #     # np.save(r"D:\tf_data_\image_{}.npy".format(batch_idx), check)
            #     # np.save(r"D:\tf_data_\preprocessed_{}.npy".format(batch_idx), label_list[batch_idx])
            #     # np.save(r"D:\tf_data_\bboxes_{}.npy".format(batch_idx), bboxes_list[batch_idx])
            #     label = np.array(label_list[batch_idx])
            #
            #     class_ = label[...,5:]
            #     class_ = np.array(class_).flatten()
            #     class_ = np.where(class_ > 0.1, class_, 0)
            #
            #     for class_idx, class_label in enumerate(class_):
            #         # print("class_label:{}".format(class_label))
            #         if class_label > 0:
            #             class_inds.append(class_idx % cfg.YOLO.NUM_CLASSES)
            #
            #     bboxes = np.array(bboxes)
            #     for bbox_idx, bbox in enumerate(bboxes):
            #         half_h = bbox[3] / 2
            #         half_w = bbox[2] / 2
            #
            #         if half_h + half_w > 0:
            #             cv2.rectangle(check, (int(bbox[0] - half_w), int(bbox[1] - half_h)),
            #                           (int(bbox[0] + half_w), int(bbox[1] + half_h)), color=(0, 255, 0), thickness=3)
            #             cv2.putText(check, text="{}".format(class_inds),
            #                         org=(int(bbox[0] + half_w)-10, int(bbox[1] + half_h)-30), thickness=2, color=(0, 255, 0),
            #                         fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7)
            #
            #     cv2.imshow("check_aug", check)
            #     cv2.waitKey(0)
            #     cv2.destroyAllWindows()
            # cv2.imwrite(os.path.join(r"D:\Public\JHS\SIMPLE_DATA_INPUT_CHECK", "{}.jpg".format(batch_idx)), check*255)

            train_loss = train_step(data[0], data[1])
            # epoch_loss += train_loss


        # for i in range(testset.steps_for_train):
        #     data = it_test.get_next()
        #     test_step(data[0], data[1])

            # CHECK TEST
            # image_data = data[0]
            # label_data = data[1]
            # bboxes_list = label_data[0][1]
            #
            # for batch_idx, bboxes in enumerate(bboxes_list):
            #     check = np.array(image_data[batch_idx]).reshape(640, 640, 3)
            #     for bbox in bboxes:
            #         half_h = bbox[3] / 2
            #         half_w = bbox[2] / 2
            #         # bbox_upper_left = (bbox[0])
            #         if np.sum(bbox) > 0:
            #             cv2.rectangle(check, (bbox[0] - half_w, bbox[1] - half_h),(bbox[0] + half_w,bbox[1]+half_h), color=(0,255,0), thickness=3)
            #     cv2.imshow("check_aug TEST", check)
            #     cv2.waitKey(0)
            #     cv2.destroyAllWindows()


        # for image_data, target in testset:
        #     test_step(image_data, target)

        # for loss graph
        """
        if steps_in_epoch >= steps_per_epoch:
            loss_tracker.append(losses_in_epoch / steps_per_epoch)
            
            ## ?????????
            # plt.plot(loss_tracker)
            # plt.xlabel('epoch')
            # plt.ylabel('loss')
            # # plt.show()
            # plt.savefig("D:/checkpoint/loss.png")
        """

        # early stopping
        """
        epoch_loss = epoch_loss / steps_per_epoch
        loss_tracker.append(epoch_loss)
        if prev_minloss > epoch_loss:   # save best weight
            prev_minloss = epoch_loss

            if epoch > 2000:  # ???????????? ??????
                model.save("D:\ckpt_best")
                print("{} epoch save best weights".format(epoch))

        if len(loss_tracker) > patience:
            print("check loss_tracker len:{}".format(len(loss_tracker)))
            if loss_tracker[0] > loss_tracker[-1]:
                loss_tracker.pop(0)
            else:
                print("total loss didn't decreased during {} epochs. train stop".format(patience))
                return
            epoch_loss = 0.
        else:
            epoch_loss += train_loss
        steps_in_epoch = 0
        """

        if (epoch+1) % 100 == 0:
            model.save("D:\yolov4-tflite-train_tf-epoch{}".format(epoch+1))
            print("{} epoch model saved".format(epoch+1))
            print("consumed time: {}".format(time.time() - start))
    model.save(r"D:\yolov4-tflite-train_tf-last")

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
