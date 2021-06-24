from absl import app, flags, logging
from absl.flags import FLAGS
import os
import shutil
import tensorflow as tf

try:
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
except RuntimeError:
    print("Physical devices cannot be modified after being initialized")

from core.yolov4 import YOLO, YOLOv4_more_tiny, decode, compute_loss, decode_train
from core.dataset import Dataset
from core.config import cfg
import numpy as np
from core import utils
from core.utils import freeze_all, unfreeze_all
import time
import matplotlib.pyplot as plt

import cv2


flags.DEFINE_string('model', 'yolov4', 'yolov4, yolov3')
flags.DEFINE_string('weights', './scripts/yolov4.weights', 'pretrained weights')
flags.DEFINE_integer('num_detection_layer', 1, '3: yolov4  |  2:yolov4-tiny  |  3:custom model')
# flags.DEFINE_boolean('show_map', True, 'true 면 학습시 map계산해서 보여줌')



def main(_argv):
    input_channel = 3
    patience = 30
    steps_in_epoch = 0
    epoch_loss = np.inf
    prev_minloss = np.inf

    trainset = Dataset(FLAGS,input_channel, is_training=True)
    # testset = Dataset(FLAGS, input_channel, is_training=False)
    logdir = "./data/log"
    isfreeze = False
    steps_per_epoch = len(trainset)
    print("steps_per_epoch:{}".format(steps_per_epoch))

    first_stage_epochs = cfg.TRAIN.FISRT_STAGE_EPOCHS
    second_stage_epochs = cfg.TRAIN.SECOND_STAGE_EPOCHS
    global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)
    warmup_steps = cfg.TRAIN.WARMUP_EPOCHS * steps_per_epoch
    total_steps = (first_stage_epochs + second_stage_epochs) * steps_per_epoch
    # train_steps = (first_stage_epochs + second_stage_epochs) * steps_per_period

    loss_tracker = []
    losses_in_epoch = 0.
    input_layer = tf.keras.layers.Input([cfg.TRAIN.INPUT_SIZE, cfg.TRAIN.INPUT_SIZE, 3])
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    IOU_LOSS_THRESH = cfg.YOLO.IOU_LOSS_THRESH

    freeze_layers = utils.load_freeze_layer(FLAGS.model, FLAGS.num_detection_layer)
    # feature_maps = YOLO(input_layer, NUM_CLASS, FLAGS.model, FLAGS.tiny)
    feature_maps = YOLOv4_more_tiny(input_layer, NUM_CLASS)

    bbox_tensors = []
    for i, fm in enumerate(feature_maps):        # fm shape: (None, featw, feath, filters)
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
    elif cfg.YOLO.NUM_YOLOLAYERS == 1: # custom yolo
        bbox_tensors = []
        bbox_tensor = decode_train(feature_maps[0], cfg.TRAIN.INPUT_SIZE // cfg.YOLO.STRIDES_CUSTOM[0], NUM_CLASS, STRIDES, ANCHORS, 0, XYSCALE)
        bbox_tensors.append(feature_maps[0])
        bbox_tensors.append(bbox_tensor)

    model = tf.keras.Model(input_layer, bbox_tensors)
    model.summary()


    if FLAGS.weights == None:
        print("Training from scratch")
    else:
        if FLAGS.weights.split(".")[len(FLAGS.weights.split(".")) - 1] == "weights":
            utils.load_weights(model, FLAGS.weights, FLAGS.model, FLAGS.num_detection_layer)
        else:
            model.load_weights(FLAGS.weights)

        print('Restoring weights from: %s ... ' % FLAGS.weights)

    optimizer = tf.keras.optimizers.Adam()

    if os.path.exists(logdir): shutil.rmtree(logdir)
    writer = tf.summary.create_file_writer(logdir)

    # define training step function
    # @tf.function
    def train_step(image_data, target):
        with tf.GradientTape() as tape:
            pred_result = model(image_data, training=True)
            ciou_loss = conf_loss = prob_loss = 0

            # optimizing process
            for i in range(len(freeze_layers)):
                conv, pred = pred_result[i * 2], pred_result[i * 2 + 1]  # target[i][1]:(32, 150, 4)
                # print("conv shape:{} pred shape:{}".format(tf.keras.backend.int_shape(conv), tf.keras.backend.int_shape(pred)))
                # print("target[i][0]:{} target[i][1]:{}".format(np.array(target[i][0]).shape, np.array(target[i][1]).shape))
                loss_items = compute_loss(pred, conv, target[i][0], target[i][1], STRIDES=STRIDES, NUM_CLASS=NUM_CLASS, IOU_LOSS_THRESH=IOU_LOSS_THRESH, i=i)
                ciou_loss += loss_items[0]
                conf_loss += loss_items[1]
                prob_loss += loss_items[2]

            total_loss = ciou_loss + conf_loss + prob_loss

            gradients = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            tf.print("=> STEP %4d/%4d   lr: %.6f   ciou_loss: %4.2f   conf_loss: %4.2f   "
                     "prob_loss: %4.2f   total_loss: %4.2f" % (global_steps, total_steps, optimizer.lr.numpy(),
                                                               ciou_loss, conf_loss,
                                                               prob_loss, total_loss))
            # update learning rate
            global_steps.assign_add(1)
            if global_steps < warmup_steps:
                lr = global_steps / warmup_steps * cfg.TRAIN.LR_INIT
            else:
                lr = cfg.TRAIN.LR_END + 0.5 * (cfg.TRAIN.LR_INIT - cfg.TRAIN.LR_END) * (
                    (1 + tf.cos((global_steps - warmup_steps) / (total_steps - warmup_steps) * np.pi))
                )
            optimizer.lr.assign(lr.numpy())

            # writing summary data
            with writer.as_default():
                tf.summary.scalar("lr", optimizer.lr, step=global_steps)
                tf.summary.scalar("loss/total_loss", total_loss, step=global_steps)
                tf.summary.scalar("loss/ciou_loss", ciou_loss, step=global_steps)
                tf.summary.scalar("loss/conf_loss", conf_loss, step=global_steps)
                tf.summary.scalar("loss/prob_loss", prob_loss, step=global_steps)
            writer.flush()
            # return total_loss

    @tf.function
    def test_step(image_data, target):
        with tf.GradientTape() as tape:
            pred_result = model(image_data, training=True)
            ciou_loss = conf_loss = prob_loss = 0

            preds = None
            gt_infos = None
            gt_bboxes = None
            # optimizing process
            for i in range(len(freeze_layers)):
                conv, pred = pred_result[i * 2], pred_result[i * 2 + 1]
                loss_items = compute_loss(pred, conv, target[i][0], target[i][1], STRIDES=STRIDES, NUM_CLASS=NUM_CLASS, IOU_LOSS_THRESH=IOU_LOSS_THRESH, i=i)

                ciou_loss += loss_items[0]
                conf_loss += loss_items[1]
                prob_loss += loss_items[2]

                if FLAGS.show_map:
                    batch_, _, _, _, ch_ = tf.keras.backend.int_shape(pred)
                    if preds == None:
                        preds = tf.reshape(pred, (batch_, -1, ch_))
                    else:
                        preds = tf.concat([preds, tf.reshape(preds, (batch_, -1, ch_))], axis=1)

                    if gt_infos == None:
                        gt_infos = tf.reshape(target[i][0], (batch_, -1, ch_))
                        gt_bboxes = target[i][1]
                    else:
                        gt_infos = tf.concat([gt_infos, tf.reshape(target[i][0], (batch_, -1, ch_))], axis=1)
                        gt_bboxes = tf.concat([gt_bboxes, tf.reshape(target[i][1], (batch_, -1, 4))], axis=1)


            if FLAGS.show_map:
                map = compute_map(preds, gt_bboxes, gt_infos, NUM_CLASS)

            total_map = map
            total_loss = ciou_loss + conf_loss + prob_loss

            tf.print("=> TEST STEP %4d   ciou_loss: %4.2f   conf_loss: %4.2f   "
                     "prob_loss: %4.2f   total_loss: %4.2f" % (global_steps, ciou_loss, conf_loss,
                                                               prob_loss, total_loss))
            tf.print("=> TEST STEP %4d map: %4.2f" % (total_map))
            # tf.print("=> TEST STEP %4d   giou_loss: %4.2f   conf_loss: %4.2f   "
            #          "prob_loss: %4.2f   total_loss: %4.2f" % (global_steps, giou_loss, conf_loss,
            #                                                    prob_loss, total_loss))

    for epoch in range(first_stage_epochs + second_stage_epochs):
        start_ch = time.time()
        train_loss = 0.
        if epoch < first_stage_epochs:
            if not isfreeze:
                isfreeze = True
                for name in freeze_layers:
                    try:
                        freeze = model.get_layer(name)
                        freeze_all(freeze)
                    except ValueError:
                        pass
        elif epoch >= first_stage_epochs:
            if isfreeze:
                isfreeze = False
                for name in freeze_layers:
                    try: # try구문 추가
                        freeze = model.get_layer(name)
                        freeze_all(freeze)
                    except ValueError:
                        pass


        for image_data, target in trainset:
            # bboxes_list = target[0][1]
            # label_data = target[0][0]
            #
            # for batch_idx, bboxes in enumerate(bboxes_list):
            #     # print("bboxes:{}".format(bboxes))
            #     class_inds = []
            #     check = np.array(image_data[batch_idx]).reshape(cfg.TRAIN.INPUT_SIZE, cfg.TRAIN.INPUT_SIZE, 3)
            #     # print(r"D:\tf_data_not\preprocessed_{}.npy".format(batch_idx))
            #     # np.save(r"D:\tf_data_not\image_{}.npy".format(batch_idx), check)
            #     # np.save(r"D:\tf_data_not\preprocessed_{}.npy".format(batch_idx), label_data[batch_idx])
            #     # np.save(r"D:\tf_data_not\bboxes_{}.npy".format(batch_idx), bboxes)
            #     label = label_data[batch_idx]
            #     label_class = label[...,5:]
            #     # print("label shape:{}".format(np.array(label).shape))
            #
            #     for line_label in label_class:
            #         if np.sum(line_label) > 0:
            #             print(line_label)
            #     class_ = np.array(label[..., 5:]).flatten()
            #     class_ = np.where(class_>0.1, class_, 0)
            #     for class_idx, class_label in enumerate(class_):
            #         if class_label > 0:
            #             class_inds.append(class_idx % cfg.YOLO.NUM_CLASSES)
            #
            #     for bbox_idx, bbox in enumerate(bboxes):
            #         half_h = bbox[3] / 2
            #         half_w = bbox[2] / 2
            #
            #         if np.sum(bbox) > 0:
            #             # print("class:{}".format(class_inds))
            #             cv2.rectangle(check, (int(bbox[0] - half_w), int(bbox[1] - half_h)),
            #                           (int(bbox[0] + half_w), int(bbox[1] + half_h)), color=(0, 255, 0), thickness=3)
            #             cv2.putText(check, text="{}".format(class_inds[bbox_idx]),
            #                         org=(int(bbox[0] + half_w)-10, int(bbox[1] + half_h)-30), thickness=2, color=(0, 255, 0),
            #                         fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7)
            #
            #     cv2.imshow("check_aug", check)
            #     cv2.waitKey(0)
            #     cv2.destroyAllWindows()
            #     cv2.imwrite(os.path.join(r"D:\Public\JHS\SIMPLE_DATA_INPUT_CHECK", "{}.jpg".format(batch_idx)), check*255)

            train_step(image_data, target)

        # for image_data, target in testset:
        #     test_step(image_data, target)

        ### for loss graph
        """
        if steps_in_epoch >= steps_per_epoch:
            loss_tracker.append(losses_in_epoch / steps_per_epoch)

            plt.plot(loss_tracker)
            plt.xlabel('epoch')
            plt.ylabel('loss')
            # plt.show()
            plt.savefig("D:/checkpoint/loss.png")
        """

        # print("steps_in_epoch:{}, steps_per_epoch:{}".format(global_steps, steps_per_epoch))
        # print("prev_minloss:{}, epoch_loss:{}".format(prev_minloss, epoch_loss))
        # # early stopping
        # if len(losses_in_epoch) >= steps_per_epoch:
        #
        #     epoch_loss = losses_in_epoch / steps_per_epoch
        #     if prev_minloss > epoch_loss:   # save best weight
        #         prev_minloss = epoch_loss
        #
        #         if epoch > 800:  # 최소학습 에폭
        #             model.save("D:\ckpt_best")
        #             print("{} epoch save best weights".format(epoch))
        #
        #     if len(loss_tracker) > patience:
        #         print("check loss_tracker len:{}".format(len(loss_tracker)))
        #         if loss_tracker[0] > np.min(loss_tracker[1:]):
        #             loss_tracker.pop(0)
        #         else:
        #             print("total loss didn't decreased during {} epochs. train stop".format(patience))
        #             return
        #     steps_in_epoch = 0
        #     epoch_loss = train_loss
        # else:
        #     epoch_loss += train_loss

        if (epoch+1) % 500 == 0:
            model.save(r"D:\notf_ckpt-epoch{}".format(epoch))
            print("{} epoch model saved".format(epoch))
    model.save(r"D:\notf_ckpt-last")

if __name__ == '__main__':
    try:
        start = time.time()
        app.run(main)
        print("train time: {} s".format(time.time() - start))
    except SystemExit:
        pass
