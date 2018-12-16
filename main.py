import tensorflow as tf
import numpy as np
import datasets
from mlc_model import Model

pretrain_vgg_19_ckpt_path = '/home/wanglei/workshop/b_pre_train_model/vgg/vgg_19.ckpt'
train_tfrecord_name_path = 'data/train_tfrecord_name.txt'
test_tfrecord_name_path = 'data/test_tfrecord_name.txt'
summary_path = 'data/summary'                               # data/summary to save events tf.summary
model_path_save = 'data/model/my-test'                      # data/model to save my-test-xxx.ckpt

num_epochs = 101
train_val_num = 86524
test_num = 25596

def train():
    md = Model(is_training=True)

    print('---Read Data...')
    image_batch, label_batch, mask_beta_batch, mask_lambda_batch = datasets.get_train_batch(train_tfrecord_name_path, md.batch_size)
    # print(image_batch)

    print('---Training Model...')
    init_fn = tf.contrib.slim.assign_from_checkpoint_fn(pretrain_vgg_19_ckpt_path, tf.contrib.slim.get_model_variables('vgg_19'))  # 'vgg_19'

    saver = tf.train.Saver(max_to_keep=501)
    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter(summary_path, sess.graph)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        init_fn(sess)

        coord = tf.train.Coordinator()  # queue manage
        threads = tf.train.start_queue_runners(coord=coord)

        iter = 0
        loss_list = []
        for epoch in range(num_epochs):
            for _ in range(train_val_num / md.batch_size):
                images, labels, mask_betas, mask_lambdas = sess.run([image_batch, label_batch, mask_beta_batch, mask_lambda_batch])
                feed_dict = {md.images: images,
                             md.labels: labels,
                             md.mask_beta: mask_betas,
                             md.mask_lambda: mask_lambdas}
                _, _summary, _global_step, _loss, = sess.run([md.step_op, md.summary, md.global_step, md.loss], feed_dict=feed_dict)
                train_writer.add_summary(_summary, _global_step)
                loss_list.append(_loss)

                iter += 1
                if iter % 1000 == 0:
                    print('epoch = %s, iter = %s, loss = %s' % (epoch, iter, np.mean(loss_list)))
                    loss_list = []
                if iter % 1000 == 0:
                    saver.save(sess, model_path_save, global_step=iter)

        coord.request_stop()
        coord.join(threads)

    print('Train end.')

def eval(model_path):
    md = Model(is_training=False)

    print('---Read Data...')
    image_batch, label_batch, mask_beta_batch, mask_lambda_batch = datasets.get_train_batch(test_tfrecord_name_path, md.batch_size)

    temp_label_tp = tf.placeholder(shape=[None], dtype=tf.float32)
    predictions_tp = tf.placeholder(shape=[None], dtype=tf.float32)
    _, auc_op = tf.metrics.auc(temp_label_tp, predictions_tp)

    print('---Test Model...')
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        saver.restore(sess, model_path)

        coord = tf.train.Coordinator()                      # queue manage
        threads = tf.train.start_queue_runners(coord=coord)

        loss_list = []

        iter = 0
        epochs = test_num / md.batch_size
        auc_predict = np.zeros([epochs * md.batch_size, md.label_num])
        auc_label = np.zeros([epochs * md.batch_size, md.label_num])
        threshold = 0.5 * np.ones([md.label_num])
        # threshold = [0.0991, 0.0318, 0.0165, 0.0203, 0.0261, 0.097, 0.0338, 0.0171, 0.0104, 0.055, 0.0483, 0.1592, 0.0017, 0.5841, 0.0138]

        true_positive_count = np.zeros([md.label_num])
        precision_positive_count = np.zeros([md.label_num])
        actual_positive_count = np.zeros([md.label_num])
        for _ in range(epochs):
            images, labels, mask_betas, mask_lambdas = sess.run([image_batch, label_batch, mask_beta_batch, mask_lambda_batch])
            feed_dict = {md.images: images,
                         md.labels: labels,
                         md.mask_beta: mask_betas,
                         md.mask_lambda: mask_lambdas}
            _loss, _predictions, _auc = sess.run([md.loss, md.predictions, md.auc], feed_dict=feed_dict)
            loss_list.append(_loss)

            for i in range(md.batch_size):
                auc_predict[iter] = _predictions[i]
                auc_label[iter] = labels[i]
                iter += 1
                # print(_predictions[i])
                # print(labels[i])
                # print('*' * 100)

                for j in range(md.label_num):
                    if labels[i][j] == 1 and _predictions[i][j] >= threshold[j]:
                        true_positive_count[j] += 1
                    if labels[i][j] == 1:
                        actual_positive_count[j] += 1
                    if _predictions[i][j] >= threshold[j]:
                        precision_positive_count[j] += 1

        coord.request_stop()
        coord.join(threads)

        # 1. loss and auc
        auc_list = []
        for i in range(md.label_num):
            temp_auc = sess.run(auc_op, feed_dict={temp_label_tp: auc_label[:, i], predictions_tp: auc_predict[:, i]})
            auc_list.append(temp_auc)
        # print(auc_list)
        print('mean loss = %s, mean auc = %s, %s' % (np.mean(loss_list), np.mean(auc_list), auc_list))

        # 2. recall and prediction
        # recall = true_positive_count / actual_positive_count
        # for j in range(md.label_num):
        #     recall[j] = round(recall[j], 4)
        # print('mean recall = %s, %s' % (np.mean(recall), recall.tolist()))
        # precision = true_positive_count / precision_positive_count
        # for j in range(md.label_num):
        #     precision[j] = round(precision[j], 4)
        # print('mean precision = %s, %s' % (np.mean(precision), precision.tolist()))
        # f1_score = 2 * recall * precision / (recall + precision)
        # for j in range(md.label_num):
        #     f1_score[j] = round(f1_score[j], 4)
        # print('mean F1-score = %s, %s' % (np.mean(f1_score), f1_score.tolist()))

        # 3. to find good threshold for 15 label
        # 3.1 use mean to get threshold
        threshold_mean = np.mean(auc_predict, axis = 0)
        for j in range(md.label_num):
            threshold_mean[j] = round(threshold_mean[j], 4)
        print('mean throshold = %s' % threshold_mean.tolist())

        # 3.2 from train label nums to get threshold
        # label_count = [8659, 2637, 1378, 1707, 2242, 8208, 2852, 1423, 876, 4708, 4034, 13782, 141, 50500, 1251]
        # threshold = np.zeros([md.label_num])
        # for j in range(md.label_num):
        #     temp_array = np.array(sorted(auc_predict.tolist(), key=lambda auc_predict: auc_predict[j]))
        #     threshold[j] = temp_array[label_count[j], j]
        #     threshold[j] = round(threshold[j], 4)
        # print('throshold = %s' % threshold.tolist())

    print('Test end.')

# train()

model_path = 'data/model/my-test-82000'
eval(model_path)