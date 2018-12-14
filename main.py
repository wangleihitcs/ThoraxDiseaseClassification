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

def test():
    md = Model(is_training=False)

    print('---Read Data...')
    image_batch, label_batch, mask_beta_batch, mask_lambda_batch = datasets.get_train_batch(test_tfrecord_name_path, md.batch_size)

    model_path = 'data/model/my-test-100000'


    temp_label_tp = tf.placeholder(shape=[md.label_num], dtype=tf.float32)
    predictions_tp = tf.placeholder(shape=[md.label_num], dtype=tf.float32)
    _, auc_op = tf.metrics.auc(temp_label_tp, predictions_tp)

    print('---Test Model...')
    saver = tf.train.Saver(max_to_keep=401)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        saver.restore(sess, model_path)

        coord = tf.train.Coordinator()                      # queue manage
        threads = tf.train.start_queue_runners(coord=coord)

        count_correct = np.zeros([md.label_num])
        count_all = np.zeros([md.label_num])
        auc_label = np.zeros([md.label_num])

        auc_list = []
        for _ in range(test_num / md.batch_size):
            images, labels, mask_betas, mask_lambdas = sess.run([image_batch, label_batch, mask_beta_batch, mask_lambda_batch])
            feed_dict = {md.images: images,
                         md.labels: labels,
                         md.mask_beta: mask_betas,
                         md.mask_lambda: mask_lambdas}
            _loss, _predictions, _auc = sess.run([md.loss, md.predictions, md.auc], feed_dict=feed_dict)

            temp_label = np.zeros([md.label_num])
            for i in range(md.batch_size):
                for j in range(md.label_num):
                    if labels[i][j] == 1:
                        count_all[j] += 1

                        # temp_label[j] = 1
                        # temp_auc = sess.run(auc_op, feed_dict={temp_label_tp: temp_label, predictions_tp:_predictions[i]})
                        # auc_label[j] += temp_auc
                        # temp_label[j] = 0

                    if labels[i][j] == 1 and _predictions[i][j] >= 0.2:
                        count_correct[j] += 1

            auc_list.append(_auc)
            # break
        print(count_correct)
        print(count_all)
        print(count_correct / count_all)
        print('*' * 100)
        print(np.mean(auc_list))
        print('*' * 100)
        # print(auc_label)
        # print(auc_label / count_all)
        # print('*' * 100)
        coord.request_stop()
        coord.join(threads)

    print('Test end.')

train()
# test()