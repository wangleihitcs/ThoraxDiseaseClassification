import tensorflow as tf
import json
import os
import numpy as np
from utils import image_utils

def get_train_batch(tfrecord_list_path, batch_size, image_size=224, label_num=15):
    with open(tfrecord_list_path, 'r') as file:
        lines = file.readlines()
        tfrecord_path_list = [line.strip() for line in lines]
    # print(tfrecord_path_list)

    # 1. get filename_queue
    filename_queue = tf.train.string_input_producer(tfrecord_path_list, shuffle=True)

    # 2. get image pixels, label
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features = {
            'image_pixels': tf.FixedLenFeature([image_size*image_size*3], tf.float32),
            'label_one_hot': tf.FixedLenFeature([label_num], tf.int64),
            'mask_beta': tf.FixedLenFeature([label_num], tf.float32),
            'mask_lambda': tf.FixedLenFeature([label_num], tf.float32)
        }
    )
    image = tf.reshape(features['image_pixels'], [image_size, image_size, 3])
    label_one_hot = features['label_one_hot']
    mask_beta = features['mask_beta']
    mask_lambda = features['mask_lambda']

    # 3. get train batch
    min_after_dequeue = 10000
    image_batch, label_batch, mask_beta_batch, mask_lambda_batch = tf.train.shuffle_batch(
        [image, label_one_hot, mask_beta, mask_lambda],
        batch_size = batch_size,
        capacity = min_after_dequeue + 3 * batch_size,
        min_after_dequeue = min_after_dequeue
    )

    return image_batch, label_batch, mask_beta_batch, mask_lambda_batch

# get /data/tfrecord/xxx.tfrecord
def get_train_tfrecord(imgs_path, data_entry_path, data_label_path, split_list_path, image_size=224, label_num=15, mode='train', D=40):
    with open(data_entry_path, 'r') as file:
        data_dict = json.load(file)
    with open(data_label_path, 'r') as file:
        data_label_dict = json.load(file)
        finding_labels = data_label_dict.keys()
    with open(split_list_path, 'r') as file:
        lines = file.readlines()
        split_id_list = [line.strip() for line in lines]

    Q = 112120  # all image num
    Qn = 60361  # all no disease num
    for i in range(D):
        subsets_num = split_id_list.__len__() / D + 1
        sub_split_id_list = split_id_list[i * subsets_num: (i + 1) * subsets_num]

        tfrecord_name = 'data/tfrecord/' + mode + '-%02d.tfrecord' % i
        writer = tf.python_io.TFRecordWriter(tfrecord_name)

        for image_index in sub_split_id_list:
            img_file_path = os.path.join(imgs_path, image_index)
            image = image_utils.getImages(img_file_path, image_size)
            image_pixels = image.reshape([image_size*image_size*3])

            label_one_hot = np.zeros(shape=[label_num], dtype=np.int64)
            labels = data_dict[image_index]
            mask_beta = np.zeros(shape=[label_num], dtype=np.float32)
            mask_lambda = np.zeros(shape=[label_num], dtype=np.float32)
            for label in labels:
                for j in range(label_num):
                    mask_lambda[j] = (0.0 + Q - data_label_dict[finding_labels[j]]) / Q
                    if label == finding_labels[j]:
                        label_one_hot[j] = 1
                        mask_beta[j] = (0.0 + Qn) / Q
                    else:
                        mask_beta[j] = (0.0 + Q - Qn) / Q

            example = tf.train.Example(
                features = tf.train.Features(
                    feature = {
                        'image_pixels': tf.train.Feature(float_list = tf.train.FloatList(value = image_pixels)),
                        'label_one_hot': tf.train.Feature(int64_list = tf.train.Int64List(value = label_one_hot)),
                        'mask_beta': tf.train.Feature(float_list = tf.train.FloatList(value = mask_beta)),
                        'mask_lambda': tf.train.Feature(float_list = tf.train.FloatList(value = mask_lambda))
                    }
                )
            )
            serialized = example.SerializeToString()
            writer.write(serialized)

        print('%s write to tfrecord success!' % tfrecord_name)

# get 'data/train_tfrecord_name.txt', 'data/test_tfrecord_name.txt'
def get_tfrecord_split():
    D = 40
    train_tfrecord_name_list = []
    for i in range(D):
        tfrecord_name = 'data/tfrecord/train-%02d.tfrecord\n' % i
        train_tfrecord_name_list.append(tfrecord_name)

    D = 15
    test_tfrecord_name_list = []
    for i in range(D):
        tfrecord_name = 'data/tfrecord/test-%02d.tfrecord\n' % i
        test_tfrecord_name_list.append(tfrecord_name)

    with open('data/train_tfrecord_name.txt', 'w') as file:
        file.writelines(train_tfrecord_name_list)
    with open('data/test_tfrecord_name.txt', 'w') as file:
        file.writelines(test_tfrecord_name_list)
    print('tfrecord split.')

def main():
    imgs_path = '/home/wanglei/workshop/NIHChestX-Ray/images'
    data_entry_path = 'data/data_entry.json'
    data_label_path = 'data/data_label.json'
    train_val_list_path = 'data/train_val_list.txt'
    # get_train_tfrecord(imgs_path, data_entry_path, data_label_path, train_val_list_path, mode='train', D=40)
    test_list_path = 'data/test_list.txt'
    # get_train_tfrecord(imgs_path, data_entry_path, data_label_path, test_list_path, mode='test', D=15)

    # get_tfrecord_split()
# main()