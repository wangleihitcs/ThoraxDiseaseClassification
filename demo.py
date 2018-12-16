import tensorflow as tf
import numpy as np
import json

from mlc_model import Model
from utils import image_utils

image_path = 'data/CXR3_IM-1384-1001.png'

FLAGS = tf.app.flags.FLAGS
tf.flags.DEFINE_string('img', 'data/examples/CXR3_IM-1384-1001.png', 'The test image path')

model_path = 'data/model/my-test-68000'
data_label_path = 'data/data_label.json'

md = Model(is_training=False, batch_size=1)

with open(data_label_path, 'r') as file:
    labels = json.load(file).keys()
id2label = {}
for j in range(md.label_num):
    id2label[j] = labels[j]

saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, model_path)

    images = np.zeros([md.batch_size, md.image_size, md.image_size, 3])
    images[0] = image_utils.getImages(FLAGS.img, md.image_size)
    paddings = np.zeros([md.batch_size, md.label_num])
    feed_dict = {md.images: images,
                 md.labels: paddings,
                 md.mask_beta: paddings,
                 md.mask_lambda: paddings}
    predictions = sess.run(md.predictions, feed_dict=feed_dict)

    for j in range(md.label_num):
        print('disease \'%s\', prob = %s' % (labels[j], round(predictions[0][j], 5)))

