import tensorflow as tf
from nets import vgg

class Model(object):
    def __init__(self, is_training=True, batch_size=64):
        self.batch_size = batch_size
        self.image_size = 224
        self.label_num = 15
        self.initial_learning_rate = 1e-4
        self.dropout_keep_prob = 0.5
        self.is_training = is_training

        self.images = tf.placeholder(shape=[self.batch_size, self.image_size, self.image_size, 3], dtype=tf.float32)
        self.labels = tf.placeholder(shape=[self.batch_size, self.label_num], dtype=tf.float32)
        self.mask_beta = tf.placeholder(shape=[self.batch_size, self.label_num], dtype=tf.float32)
        self.mask_lambda = tf.placeholder(shape=[self.batch_size, self.label_num], dtype=tf.float32)

        self.build_cnn()
        self.build_metrics()
        self.build_optimizer()
        if is_training:
            self.build_summary()

    def build_cnn(self):
        with tf.contrib.slim.arg_scope(vgg.vgg_arg_scope()):
            _, end_points = vgg.vgg_19(inputs=self.images)
            net = end_points['vgg_19/fc7']     # shape = [batch size, 1, 1, 4096]

        with tf.variable_scope('mlc'):
            net = tf.contrib.slim.dropout(net, self.dropout_keep_prob, is_training=self.is_training, scope='dropout7')
            net = tf.contrib.slim.conv2d(net, 1024, [1, 1], activation_fn=tf.nn.relu, normalizer_fn=None, scope='fc8')  # shape = [batch size, 1, 1, 1024]
            net = tf.contrib.slim.dropout(net, self.dropout_keep_prob, is_training=self.is_training, scope='dropout8')
            net = tf.contrib.slim.conv2d(net, self.label_num, [1, 1], activation_fn=None, normalizer_fn=None,   scope='fc9')  # shape = [batch size, 1, 1, 15]
            logits = tf.squeeze(net, [1, 2])  # shape = [batch size, 15]

        self.logits = logits
        self.predictions = tf.nn.sigmoid(logits)
        self.conv5_3_feats = end_points['vgg_19/conv5/conv5_3']
        print('cnn built.')

    def build_metrics(self):
        # 1. build loss
        sigmoid_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=self.logits) # shape = [batch size, 15]
        sigmoid_loss_weighted = sigmoid_loss * self.mask_beta * self.mask_lambda

        mlc_loss = tf.reduce_sum(sigmoid_loss) / self.batch_size
        mlc_loss_weighted = tf.reduce_sum(sigmoid_loss_weighted) / self.batch_size
        reg_loss = tf.losses.get_regularization_loss()

        self.mlc_loss = mlc_loss
        self.mlc_loss_weighted = mlc_loss_weighted
        self.loss = mlc_loss_weighted + 0.0 * reg_loss
        self.reg_loss = reg_loss

        # 2. build auc
        _, auc_op = tf.metrics.auc(labels=self.labels, predictions=self.predictions)
        self.auc = auc_op
        print('metrics built.')

    def build_optimizer(self):
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        learning_rate = tf.constant(self.initial_learning_rate)

        def _learning_rate_decay_fn(learning_rate, global_step):
            return tf.train.exponential_decay(
                learning_rate=learning_rate,
                global_step=global_step,
                decay_steps=100000,
                decay_rate=0.9,
                staircase=True
            )

        # learning_rate_decay_fn = _learning_rate_decay_fn
        learning_rate_decay_fn = None

        vgg_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='vgg_19')  # get cnn layers' vars
        all_var_list = tf.trainable_variables()  # get all layers' vars
        other_var_list = [var for var in all_var_list if not var in vgg_var_list]

        with tf.variable_scope('optimizer', reuse=tf.AUTO_REUSE):
            optimizer = tf.train.AdamOptimizer(
                learning_rate=learning_rate,
                beta1=0.9,
                beta2=0.999,
                epsilon=1e-6
            )

            self.step_op = tf.contrib.layers.optimize_loss(
                loss=self.loss,
                global_step=self.global_step,
                learning_rate=learning_rate,
                optimizer=optimizer,
                clip_gradients=5.0,
                learning_rate_decay_fn=learning_rate_decay_fn,
                # variables=other_var_list
            )
        print('optimizer built.')

    def build_summary(self):
        with tf.name_scope('metrics'):
            tf.summary.scalar('mlc loss', self.mlc_loss)
            tf.summary.scalar('mlc loss weighted', self.mlc_loss_weighted)
            tf.summary.scalar('reg loss', self.reg_loss)

        with tf.name_scope('heatmap'):
            alpha_mean = tf.reduce_mean(self.conv5_3_feats, axis=3)
            conv5_3_feats = tf.reshape(alpha_mean, [self.batch_size, 14, 14, 1])
            tf.summary.image('conv5_3 feats', conv5_3_feats, max_outputs=4)

        with tf.name_scope('images'):
            tf.summary.image('oral images', self.images, max_outputs=4)

        self.summary = tf.summary.merge_all()
        print('summary built.')


