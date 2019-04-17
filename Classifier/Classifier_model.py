import tensorflow as tf
from Classifier.hyperparams import Hyperparams as hp

class classifier_model(object):

    def __init__(self, _train = True):
        self.train = _train
        self.x = tf.placeholder(tf.float32, [None, hp.input_dim], 'x')
        self.y = tf.placeholder(tf.float32, [None, 1], 'y')
        self.prediction_grid = tf.placeholder(tf.float32, [None, hp.input_dim], 'prediction_grid')
        self.build_model()

    def build_model(self):

        self.fc1 = tf.layers.dense(self.x, hp.hidden_dim, name='fc1')
        self.fc1 = tf.contrib.layers.dropout(self.fc1, hp.drop_rate)
        self.fc1 = tf.nn.relu(self.fc1)
        # Final linear projection
        self.logits = tf.layers.dense(self.fc1, hp.class_num, name='fc2')

        self.preds = tf.to_int32(tf.arg_max(tf.nn.softmax(self.logits, axis=-1), dimension=-1))
        self.y_label = tf.to_int32(tf.argmax(self.y, -1))
        self.true_preds = tf.equal(self.preds, self.y_label)

        self.acc = tf.reduce_mean(tf.cast(tf.equal(self.y_label, self.preds), tf.float32))

        if self.train:
            self.y_smoothed = self.label_smoothing(tf.cast(self.y, dtype=tf.float32))
            self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y_smoothed)
            self.loss = tf.reduce_mean(self.loss)
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=hp.lr, beta1=0.9, beta2=0.98, epsilon=1e-8)
            self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)

            # Summary
            tf.summary.scalar('mean_loss', self.loss)
            self.merged = tf.summary.merge_all()

    def label_smoothing(self, inputs, epsilon=0.1):
        K = inputs.get_shape().as_list()[-1]  # number of channels
        return ((1 - epsilon) * inputs) + (epsilon / K)