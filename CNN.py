import tensorflow as tf
import matplotlib.pyplot as plt

class CNN:
    '''
     - X : 4차원 이미지 데이터, 사이즈에 유동적
     - y : 1차원 class 데이터
    '''
    def __init__(self, sess, input_shape, n_class,
                 activation_fn=tf.nn.relu, initializer=tf.contrib.layers.xavier_initializer()):

        _, w, h, d = input_shape
        self._sess = sess
        self._x = tf.placeholder(tf.float32, [None, w, h, d])
        self._y = tf.placeholder(tf.int32, [None])
        y_one_hot = tf.one_hot(self._y, n_class)
        y_one_hot = tf.reshape(y_one_hot, [-1, n_class])

        W1 = tf.get_variable(name="W1", shape=[3, 3, d, 32], dtype=tf.float32, initializer=initializer)
        L1 = tf.nn.conv2d(self._x, W1, strides=[1, 1, 1, 1], padding='SAME')
        L1 = activation_fn(L1)
        L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        w = int(w / 2 + 0.5)
        h = int(h / 2 + 0.5)

        W2 = tf.get_variable(name="W2", shape=[3, 3, 32, 64], dtype=tf.float32, initializer=initializer)
        L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
        L2 = activation_fn(L2)
        L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        w = int(w / 2 + 0.5)
        h = int(h / 2 + 0.5)

        L2_flat = tf.reshape(L2, [-1, w * h * 64])

        W3 = tf.get_variable("W3", shape=[w * h * 64, n_class], initializer=initializer)
        b = tf.Variable(tf.random_normal([n_class]))
        logits = tf.matmul(L2_flat, W3) + b
        self._prediction = tf.argmax(input= logits, axis = -1)

        xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_one_hot)
        self._loss = tf.reduce_mean(xentropy, name="loss")

class Solver:
    '''
    data를 이 안에 넣을까?
    '''
    def __init__(self, sess, name, model, optimizer=tf.train.AdamOptimizer):
        self._sess = sess
        self._model = model
        self._lr = tf.placeholder(dtype=tf.float32)
        self._loss_history = []
        self._train_acc_history = []
        self._test_acc_history = []

        with tf.variable_scope(name):
            self._optimizer = optimizer(self._lr)
            self._training_op = self._optimizer.minimize(self._model._loss)

    def train(self, x_data, y_data, lr = 1e-2, test_x_data = None, test_y_data = None, verbose = True):
        feed_train = {self._model._x: x_data, self._model._y: y_data, self._lr: lr}
        _, recent_loss = self._sess.run(fetches=[self._training_op, self._model._loss], feed_dict=feed_train)
        self._loss_history.append(recent_loss)
        self._train_acc_history.append(self.accuracy(x_data, y_data))
        self._test_acc_history.append(self.accuracy(test_x_data, test_y_data))
        # print process
        #if test_x_data is not None:
        #    self._val_acc_history.append(self.accuracy(test_x_data, test_y_data))
        #if verbose:
        #    if test_x_data is not None:
        #       self.print_information(x_data, y_data, test_x_data, test_y_data)
        #    else:
        #        self.print_information(x_data, y_data)

    def loss(self, x_data, y_data):
        feed_loss = {self._model._x: x_data, self._model._y: y_data}
        return self._sess.run(fetches=self._model._loss, feed_dict=feed_loss)

    def predict(self, x_data):
        feed_predict = {self._model._x: x_data}
        return self._sess.run(fetches=self._model._prediction, feed_dict=feed_predict)

    def print_accuracy(self, x_data, y_data):
        result = y_data == self.predict(x_data=x_data)
        print('accuracy : {:.4f}'.format(sum(result) / len(result)))

    def print_information(self, x_data, y_data, test_x_data = None, test_y_data = None):
        if test_x_data is None:
            print('loss : {:.4f}, train_accuracy : {:.4f}'.format(
                self.loss(x_data, y_data), self.accuracy(x_data, y_data)))
        else:
            test_result = test_y_data == self.predict(test_x_data)
            print('loss : {:.4f}, train_accuracy : {:.4f}, test_accuracy : {:.4f}'.format(
                self.loss(x_data, y_data), self.accuracy(x_data, y_data), self.accuracy(test_x_data, test_y_data)))

    def accuracy(self, x_data, y_data):
        if x_data is None:
            return 0
        result = y_data == self.predict(x_data)
        return sum(result) / len(result)

    def print_result(self):
        plt.plot(self._loss_history)
        plt.title('loss')
        plt.show()

        l = range(len(self._train_acc_history))
        plt.plot(l, self._train_acc_history, 'b', l, self._test_acc_history, 'r')
        plt.show()
