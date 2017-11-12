import tensorflow as tf
import numpy as np
import pickle
import CNN

epochs = 100
batch_size = 128
data = pickle.load(open('bnb.p', 'rb'))

sess = tf.Session()
dnn_model = CNN.CNN(sess = sess, input_shape=data['train_image'].shape, n_class = 2)
adam_opt = CNN.Solver(sess = sess, name = 'Adam', model = dnn_model, optimizer = tf.train.AdamOptimizer)

sess.run(tf.global_variables_initializer())
for epoch in range(epochs):
    total_batch = int(data['train_label'].shape[0] / batch_size)

    for step in range(total_batch):
        ind = np.random.randint(len(data['train_label']), size = 100)
        batch_xs, batch_ys = data['train_image'][ind], data['train_label'][ind]
        val_xs, val_ys = data['test_image'], data['test_label']

        adam_opt.train(x_data=batch_xs, y_data=batch_ys, lr=1e-4)
        val_loss = adam_opt.loss(x_data=val_xs, y_data=val_ys)

    if epoch % 10 == 0:
        adam_opt.print_information(data['train_image'], data['train_label'], data['test_image'], data['test_label'])

adam_opt.print_result()