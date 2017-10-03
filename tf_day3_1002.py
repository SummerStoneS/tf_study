import tensorflow as tf
import os

BATCH_SIZE = 50

global_step = tf.Variable(0.0)


# 读取文件
def read_csv(batch_size, file_name, record_defaults):
    filename_queue = tf.train.string_input_producer([''.join(['C:/Users/Ruofei Shen/Desktop/Serena/serena_code/dataset/', file_name])])
    reader = tf.TextLineReader(skip_header_lines=1)
    key, value = reader.read(filename_queue)
    decoded = tf.decode_csv(value, record_defaults=record_defaults)
    return tf.train.shuffle_batch(decoded, batch_size=batch_size, capacity=batch_size*50, min_after_dequeue=batch_size)


def inputs():
    passenger_id, survived, pclass, name, sex, age, sibsp, parch, ticket, fare, cabin, embarked = \
        read_csv(BATCH_SIZE, 'titanic.csv', [[0.0], [0.0], [0], [''], [''], [0.0], [0.0], [0.0], [''], [0.0], [''], ['']])
    # 转换属性数据
    is_first_class = tf.to_float(tf.equal(pclass, [1]))
    is_second_class = tf.to_float(tf.equal(pclass, [2]))
    is_third_class = tf.to_float(tf.equal(pclass, [3]))

    gender = tf.to_float(tf.equal(sex, ["female"]))
    age = tf.nn.l2_normalize(age, dim=0)
    features = tf.transpose(tf.stack([is_first_class, is_second_class, is_third_class, gender, age]))
    survived = tf.reshape(survived, [BATCH_SIZE, 1])
    print(features.shape)
    return features, survived


# w = tf.Variable(tf.zeros([5,1]), name='weights')
w = tf.Variable([0.6, -0.2, -1.26, 2.5, -0.26], name='weights')
w = tf.reshape(w, [5, 1])
b = tf.Variable(-0.9, name='bias')


def inference(x):
    y_hat = tf.matmul(x, w)+b
    # return tf.sigmoid(y_hat)
    return y_hat


def loss(x, y):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=inference(x), labels=y))


def train(total_loss):
    learning_rate = 0.01
    return tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    features, survived = inputs()
    total_loss = loss(features, survived)
    train_op = train(total_loss)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for step in range(10000):
        sess.run([train_op])
        if step % 100 == 0:
            print('loss:', sess.run(total_loss))
            print('wb:', sess.run([w, b]))

    coord.request_stop()
    coord.join(threads)


# 手写数字
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("mnist/", one_hot=True)


def inputs():
    x = tf.placeholder(tf.float32, [None, 784])
    y = tf.placeholder(tf.float32, [None, 10])
    return x,y

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))


def inference(x):
    return tf.nn.softmax(tf.matmul(x, W) + b)


def loss(x, y):
    return tf.reduce_mean(-tf.reduce_sum(y * tf.log(inference(x)), reduction_indices=[1]))


def train(total_loss):
    learning_rate = 0.5
    return tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    x, y = inputs()
    total_loss = loss(x, y)
    train_op = train(total_loss)
    for _ in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_op, feed_dict={x: batch_xs, y: batch_ys})

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(inference(x), 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))