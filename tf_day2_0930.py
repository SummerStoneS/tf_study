"""
    create on: 2017/9/30
"""
# 有监督学习框架
import tensorflow as tf
import os

def inference(x):
    # 计算模型在x上的输出，返回结果

def loss(x, y):
    # 根据x对应的实际y值和模型给出的y值计算损失

def inputs():
    # 读取训练数据x和y

def train(total_loss):
    # 依据计算的总损失训练或调整模型参数

def evaluate(sess, x, y):
    # 对训练得到的模型进行评估

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    x,y = inputs()
    total_loss = loss(x,y)
    train_op = train(total_loss)

    coord = tf.train.Coordinator()  # 可以在发生错误的情况下正确地关闭这些线程
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)   # 函数将会启动输入管道的线程，填充样本到队列中，以便出队操作可以从队列中拿到样本

    training_steps = 1000
    for step in range(training_steps):
        sess.run([train_op])
        if step % 10 == 0:
            print("loss:",sess.run([total_loss]))
    evaluate(sess,x,y)
    coord.request_stop()
    coord.join(threads)


# http://blog.sina.com.cn/s/blog_e22771170102wcfv.html

# checkpoint
# 创建一个Saver对象
saver = tf.train.Saver()

# 启动Session，在训练过程中阶段性创建checkpoint，保存变量值
with tf.Session() as sess:
    # ......
    start_step = 0
    # 检查是否有checkpoint
    checkpoint = tf.train.get_checkpoint_state(os.path.dirname(__file__))
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        start_step = int(checkpoint.model_checkpoint_path.rsplit('-',1))[1]

    for step in range(training_steps):
        sess.run([train_op])
        if step % 1000 == 0:
            saver.save(sess, 'model', global_step=step)
    # ...
    saver.save(sess, 'model', global_step=training_steps)

# 线性回归
w = tf.Variable(tf.zeros([2,1]), name='weights')
b = tf.Variable(0., name='bias')

def inference(x):
    return tf.matmul(x,w)+b

def loss(x, y):
    y_hat = inference(x)
    return tf.reduce_sum(tf.squared_difference(y,y_hat))

def inputs():
    x = tf.random_normal([50,2], mean=0.0, stddev=1.0)
    w = tf.constant([[0.3], [7]])
    y = tf.matmul(x, w) + 2
    return x,y

def train(total_loss):
    learning_rate = 0.0001
    return tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    x,y = inputs()
    total_loss = loss(x, y)
    train_op = train(total_loss)

    training_steps = 1000
    for step in range(training_steps):
        sess.run([train_op])
        if step % 10 == 0:
            print('loss:', sess.run(total_loss))
            print('wb:', sess.run([w, b]))


# logistic 回归模型
w = tf.Variable(tf.zeros([4,1]), name="weights")
b = tf.Variable(0., name='bias')

def inference(x):
    y = tf.matmul(x, w)+b
    return tf.sigmoid(y)

def loss(x, y):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(inference(x), y))





