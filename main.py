import tensorflow as tf
from tensorflow.keras import layers, models
from matplotlib import pyplot as plt
import numpy as np
import os

'''
this project is the demo code for predicting the controlling point from a set of points of Bezier curve.
the source code is from https://blog.csdn.net/zhigongjz/article/details/110119930
if the dimension is extended to 3D, the controlling points can be predicted from 3D points clouds.
And you can custom the n as you like. that means you can use any order of Bezier surface to approximate the 3D point cloud.
'''

b_xs = []
b_ys = []
# xs表示原始数据
# n表示阶数
# k表示索引
def one_bezier_curve(a, b, t):
    return (1 - t) * a + t * b
 
 
def n_bezier_curve(xs, n, k, t):
    if n == 1:
        return one_bezier_curve(xs[k], xs[k + 1], t)
    else:
        return (1 - t) * n_bezier_curve(xs, n - 1, k, t) + t * n_bezier_curve(xs, n - 1, k + 1, t)
 
def bezier_curve(xs, ys, num, b_xs, b_ys):
    n = 3  # 采用5次bezier曲线拟合
    t_step = 1.0 / (num - 1)
    # t_step = 1.0 / num
    print(t_step)
    t = np.arange(0.0, 1 + t_step, t_step)
    print(len(t))
    for each in t:
        b_xs.append(n_bezier_curve(xs, n, 0, each))
        b_ys.append(n_bezier_curve(ys, n, 0, each))
 
class AnnNet(object):
    def __init__(self):
        model = models.Sequential()
        model.add(layers.Dense(64, activation='sigmoid', input_shape=(10,)))
        model.add(layers.Dense(32, activation='sigmoid'))
        model.add(layers.Dense(4))
        model.summary()
        self.model = model
 
    def GenerateArray(self):
        while 1:
            x= []
            y= []
            for i in range(10):
                point = np.random.rand(8) * 10
                times = np.random.rand(3)
                times = [0.2, 0.5, 0.8]
                pt1x = ((1 - times[0]) ** 3) * point[0] + 3 * times[0] * ((1 - times[0]) ** 2) * point[2]  + 3 * (times[0] ** 2) * (1 - times[0]) * point[4] + (times[0] ** 3) * point[6]
                pt1y = ((1 - times[0]) ** 3) * point[1] + 3 * times[0] * ((1 - times[0]) ** 2) * point[3]  + 3 * (times[0] ** 2) * (1 - times[0]) * point[5] + (times[0] ** 3) * point[7]
                pt2x = ((1 - times[1]) ** 3) * point[0] + 3 * times[1] * ((1 - times[1]) ** 2) * point[2]  + 3 * (times[1] ** 2) * (1 - times[1]) * point[4] + (times[1] ** 3) * point[6]
                pt2y = ((1 - times[1]) ** 3) * point[1] + 3 * times[1] * ((1 - times[1]) ** 2) * point[3]  + 3 * (times[1] ** 2) * (1 - times[1]) * point[5] + (times[1] ** 3) * point[7]
                pt3x = ((1 - times[2]) ** 3) * point[0] + 3 * times[2] * ((1 - times[2]) ** 2) * point[2]  + 3 * (times[0] ** 2) * (1 - times[2]) * point[4] + (times[2] ** 3) * point[6]
                pt3y = ((1 - times[2]) ** 3) * point[1] + 3 * times[2] * ((1 - times[2]) ** 2) * point[3]  + 3 * (times[0] ** 2) * (1 - times[2]) * point[5] + (times[2] ** 3) * point[7]
                x.append([point[0], point[1], pt1x, pt1y, pt2x, pt2y, pt3x, pt3y, point[6], point[7]])
                y.append([point[2], point[3], point[4], point[5]])
            X = np.array(x)
            Y = np.array(y)
            yield (X, Y)
 
    def LossFunc(self, ytrue, ypred):
        loss = tf.reduce_mean(tf.square(ytrue-ypred))
        return loss
 
    def Train(self):
        print('train start!')
        self.check_path = 'save.ckpt'
        if os.path.exists(self.check_path + '.index'):
            self.model.load_weights(self.check_path)
        self.model.compile(optimizer='adam', loss = self.LossFunc)
        save_model_cb = tf.keras.callbacks.ModelCheckpoint(self.check_path, save_weights_only=True, verbose=1, period=1)
        back = self.model.fit_generator(self.GenerateArray(), steps_per_epoch=1000, epochs=100, callbacks=[save_model_cb])
        print('train end!')
 
    def Test(self):
        print('test start!')
        self.check_path = 'save.ckpt'
        if os.path.exists(self.check_path + '.index'):
            self.model.load_weights(self.check_path)
        x= []
        y= []
        point = np.random.rand(8) * 10
        times = [0.2, 0.5, 0.8]
        pt1x = ((1 - times[0]) ** 3) * point[0] + 3 * times[0] * ((1 - times[0]) ** 2) * point[2]  + 3 * (times[0] ** 2) * (1 - times[0]) * point[4] + (times[0] ** 3) * point[6]
        pt1y = ((1 - times[0]) ** 3) * point[1] + 3 * times[0] * ((1 - times[0]) ** 2) * point[3]  + 3 * (times[0] ** 2) * (1 - times[0]) * point[5] + (times[0] ** 3) * point[7]
        pt2x = ((1 - times[1]) ** 3) * point[0] + 3 * times[1] * ((1 - times[1]) ** 2) * point[2]  + 3 * (times[1] ** 2) * (1 - times[1]) * point[4] + (times[1] ** 3) * point[6]
        pt2y = ((1 - times[1]) ** 3) * point[1] + 3 * times[1] * ((1 - times[1]) ** 2) * point[3]  + 3 * (times[1] ** 2) * (1 - times[1]) * point[5] + (times[1] ** 3) * point[7]
        pt3x = ((1 - times[2]) ** 3) * point[0] + 3 * times[2] * ((1 - times[2]) ** 2) * point[2]  + 3 * (times[0] ** 2) * (1 - times[2]) * point[4] + (times[2] ** 3) * point[6]
        pt3y = ((1 - times[2]) ** 3) * point[1] + 3 * times[2] * ((1 - times[2]) ** 2) * point[3]  + 3 * (times[0] ** 2) * (1 - times[2]) * point[5] + (times[2] ** 3) * point[7]
        x.append([point[0], point[1], pt1x, pt1y, pt2x, pt2y, pt3x, pt3y, point[6], point[7]])
        y.append([point[2], point[3], point[4], point[5]])
        X = np.array(x)
        Y = np.array(y)
        ypred = self.model.predict(X)
        print(Y)
        print(ypred)
        num = 20
        xs = [point[0], point[2], point[4], point[6]]
        ys = [point[1], point[3], point[5], point[7]]
        b_xs = []
        b_ys = []
        bezier_curve(xs, ys, num, b_xs, b_ys)  # 将计算结果加入到列表中
        plt.figure()
        plt.plot(b_xs, b_ys, 'r')  # bezier曲线
        plt.plot(xs, ys, '.r')  # 控制点
 
        b_xs = []
        b_ys = []
        xs = [point[0], ypred[0][0], ypred[0][2], point[6]]
        ys = [point[1], ypred[0][1], ypred[0][3], point[7]]
        bezier_curve(xs, ys, num, b_xs, b_ys)  # 将计算结果加入到列表中
        plt.plot(b_xs, b_ys, 'y')  # bezier曲线
        plt.plot(xs, ys, '.y')  # 控制点
        plt.show()
 
        print('test end!')
 
if __name__ == '__main__':
    net = AnnNet()
    net.Train()
    net.Test()
 