# -*- coding: utf-8 -*-

import chainer
import chainer.functions as F
import chainer.optimizers
import numpy
import Image
import pickle
import img_classify_common as imc

w1 = 256 * 256 * 3
w2 = 64
w3 = 2

model = chainer.FunctionSet(
    fc1 = F.Linear(w1, w2),
    fc2 = F.Linear(w2, w3),
)

instance = imc.ImgClassify(model)

train_loss = []
train_accuracy = []
test_loss = []
test_accuracy = []

x_train, t_train = instance.getData('train.txt')
x_test, t_test = instance.getData('test.txt')

max_epoch = 2

for epoch in range(1, max_epoch + 1):
    e, a = instance.mini_batch_learn(x_train, t_train)
    train_loss.append(e)
    train_accuracy.append(a)
    print "epoch " + str(epoch) + ": train accuracy " + str(a)

    e, a = instance.mini_batch_test(x_test, t_test)
    test_loss.append(e)
    test_accuracy.append(a)
    print "epoch " + str(epoch) + ": test  accuracy " + str(a)

instance.saveModel()

