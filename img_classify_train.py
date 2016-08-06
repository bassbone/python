# -*- coding: utf-8 -*-

import chainer
import chainer.functions as F
import chainer.optimizers
import numpy
import Image
import pickle

model = chainer.FunctionSet(
    fc1 = F.Linear(256 * 256 * 3, 64),
    fc2 = F.Linear(64, 2),
)

def forward(x):
    u2 = model.fc1(x)
    z2 = F.relu(u2)
    u3 = model.fc2(z2)
    return u3

def output(x):
    h = forward(x)
    return F.softmax(h)

def predict(x):
    y = output(x)
    d = numpy.argmax(y.data)
    return d

def loss(h, t):
    return F.softmax_cross_entropy(h, t)

optimizer = chainer.optimizers.Adam()
optimizer.setup(model.collect_parameters())

def mini_batch_learn(x_train, t_train, mini_batch_size = 10):
    sum_loss = 0
    sum_accuracy = 0

    train_count = len(x_train)
    #train_count = 100
    perm = numpy.random.permutation(train_count)

    for i in range(0, train_count, mini_batch_size):
        print i
        x_batch = []
        t_batch = []
        for j in perm[i:i + mini_batch_size]:
            x_batch.append(x_train[j])
            t_batch.append(t_train[j])
        x_batch = numpy.array(x_batch, dtype=numpy.float32)
        t_batch = numpy.array(t_batch, dtype=numpy.int32)

        optimizer.zero_grads()

        x = chainer.Variable(x_batch)
        t = chainer.Variable(t_batch)
        h = forward(x)
        e = loss(h, t)
        a = F.accuracy(h, t)

        e.backward()
        optimizer.weight_decay(0.001)
        optimizer.update()

        sum_loss += float(e.data) * len(t_batch)
        sum_accuracy += float(a.data) * len(t_batch)

    train_loss = sum_loss / train_count
    train_accuracy = sum_accuracy / train_count
    return train_loss, train_accuracy

def mini_batch_test(x_test, t_test, mini_batch_size = 10):
    sum_loss = 0
    sum_accuracy = 0

    test_count = len(x_test)
    #test_count = 20
    
    for i in range(0, test_count, mini_batch_size):
        print i
        x_batch = []
        t_batch = []
        for j in range(i, i + mini_batch_size):
            x_batch.append(x_train[j])
            t_batch.append(t_train[j])
        x_batch = numpy.array(x_batch, dtype=numpy.float32)
        t_batch = numpy.array(t_batch, dtype=numpy.int32)

        x = chainer.Variable(x_batch)
        t = chainer.Variable(t_batch)
        h = forward(x)
        e = loss(h, t)
        a = F.accuracy(h, t)
        
        sum_loss += float(e.data) * len(t_batch)
        sum_accuracy += float(a.data) * len(t_batch)

    test_loss = sum_loss / test_count
    test_accuracy = sum_accuracy / test_count
    return test_loss, test_accuracy

def getTrainData():
    x = []
    t = []
    train_text = '/tmp/train.txt'
    for line in open(train_text):
        pair = line.strip().split()
        x.append(numpy.asarray(Image.open(pair[0])))
        t.append(pair[1])
    return x, t

def getTestData():
    x = []
    t = []
    test_text = '/tmp/test.txt'
    for line in open(test_text):
        pair = line.strip().split()
        x.append(numpy.asarray(Image.open(pair[0])))
        t.append(pair[1])
    return x, t

train_loss = []
train_accuracy = []
test_loss = []
test_accuracy = []

x_train, t_train = getTrainData()
x_test, t_test = getTestData()

max_epoch = 5

for epoch in range(1, max_epoch + 1):
    e, a = mini_batch_learn(x_train, t_train)
    train_loss.append(e)
    train_accuracy.append(a)

    e, a = mini_batch_test(x_test, t_test)
    test_loss.append(e)
    test_accuracy.append(a)

print train_loss
print train_accuracy
print test_loss
print test_accuracy

f = open('/tmp/model', 'wb')
pickle.dump(model, f, -1)
f.close()

