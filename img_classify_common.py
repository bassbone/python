# -*- coding: utf-8 -*-

import chainer
import chainer.functions as F
import chainer.optimizers
import numpy
import Image
import pickle

class ImgClassify:

    def __init__(self, w1, w2, w3):
        self._w1 = w1
        self._w2 = w2
        self._w3 = w3

        self._model = chainer.FunctionSet(
            fc1 = F.Linear(self._w1, self._w2),
            fc2 = F.Linear(self._w2, self._w3),
        )

        self._optimizer = chainer.optimizers.Adam()
        self._optimizer.setup(self._model.collect_parameters())

    def forward(x):
        u2 = self._model.fc1(x)
        z2 = F.relu(u2)
        u3 = self._model.fc2(z2)
        return u3

    def output(x):
        h = self.forward(x)
        return F.softmax(h)

    def predict(x):
        y = self.output(x)
        d = numpy.argmax(y.data)
        return d

    def loss(h, t):
        return F.softmax_cross_entropy(h, t)

    def mini_batch_learn(x_train, t_train, mini_batch_size = 10):
        sum_loss = 0
        sum_accuracy = 0

        train_count = len(x_train)
        #train_count = 100 # DEBUG
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

            self._optimizer.zero_grads()

            x = chainer.Variable(x_batch)
            t = chainer.Variable(t_batch)
            h = self._forward(x)
            e = self._loss(h, t)
            a = F.accuracy(h, t)

            e.backward()
            self._optimizer.weight_decay(0.001)
            self._optimizer.update()

            sum_loss += float(e.data) * len(t_batch)
            sum_accuracy += float(a.data) * len(t_batch)

        train_loss = sum_loss / train_count
        train_accuracy = sum_accuracy / train_count
        return train_loss, train_accuracy

    def mini_batch_test(x_test, t_test, mini_batch_size = 10):
        sum_loss = 0
        sum_accuracy = 0

        test_count = len(x_test)
        #test_count = 20 # DEBUG
    
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
            h = self._forward(x)
            e = self._loss(h, t)
            a = F.accuracy(h, t)
        
            sum_loss += float(e.data) * len(t_batch)
            sum_accuracy += float(a.data) * len(t_batch)

        test_loss = sum_loss / test_count
        test_accuracy = sum_accuracy / test_count
        return test_loss, test_accuracy

    def getData(filename):
        x = []
        t = []
        for line in open('/tmp/' + filename):
            pair = line.strip().split()
            x.append(numpy.asarray(Image.open(pair[0])))
            t.append(pair[1])
        return x, t

    def saveModel():
        f = open('/tmp/model', 'wb')
        pickle.dump(model, f, -1)
        f.close()

