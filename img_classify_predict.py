# -*- coding: utf-8 -*-

import chainer
import chainer.functions as F
import chainer.optimizers
import numpy
import Image
import pickle

# load model
f = open('/tmp/model')
model = pickle.load(f)
f.close()


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
    print y.data
    d = numpy.argmax(y.data)
    return d

def loss(h, t):
    return F.softmax_cross_entropy(h, t)

optimizer = chainer.optimizers.Adam()
optimizer.setup(model.collect_parameters())

x_tmp = []
x_tmp.append(numpy.asarray(Image.open('/tmp/tmp.jpg')))
x_data = numpy.array(x_tmp, dtype=numpy.float32)
x = chainer.Variable(x_data)
digit = predict(x)

print digit

