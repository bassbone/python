#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer
import chainer.functions as F
import chainer.optimizers
import numpy
import Image
import pickle
import cgi
import cgitb
import os
import sys

cgitb.enable()

print('Content-type: text/html; charset=UTF-8\r\n')

tmp_file = '/tmp/ajdfsafaa.jpg'

form = cgi.FieldStorage()
if form.has_key('item_image'):
    item = form['item_image']
    if item.file:
        tmp_f = open(tmp_file, 'wb')
        while True:
            chunk = item.file.read(1000000)
            if not chunk:
                break
            tmp_f.write(chunk)
        tmp_f.close()

tmp_image = Image.open(tmp_file).resize((256,256))

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
    d = numpy.argmax(y.data)
    return d

def loss(h, t):
    return F.softmax_cross_entropy(h, t)

optimizer = chainer.optimizers.Adam()
optimizer.setup(model)

x_tmp = []
x_tmp.append(numpy.asarray(tmp_image))
x_data = numpy.array(x_tmp, dtype=numpy.float32)
x = chainer.Variable(x_data)
digit = predict(x)

print 'あなたのアップロードした画像は' + ['Dress', 'HandBag'][digit] + 'の可能性が高いです。'

