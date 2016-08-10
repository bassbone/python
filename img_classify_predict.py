# -*- coding: utf-8 -*-

import pickle
import numpy
import Image
import chainer
import img_classify_common as imc

# load model
f = open('/tmp/model')
model = pickle.load(f)
f.close()

instance = imc.ImgClassify(model)

x_tmp = []
x_tmp.append(numpy.asarray(Image.open('/tmp/tmp.jpg')))
x_data = numpy.array(x_tmp, dtype=numpy.float32)
x = chainer.Variable(x_data)
digit = instance.predict(x)

print digit

