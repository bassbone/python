# -*- coding: utf-8 -*-

import os, glob, Image

from_dir = "/tmp/download/"
to_dir = "/tmp/resize/"

def getdirs(path):
    dirs = []
    for item in os.listdir(path):
        if os.path.isdir(os.path.join(path, item)):
            dirs.append(item)
    return dirs


for dir in getdirs(from_dir):
    if not os.path.exists(to_dir + dir):
        os.mkdir(to_dir + dir)
    for file in glob.glob(from_dir + dir + "/*.jpg"):
        try:
            print file.rsplit("/")[4]
            Image.open(file).resize((256,256)).save(to_dir + dir + "/" + file.rsplit('/')[4])
        except:
            continue
