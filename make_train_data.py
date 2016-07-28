# original http://d.hatena.ne.jp/shi3z/20150709/1436397615

import sys
import commands
import subprocess

def cmd(cmd):
	return commands.getoutput(cmd)

pwd = "/tmp"

#labels
dirs = cmd("ls "+ pwd + "/" + sys.argv[1])
labels = dirs.splitlines()

#copy images and make train.txt
imageDir = pwd + "/images"
cmd("mkdir " + pwd + "/images")
train = open(pwd + '/train.txt','w')
test = open(pwd + '/test.txt','w')
labelsTxt = open(pwd + '/labels.txt','w')

classNo=0
cnt = 0
#label = labels[classNo]
for label in labels:
	workdir = pwd + "/" + sys.argv[1] + "/" + label
	imageFiles = cmd("ls " + workdir + "/*.jpg")
	images = imageFiles.splitlines()
	print(label)
	labelsTxt.write(label + "\n")
	startCnt = cnt
	length = len(images)
	for image in images:
		imagepath = imageDir + "/image%07d" % cnt + ".jpg"
		cmd("cp " + image + " " + imagepath)
		if cnt - startCnt < length * 0.75:
			train.write(imagepath + " %d\n" % classNo)
		else:
			test.write(imagepath + " %d\n" % classNo)
		cnt += 1
	
	classNo += 1

train.close()
test.close()
labelsTxt.close()

