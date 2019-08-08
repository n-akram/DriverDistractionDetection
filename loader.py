'''
data loader for driver sistraction detection.
Date: 22.07.2019
'''

from keras.utils import to_categorical
import csv
import numpy as np
import os
import sys
import cv2

batch_size = 32
validationShare = 0.33

base_path = "data/"

def convertClassIntoInteger(c):
    return(int(c[1]))
    
def concatenateTrainingPath(c, im):
    global base_path
    return(base_path+ 'imgs/train/' + c + '/' + im)

def categorical(y):
	if "-m" in sys.argv:
		return(to_categorical(y, num_classes=5))
	else:
		return(to_categorical(y, num_classes=10))

def getTotalList():
    line_count = 0
    total_list = []
    mobile_only = ["c0","c1", "c2", "c3", "c4"]
    with open('data/driver_imgs_list.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if line_count != 0:
                row.append(convertClassIntoInteger(row[1]))
                #row.append(categorical(row[3]))
                row[2] = concatenateTrainingPath(row[1], row[2])
            if ("-m" in sys.argv) and (line_count!=0):
                if row[1] in mobile_only:
                	row.append(categorical(row[3]))
                	total_list.append(row)
            elif line_count!=0:
            	row.append(categorical(row[3]))
            	total_list.append(row)
            line_count += 1
    return(total_list)

def getSegregatedList():
	global validationShare
	total_list = getTotalList() 
	valiNum = int(len(total_list) * validationShare)
	np.random.shuffle(total_list)
	valiSet, trainSet = [], []
	i = 0
	for row in total_list:
		if i > valiNum:
			trainSet.append(row)
		else:
			valiSet.append(row)
		i += 1
	return(valiSet, trainSet)

def getTestSetList():
	global base_path
	lst = os.listdir(base_path + 'imgs/test')
	fullLst = []
	for ele in lst:
		fullLst.append(base_path + 'imgs/test/' + ele)
	return(fullLst)

def getArrays():
	valiSet, trainSet = getSegregatedList()
	x_train, y_train = [], []
	x_val, y_val = [], []
	dim = (150, 150) 
	for ele  in trainSet:
		i = cv2.imread(ele[2])
		i = cv2.resize(i, dim)
		norm_image = cv2.normalize(i, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
		x_train.append(norm_image)
		y_train.append(ele[4])
	for ele  in valiSet:
		i = cv2.imread(ele[2])
		i = cv2.resize(i, dim)
		norm_image = cv2.normalize(i, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
		x_val.append(norm_image)
		y_val.append(ele[4])
	return(x_train, y_train, x_val, y_val)

testSet = getTestSetList()

def changeDefaults(bs=30,vS=0.2, bP = "data/"):
	global base_path
	global validationShare
	global base_path

	base_path = bs
	validationShare = vS
	base_path = bP

#x_train, y_train=getArrays()

#print(x_train[0].shape)
#print(len(valiSet), "valiSet")
#print(len(trainSet), "trainSet")

#print(valiSet[0])

#img = cv2.imread(testSet[0])
#norm_image = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

#print(testSet[0])
#cv2.imshow('image',norm_image)
#cv2.waitKey(0)
