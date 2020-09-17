from torch.utils.data import Dataset
from .preprocess import *
import os
import glob
import numpy as np
import random
import cv2
import torchvision.transforms as transforms
import torch
from PIL import Image

def load_file(filename):
    video = []
    cap = cv2.VideoCapture(filename)

    while(cap.isOpened()):
        ret, frame = cap.read() # BGR
        if ret:
            video.append(frame)#cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        else:
            break
    cap.release()
    video = np.array(video)
    cap =  video[...,::-1]
    #arrays = np.stack([cap[_]
                      #for _ in range(len())], axis=0)
    #cap = cap / 255.
    return cap

class MyDataset():
    def __init__(self, folds, path):
        self.folds = folds
        self.path = path
        with open('balanced_300_new.txt') as myfile:
            self.data_dir = myfile.read().splitlines()
        self.filenames = glob.glob(os.path.join(self.path, '*', self.folds, '*.mp4'))
        #self.filenames = glob.glob(os.path.join(self.path, self.folds, '*.mp4'))
        self.filenames = [x for x in self.filenames if x.split('/')[-3] in self.data_dir]
        self.list = {}
        for i, x in enumerate(self.filenames):
            target = x.split('/')[-3]
            for j, elem in enumerate(self.data_dir):
                if elem == target:
                    self.list[i] = [x]
                    self.list[i].append(j)
        print('Load {} part'.format(self.folds))
        print(len(self.list))

    def __getitem__(self, idx):
        inputs = load_file(self.list[idx][0])
        length = len(inputs)
        inputs = bbc(inputs, 29, True)
        labels = torch.LongTensor([self.list[idx][1]])
        return inputs, labels, length

    def __len__(self):
        return len(self.filenames)
