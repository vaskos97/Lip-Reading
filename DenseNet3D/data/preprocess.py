import imageio
#from scipy import ndimage
#from scipy.misc import imresize
import torchvision.transforms.functional as functional
import torchvision.transforms as transforms
import torch
from .statefultransforms import StatefulRandomCrop, StatefulRandomHorizontalFlip
import os
import numpy as np
import glob


def bbc(vidframes, padding, augmentation=True):
    temporalvolume = torch.zeros((3,padding,112,112))

    #croptransform = transforms.CenterCrop((112, 112))

    if(augmentation):
        #crop = StatefulRandomCrop((122, 122), (112, 112))
        flip = StatefulRandomHorizontalFlip(0.5)

        croptransform = transforms.Compose([
            #crop,
            flip
        ])

    for i in range(len(vidframes)):
        result = transforms.Compose([
            transforms.ToPILImage(),
            #transforms.Resize((112, 112)),
            croptransform,
            transforms.ToTensor(),
            transforms.Normalize([0,0,0],[1,1,1]),
        ])(vidframes[i])

        temporalvolume[:,i] = result
    '''
    for i in range(len(vidframes), padding):
        temporalvolume[0][i] = temporalvolume[0][len(vidframes)-1]
    '''
    return temporalvolume
    
