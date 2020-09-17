import imageio
#from scipy import ndimage
#from scipy.misc import imresize
import torchvision.transforms.functional as functional
import torchvision.transforms as transforms
import torch
from .statefultransforms import StatefulRandomCrop, StatefulRandomHorizontalFlip
from .cutout import Cutout
import os
import numpy as np
import glob


def bbc(vidframes, padding, augmentation=True):
    temporalvolume = torch.zeros((3,padding,112, 112))

    #croptransform = transforms.CenterCrop((88, 88))

    if(augmentation):
        #crop = StatefulRandomCrop((256, 256), (88, 88))
        flip = StatefulRandomHorizontalFlip(0.5)

        croptransform = transforms.Compose([
            #crop,
            flip
        ])

    for i in range(len(vidframes)):
        result = transforms.Compose([
            transforms.ToPILImage(),
            #transforms.Grayscale(num_output_channels=1),
            #transforms.Resize((112, 112)),
            #transforms.RandomCrop((88, 88)),
            croptransform,
            transforms.ToTensor(),
            #transforms.Normalize((0.0,), (255.0,)),
            transforms.Normalize([0,0,0],[1,1,1])
            #transforms.Normalize((0.0996,), (0.4952,))
            #RandomCrop(crop_size)
            #Cutout(n_holes=32, length=8)
        ])(vidframes[i])

        temporalvolume[:,i] = result
    '''
    for i in range(len(vidframes), padding):
        temporalvolume[0][i] = temporalvolume[0][len(vidframes)-1]
    '''
    return temporalvolume
