from __future__ import print_function
from models.model import Lipreading
from models.Dense3D import Dense3D
import torch
import toml
from training import Trainer
from validation import Validator
import torch.nn as nn
import os
import sys
from collections import OrderedDict   
import csv
import numpy as np
import json
import scipy.io as sio
from collections import defaultdict
import matplotlib.pyplot as plt
#from torchsummary import summary


print("Loading options...")
with open(sys.argv[1], 'r') as optionsFile:
    options = toml.loads(optionsFile.read())

if(options["general"]["usecudnnbenchmark"] and options["general"]["usecudnn"]):
    print("Running cudnn benchmark...")
    torch.backends.cudnn.benchmark = True

os.environ['CUDA_VISIBLE_DEVICES'] = options["general"]['gpuid']
    
torch.manual_seed(options["general"]['random_seed'])

#Create the model.
'''model = Lipreading(hidden_dim=256, backbone_type='resnet',
               num_classes=300,
               relu_type='prelu',
               tcn_options={"dropout": 0.2,
                            "dwpw": False,
                            "kernel_size": [3,5,7],
                            "num_layers": 4,
                            "width_mult": 1
                           },
               width_mult=1.0, extract_feats=False)'''

model = Dense3D(options)


'''device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0 
model = model.to(device)

summary(model, (3,29,112,112))'''

'''pretrainedmodels = ["{}_{:0>8}.pt".format('DenseNet3D/weights/dataset_new_decay', epoch) for epoch in range(5, 21)]

for epoch, pretrainedmodel in enumerate(pretrainedmodels):
    pretrained_dict = torch.load(pretrainedmodel)
    # load only exists weights
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys() and v.size() == model_dict[k].size()}
    #print('matched keys:',len(pretrained_dict))
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    
    criterion = model.loss()        
    if(options["general"]["usecudnn"]):        
        torch.cuda.manual_seed(options["general"]['random_seed'])
        torch.cuda.manual_seed_all(options["general"]['random_seed'])

    if(options["training"]["train"]):
        trainer = Trainer(options)
    if(options["validation"]["validate"]):   
        validator = Validator(options, 'validation')
    if(options['test']['test']):   
        tester = Validator(options, 'test')
    
    if(options["validation"]["validate"]):        
        result = validator(model)
        print(result[0], file=open('test_special.txt', 'a'))
            
    
    
Trainer.writer.close()'''


if(options["general"]["loadpretrainedmodel"]):
    # remove paralle module
    pretrained_dict = torch.load(options["general"]["pretrainedmodelpath"])
    # load only exists weights
    model_dict = model.state_dict()
    print(len(model_dict.keys()))
    #print('PRETRAIN')
    #print(pretrained_dict['model_state_dict'].keys())
    #print('MODEL')
    #print(model_dict.keys())
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys() and v.size() == model_dict[k].size()}
    #pretrained_dict = {k: v for k, v in pretrained_dict['model_state_dict'].items() if k in model_dict.keys() and v.size() == model_dict[k].size()}
    print('matched keys:',len(pretrained_dict))
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)


#Move the model to the GPU.
#criterion = model.loss()        
if(options["general"]["usecudnn"]):        
    torch.cuda.manual_seed(options["general"]['random_seed'])
    torch.cuda.manual_seed_all(options["general"]['random_seed'])

if(options["training"]["train"]):
    trainer = Trainer(options)
if(options["validation"]["validate"]):   
    validator = Validator(options, 'validation')
if(options['test']['test']):   
    tester = Validator(options, 'test')
    

for epoch in range(options["training"]["startepoch"], options["training"]["epochs"]):
    print("EPOCH {}".format(epoch))
    if(options["training"]["train"]):
        trainer(model, epoch)
    if(options["validation"]["validate"]):        
        result = validator(model)
        #print(result)
        print('-'*21)
        print('{:<10}|{:>10}'.format('Top #', 'Accuracy'))
        for i in range(5):
            print('{:<10}|{:>10}'.format(i, result[i]))
        print('-'*21)
            
    if(options['test']['test']):
        result = tester(model)
        print('-'*21)
        print('{:<10}|{:>10}'.format('Top #', 'Accuracy'))
        for i in range(5):
            print('{:<10}|{:>10}'.format(i, result[i]))
        print('-'*21)
    
Trainer.writer.close()
