from torch.autograd import Variable
import torch
import torch.optim as optim
from datetime import datetime, timedelta
from data import MyDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import os
import numpy as np
import torch.nn.functional as F


class Validator():
    def __init__(self, options, mode):
    
        self.usecudnn = options["general"]["usecudnn"]

        self.batchsize = options["input"]["batchsize"] 
        self.validationdataset = MyDataset('val', options["validation"]["data_root"])
        #self.validationdataset = MyDataset('val', os.path.join(options["validation"]["data_root"], word))
                                                
        self.tot_data = len(self.validationdataset)
        self.validationdataloader = DataLoader(
                                    self.validationdataset,
                                    batch_size=options["input"]["batchsize"],
                                    shuffle=options["input"]["shuffle"],
                                    num_workers=options["input"]["numworkers"],
                                    drop_last=False
                                )
        self.mode = mode
        
    def __call__(self, model):
        with torch.no_grad():
            print("Starting {}...".format(self.mode))
            count = np.zeros((100))
            validator_function = model.validator_function()
            model.eval()
            if(self.usecudnn):
                net = nn.DataParallel(model).cuda()
                
            num_samples = 0
            running_corrects = 0
            for i_batch, (inputs, targets, length) in enumerate(self.validationdataloader):
                input = Variable(inputs).cuda()
                labels = Variable(targets).cuda()
                length = Variable(length).cuda()
                
                model = model.cuda()

                outputs = net(input)
                (vector, top1) = validator_function(outputs, length, labels)
                _, maxindices = vector.cpu().max(1)
                argmax = (-vector.cpu().numpy()).argsort()
                for i in range(input.size(0)):
                    p = list(argmax[i]).index(labels[i])
                    count[p:] += 1                    
                num_samples += input.size(0)
                
                """logits = model(input, lengths=length)
                _, preds = torch.max(F.softmax(logits, dim=1).data, dim=1)
                running_corrects += preds.eq(labels.cuda().view_as(preds)).sum().item() """
                

                #print('i_batch/tot_batch:{}/{},corret/tot:{}/{},current_acc:{}'.format(i_batch,len(self.validationdataloader),count[0],len(self.validationdataset),1.0*count[0]/num_samples)) 
            #print('{} in total\tCR: {}'.format(len(self.validationdataset), running_corrects/len(self.validationdataset)))

        #return running_corrects/len(self.validationdataset)
        return count/num_samples
