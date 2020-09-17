from torch.autograd import Variable
import torch
import torch.optim as optim
from datetime import datetime, timedelta
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import torchvision.transforms.functional as functional
import torchvision.transforms as transforms
from data import MyDataset
import torch.nn as nn
import os
import pdb
import math
from optimizer import Lookahead


def timedelta_string(timedelta):
    totalSeconds = int(timedelta.total_seconds())
    hours, remainder = divmod(totalSeconds,60*60)
    minutes, seconds = divmod(remainder,60)
    return "{:0>2} hrs, {:0>2} mins, {:0>2} secs".format(hours, minutes, seconds)

def output_iteration(loss, i, time, totalitems):

    avgBatchTime = time / (i+1)
    estTime = avgBatchTime * (totalitems - i)
    
    print("Iteration: {:0>8},Elapsed Time: {},Estimated Time Remaining: {},Loss:{}".format(i, timedelta_string(time), timedelta_string(estTime),loss))

class Trainer():

    tot_iter = 0
    writer = SummaryWriter()    
    
    def __init__(self, options):
                                
        self.usecudnn = options["general"]["usecudnn"]

        self.batchsize = options["input"]["batchsize"]

        self.statsfrequency = options["training"]["statsfrequency"]       

        self.learningrate = options["training"]["learningrate"]

        self.modelType = options["training"]["learningrate"]

        self.weightdecay = options["training"]["weightdecay"]
        self.momentum = options["training"]["momentum"]
        
        self.save_prefix = options["training"]["save_prefix"]
        
        self.dsets = MyDataset('train', options["training"]["data_root"])
        
        self.dset_loaders = DataLoader(self.dsets, 
                                      batch_size=options["input"]["batchsize"], 
                                      shuffle=options["input"]["shuffle"], 
                                      num_workers=options["input"]["numworkers"])
                                    
        self.dset_sizes = len(self.dsets)
        

    def learningRate(self, epoch):
        decay = math.floor((epoch) / 5)
        return self.learningrate * pow(0.5, decay)

    def __call__(self, model, epoch):
        #set up the loss function.
        '''num_of_pixels = len(self.dsets) * 29 * 112 * 112
        total_sum = 0
        for batch in self.dset_loaders: 
            total_sum += batch[0].sum()
        mean = total_sum / num_of_pixels
        sum_of_squared_error = 0
        for batch in self.dset_loaders: 
            sum_of_squared_error += ((batch[0] - mean).pow(2)).sum()
        std = torch.sqrt(sum_of_squared_error / num_of_pixels)
        print(mean, std)'''
        model.train()
        criterion = model.loss()
        if(self.usecudnn):
            net = nn.DataParallel(model).cuda()
        criterion = criterion.cuda()
        
        decay = math.floor((epoch ) / 5)
               
        optimizer = optim.Adam(
                        model.parameters(),
                        lr = self.learningrate, amsgrad=True)
        #optimizer = Lookahead(optimizer=optimizer,k=5,alpha=0.5)
        
        #transfer the model to the GPU.       
            
        startTime = datetime.now()
        print("Starting training...")
        for i_batch, (inputs, targets, length) in enumerate(self.dset_loaders):
            
            optimizer.zero_grad()
            inputs = Variable(inputs)
            targets = Variable(targets)
            length = Variable(length)
            #print(length.size(0))
            
            if(self.usecudnn):
                input = inputs.cuda()
                labels = targets.cuda()
                

            outputs = net(input)
            #outputs = net(input, lengths=length)
            loss = criterion(outputs, length, labels.squeeze(1))
            #loss = criterion(outputs, labels.squeeze(1))
            loss.backward()
            optimizer.step()
            sampleNumber = i_batch * self.batchsize

            if(sampleNumber % self.statsfrequency == 0):
                currentTime = datetime.now()
                output_iteration(loss.cpu().detach().numpy(), sampleNumber, currentTime - startTime, self.dset_sizes)
                Trainer.writer.add_scalar('Train Loss', loss, Trainer.tot_iter)
            Trainer.tot_iter += 1

        print("Epoch completed, saving state...")
        torch.save(model.state_dict(), "{}_{:0>8}.pt".format(self.save_prefix, epoch))       
