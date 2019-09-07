from glob import glob
from PIL import Image

import pandas as pd
import numpy as np

import gc

import torch
import torch.nn as nn
from torch.autograd import Variable

class Net(nn.Module):
    """Net
    
    Pytorch model for image classificaiton
    
    """
    def __init__(self, lr = 0.1, momentum=0.9):
        """init
    
        Defines the neural network architecture/parameter

        Args:
            lr (float): learning rate
            momentum (float): Momentum for SDG optimizer
            path (string): Path of train data
        """
        super(Net, self).__init__()
        self.nn1 = nn.Conv2d(3, 512, 3)
        self.nn1_bn = nn.BatchNorm2d(512)
        self.rl1 = nn.ReLU()
        self.nn2 = nn.Conv2d(512, 256, 3)
        self.nn2_bn = nn.BatchNorm2d(256)
        self.rl2 = nn.ReLU()
        self.nn3 = nn.Conv2d(256, 128, 3)
        self.nn3_bn = nn.BatchNorm2d(128)
        self.rl3 = nn.ReLU()
        self.linear = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr = lr, momentum=momentum)
                
    def forward(self, x):
        """forward
    
        Defines the neural network architecture/parameter and its forward computation

        Args:
            x (tensor): input data
        Returns:
            x (tensor): Output prediction
        """
        x = self.nn1(x)
        x = self.nn1_bn(x)
        x = self.rl1(x)
        x = self.nn2(x)
        x = self.nn2_bn(x)
        x = self.rl2(x)
        x = self.nn3(x)
        x = self.nn3_bn(x)
        x = self.rl3(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        x = self.linear(x)
        x = self.sigmoid(x)
        return(x)

def train_nn(net, trainloader, valloader, epochs = 10, batch_size = 64):
    """forward
    
    Defines the neural network architecture/parameter and its forward computation

    Args:
        net (nn.Module): Network
        x_train (nparray): Numpy array of training images
        y_train (nparray): Numpy array of training labels
        x_val (nparray): Numpy array of validation images
        y_val (nparray): Numpy array of validation labels
        epochs (int): Number of epochs
        batch_size (int): Number of datapoints per batch
    """
    i = 0
    for epoch in range(epochs):
        net.train()
        for i, data in enumerate(trainloader, 0):
            batch_x = data[0].float().cuda()
            batch_y = data[1].float().cuda()
            net.optimizer.zero_grad()
            outputs = net(batch_x)
            loss = net.criterion(outputs, batch_y)
            loss.backward()
            net.optimizer.step()
            i += 1
            if i % 10 == 0:
               print('Epoch: ' + str(epoch) + '/' + str(epochs) + ' Batch: ' + str(i))
        val_pred = []
        val_target = []
        net.eval()
        val_accuracy = 0
        with torch.no_grad():
            for i, data in enumerate(valloader, 0):
                batch_x = data[0].float().cuda()
                net.optimizer.zero_grad()
                outputs = net(batch_x)
                val_pred.append(outputs.cpu().detach().numpy())
                val_target.append(data[1].cpu().detach().numpy())
        val_pred = np.concatenate(val_pred).ravel()
        val_target = np.concatenate(val_target).ravel()
        val_accuracy = np.mean((val_pred>=0.5).astype(np.int)==val_target)
        print('Epoch: ' + str(epoch) + '/' + str(epochs) + ' Val Accuracy:' + str(round(val_accuracy,3)))
        
def predict_test_set(net, path, batch_size = 64):
    """predicts the test darta
    
    Reads the test data in batches and predict it
    
    Args:
        net (nn.Module): Network
        path (string): Folder of test data
        batch_size: Number of images at once
    Returns:
        output (list): prediction and file names
    """
    files = glob(path)
    output = []
    output_files = []
    batch_no = int(len(files) // batch_size) + 1
    x_test = np.zeros((batch_size, 96, 96, 3))
    i = 0
    with torch.no_grad():
        for file in files:
            im = Image.open(file)
            np_im = np.asarray(im)
            x_test[i,:,:,:] = np_im
            i+=1
            output_files.append(file)
            if i == batch_size:
                x_test = np.rollaxis(x_test, 3, 1)
                batch_x = Variable(torch.from_numpy(x_test), volatile=True).float().cuda()
                pred = net(batch_x)
                pred = pred.cpu().detach().numpy()
                output.append([output_files, pred])
                output_files = []
                x_test = np.zeros((batch_size, 96, 96, 3))
                i = 0
    return(output)