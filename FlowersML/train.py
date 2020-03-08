import utils
import model_utils

import torch
from torch.autograd import Variable
from torch import nn
from torch import optim
import numpy as np
import matplotlib.pyplot as plt
import json
from torchvision import datasets, transforms,models
from PIL import Image
import argparse
import os

def train():
    parser = argparse.ArgumentParser(description='Train netwotk.')
    parser.add_argument('data_dir',help='train and test datas directory.')
    parser.add_argument('checkpoint_name',help='name to save the model, if none provided the model is not saved')
    parser.add_argument('--architecture', default='densenet121', help='architecture to be used')
    parser.add_argument('--save_dir', help='name to save the model, if none provided the model is not saved',default='')
    parser.add_argument('--hidden_units', type=int, help='hidden units for the model, default is 512', default=512)
    parser.add_argument('--learningRate', type=float, help='Learning rate to train the model.0.001 is default',default=0.001)
    parser.add_argument('--epochs', type=int, help='epochs when the model is training',default=2)

    args = parser.parse_args()

    data_dir = args.data_dir
    checkpoint_name = args.checkpoint_name
    architecture = args.architecture
    save_dir = args.save_dir
    hidden_units = args.hidden_units
    learningRate = args.learningRate
    epochs = args.epochs

    print('start training. data_dir is: ' + data_dir)

    train_dir = data_dir + '/train'
    test_dir = data_dir + '/test'
    
    traindata, trainloader = utils.load_data(True,train_dir)
    testdata, testloader = utils.load_data(False,test_dir)

    #Use GPU if it's available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #define the model
    model = model_utils.create_model(hidden_units,architecture)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learningRate)
    
    #model training execution
    model_utils.train_model(model,trainloader, testloader,criterion,optimizer,device,epochs)

    #save the model
    
    if(save_dir!= ''):
        checkpoint_name = os.path.join(save_dir, checkpoint_name)
    model_utils.save_model(model, traindata,optimizer,checkpoint_name,epochs,architecture)

train()


