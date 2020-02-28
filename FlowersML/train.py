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


def train_model():
    
    parser = argparse.ArgumentParser(description='Train netwotk.')
    parser.add_argument('data_dir',required=True,help='train and test datas directory')
    parser.add_argument('checkpoint_name',required=False,help='name to save the model, if none provided the model is not saved')
    parser.add_argument('architecture',required=False, default='densenet121', help='architecture to be used')
    parser.add_argument('save_dir',required=False, help='name to save the model, if none provided the model is not saved')
    parser.add_argument('hidden_units',required=False, default=512, type=int, help='hidden units for the model, default is 512')
    parser.add_argument('learningRate',required=False, default=0.001, type=float, help='Learning rate to train the model.0.001 is default')
    parser.add_argument('epochs',required=False, default=2, type=int, help='epochs when the model is training')

    args = parser.parse_args()

    data_dir = args.data_dir
    checkpoint_name = args.checkpoint_name
    architecture = args.architecture
    save_dir = args.save_dir
    hidden_units = args.hidden_units
    learningRate = args.learningRate
    epochs = args.epochs

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
        model_utils.save_model(model, traindata,optimizer,checkpoint_name)


