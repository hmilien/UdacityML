import torch
from torch.autograd import Variable
from torch import nn
from torch import optim
from torchvision import datasets, transforms,models
from PIL import Image
import numpy as np
import json

def load_data(isTrainMode, dirPath):
    if(isTrainMode):
        transforms = transforms.Compose([transforms.Resize([224,224]),
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    else:                                                                
        transforms = transforms.Compose([transforms.Resize([224,224]),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

    # TODO: Load the datasets with ImageFolder
    imageData = datasets.ImageFolder(dirPath, transform=transforms)
    if(isTrainMode):
        return imageData, torch.utils.data.DataLoader(imageData, batch_size=64, shuffle=True)
    else:
        return imageData, torch.utils.data.DataLoader(imageData, batch_size=32)

def load_category(cat_to_name):
    if(cat_to_name != ''):
        with open('cat_to_name.json', 'r') as f:
            cat_to_name = json.load(f)
    
def process_image(img):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
     # TODO: Process a PIL image for use in a PyTorch model
    pil_image = Image.open(img)
    pil_image.load()
   
    #Resizing
    if pil_image.size[0] > pil_image.size[1]:
        pil_image.thumbnail((10000, 256))
    else:
        pil_image.thumbnail((256, 10000))
  
    #Cropping
    size = pil_image.size
    pil_image = pil_image.crop((size[0]//2 -(224/2),
                                size[1]//2 - (224/2),
                                size[0]//2 +(224/2),
                                size[1]//2 + (224/2)))
  
    np_image = np.array(pil_image)/255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image =(np_image - mean)/std
    np_image = np_image.transpose((2, 0, 1))
    return np_image

                                    

