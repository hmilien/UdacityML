import utils
import model_utils
import torch
from torch.autograd import Variable
from torch import nn
from torch import optim
from torchvision import datasets, transforms,models
import argparse

def predict():
    parser = argparse.ArgumentParser(description='Display model predication for an image.')
    parser.add_argument('image_path',help='Image path to make the prediction upon')
    parser.add_argument('checkpoint_name',help='name to save the model, no prediction will be made')
    parser.add_argument('--topk', type=int, help='topk probability when the model is predicting',default=2, )
    parser.add_argument('--cat_to_name',help='category to name encoding',default='')
    parser.add_argument('--gpu', type=bool, help='use gpu',default=False)

    args = parser.parse_args()
    image_path = args.image_path
    checkpoint_name = args.checkpoint_name
    topk = args.topk
    cat_to_name = args.cat_to_name
    gpu = args.gpu

    if(cat_to_name != ''):
        cat_to_name = utils.load_category(cat_to_name)        
    print_prediction(image_path,checkpoint_name,cat_to_name, topk,gpu)


   
def eval( image_path, checkpoint_name, topk,gpu):

    model = model_utils.load_checkpoint(checkpoint_name,gpu)
    model.eval()
    if(gpu == False):
        model.cpu()

    image = utils.process_image(image_path)
    
    image = torch.from_numpy(image).unsqueeze(0)
    image = image.float()
    output = model.forward(image)
  
    top_prob, top_labels = torch.topk(output, topk)
    top_prob = top_prob.exp()
    top_prob_array = top_prob.data.numpy()[0]
    
    inv_class_to_idx = {v: k for k, v in model.class_to_idx.items()}
    
    top_labels_data = top_labels.data.numpy()
    top_labels_list = top_labels_data[0].tolist()  
    
    top_classes = [inv_class_to_idx[x] for x in top_labels_list]
    
    return top_prob_array, top_classes


def print_prediction(img, model, mapper,topk,gpu):
    probs, classes = eval(img, model,topk,gpu)
    print(probs)
    if(mapper != ''):
        img_filename = img.split('/')[-2]
        for index in range(len(classes)):
            classes[index] = mapper[img_filename]
    print(classes)


predict()



