import utils
import model_utils
import torch
from torch.autograd import Variable
from torch import nn
from torch import optim
from torchvision import datasets, transforms,models

def predict():
    parser = argparse.ArgumentParser(description='Display model predication for an image.')
    parser.add_argument('image_path',required=True,help='Image path to make the prediction upon')
    parser.add_argument('checkpoint_name',required=False,help='name to save the model, if none provided the model is not saved')
    parser.add_argument('topk',required=False, default=2, type=int, help='topk probability when the model is predicting')
    parser.add_argument('cat_to_name',required=False, default='', help='category to name encoding')

    args = parser.parse_args()
    image_path = args.image_path
    checkpoint_name = args.checkpoint_name
    topk = args.topk
    cat_to_name = args.cat_to_name

    if(cat_to_name != ''):
        cat_to_name = utils.load_category(load_category)        
    print_prediction(image_path,checkpoint_name,cat_to_name, topk)


   
def eval( image_path, checkpoint_name, topk=3,gpu =False):

    model = model_utils.load_checkpoint(checkpoint_name)
    model.eval()
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


def print_prediction(img, model, mapper):
    probs, classes = eval(img, model)
    print(probs)
    if(mapper != ''):
        img_filename = img.split('/')[-2]
        for index in range(len(classes):
            classes[index] = mapper[img_filename]
    print(classes)




