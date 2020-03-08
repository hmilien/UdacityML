import workspace_utils
from workspace_utils import active_session

import torch
from torch.autograd import Variable
from torch import nn
from torch import optim
from torchvision import datasets, transforms,models


def save_model(model,train_data,optimizer, checkpoint_name,epochs, architecture):
    model.class_to_idx = train_data.class_to_idx

    checkpoint = { 'input_size' : 1024,
                'output_size': 102,
                'epochs' : epochs,
                'architecture' : architecture,
                'optim_state_dict' : optimizer.state_dict(),
                'class_to_idx':train_data.class_to_idx,
                'state_dict' : model.state_dict()} 

    torch.save(checkpoint,checkpoint_name)
    
def load_checkpoint(filepath, gpu):
    if(gpu):
        checkpoint = torch.load(filepath)
    else:
         checkpoint = torch.load(filepath,map_location='cpu')

    architecture = checkpoint['architecture']
    if(architecture == 'densenet121'):
        model = models.densenet121(pretrained=True)
    else:
        model = models.densenet169(pretrained=True)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.class_to_idx = checkpoint['class_to_idx']

    hidden = checkpoint ['input_size']
    output = checkpoint['output_size']
    model.classifier = nn.Sequential(nn.Linear(1024, hidden),
                                nn.ReLU(),
                                nn.Dropout(0.2),
                                nn.Linear(hidden,output),
                                nn.LogSoftmax(dim=1))
    return model

def create_model(hidden_units, architecture):
    if(architecture == 'densenet121'):
         model = models.densenet121(pretrained=True)
    else:
        model = models.densenet169(pretrained=True)
   
       # Turn off gradients for the model
    for param in model.parameters():
        param.requires_grad = False
        
    model.classifier = nn.Sequential(nn.Linear(1024, hidden_units),
                                    nn.ReLU(),
                                    nn.Dropout(0.2),
                                    nn.Linear(hidden_units,102),
                                    nn.LogSoftmax(dim=1))
    return model
    
def train_model(model,trainloader, testloader, criterion, optimizer,device, epochs=2):
    # Only train the classifier parameters, feature parameters are frozen
    model.to(device)
    steps = 0
    running_loss = 0
    print_every = 5

    with active_session():
        for epoch in range(epochs):
            for images,labels in trainloader:
                steps += 1
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                logps = model.forward(images)
                loss = criterion(logps, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
                # evaluate and print acuracy
                if steps % print_every == 0:
                    test_loss = 0
                    accuracy = 0
                    model.eval()
                    with torch.no_grad():
                        for inputs, labels in testloader:
                            inputs, labels = inputs.to(device), labels.to(device)
                            logps = model.forward(inputs)
                            batch_loss = criterion(logps, labels)

                            test_loss += batch_loss.item()

                            # Calculate accuracy
                            ps = torch.exp(logps)
                            top_p, top_class = ps.topk(1, dim=1)
                            equals = top_class == labels.view(*top_class.shape)
                            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                    print(f"Epoch {epoch+1}/{epochs}.. "
                        f"Train loss: {running_loss/print_every:.3f}.. "
                        f"Test loss: {test_loss/len(testloader):.3f}.. "
                        f"Test accuracy: {accuracy/len(testloader):.3f}")
                    running_loss = 0
                    model.train()