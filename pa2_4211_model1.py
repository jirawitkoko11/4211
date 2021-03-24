import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import torch.optim as optim
from torch.utils.data import random_split


PATH = './pa2'
class CustomDataset(Dataset):
  def __init__(self,df_path,txt_path,img_path):
    df = pd.read_csv(df_path)
    txt = pd.read_csv(txt_path, header = None)
    txt.columns = ['id','path']
    self.df = df
    self.txt = txt
    self.imgPath = img_path

  def __getitem__(self,idx):
    ind1 = self.df.iloc[idx].id1 
    ind2 = self.df.iloc[idx].id2
    label = self.df.iloc[idx].target
    img1_path = os.path.join(PATH,self.txt.iloc[ind1-1].path)
    img2_path = os.path.join(PATH,self.txt.iloc[ind2-1].path)
    
    return self.transformPic(img1_path),self.transformPic(img2_path),label
    
  def __len__(self):
    return len(self.df)

  def transformPic(self,pic_path):
    Resize = transforms.Compose([transforms.Resize([32,32]),
                                 transforms.ToTensor(),
                                 transforms.Normalize(0.5,0.5)])
    img = Image.open(pic_path).convert('L')
    return Resize(img)



train = CustomDataset(os.path.join(PATH,'train.csv'),os.path.join(PATH,'index.txt'),'asd')
valid = CustomDataset(os.path.join(PATH,'train.csv'),os.path.join(PATH,'index.txt'),'asd')
test = CustomDataset(os.path.join(PATH,'train.csv'),os.path.join(PATH,'index.txt'),'asd')
loader = torch.utils.data.DataLoader(train, batch_size=128, shuffle=True, num_workers=2)
valid_loader = torch.utils.data.DataLoader(valid, batch_size=128, shuffle=False, num_workers=2)



class Siamese(nn.Module):
  def __init__(self):
    super(Siamese, self).__init__()
    #####Convolutional layers###############
    self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size= 3,stride=1, padding=1)
    self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size= 3,stride=1, padding=1)
    self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size= 3,stride=1, padding=1)
    self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size= 3,stride=1, padding=1)
    self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size= 3,stride=1, padding=1)
    self.conv6 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size= 3,stride=1, padding=1)

    self.bn1 = nn.BatchNorm1d(32*32*32)
    self.bn2 = nn.BatchNorm1d(32*32*32)
    self.bn3 = nn.BatchNorm1d(16*16*64)
    self.bn4 = nn.BatchNorm1d(16*16*128)
    self.bn5 = nn.BatchNorm1d(16*16*256)
    self.bn6 = nn.BatchNorm1d(16*16*512)

    self.maxPool = nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
    self.avgPool = nn.AvgPool2d(kernel_size=16,padding=0)
    ##########end of Convolutional layers###########
    
    ##########linear layers###################
    self.fc1 = nn.Linear(512,512)
    self.drop = nn.Dropout(0.5)
    self.fc2 = nn.Linear(512,1)

    self.sigmoid = nn.Sigmoid()

    ############end of linear layers################
  def forward(self,pic1,pic2):
    x = self.aggregation(pic1,pic2)
    x = F.relu(x)
    x = self.drop(x)
    x = self.sigmoid(self.fc2(x))
    return x.view(-1)

  def getfeature(self,pic):
    batch_in = pic.shape[0]
    x = F.relu(self.bn1(self.conv1(pic).view(batch_in,32*32*32)).view(batch_in,32,32,32))
    x = F.relu(self.bn2(self.conv2(x).view(batch_in,32*32*32)).view(batch_in,32,32,32))
    x = self.maxPool(x)
    x = F.relu(self.bn3(self.conv3(x).view(batch_in,64*16*16)).view(batch_in,64,16,16))
    x = F.relu(self.bn4(self.conv4(x).view(batch_in,128*16*16)).view(batch_in,128,16,16))
    x = F.relu(self.bn5(self.conv5(x).view(batch_in,256*16*16)).view(batch_in,256,16,16))
    x = F.relu(self.bn6(self.conv6(x).view(batch_in,512*16*16)).view(batch_in,512,16,16))
    x = self.avgPool(x)
    x = x.view(-1,1*512)
    return x

    #############Aggregation##################
  def aggregation(self,pic1,pic2):
    pic1_vector = self.getfeature(pic1)
    pic2_vector = self.getfeature(pic2)
    diff = torch.abs(torch.sub(pic1_vector,pic2_vector))
    return diff
    #############end of Aggregation##################




def save_checkpoint(save_path, model, optimizer, val_loss):
    if save_path==None:
        return
    save_path = save_path 
    state_dict = {'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'val_loss': val_loss}

    torch.save(state_dict, save_path)

    print(f'Model saved to ==> {save_path}')

def plotLoss(train,val,step, save_path):
  plt.plot(step,train, label = "train loss")
  plt.plot(step,val, label = "validation loss")
  plt.xlabel('step')
  plt.ylabel('loss')
  plt.title('train vs validation loss')
  plt.legend()
  plt.savefig(save_path,dpi=300)


def TRAIN(net, train_loader, valid_loader,  num_epochs, eval_every, total_step, criterion, optimizer, val_loss, device, save_path, plot_path):
    trainset_loss = []
    valset_loss = []
    step = []
    running_loss = 0.0
    running_corrects = 0
    running_num = 0
    global_step = 0
    if val_loss==None:
        best_val_loss = float("Inf")  
    else: 
        best_val_loss=val_loss
    

    for epoch in range(num_epochs):  # loop over the dataset multiple times

        for i, (pic1 , pic2, labels) in enumerate(train_loader):
            net.train()
            pic1 = pic1.to(device)
            pic2 = pic2.to(device)
            labels = labels.to(device).float()

            '''Training of the model'''
            # Forward pass
            outputs = net(pic1,pic2)
            preds = outputs.data
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1

            running_loss += loss.item()
            running_num += len(labels)

            '''Evaluating the model every x steps'''
            if global_step % eval_every == 0:
                with torch.no_grad():
                    net.eval()
                    val_running_loss = 0.0
                    for val_pic1,val_pic2, val_labels in valid_loader:
                        val_pic1, val_pic2 , val_labels = val_pic1.to(device), val_pic2.to(device), val_labels.to(device).float()
                        val_outputs = net(val_pic1,val_pic2)
                        val_loss = criterion(val_outputs, val_labels)         
                        val_running_loss += val_loss.item()

                    average_train_loss = running_loss / eval_every
                    average_val_loss = val_running_loss / len(valid_loader)
                    trainset_loss.append(average_train_loss)
                    valset_loss.append(average_val_loss)
                    step.append(global_step)
                    print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}'
                          .format(epoch+1, num_epochs, global_step, total_step, average_train_loss, average_val_loss))

                    running_loss = 0.0
                    running_num = 0
                    running_corrects = 0
                    
                    if average_val_loss < best_val_loss:
                        best_val_loss = average_val_loss
                        save_checkpoint(save_path, net, optimizer, best_val_loss)

    plotLoss(trainset_loss,valset_loss,step,plot_path)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
model = Siamese()
model = model.to(device)
num_epochs = 20
eval_every = 10
total_step = len(loader)*num_epochs
best_val_loss = None
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


TRAIN(model, loader, valid_loader, num_epochs, eval_every,
      total_step, criterion, optimizer, best_val_loss, device, 'model1_net.pt','Loss_M1.png')



def pltThreshold(thres,acc,bestT,bestA,save_path):
  plt.plot(thres,acc)
  plt.xlabel('threshold')
  plt.ylabel('accuracy')
  plt.title('threshold vs accuracy\n best theta = %.3f which gives %.4f accuracy',(bestT,bestA))
  plt.savefig(save_path,dpi=300)

def getThreshold(valid_loader,model,save_path):
  theta = np.linspace(0.4,0.9,50, False)
  acc  = []
  bestT = 0
  bestA = 0
  for thres in theta:
    total_correct = 0
    total_cmp = 0
    with torch.no_grad():
      model.eval()
      for val_pic1,val_pic2, val_labels in valid_loader:
        val_pic1, val_pic2 , val_labels = val_pic1.to(device), val_pic2.to(device), val_labels.to(device)
        val_outputs = model(val_pic1,val_pic2)
        i = 0
        for out in val_outputs:
          total_cmp += 1
          check = 0
          if out >= thres:
            check = 1
          if check == val_labels[i]:
            total_correct += 1
    running_acc = total_correct/total_cmp
    #print(str(thres)+"  "+str(running_acc))
    if(bestA < running_acc):
      bestA = running_acc
      bestT = thres
    acc.append(total_correct/total_cmp)
  pltThreshold(theta,acc,bestT,bestA,save_path)
  return acc

def load_checkpoint(model, optimizer, save_path):
    state_dict = torch.load(save_path)
    model.load_state_dict(state_dict['model_state_dict'])
    optimizer.load_state_dict(state_dict['optimizer_state_dict'])
    val_loss = state_dict['val_loss']
    print(f'Model loaded from <== {save_path}')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = Siamese()
optimizer = optim.Adam(model.parameters(), lr=0.001)
load_checkpoint(model,optimizer,'model1_net.pt')
getThreshold(valid_loader,model.to(device),'T1.png')






##########################################################improve the model ###############################################################

###########################################      CHANGE OPTIMIZER     ##############################

#############model 1 ############################
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
model = Siamese()
model = model.to(device)
num_epochs = 20
eval_every = 10
total_step = len(loader)*num_epochs
best_val_loss = None
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay = 0.001)  #### change learning rate to 0.01 and add weight_decay
TRAIN(model, loader, valid_loader, num_epochs, eval_every,
      total_step, criterion, optimizer, best_val_loss, device ,'model2_net.pt','Loss_M2.png')

load_checkpoint(model,optimizer,'model2_net.pt')
getThreshold(valid_loader,model.to(device),'T2.png')
##########################end of model 1


#############model 2 ############################
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
model = Siamese()
model = model.to(device)
num_epochs = 1
eval_every = 4
total_step = len(loader)*num_epochs
best_val_loss = None
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay = 0.01)  ######### add only weight_decay


TRAIN(model, loader, valid_loader, num_epochs, eval_every,
      total_step, criterion, optimizer, best_val_loss, device ,'model3_net.pt','Loss_M3.png')

load_checkpoint(model,optimizer,'model3_net.pt')
getThreshold(valid_loader,model.to(device),'T3.png')
##########################end of model 2


#############model 3 ############################
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
model = Siamese()
model = model.to(device)
num_epochs = 25  #####increase the number of epoch
eval_every = 10
total_step = len(loader)*num_epochs
best_val_loss = None
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay = 0.001) ######add weight_decay

TRAIN(model, loader, valid_loader, num_epochs, eval_every,
      total_step, criterion, optimizer, best_val_loss, device ,' model4_net.pt','Loss_M4.png')

load_checkpoint(model,optimizer,'model4_net.pt')
getThreshold(valid_loader,model.to(device),'T4.png')
##########################end of model 3

########################################################end of change optimizer #####################################

################################################### Change network ##############################################
class Siamese2(nn.Module):
  def __init__(self):
    super(Siamese2, self).__init__()
    #####Convolutional layers###############
    self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size= 3,stride=1, padding=1)
    self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size= 3,stride=1, padding=1)
    self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size= 3,stride=1, padding=1)
    self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size= 3,stride=1, padding=1)
    self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size= 3,stride=1, padding=1)
    self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size= 3,stride=1, padding=1) ##new
    self.conv7 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size= 3,stride=1, padding=1)


    self.bn1 = nn.BatchNorm1d(32*32*32)
    self.bn2 = nn.BatchNorm1d(32*32*32)
    self.bn3 = nn.BatchNorm1d(16*16*64)
    self.bn4 = nn.BatchNorm1d(16*16*128)
    self.bn5 = nn.BatchNorm1d(16*16*256)
    self.bn6 = nn.BatchNorm1d(16*16*256) #new
    self.bn7 = nn.BatchNorm1d(16*16*512)

    self.maxPool = nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
    self.avgPool = nn.AvgPool2d(kernel_size=16,padding=0)
    ##########end of Convolutional layers###########
    
    ##########linear layers###################
    self.fc1 = nn.Linear(512,512)
    self.drop = nn.Dropout(0.5)
    self.fc2 = nn.Linear(512,1)

    self.sigmoid = nn.Sigmoid()

    ############end of linear layers################
  def forward(self,pic1,pic2):
    x = self.aggregation(pic1,pic2)
    x = F.relu(x)
    x = self.drop(x)
    x = self.sigmoid(self.fc2(x))
    return x.view(-1)

  def getfeature(self,pic):
    batch_in = pic.shape[0]
    x = F.relu(self.bn1(self.conv1(pic).view(batch_in,32*32*32)).view(batch_in,32,32,32))
    x = F.relu(self.bn2(self.conv2(x).view(batch_in,32*32*32)).view(batch_in,32,32,32))
    x = self.maxPool(x)
    x = F.relu(self.bn3(self.conv3(x).view(batch_in,64*16*16)).view(batch_in,64,16,16))
    x = F.relu(self.bn4(self.conv4(x).view(batch_in,128*16*16)).view(batch_in,128,16,16))
    x = F.relu(self.bn5(self.conv5(x).view(batch_in,256*16*16)).view(batch_in,256,16,16))
    x = F.relu(self.bn6(self.conv6(x).view(batch_in,256*16*16)).view(batch_in,256,16,16)) ##new
    x = F.relu(self.bn7(self.conv7(x).view(batch_in,512*16*16)).view(batch_in,512,16,16))
    x = self.avgPool(x)
    x = x.view(-1,1*512)
    return x

    #############Aggregation##################
  def aggregation(self,pic1,pic2):
    pic1_vector = self.getfeature(pic1)
    pic2_vector = self.getfeature(pic2)
    diff = torch.abs(torch.sub(pic1_vector,pic2_vector))
    return diff
    #############end of Aggregation##################

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
model = Siamese2()
model = model.to(device)
num_epochs = 20
eval_every = 10
total_step = len(loader)*num_epochs
best_val_loss = None
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


TRAIN(model, loader, valid_loader, num_epochs, eval_every,
      total_step, criterion, optimizer, best_val_loss, device ,'model5_net.pt', 'Loss_M5.png')

load_checkpoint(model,optimizer,'model5_net.pt')
getThreshold(valid_loader,model.to(device),'T5.png')

#####################################end of model 5

####################################model 6
class Siamese3(nn.Module):
  def __init__(self):
    super(Siamese3, self).__init__()
    #####Convolutional layers###############
    self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size= 3,stride=1, padding=1)
    self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size= 3,stride=1, padding=1)
    self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size= 3,stride=1, padding=1)
    self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size= 3,stride=1, padding=1)
    self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size= 3,stride=1, padding=1)
    self.conv6 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size= 3,stride=1, padding=1)
    self.conv7 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size= 3,stride=1, padding=1) #new

    self.bn1 = nn.BatchNorm1d(32*32*32)
    self.bn2 = nn.BatchNorm1d(32*32*32)
    self.bn3 = nn.BatchNorm1d(16*16*64)
    self.bn4 = nn.BatchNorm1d(16*16*128)
    self.bn5 = nn.BatchNorm1d(16*16*256)
    self.bn6 = nn.BatchNorm1d(16*16*512)
    self.bn7 = nn.BatchNorm1d(16*16*512) #new

    self.maxPool = nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
    self.avgPool = nn.AvgPool2d(kernel_size=16,padding=0)
    ##########end of Convolutional layers###########
    
    ##########linear layers###################
    self.fc1 = nn.Linear(512,512)
    self.drop = nn.Dropout(0.5)
    self.fc2 = nn.Linear(512,1)

    self.sigmoid = nn.Sigmoid()

    ############end of linear layers################
  def forward(self,pic1,pic2):
    x = self.aggregation(pic1,pic2)
    x = F.relu(x)
    x = self.drop(x)
    x = self.sigmoid(self.fc2(x))
    return x.view(-1)

  def getfeature(self,pic):
    batch_in = pic.shape[0]
    x = F.relu(self.bn1(self.conv1(pic).view(batch_in,32*32*32)).view(batch_in,32,32,32))
    x = F.relu(self.bn2(self.conv2(x).view(batch_in,32*32*32)).view(batch_in,32,32,32))
    x = self.maxPool(x)
    x = F.relu(self.bn3(self.conv3(x).view(batch_in,64*16*16)).view(batch_in,64,16,16))
    x = F.relu(self.bn4(self.conv4(x).view(batch_in,128*16*16)).view(batch_in,128,16,16))
    x = F.relu(self.bn5(self.conv5(x).view(batch_in,256*16*16)).view(batch_in,256,16,16))
    x = F.relu(self.bn6(self.conv6(x).view(batch_in,512*16*16)).view(batch_in,512,16,16))
    x = F.relu(self.bn6(self.conv7(x).view(batch_in,512*16*16)).view(batch_in,512,16,16)) ##new
    x = self.avgPool(x)
    x = x.view(-1,1*512)
    return x

    #############Aggregation##################
  def aggregation(self,pic1,pic2):
    #print(pic1.shape)
    pic1_vector = self.getfeature(pic1)
    pic2_vector = self.getfeature(pic2)
    diff = torch.abs(torch.sub(pic1_vector,pic2_vector))
    return diff
    #############end of Aggregation##################
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
model = Siamese3()
model = model.to(device)
num_epochs = 20
eval_every = 10
total_step = len(loader)*num_epochs
best_val_loss = None
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


TRAIN(model, loader, valid_loader, num_epochs, eval_every,
      total_step, criterion, optimizer, best_val_loss, device ,'model6_net.pt', 'Loss_M6.png')

load_checkpoint(model,optimizer,'model6_net.pt')
getThreshold(valid_loader,model.to(device),'T6.png')
#####################################end of model 6

###########model 7
class Siamese4(nn.Module):
  def __init__(self):
    super(Siamese4, self).__init__()
    #####Convolutional layers###############
    self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size= 3,stride=1, padding=1)
    self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size= 3,stride=1, padding=1)
    self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size= 3,stride=1, padding=1)
    self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size= 3,stride=1, padding=1) #####new
    self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size= 3,stride=1, padding=1)
    self.conv6 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size= 3,stride=1, padding=1)
    self.conv7 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size= 3,stride=1, padding=1)

    self.bn1 = nn.BatchNorm1d(32*32*32)
    self.bn2 = nn.BatchNorm1d(32*32*32)
    self.bn3 = nn.BatchNorm1d(16*16*64)
    self.bn4 = nn.BatchNorm1d(16*16*64)
    self.bn5 = nn.BatchNorm1d(16*16*128)
    self.bn6 = nn.BatchNorm1d(16*16*256)
    self.bn7 = nn.BatchNorm1d(16*16*512)

    self.maxPool = nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
    self.avgPool = nn.AvgPool2d(kernel_size=16,padding=0)
    ##########end of Convolutional layers###########
    
    ##########linear layers###################
    self.fc1 = nn.Linear(512,512)
    self.drop = nn.Dropout(0.5)
    self.fc2 = nn.Linear(512,1)

    self.sigmoid = nn.Sigmoid()

    ############end of linear layers################
  def forward(self,pic1,pic2):
    x = self.aggregation(pic1,pic2)
    #print(x.shape)
    x = F.relu(x)
    #print(x.shape)
    x = self.drop(x)
    #print(x.shape)
    x = self.sigmoid(self.fc2(x))
    y = x
    return x.view(-1)

  def getfeature(self,pic):
    batch_in = pic.shape[0]
    x = F.relu(self.bn1(self.conv1(pic).view(batch_in,32*32*32)).view(batch_in,32,32,32))
    x = F.relu(self.bn2(self.conv2(x).view(batch_in,32*32*32)).view(batch_in,32,32,32))
    x = self.maxPool(x)
    x = F.relu(self.bn3(self.conv3(x).view(batch_in,64*16*16)).view(batch_in,64,16,16))
    x = F.relu(self.bn4(self.conv4(x).view(batch_in,64*16*16)).view(batch_in,64,16,16))
    x = F.relu(self.bn5(self.conv5(x).view(batch_in,128*16*16)).view(batch_in,128,16,16))
    x = F.relu(self.bn6(self.conv6(x).view(batch_in,256*16*16)).view(batch_in,256,16,16))
    x = F.relu(self.bn7(self.conv7(x).view(batch_in,512*16*16)).view(batch_in,512,16,16))
    x = self.avgPool(x)
    x = x.view(-1,1*512)
    # print(x.shape)
    return x

    #############Aggregation##################
  def aggregation(self,pic1,pic2):
    #print(pic1.shape)
    pic1_vector = self.getfeature(pic1)
    pic2_vector = self.getfeature(pic2)
    diff = torch.abs(torch.sub(pic1_vector,pic2_vector))
    return diff
    #############end of Aggregation##################
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
model = Siamese4()
model = model.to(device)
num_epochs = 20
eval_every = 10
total_step = len(loader)*num_epochs
best_val_loss = None
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


TRAIN(model, loader, valid_loader, num_epochs, eval_every,
      total_step, criterion, optimizer, best_val_loss, device ,'model7_net.pt', 'Loss_M7.png')

load_checkpoint(model,optimizer,'model7_net.pt')
getThreshold(valid_loader,model.to(device),'T7.png')
########################end of model 7 ##############

