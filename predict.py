from utilitiesAndClasses import *


def Predict(test_loader,model,threshold):
  pred = []
  with torch.no_grad():
    model.eval()
    for test_pic1,test_pic2, test_labels in test_loader:
      test_pic1, test_pic2 , test_labels = test_pic1.to(device), test_pic2.to(device), test_labels.to(device)
      test_outputs = model(test_pic1,test_pic2)
      i = 0
      for out in test_outputs:
        check = 0
        if out >= threshold:
            check = 1
        pred.append(check)

    pd.DataFrame(pred).to_csv('pred.csv')
    print(pred)

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
    self.bn4 = nn.BatchNorm1d(16*16*64) ##new
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
    x = self.fc1(x)
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
    x = F.relu(self.bn4(self.conv4(x).view(batch_in,64*16*16)).view(batch_in,64,16,16))  #new
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
print("predicting")
model = Siamese4()
optimizer = optim.Adam(model.parameters(), lr=0.01 , weight_decay = 0.001)
model = model.to(device)
load_checkpoint(model,optimizer,'model7_net.pt')
Predict(test_loader,model.to(device),0.535)