from utilitiesAndClasses import *

def pltThreshold(thres,acc,bestT,bestA,save_path):
  plt.plot(thres,acc)
  plt.xlabel('threshold')
  plt.ylabel('accuracy')
  plt.title('threshold vs accuracy\n best theta = %.3f which gives %.4f accuracy' % (bestT,bestA))
  plt.savefig(save_path,dpi=300)
  plt.close()

def getThreshold(valid_loader,model,save_path):
  theta = np.linspace(0.30,0.6,20, False)
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
    print(str(thres)+"  "+str(running_acc))
    if(bestA < running_acc):
      bestA = running_acc
      bestT = thres
    acc.append(total_correct/total_cmp)
  pltThreshold(theta,acc,bestT,bestA,save_path)

def load_checkpoint(model, optimizer, save_path):
    state_dict = torch.load(save_path)
    model.load_state_dict(state_dict['model_state_dict'])
    optimizer.load_state_dict(state_dict['optimizer_state_dict'])
    val_loss = state_dict['val_loss']
    print(f'Model loaded from <== {save_path}')


print("findding the best threshold for the model")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = Siamese()
optimizer = optim.Adam(model.parameters(), lr=0.001)
load_checkpoint(model,optimizer,'model1_net.pt')
getThreshold(valid_loader,model.to(device),'T1.png')
