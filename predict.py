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

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
print("predicting")
model = Siamese()
optimizer = optim.Adam(model.parameters(), lr=0.01)
model = model.to(device)
load_checkpoint(model,optimizer,'model5_net.pt')
Predict(test_loader,model.to(device))