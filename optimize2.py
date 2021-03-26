from utilitiesAndClasses import *

#############model 2 ############################
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
model = Siamese()
model = model.to(device)
num_epochs = 20
eval_every = 10
total_step = len(loader)*num_epochs
best_val_loss = None
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay = 0.001)  ######### add weight_decay and change learning rate to 0.01


TRAIN(model, loader, valid_loader, num_epochs, eval_every,
      total_step, criterion, optimizer, best_val_loss, device ,'model3_net.pt','Loss_M3.png')

print("findding the best threshold for the model")
load_checkpoint(model,optimizer,'model3_net.pt')
getThreshold(valid_loader,model.to(device),'T3.png')
##########################end of model 2