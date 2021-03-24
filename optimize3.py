from utilitiesAndClasses import *

#############model 3 ############################
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
model = Siamese()
model = model.to(device)
num_epochs = 20
eval_every = 10
total_step = len(loader)*num_epochs
best_val_loss = None
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.005 , weight_decay = 0.00001) #change learning rate to 0.01, add weight_decay

TRAIN(model, loader, valid_loader, num_epochs, eval_every,
      total_step, criterion, optimizer, best_val_loss, device ,'model4_net.pt','Loss_M4.png')

print("findding the best threshold for the model")
load_checkpoint(model,optimizer,'model4_net.pt')
getThreshold(valid_loader,model.to(device),'T4.png')
##########################end of model 3