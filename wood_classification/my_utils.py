import torch
import numpy as np
import time
import pandas as pd

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds==labels).item()/len(preds))

@torch.no_grad()
def evaluate_model(model, val_dl, criterion):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    
    losses = []
    accuracies = []
    
    for x_b, y_b, _ in val_dl:
       
        x_b, y_b = x_b.to(device), y_b.to(device)
        
        outputs = model(x_b)
        # loss = criterion(outputs.logits, y_b)
        # acc = accuracy(outputs.logits, y_b)
        loss = criterion(outputs, y_b)
        acc = accuracy(outputs, y_b)

        
        losses.append(loss.item())
        accuracies.append(acc.item())
    
    return (np.mean(losses), np.mean(accuracies))


def fit_search(model, train_dl, val_dl, optimizer, loss_fn, epochs, device="cpu", batch_size=32, logits=True):
    model.train()
    scaler = torch.cuda.amp.GradScaler()

    # Start Training
    for epoch in range(epochs):
   
        running_loss = 0
        for i, (x_b, y_b, _) in enumerate(train_dl):
            x_b, y_b = x_b.to(device), y_b.to(device)

            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                outputs = model(x_b)
                if(logits):
                    loss = loss_fn(outputs.logits, y_b)
                else:
                    loss = loss_fn(outputs, y_b)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        print(f'[Epoch: {epoch+1} | loss: {running_loss / i:.3f}')

    val_accs = []
    val_losses = []


    with torch.cuda.amp.autocast():
        for v in range(3):
            val_loss, val_acc = evaluate_model(model, val_dl, loss_fn)
            val_accs.append(val_acc)
            val_losses.append(val_loss)

    return (np.mean(val_accs), np.mean(val_losses))

def save_training_result(v_loss, v_acc, t_loss, t_acc, filename):
    # dict = {'Val_Loss': v_loss, 'Val_Accuracy': v_acc, 'Test_Loss': t_loss, 'Test_Accuracy': t_acc, 'Train_Time': train_time}
    dict = {'Train_Loss': t_loss, 'Train_Accuracy': t_acc,'Val_Loss': v_loss, 'Val_Accuracy': v_acc}
    df = pd.DataFrame(dict)
    df.to_csv(f'./results/{filename}')


def fit2(model, train_dl, optimizer, loss_fn, epochs, scheduler=None, device="cpu", save=False, save_name="", logits=True):
    
    scaler = torch.cuda.amp.GradScaler()
    save_on_epoch = [10, 20, 30, 40, 50, 60]

    # Evaluate Model Before Training
    if(save == True):
        # List to store losses and accuracies
        print('=== (Save Results Mode : ON) ===')
        #
        train_losses = []
        train_accuracies = []
        #

                    
    print('\n---------------------------')
    print('>>>>> Training Starts <<<<<')
    print('---------------------------\n')
    # Start Training
    for epoch in range(epochs):
        start_time = time.time()       
        losses = []
        running_loss = 0
        for i, (x_b, y_b, _) in enumerate(train_dl):
            x_b, y_b = x_b.to(device), y_b.to(device)

            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                outputs = model(x_b)
                if(logits):
                    loss = loss_fn(outputs.logits, y_b)
                else:
                    loss = loss_fn(outputs, y_b)
                losses.append(loss.item())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

            if i%64 == (63) and i > 0:
                print(f'[Epoch: {epoch+1} | Batch: {i + 1:5d}] loss: {running_loss / 64:.3f}')
                running_loss = 0.0
        
        if(scheduler is not None):
            # avg_loss = np.mean(losses)
            scheduler.step()
        end_time = time.time()-start_time

        print(f'=== (Time/Epoch : {end_time/60:.2f} minutes) ===')
        
        if (save == True):

#             with torch.cuda.amp.autocast():
#                 print('=== [ Evaluating.....] ===')

#                 train_loss, train_acc = evaluate_model(model, train_dl, loss_fn)
#                 train_losses.append(train_loss)
#                 train_accuracies.append(train_acc)

            if((epoch+1) in save_on_epoch):
                print(f'=== [ Saving Epoch {epoch+1} ] ===')
            # save_training_result(val_losses, val_accuracies, test_losses, test_accuracies, train_times, filename=f'/{save_name}/epoch_{epoch+1}.csv')
#                 save_training_result2(train_losses, train_accuracies, filename=f'/{save_name}/epoch_{epoch+1}.csv')
                save_checkpoint(epoch+1, model, optimizer, f'/{save_name}/epoch_{epoch+1}.pt')

    print('\n---------------------------')
    print('>>>>> Training  Ended <<<<<')
    print('---------------------------')

def save_checkpoint(epoch, model, optimizer, filename):
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, f'./check_points/{filename}')
    
def save_training_result2(t_loss, t_acc, filename):
    # dict = {'Val_Loss': v_loss, 'Val_Accuracy': v_acc, 'Test_Loss': t_loss, 'Test_Accuracy': t_acc, 'Train_Time': train_time}
    dict = {'Train_Loss': t_loss, 'Train_Accuracy': t_acc}
    df = pd.DataFrame(dict)
    df.to_csv(f'./results/{filename}')