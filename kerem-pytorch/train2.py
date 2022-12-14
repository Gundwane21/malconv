import sys
import os
from tqdm import tqdm
import torch.nn as nn
import torch
import torch.optim as optim
from src.util import print_and_log, save_checkpoint ,load_checkpoint, create_config_stamp , create_timestamp,make_directory
from src.util import ExeDataset,write_pred
from src.model import MalConv
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Hyperparameters etc.
LEARNING_RATE = 0.0001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 25
NUM_EPOCHS = 10
NUM_WORKERS = 0

PIN_MEMORY = True
LOAD_MODEL = False

PARENT_DIR = '/content/drive/MyDrive/runs/kerem-pytorch'
DATASET = 'microsoft-big'
DATA_COMB = 'small-random'

train_data_path = '/content/drive/MyDrive/microsoft_big/train_small_1_10'                    # Training data
valid_data_path = train_data_path
test_data_path = train_data_path
first_n_byte = 2000000
batch_size = BATCH_SIZE
use_gpu =  True if torch.cuda.is_available() else False
train_label_path = '/content/drive/MyDrive/microsoft_big' # Training label  

use_cpu = 1
window_size = 500          # Kernel size & stride for Malconv (defualt : 500)

from sklearn.metrics import confusion_matrix , classification_report
# evaluation function which will run for every EPOCH
def model_test(model, loader):
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for index, (data, target) in enumerate(loader):
            data = data.to(device=DEVICE)
            target = target.type(torch.LongTensor).to(device=DEVICE)
            prediction = model(data)
            y_true.extend(target.cpu().numpy())
            _, predicted = torch.max(prediction, 1)
            y_pred.extend(predicted.cpu().numpy())            

    cm = confusion_matrix(y_true,y_pred)
    cr = classification_report(y_true,y_pred, labels = [1,2,3,4,5,6,7,8,9])
    print("CM")
    print(cm)
    print("CR")
    print(cr)
    return cm ,cr


# evaluation function which will run for every EPOCH
def model_eval(model, loader, loss_fn):
    total_loss = 0
    model.eval()
    with torch.no_grad():
        for index, (data, target) in enumerate(loader):
            data = data.to(device=DEVICE)
            target = target.type(torch.LongTensor).to(device=DEVICE)
            with torch.cuda.amp.autocast():
                predictions = model(data)
                loss = loss_fn(predictions, target)

                total_loss += loss


    average_loss = total_loss / len(loader)
    model.train()

    return average_loss

# training function which will run for every EPOCH
def train_fn(loader, model, optimizer, loss_fn):
    loop = tqdm(loader, file=sys.stdout)
    total_loss = 0

    for index, (data, target) in enumerate(loop):
        data = data.to(device=DEVICE)
        target = target.type(torch.LongTensor).to(device=DEVICE)

#        print("input data")
#        print(data)
#        print("targets")
#        print(target)


        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            
            loss = loss_fn(predictions, target)
            total_loss += loss

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    
        # update tqdm loop
        loop.set_postfix(loss=loss.item())

    average_loss = total_loss / len(loader)
    return average_loss

def draw_graph(loss_train,loss_val,savepath):
  import matplotlib.pyplot as plt

  epochs = range(len(loss_train) )
  print(epochs)
  print(type(loss_train))
  print(loss_train)
  print(type(loss_val))
  print(loss_val)

  plt.plot(epochs, loss_train, 'g', label='Training loss')
  plt.plot(epochs, loss_val, 'b', label='validation loss')
  plt.title('Training and Validation loss')
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.legend()
  # plt.show()
  plt.savefig(os.path.join(savepath,"loss_graph.png" ))


# get the data and initiate loaders for train and validation
def get_loaders():
    all_labels_path = os.path.join(train_label_path,'trainLabels.csv')
    all_files_path = train_data_path

    all_labels_df = pd.read_csv(all_labels_path)
    all_files = os.listdir(all_files_path)
    for i in range(len(all_files)):
      all_files[i] = all_files[i].split('.')[0]

    print(len(all_labels_df))


    print(all_labels_df)
    print(all_files)


    # remove the labels not in our limited trainings
    all_labels_df.drop(all_labels_df[~all_labels_df.Id.isin(all_files)].index, inplace=True)
    all_labels= all_labels_df.reset_index()
    print(len(all_labels_df))

    X= all_labels_df['Id']
    y= all_labels_df['Class']


    # using the train test split function
    X_train, X_test,y_train, y_test = train_test_split(X,y ,
                                    test_size=0.20, 
                                    shuffle=True)
    # using the train test split function
    X_train, X_val,y_train, y_val = train_test_split(X_train,y_train ,
                                    test_size=0.20, 
                                    shuffle=True)


    
    X_train = X_train.to_list()
    y_train = y_train.to_list()
    X_val = X_val.to_list()
    y_val = y_val.to_list()
    X_test = X_test.to_list()
    y_test = y_test.to_list()

    for i in range(len(X_train)):
      X_train[i] = X_train[i]+'.bytes'

    for i in range(len(X_val)):
      X_val[i] = X_val[i]+'.bytes'

    for i in range(len(X_test)):
      X_test[i] = X_test[i]+'.bytes'

    print(f'train size: {len(X_train)}')
    print(f'valid size: {len(X_val)}')
    print(f'test size: {len(X_test)}')

    train_loader = DataLoader(ExeDataset(X_train, train_data_path, y_train,first_n_byte),
                                batch_size=batch_size, shuffle=True, num_workers=use_cpu)
    valid_loader = DataLoader(ExeDataset(X_val, valid_data_path, y_val,first_n_byte),
                            batch_size=batch_size, shuffle=False, num_workers=use_cpu)
    test_loader = DataLoader(ExeDataset(X_test, test_data_path, y_test,first_n_byte),
                            batch_size=batch_size, shuffle=False, num_workers=use_cpu)

    return train_loader, valid_loader, test_loader

# driver of the code
def main():

    train_losses = []
    val_losses = []
    train_loader,val_loader,test_loader  = get_loaders()

    model = MalConv(input_length=first_n_byte,window_size=window_size)
    criterion = nn.CrossEntropyLoss()
    softmax = nn.Softmax()

    if use_gpu:
        model = model.cuda()
        criterion = criterion.cuda()
        softmax = softmax.cuda()

    loss_fn = criterion 
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # scaler = torch.cuda.amp.GradScaler()

    # for early stopping
    min_val_loss = np.Inf
    min_epoch = 0
    min_delta = 0
    patience = 50


    # create output directory
    config_stamp = create_config_stamp(type(optimizer).__name__, LEARNING_RATE, patience)
    timestamp = create_timestamp()
    paths = make_directory(PARENT_DIR, DATASET, DATA_COMB, config_stamp, timestamp)
    # create log file to keep info
    log_file_path = paths["output_path"] + "/execution.log"
    log_file = open(log_file_path, "w")

    if LOAD_MODEL:
        checkpoint_file_path = paths["output_path"] + "/checkpoint.pth.tar"
        load_checkpoint(torch.load(checkpoint_file_path), model)
    # compute_scores(val_loader, model, device=DEVICE)

    # training
    for epoch in range(NUM_EPOCHS):
        print_and_log(f"\nTraining EPOCH {epoch + 1}", log_file)
        train_loss = train_fn(train_loader, model, optimizer, loss_fn)
        print_and_log(f"\nAverage Training Loss for EPOCH {epoch + 1}: {train_loss}", log_file)
        train_losses.append(float(train_loss.cpu()))

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        checkpoint_file_path = paths["output_path"] + "/checkpoint.pth.tar"
        save_checkpoint(checkpoint, filename=checkpoint_file_path)

        val_loss = model_eval(model, val_loader, loss_fn)
        val_losses.append(float(val_loss.cpu()))
        print_and_log(f"\nAverage Validation Loss for EPOCH {epoch + 1}: {val_loss}", log_file)

        if min_val_loss > val_loss.item() + min_delta:
            min_val_loss = val_loss.item()
            min_epoch = epoch + 1
            counter = 0
        else:
            counter += 1

        if counter >= patience:
            print_and_log(f"\nMinimum Validation Loss: {min_val_loss} at EPOCH {min_epoch}", log_file)
            print_and_log(f"\nTotal Epoch Count: {epoch + 1}", log_file)
            break

    log_file.close()
    # create txt file to keep scores
    txt_file_path = paths["output_path"] + "/scores.txt"
    txt_file = open(txt_file_path, "w")

    draw_graph(train_losses, val_losses, paths["output_path"])

    model_test(model, test_loader)

if __name__ == "__main__":
    main()    
