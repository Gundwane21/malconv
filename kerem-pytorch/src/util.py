import os

import numpy as np
import torch
from torch.utils.data import Dataset

from torch.utils.data import DataLoader
from datetime import datetime
import numpy as np

def write_pred(test_pred,test_idx,file_path):
    test_pred = [item for sublist in test_pred for item in sublist]
    with open(file_path,'w') as f:
        for idx,pred in zip(test_idx,test_pred):
            print(idx.upper()+','+str(pred[0]),file=f)

# Dataset preparation
class ExeDataset(Dataset):
    def __init__(self, fp_list, data_path, label_list, first_n_byte=2000000):
        self.fp_list = fp_list
        self.data_path = data_path
        self.label_list = label_list
        self.first_n_byte = first_n_byte

    def __len__(self):
        return len(self.fp_list)

    def __getitem__(self, idx):
        try:
            with open(os.path.join(self.data_path,self.fp_list[idx]),'rb') as f:
                tmp = [i+1 for i in f.read()[:self.first_n_byte]]
                tmp = tmp+[0]*(self.first_n_byte-len(tmp))
        except:
            with open(os.path.join(self.data_path,self.fp_list[idx].lower()),'rb') as f:
                tmp = [i+1 for i in f.read()[:self.first_n_byte]]
                tmp = tmp+[0]*(self.first_n_byte-len(tmp))

        return np.array(tmp),np.array([self.label_list[idx]])



def save_checkpoint(state, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    print("Checkpoint saved")


def load_checkpoint(checkpoint, model):
    model.load_state_dict(checkpoint["state_dict"])
    print("Checkpoint loaded")


def print_and_log(data, file=None):
    print(data)
    if file is not None:
        file.write(data)

def create_config_stamp(optim_name, learning_rate, patience):
    return f"{optim_name}_{learning_rate}_{patience}"

def create_timestamp():
    return datetime.now().strftime("%d-%m-%Y_%H-%M-%S")    

def make_directory(parent_dir, dataset, data_comb, config_stamp, timestamp):
    directory = f"{dataset}_{data_comb}_{config_stamp}_{timestamp}"
    output_path = os.path.join(parent_dir, directory)
    predictions_path = os.path.join(output_path, "predictions")
    golds_path = os.path.join(output_path, "golds")
    arrays_path = os.path.join(output_path, "arrays")
    processed_path = os.path.join(output_path, "processed")
    try:
        os.mkdir(output_path)
        print(f"\n{output_path} successfully created")
        os.mkdir(predictions_path)
        print(f"\n{predictions_path} successfully created")
#        os.mkdir(golds_path)
#        print(f"\n{golds_path} successfully created")
        os.mkdir(arrays_path)
        print(f"\n{arrays_path} successfully created")
        os.mkdir(processed_path)
        print(f"\n{processed_path} successfully created")
    except OSError as error:
        print(error)
    return {
        "output_path": output_path,
        "predictions_path": predictions_path,
#        "golds_path": golds_path,
        "arrays_path": arrays_path,
        "processed_path": processed_path
    }
