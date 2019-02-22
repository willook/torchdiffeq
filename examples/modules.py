from torch.utils.data import Dataset
from glob import glob
import numpy as np

def read_mfccs_and_phones(npz_file):
    np_arrays = np.load(npz_file)

    mfccs = np_arrays['mfccs']
    phns = np_arrays['phns']

    np_arrays.close()

    return mfccs, phns

class TIMIT(Dataset):

    
    def __init__(self, root, train = True, debug = False):
        self.root = root
        self.train = train
        self.debug = debug
        if self.train:
            self.train_data = []
            self.train_label = []
            
            self.train_root = root+"/TRAIN/*/*/*.npz"    
            self.train_list = glob(self.train_root)
            
            for data_path in self.train_list:
                mfccs, phns = read_mfccs_and_phones(data_path)
                self.train_data.append(mfccs)
                self.train_label.append(phns)
            

        else:
            self.test_data = []
            self.test_label = []
            
            self.test_root = root+"/TEST/*/*/*.npz"
            self.test_list = glob(self.test_root)
            if self.debug:
                print("test dataset:", len(self.test_list))
            for data_path in self.test_list:
                mfccs, phns = read_mfccs_and_phones(data_path)
                self.test_data.append(mfccs)
                self.test_label.append(phns)
            

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)
        
    def __getitem__(self, idx):
        if self.train:
            return self.train_data[idx],self.train_label[idx]
        else:
            return self.test_data[idx], self.test_label[idx]     
        
