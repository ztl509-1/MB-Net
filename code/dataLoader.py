import torch
import os
from torch.utils.data import Dataset
import glob
import pywt
import numpy as np

class FallDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.data, self.label = self.readRadarData(self.path)

    def readRadarData(self, path):
        folder_list = os.listdir(path)
        print(folder_list)

        all_data = []
        label = []
        l = None
        wavename = 'db1'
        for folder in folder_list:
            if folder[:3] == 'fal':
                l = 0
            else:
                l = 1
            folder_path = os.path.join(path, folder)
            files_path = glob.glob('{}/*'.format(folder_path))
            for file_path in files_path:
                data = np.zeros([2, 4800])
                data1 = np.load(file_path, allow_pickle=True)

                cA, cD2, cD1 = pywt.wavedec(data1[0], wavename, level=2)
                data_0_250Hz = (cA - cA.min()) / (cA.max() - cA.min()) * 2 - 1  # normalize to [-1,1]
                data_250_500Hz = (cD2 - cD2.min()) / (cD2.max() - cD2.min()) * 2 - 1
                data_500_1000Hz = (cD1 - cD1.min()) / (cD1.max() - cD1.min()) * 2 - 1
                data[0] = np.hstack((data_0_250Hz[:1200], data_250_500Hz[:1200], data_500_1000Hz[:2400]))

                cA, cD2, cD1 = pywt.wavedec(data1[1], wavename, level=2)
                data_0_250Hz = (cA - cA.min()) / (cA.max() - cA.min()) * 2 - 1  # normalize to [-1,1]
                data_250_500Hz = (cD2 - cD2.min()) / (cD2.max() - cD2.min()) * 2 - 1
                data_500_1000Hz = (cD1 - cD1.min()) / (cD1.max() - cD1.min()) * 2 - 1
                data[1] = np.hstack((data_0_250Hz[:1200], data_250_500Hz[:1200], data_500_1000Hz[:2400]))

                all_data.append(data)
                label.append(l)
        print("finish")
        all_data = np.array(all_data)
        label = np.array(label)
        print("max_label: ", label.max())
        index = np.arange(len(label))
        np.random.shuffle(index)
        all_data = all_data[index]
        label = label[index]

        all_data = torch.Tensor(all_data)
        label = torch.Tensor(label)

        print(all_data.shape)
        print(label.shape)

        return all_data, label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]
