'''
Refer to https://github.com/ZZUTK/Face-Aging-CAAE/blob/master/FaceAging.py
'''

from torch.utils.data import Dataset
import glob
import cv2
import numpy as np
from torchvision import transforms


class UTKDataset(Dataset):
    def __init__(self, root, transform=None):
        self.path = root
        self.num_categories = 10
        self.image_value_range = (-1, 1)
        self.transform = transform
        file_list = glob.glob(self.path + "/*.jpg")
        self.data = []
        self.sample_label_age = np.ones(
                    shape=(len(file_list), self.num_categories),
                    dtype=np.float32
                ) * self.image_value_range[0]

        for i, f in enumerate(file_list):
            age = int(f.split('/')[-1].split('_')[0])
            if 0 <= age <= 5:
                age = 0
            elif 6 <= age <= 10:
                age = 1
            elif 11 <= age <= 15:
                age = 2
            elif 16 <= age <= 20:
                age = 3
            elif 21 <= age <= 30:
                age = 4
            elif 31 <= age <= 40:
                age = 5
            elif 41 <= age <= 50:
                age = 6
            elif 51 <= age <= 60:
                age = 7
            elif 61 <= age <= 70:
                age = 8
            else:
                age = 9

            self.data.append(f)
            self.sample_label_age[i, age] = self.image_value_range[-1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path = self.data[idx]
        img = cv2.imread(path)
        img = transforms.ToPILImage()(img)
        img = self.transform(img)

        age = self.sample_label_age[idx]
        return img, age
