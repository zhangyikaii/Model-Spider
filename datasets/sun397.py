import os

from PIL import Image
from torch.utils.data import Dataset
from .load_dataset import load_pickle


class SUN397(Dataset):
    """
    SUN Database: http://vision.princeton.edu/projects/2010/SUN/SUN397.tar.gz
    """
    def __init__(self, root, train, transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.split = 'trainval' if train else 'test'
        data = load_pickle(os.path.join(self.root, 'annotations.pkl'))[self.split]
        self.samples = [(os.path.join(self.root, 'SUN397', i[0]), i[1]) for i in data]

    def __getitem__(self, i):
        image, label = self.samples[i]
        image = Image.open(image).convert('RGB')
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

    def __len__(self):
        return len(self.samples)
