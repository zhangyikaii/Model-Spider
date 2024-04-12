import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class dSprites(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, dir, transform=None):
        """
        Args:
            dir (string): Directory containing the dSprites dataset
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.dir = dir
        self.filename = 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'
        self.filepath = f'{self.dir}/{self.filename}'
        dataset_zip = np.load(self.filepath, allow_pickle=True, encoding='bytes')

        self.imgs = dataset_zip['imgs']
        self.latents_values = dataset_zip['latents_values']
        self.latents_classes = dataset_zip['latents_classes']
        self.metadata = dataset_zip['metadata'][()]
        np.random.seed(42)
        indices = np.random.choice(self.imgs.shape[0], size=200000, replace=False)
        self.imgs = self.imgs[indices]
        self.latents_values = self.latents_values[indices]

        self.transform = transform
        print()

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        sample = self.imgs[idx].astype(np.uint8)

        sample = sample.reshape(sample.shape[0], sample.shape[1], 1).repeat(3, axis=2)
        sample = transforms.ToPILImage()(sample)
        if self.transform:
            sample = self.transform(sample)
            
        return sample, self.latents_values[idx][2:].astype(np.float32)

if __name__ == "__main__":
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
        mean=0.485, std=0.229
        )
    ])
    train_dataset = dSprites('/data/zhangyk/data/dsprites/', transform=train_transform)
    print()