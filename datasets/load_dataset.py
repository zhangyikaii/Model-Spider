import os
import pickle
import torchvision
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader


class MyImageFolderDataset(Dataset):
    def __init__(self, samples, transform):
        super().__init__()
        self.transform = transform
        self.samples = samples

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = default_loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target

    def __len__(self):
        return len(self.samples)


def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)


def get_dataset(name, data_path, train_transform, val_transform, way=0, shot=0):
    def imagefolder_dataset(train_prefix, test_prefix):
        return torchvision.datasets.ImageFolder(os.path.join(data_path, train_prefix), transform=train_transform), torchvision.datasets.ImageFolder(os.path.join(data_path, test_prefix), transform=val_transform)
    if name == 'ImageNet':
        train_dataset, val_dataset = imagefolder_dataset('train', 'val')
        num_classes = 1000
    elif name == 'CIFAR10':
        train_dataset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=train_transform)
        val_dataset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=val_transform)
        num_classes = 10
    elif name == 'CIFAR100':
        train_dataset = torchvision.datasets.CIFAR100(root=data_path, train=True, download=True, transform=train_transform)
        val_dataset = torchvision.datasets.CIFAR100(root=data_path, train=False, download=True, transform=val_transform)
        num_classes = 100
    elif name == 'Aircraft':
        from datasets.aircraft import Aircraft
        train_dataset = Aircraft(data_path, transform=train_transform, train=True, download=True)
        val_dataset = Aircraft(data_path, transform=val_transform, train=False, download=True)
        num_classes = 100
    elif name == 'Caltech101':
        from datasets.caltech101 import Caltech101
        train_dataset = Caltech101(data_path, transform=train_transform, train=True)
        val_dataset = Caltech101(data_path, transform=val_transform, train=False)
        num_classes = 101
    elif name == 'Cars':
        print(data_path)
        from datasets.cars import Cars
        train_dataset = Cars(data_path, transform=train_transform, train=True, download=True)
        val_dataset = Cars(data_path, transform=val_transform, train=False, download=True)
        num_classes = 196
    elif name == 'CUB2011':
        from datasets.cub2011 import CUB2011
        train_dataset = CUB2011(data_path, transform=train_transform, train=True, download=True)
        val_dataset = CUB2011(data_path, transform=val_transform, train=False, download=True)
        num_classes = 200
    elif name == 'Dogs':
        from datasets.dogs import Dogs
        train_dataset = Dogs(data_path, transform=train_transform, train=True, download=True)
        val_dataset = Dogs(data_path, transform=val_transform, train=False, download=True)
        num_classes = 120
    elif name == 'DTD':
        from datasets.dtd import DTD
        train_dataset = DTD(data_path, transform=train_transform, train=True)
        val_dataset = DTD(data_path, transform=val_transform, train=False)
        num_classes = 47
    elif name == 'EuroSAT':
        from datasets.eurosat import EuroSAT
        train_dataset = EuroSAT(data_path, transform=train_transform, train=True)
        val_dataset = EuroSAT(data_path, transform=val_transform, train=False)
        num_classes = 10
    elif name == 'Flowers':
        from datasets.flowers import Flowers
        train_dataset = Flowers(data_path, transform=train_transform, train=True)
        val_dataset = Flowers(data_path, transform=val_transform, train=False)
        num_classes = 102
    elif name == 'Food':
        train_dataset, val_dataset = imagefolder_dataset('train', 'test')
        num_classes = 101
    elif name == 'Pet':
        from datasets.oxford_iiit_pet import OxfordIIITPet
        train_dataset = OxfordIIITPet(root=data_path, split='trainval', download=True, transform=train_transform)
        val_dataset = OxfordIIITPet(root=data_path, split='test', download=True, transform=val_transform)
        # train_dataset = torchvision.datasets.OxfordIIITPet(root=data_path, split='trainval', download=True, transform=train_transform)
        # val_dataset = torchvision.datasets.OxfordIIITPet(root=data_path, split='test', download=True, transform=val_transform)
        num_classes = 37
    elif name == 'STL10':
        train_dataset = torchvision.datasets.STL10(root=data_path, split='train', download=True, transform=train_transform)
        val_dataset = torchvision.datasets.STL10(root=data_path, split='test', download=True, transform=val_transform)
        num_classes = 10
    elif name == 'SVHN':
        train_dataset = torchvision.datasets.SVHN(root=data_path, split='train', download=True, transform=train_transform)
        val_dataset = torchvision.datasets.SVHN(root=data_path, split='test', download=True, transform=val_transform)
        num_classes = 10
    elif name == 'SUN397':
        from datasets.sun397 import SUN397
        train_dataset = SUN397(data_path, transform=train_transform, train=True)
        val_dataset = SUN397(data_path, transform=val_transform, train=False)
        num_classes = 397
    elif name in ['OfficeHome', 'PACS', 'DomainNet', 'VLCS']:
        if name == 'OfficeHome':
            raise Exception('train&test set not yet splitting')
            from datasets.officehome import OfficeHome
            cur_dataset_class = OfficeHome
            # domains = ["Art", "Clipart", "Product"]
            domains = ["Art", "Clipart", "Product", "Real World"]
            num_classes = 65
        elif name == 'PACS':
            from datasets.pacs import PACS
            cur_dataset_class = PACS
            # domains = ["art_painting", "cartoon", "photo"]
            domains = ["art_painting", "cartoon", "photo", "sketch"]
            num_classes = 7
        elif name == 'DomainNet':
            from datasets.domainnet import DomainNet
            cur_dataset_class = DomainNet
            domains = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]
            num_classes = 345
        elif name == 'VLCS':
            from datasets.vlcs import VLCS
            cur_dataset_class = VLCS
            # domains = ["caltech", "labelme", "pascal"]
            domains = ["caltech", "labelme", "pascal", "sun"]
            num_classes = 5
        from datasets.udomain import DatasetWrapper
        full_dataset = cur_dataset_class(data_path, domains, domains)
        train_dataset = DatasetWrapper(full_dataset.train_x, transform=train_transform)
        val_dataset = DatasetWrapper(full_dataset.test, transform=val_transform)
    elif 'DomainNet-' in name:
        from datasets.domainnet import DomainNet
        from datasets.udomain import DatasetWrapper
        domains = [name[name.find('DomainNet-') + len('DomainNet-'):]]
        num_classes = 345
        full_dataset = DomainNet(data_path, domains, domains)
        train_dataset = DatasetWrapper(full_dataset.train_x, transform=train_transform)
        val_dataset = DatasetWrapper(full_dataset.test, transform=val_transform)
    elif name == 'NABirds':
        from datasets.nabirds import NABirds
        train_dataset = NABirds(data_path, transform=train_transform, train=True, download=False)
        val_dataset = NABirds(data_path, transform=val_transform, train=False, download=False)
        num_classes = 555
    elif name == 'SmallNORB':
        from datasets.smallnorb import SmallNORB
        train_dataset = SmallNORB(data_path, transform=train_transform, train=True, download=True)
        val_dataset = SmallNORB(data_path, transform=val_transform, train=False, download=True)
        num_classes = 5
    elif name == 'PCAM':
        from datasets.pcam import PCAM
        train_dataset = PCAM(root=data_path, split='train', transform=train_transform, download=False)
        val_dataset = PCAM(root=data_path, split='test', transform=val_transform, download=False)
        num_classes = 2
    elif name == 'Resisc45' or name == 'AID':
        if not os.path.isfile(os.path.join(data_path, 'train.pkl')) and not os.path.isfile(os.path.join(data_path, 'test.pkl')):
            raise Exception('split file error')

        train_samples = load_pickle(os.path.join(data_path, 'train.pkl'))
        train_samples = [(os.path.join(data_path, i[0]), i[1]) for i in train_samples]
        val_samples = load_pickle(os.path.join(data_path, 'test.pkl'))
        val_samples = [(os.path.join(data_path, i[0]), i[1]) for i in val_samples]
        train_dataset = MyImageFolderDataset(train_samples, transform=train_transform)
        val_dataset = MyImageFolderDataset(val_samples, transform=val_transform)
        if name == 'Resisc45':
            num_classes = 45
        elif name == 'AID':
            num_classes = 30
    elif name == 'dSprites':
        from datasets.dsprites import dSprites
        train_dataset = dSprites(data_path, transform=train_transform)
        val_dataset = dSprites(data_path, transform=val_transform)
        num_classes = -1
    elif name == 'UTKFace':
        from datasets.utkface import UTKDataset
        train_dataset = UTKDataset(data_path, transform=train_transform)
        val_dataset = UTKDataset(data_path, transform=val_transform)
        num_classes = -1
    else:
        raise NotImplementedError

    print(f'Dataset: {name} - [train {len(train_dataset)}] [test {len(val_dataset)}] [num_classes {num_classes}]')
    return train_dataset, val_dataset, num_classes
