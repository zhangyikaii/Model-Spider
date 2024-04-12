import torch
from torch.utils.data import Dataset

import os
import random
import pickle
import logging
from copy import deepcopy

from .learnware_info import DATASET2DIR, DATA_SPECIFIC_RANK, BKB_SPECIFIC_RANK, BKB_SPECIFIC_RANK2ID


def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)


class LearnwareDataset(Dataset):
    __heterogeneous_sampled_minnum__ = None
    __heterogeneous_sampled_maxnum__ = None
    __heterogeneous_sampled_fixnum__ = None
    __heterogeneous_prefetch_rank__ = None

    def __init__(self, args, stype, continuous_label=False, samples=None, heterogeneous=False):
        super().__init__()
        LearnwareDataset.__heterogeneous_sampled_minnum__ = args.heterogeneous_sampled_minnum
        LearnwareDataset.__heterogeneous_sampled_maxnum__ = args.heterogeneous_sampled_maxnum
        if samples is None:
            self.samples = []
            if stype == 'train':
                cur_datasets = args.train_dataset
            elif stype == 'test':
                cur_datasets = args.test_dataset
            elif stype == 'val':
                cur_datasets = args.val_dataset
            else:
                raise Exception('stype not in [train, test, val]')
            fixed_gt_samples_num = 0
            for i_dataset in cur_datasets:
                cur_samples, cur_fixed_gt_samples = [], []
                if args.heterogeneous and stype == 'test':
                    cur_path = os.path.join(args.data_url, DATASET2DIR[f'{args.data_sub_url}_test4hete_seed1'][i_dataset])
                else:
                    cur_path = os.path.join(args.data_url, DATASET2DIR[args.data_sub_url][i_dataset])
                for i in os.listdir(cur_path):
                    # i: xxx.pkl
                    cur_ready_path = (os.path.join(cur_path, i), i_dataset)
                    if 'z_' in i:
                        cur_fixed_gt_samples.append(cur_ready_path)
                    # else:
                    #     cur_samples.append(cur_ready_path)

                if stype == 'train' and args.fixed_gt_size_threshold != 0:
                    cur_fixed_gt_samples = random.sample(cur_fixed_gt_samples, min(len(cur_fixed_gt_samples), args.fixed_gt_size_threshold))
                    fixed_gt_samples_num += len(cur_fixed_gt_samples)
                if len(cur_fixed_gt_samples) != 0:
                    cur_samples += cur_fixed_gt_samples
                    random.shuffle(cur_samples)
                if (stype == 'test' or stype == 'val') and args.test_size_threshold != 0:
                    test_copy_samples = deepcopy(cur_samples)
                    random.shuffle(test_copy_samples)
                    cur_samples = test_copy_samples[: args.test_size_threshold]

                self.samples.extend(cur_samples)

            if stype == 'train':
                logging.info(f'Train fixed samples: {fixed_gt_samples_num}')
        else:
            self.samples = samples
        self.continuous_label = continuous_label
        self.prototype_maxnum = args.prototype_maxnum
        self.heterogeneous = heterogeneous
        self.stype = stype

    def __getitem__(self, index):
        """
        return: [num_prototypes, dim], [num_learnware]
        """
        if self.continuous_label:
            cur_discrete_type = 'Finetuning'
        else:
            cur_discrete_type = 'FTRank'

        x = load_pickle(self.samples[index][0])

        def pad_x(cur_x4pad):
            if cur_x4pad.shape[0] < self.prototype_maxnum:
                cur_pad_length = self.prototype_maxnum - cur_x4pad.shape[0]
                cur_x4pad = torch.cat([cur_x4pad, torch.zeros(cur_pad_length, cur_x4pad.shape[1])])
            else:
                cur_pad_length = 0
                cur_x4pad = cur_x4pad[:self.prototype_maxnum]
            return cur_x4pad, cur_pad_length

        if isinstance(x, torch.Tensor):
            ret_x, pad_length = pad_x(x)
            return ret_x, DATA_SPECIFIC_RANK[self.samples[index][1]][cur_discrete_type], self.samples[index][1], pad_length
        elif isinstance(x, list):
            ret_x, pad_length = pad_x(x[0])
            if self.heterogeneous:
                sample_hete = {k: x[1][k] for k in BKB_SPECIFIC_RANK}
                # print(self.samples[index][0], [len(sample_hete[ii]) for ii in sample_hete.keys()])
                ret_x = (ret_x, sample_hete)
            if len(x) == 3:
                return ret_x, x[2] if self.continuous_label else x[2].to(torch.long), self.samples[index][1], pad_length
            elif len(x) == 2:
                # z_xxx.pkl：
                return ret_x, DATA_SPECIFIC_RANK[self.samples[index][1]][cur_discrete_type], self.samples[index][1], pad_length

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def collate_fn(batch):
        x_uni_hete, cur_rank, dataset_name, pad_length = zip(*batch)
        ret_x = []
        ret_hete_x = {}  # {bkb: list}
        ret_hete_x_indices = {}  # {bkb: list}
        ret_batchid2bkbid = {}

        if LearnwareDataset.__heterogeneous_sampled_fixnum__ is None:
            heterogeneous_sampled_num = random.randint(LearnwareDataset.__heterogeneous_sampled_minnum__, LearnwareDataset.__heterogeneous_sampled_maxnum__)
        else:
            heterogeneous_sampled_num = LearnwareDataset.__heterogeneous_sampled_fixnum__

        for idx, (x_uni, x_hete) in enumerate(x_uni_hete):
            ret_x.append(x_uni)
            if LearnwareDataset.__heterogeneous_prefetch_rank__ is not None:
                hete_keys = LearnwareDataset.__heterogeneous_prefetch_rank__[dataset_name[idx]][:heterogeneous_sampled_num]
            else:
                hete_keys = random.sample(list(x_hete.keys()), heterogeneous_sampled_num)
            if len(hete_keys) == 0:
                ret_batchid2bkbid[idx] = None
                continue
            ret_batchid2bkbid[idx] = [BKB_SPECIFIC_RANK2ID[cur_hete_k] for cur_hete_k in hete_keys]

            for bkbid, bkb_k in zip(ret_batchid2bkbid[idx], hete_keys):
                if bkb_k not in ret_hete_x.keys():
                    ret_hete_x[bkb_k] = [x_hete[bkb_k]]
                else:
                    ret_hete_x[bkb_k].append(x_hete[bkb_k])
                if bkb_k not in ret_hete_x_indices.keys():
                    ret_hete_x_indices[bkb_k] = [(idx, bkbid)]
                else:
                    ret_hete_x_indices[bkb_k].append((idx, bkbid))
        # print('k', [len(ret_hete_x[iii]) for iii in ret_hete_x.keys()])

        return (torch.stack(ret_x), (ret_hete_x, ret_hete_x_indices), ret_batchid2bkbid), torch.stack(cur_rank), dataset_name, torch.tensor(pad_length)
