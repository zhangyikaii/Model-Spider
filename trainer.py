import torch
import os
import random

import logging
import hashlib
import datetime
from copy import deepcopy
from pathlib import Path

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from learnware.learnware_model import LearnwareCAHeterogeneous
from learnware.learnware_loss import HierarchicalCE
from learnware.learnware_dataset import LearnwareDataset
from trainer_utils import PrepareFunc, pprint, Logger, set_seed, save_pickle, set_gpu, nan_assert, get_command_line_parser, measure_test
from learnware.learnware_info import DATASET2NUM_CLASSES, DATA_SUB_URL2DIM, BKB_SPECIFIC_RANK

torch.multiprocessing.set_sharing_strategy('file_system')

class Trainer(object):
    def parse_trainer_args(parser):
        # dataset config
        parser.add_argument('--train_dataset', nargs='*', type=str, default=None)
        parser.add_argument('--val_dataset', nargs='*', type=str, default=None)
        parser.add_argument('--test_dataset', nargs='*', type=str, default=None)
        parser.add_argument('--test_size_threshold', type=int, default=1024)
        parser.add_argument('--val_ratio', type=float, default=0.2)
        parser.add_argument('--dataset_size_threshold', type=int, default=0)
        parser.add_argument('--data_sub_url', type=str, default='swin_base_7_checkpoint')
        parser.add_argument('--data_url', type=str, default='')

        # train config
        parser.add_argument('--max_epoch', type=int, default=50)
        parser.add_argument('--batch_size', type=int, default=128)
        parser.add_argument('--lr', type=float, default=0.01)
        parser.add_argument('--optimizer', type=str, default='Adam')
        parser.add_argument('--lr_scheduler', type=str, default='cosine')
        parser.add_argument('--cosine_annealing_lr_eta_min', type=float, default=5e-6)
        parser.add_argument('--weight_decay', type=float, default=0.00005)
        parser.add_argument('--momentum', type=float, default=0.8)
        parser.add_argument('--num_workers', type=int, default=8)
        # parser.add_argument('--metric', type=str, default='euclidean')
        # parser.add_argument('--temperature', type=float, default=1)
        # parser.add_argument('--forget_rate', type=float, default=0)
        # parser.add_argument('--testaug', action='store_true', default=False)

        # model spider config
        parser.add_argument('--pretrained_url', type=str, default=None)
        parser.add_argument('--num_prototypes', type=int, default=None)
        parser.add_argument('--num_learnware', type=int, default=72)

        parser.add_argument('--fixed_gt_size_threshold', type=int, default=128)
        parser.add_argument('--attn_pool', type=str, default='cls')

        parser.add_argument('--heterogeneous', action='store_true', default=False)
        parser.add_argument('--heterogeneous_sampled_minnum', type=int, default=0)
        parser.add_argument('--heterogeneous_sampled_maxnum', type=int, default=10)
        parser.add_argument('--heterogeneous_prompt', action='store_true', default=False)
        parser.add_argument('--heterogeneous_extra_prompt', action='store_true', default=False)
        return parser

    def __init__(self, args):
        if args.time_str == '':
            args.time_str = datetime.datetime.now().strftime('%m%d-%H-%M-%S-%f')[:-3]

        train_dataset_str = '_'.join(sorted(args.train_dataset))
        if len(train_dataset_str) > 64:
            train_dataset_str = hashlib.sha1(bytes(train_dataset_str, encoding='utf8')).hexdigest()
        if args.val_dataset is None:
            args.setting_str = f"{train_dataset_str}__{'_'.join(sorted(args.test_dataset))}"
        else:
            val_dataset_str = '_'.join(sorted(args.val_dataset))
            if len(val_dataset_str) > 16:
                val_dataset_str = hashlib.sha1(bytes(val_dataset_str, encoding='utf8')).hexdigest()
            args.setting_str = f"{train_dataset_str}__{val_dataset_str}__{'_'.join(sorted(args.test_dataset))}"

        args.prototype_maxnum = max([DATASET2NUM_CLASSES[i] for i in (args.train_dataset + args.test_dataset)])
        args.prototype_maxnum_hete = max([DATASET2NUM_CLASSES[i] for i in args.train_dataset])
        self.prototype_maxnum_hete = args.prototype_maxnum_hete
        if args.heterogeneous:
            args.dim = DATA_SUB_URL2DIM['heterogeneous']
        else:
            args.dim = DATA_SUB_URL2DIM[args.data_sub_url]

        pprint(vars(args))

        self.log_handle = LogHandle(args)

        self.model = LearnwareCAHeterogeneous(
            num_learnware=args.num_learnware,
            dim=args.dim,
            hdim=args.dim,
            uni_hete_proto_dim=(args.prototype_maxnum, args.prototype_maxnum_hete),
            data_sub_url=args.data_sub_url,
            pool=args.attn_pool,
            heads=1,
            dropout=0.1,
            emb_dropout=0.1,
            heterogeneous_extra_prompt=args.heterogeneous_extra_prompt
        )
        self.model = self.model.to(torch.device('cuda'))

        trainval_dataset = LearnwareDataset(args=args, stype='train', heterogeneous=args.heterogeneous)
        trainval_samples = deepcopy(trainval_dataset.samples)
        random.shuffle(trainval_samples)
        if args.dataset_size_threshold != 0:
            trainval_samples = trainval_samples[: args.dataset_size_threshold]

        train_dataset = LearnwareDataset(args=args, stype='train', samples=trainval_samples, heterogeneous=args.heterogeneous)
        val_dataset = LearnwareDataset(args=args, stype='val', heterogeneous=args.heterogeneous)
        test_dataset = LearnwareDataset(args=args, stype='test', heterogeneous=args.heterogeneous)

        self.data_loader_train = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            shuffle=True,
            collate_fn=LearnwareDataset.collate_fn if args.heterogeneous else None
        )
        self.data_loader_val = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            shuffle=False,
            collate_fn=LearnwareDataset.collate_fn if args.heterogeneous else None
            )
        self.data_loader_test = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            shuffle=False,
            collate_fn=LearnwareDataset.collate_fn if args.heterogeneous else None
        )

        prepare_handle = PrepareFunc(args)
        self.optimizer, self.lr_scheduler = prepare_handle.prepare_optimizer(self.model)
        self.criterion = HierarchicalCE(args.num_learnware)
        logging.info(f'Size: Train [{len(train_dataset)}], Val [{len(val_dataset)}], Test [{len(test_dataset)}]')

        self.best_state = {}

        prompt_num = 1
        self.attn_pad_mask_dict = torch.tensor([
            [0 for _ in range(args.prototype_maxnum + prompt_num - i)] + [1 for _ in range(i)] for i in range(args.prototype_maxnum + prompt_num)
            ]).bool()
        base_attn_pad_mask_hete = torch.tensor([
            [0 for _ in range(self.prototype_maxnum_hete - i)] + [1 for _ in range(i)] for i in range(self.prototype_maxnum_hete + 1)
            ]).bool()  # [self.prototype_maxnum_hete + 1, self.prototype_maxnum_hete]
        base_attn_pad_mask_hete_zero = torch.zeros_like(base_attn_pad_mask_hete)

        def get_hete_k_mask(heterogeneous_sampled_num, hete_pad_length, hete_order_idx):
            cur_hete_k_mask = [base_attn_pad_mask_hete_zero for _ in range(heterogeneous_sampled_num)]
            cur_hete_k_mask[hete_order_idx] = base_attn_pad_mask_hete
            cur_hete_k_mask = torch.cat(cur_hete_k_mask, dim=-1)
            return cur_hete_k_mask

        self.attn_pad_mask_hete_k = get_hete_k_mask

        cur_attn_pad_mask_hete_range = list(range(max(1, args.heterogeneous_sampled_minnum), args.heterogeneous_sampled_maxnum + 1))
        if 1 not in cur_attn_pad_mask_hete_range:
            cur_attn_pad_mask_hete_range.append(1)

        self.attn_pad_mask_hete_dict = {i: base_attn_pad_mask_hete.repeat(1, i) for i in cur_attn_pad_mask_hete_range}

        self.do_train = True
        if args.pretrained_url is not None:
            state_dict = torch.load(args.pretrained_url)
            self.model.load_state_dict(state_dict['model'])
            logging.info(f'Loading model for test: [time_str {state_dict["time_str"]}], [epoch {state_dict["epoch"]}]')
            self.do_train = False

        self.args = args

    # Function to get an attention pad mask for a given pad_length
    def get_attn_pad_mask(self, pad_length):
        if self.args.batch_size == 1:
            return None
        pad_attn_mask = self.attn_pad_mask_dict[pad_length].unsqueeze(1)
        pad_attn_mask = pad_attn_mask.repeat(1, pad_attn_mask.shape[-1], 1)
        pad_attn_mask = pad_attn_mask | pad_attn_mask.transpose(-2, -1)
        return pad_attn_mask.to(torch.device('cuda'))

    # Function to get an attention pad mask for heterogeneous data
    def get_attn_pad_hete_mask(self, pad_length, hete_pad_length, heterogeneous_sampled_num):
        if self.args.batch_size == 1:
            return None
        # If heterogeneous_sampled_num is not 0, concatenate masks from both sources
        if heterogeneous_sampled_num != 0:
            # self.attn_pad_mask_dict[pad_length]: [batch_size, dim]
            pad_attn_mask = torch.cat([
                self.attn_pad_mask_dict[pad_length], self.attn_pad_mask_hete_dict[heterogeneous_sampled_num][hete_pad_length]
                ], dim=1).unsqueeze(1)
        else:
            # Otherwise, use the mask from the first source
            pad_attn_mask = self.attn_pad_mask_dict[pad_length].unsqueeze(1)

        # Repeat the pad_attn_mask along the specified dimensions
        pad_attn_mask = pad_attn_mask.repeat(1, pad_attn_mask.shape[-1], 1)
        pad_attn_mask = pad_attn_mask | pad_attn_mask.transpose(-2, -1)

        return pad_attn_mask.to(torch.device('cuda'))

    # Wrapper function to create a function for getting heterogeneous attention pad masks
    def hete_attn_pad_func(self, pad_length):
        def core(hete_pad_length, heterogeneous_sampled_num):
            return self.get_attn_pad_hete_mask(pad_length, hete_pad_length, heterogeneous_sampled_num)
        return core

    # Function for preprocessing heterogeneous inputs
    def preprocess_hete_inputs(self, inputs):
        cur_batch_size = inputs[0].shape[0]
        # Move the uniform input tensor to the 'cuda' device and apply a linear layer
        x_uni = inputs[0].to(torch.device('cuda'))
        x_uni = self.model.uni_linear(x_uni)
        # Split the heterogeneous inputs into x_hete (a dictionary of heterogeneous features) and x_hete_indices
        x_hete, x_hete_indices = inputs[1]  # x_hete: dict of {bkb: [list of features]}

        # If there are no heterogeneous features, return None values
        if len(x_hete) == 0 and len(x_hete_indices) == 0:
            return x_uni, None, None, 0

        # List to store intermediate heterogeneous inputs
        hete_mid_inputs = []

        # Process each batch of heterogeneous data
        for i_bkb in x_hete.keys():
            # Calculate the split dimensions for each feature
            split_dims = [cur.shape[0] for cur in x_hete[i_bkb]]
            # Concatenate the features within each batch for a specific feature space
            cur_linear_inputs = torch.cat(x_hete[i_bkb]).to(torch.device('cuda'))
            # Apply a linear layer to the concatenated features
            cur_hete_inputs = self.model.hete_linears[i_bkb](cur_linear_inputs)
            # Split the results back based on batch_index information
            cur_hete_inputs = torch.split(cur_hete_inputs, split_dims)
            # Create a dictionary mapping batch indices to heterogeneous inputs
            batchid2hete_inputs = {bid[0]: {'inp': hinp, 'bkbid': bid[1]} for bid, hinp in zip(x_hete_indices[i_bkb], cur_hete_inputs)}
            hete_mid_inputs.append(batchid2hete_inputs)

        # Initialize dictionaries and lists to organize the heterogeneous inputs
        # hete_mid_inputs：[dict: {batchid: (bkbid, cur_hete_inputs)}]
        hete_inputs = {i_batchid: [] for i_batchid in range(cur_batch_size)}
        x_hete_inp = {}
        hete_inputs_pad_length = {}
        hete_inputs_bkbid = {}

        # Organize the intermediate results
        for cur_bkb_hete_inputs in hete_mid_inputs:
            for cur_batchid in cur_bkb_hete_inputs.keys():
                hete_inputs[cur_batchid].append(cur_bkb_hete_inputs[cur_batchid])

        # Determine the number of heterogeneous samples within each batch
        cur_batch_heterogeneous_sampled_num = len(hete_inputs[0])

        # Align the prototype dimensions for the heterogeneous data
        for cur_batchid in hete_inputs.keys():
            anchor_bfeat = hete_inputs[cur_batchid][0]['inp']
            if anchor_bfeat.shape[0] < self.prototype_maxnum_hete:
                cur_hete_pad_length = self.prototype_maxnum_hete - anchor_bfeat.shape[0]
                cur_bfeat_pad = torch.zeros(cur_hete_pad_length, anchor_bfeat.shape[1]).to(torch.device('cuda'))  # [pad_length x feat_dim]
                cur_bfeat = torch.cat([
                    torch.stack([cur['inp'] for cur in hete_inputs[cur_batchid]]),
                    cur_bfeat_pad.repeat(cur_batch_heterogeneous_sampled_num, 1, 1)
                    ], dim=1)  # [heterogeneous_sampled_num (i.e., len(hete_inputs[cur_batchid])), prototype_length + pad_length, feat_dim]

            else:
                cur_hete_pad_length = 0
                cur_bfeat = torch.stack([cur['inp'][:self.prototype_maxnum_hete] for cur in hete_inputs[cur_batchid]])

            hete_inputs_bkbid[cur_batchid] = [cur['bkbid'] for cur in hete_inputs[cur_batchid]]  # collate_fn
            x_hete_inp[cur_batchid] = cur_bfeat  # [prototype_length, heterogeneous_sampled_num * feat_dim]
            hete_inputs_pad_length[cur_batchid] = cur_hete_pad_length

        # Align the prototype dimensions for the heterogeneous data
        hete_pad_length = torch.tensor([hete_inputs_pad_length[cur_batchid] for cur_batchid in range(cur_batch_size)])

        # Organize and align the prompt data
        prompt2hete = {}
        anchor_hete = x_hete_inp[0]
        prompt2hete_base_zero = [torch.zeros(anchor_hete.shape[1], anchor_hete.shape[2]).to(torch.device('cuda')) for _ in range(cur_batch_size)]
        hete_pad_length_base = torch.ones_like(hete_pad_length) * self.prototype_maxnum_hete
        prompt_id2hete_pad_length = {i_prompt: deepcopy(hete_pad_length_base) for i_prompt in range(self.args.num_learnware)}

        # Populate prompt2hete with aligned data
        for i_prompt in range(self.args.num_learnware):
            prompt2hete[i_prompt] = deepcopy(prompt2hete_base_zero)
        for i_batchid in range(cur_batch_size):
            for i_p_idx, i_prompt in enumerate(hete_inputs_bkbid[i_batchid]):
                prompt2hete[i_prompt][i_batchid] = x_hete_inp[i_batchid][i_p_idx]
                prompt_id2hete_pad_length[i_prompt][i_batchid] = hete_pad_length[i_batchid]

        # Stack the prompt data
        for i_prompt in prompt2hete.keys():
            prompt2hete[i_prompt] = torch.stack(prompt2hete[i_prompt])

        return x_uni, prompt2hete, prompt_id2hete_pad_length, cur_batch_heterogeneous_sampled_num

    def fit(self):
        if not self.do_train:
            epoch = 0
            # Test for k=0
            LearnwareDataset.__heterogeneous_sampled_fixnum__ = 0
            mt_raw_results, mt_hete_results = self.test(epoch)

            for i_mdataset in mt_raw_results.keys():
                # sorted_rank = torch.sort(torch.mean(mt_raw_results[i_mdataset], dim=0))[1].tolist()
                sorted_rank = torch.mean(mt_raw_results[i_mdataset], dim=0).tolist()
                sorted_bkb = sorted(BKB_SPECIFIC_RANK, key=lambda x: sorted_rank[BKB_SPECIFIC_RANK.index(x)], reverse=True)
                mt_raw_results[i_mdataset] = sorted_bkb

            # Test for k in [1, args.heterogeneous_sampled_maxnum]
            LearnwareDataset.__heterogeneous_prefetch_rank__ = mt_raw_results
            for sample_num in range(1, self.args.heterogeneous_sampled_maxnum + 1):
                LearnwareDataset.__heterogeneous_sampled_fixnum__ = sample_num
                _, cur_mt_hete_results = self.test(epoch)
                for key_dataset in cur_mt_hete_results.keys():
                    mt_hete_results[key_dataset][sample_num] = cur_mt_hete_results[key_dataset][sample_num]
            
            # logging
            sample_num_averages = {}
            for key_dataset, sample_results in mt_hete_results.items():
                for sample_num, result in sample_results.items():
                    sample_num_averages[sample_num] = sample_num_averages.get(sample_num, []) + [result]
            sample_num_averages = {sample_num: sum(results) / len(results) for sample_num, results in sample_num_averages.items()}
            max_sample_num = max(sample_num_averages, key=sample_num_averages.get)
            LearnwareDataset.__heterogeneous_sampled_fixnum__ = max_sample_num

            mt_raw_results, _ = self.test(epoch)
            print(f'Model Spider\'s scores on {BKB_SPECIFIC_RANK}')
            for i in mt_raw_results.keys():
                print(f'{i}:', list(mt_raw_results[i][0].detach().cpu().numpy()))
            print("best heterogeneous_sample_num:", max_sample_num)
            for key_dataset in cur_mt_hete_results.keys():
                print(f"wtau of {key_dataset}:", mt_hete_results[key_dataset][max_sample_num])
            return
  
        for epoch in range(1, self.args.max_epoch + 1):
            self.model.train()
            logging.info(f'[{epoch} / {self.args.max_epoch}] lr: {self.lr_scheduler.get_last_lr()[0]:.5f}')
            losses = []

            for step, (inputs, labels, labels_dataset, pad_length) in enumerate(self.data_loader_train):
                labels = labels.to(torch.device('cuda'))

                x_uni, x_hete, prompt_id2hete_pad_length, _ = self.preprocess_hete_inputs(inputs)

                if x_hete is not None:
                    outputs = self.model(x_uni, x_hete, prompt_id2hete_pad_length,
                                         attn_mask_func=self.hete_attn_pad_func(pad_length))
                else:
                    pad_attn_mask = self.get_attn_pad_mask(pad_length)
                    outputs = self.model(x_uni, x_hete, pad_attn_mask)

                loss = self.criterion(outputs, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                nan_assert(loss)
                losses.append(loss.item())

            self.log_handle.add_scalar('epoch_loss', sum(losses) / len(losses), epoch)
            self.lr_scheduler.step()

            # Test for k=0
            LearnwareDataset.__heterogeneous_sampled_fixnum__ = 0
            mt_raw_results, mt_hete_results = self.test(epoch)

            for i_mdataset in mt_raw_results.keys():
                # sorted_rank = torch.sort(torch.mean(mt_raw_results[i_mdataset], dim=0))[1].tolist()
                sorted_rank = torch.mean(mt_raw_results[i_mdataset], dim=0).tolist()
                sorted_bkb = sorted(BKB_SPECIFIC_RANK, key=lambda x: sorted_rank[BKB_SPECIFIC_RANK.index(x)], reverse=True)
                mt_raw_results[i_mdataset] = sorted_bkb

            # Test for k in [1, args.heterogeneous_sampled_maxnum]
            LearnwareDataset.__heterogeneous_prefetch_rank__ = mt_raw_results
            for sample_num in range(1, self.args.heterogeneous_sampled_maxnum + 1):
                LearnwareDataset.__heterogeneous_sampled_fixnum__ = sample_num
                _, cur_mt_hete_results = self.test(epoch)
                for key_dataset in cur_mt_hete_results.keys():
                    mt_hete_results[key_dataset][sample_num] = cur_mt_hete_results[key_dataset][sample_num]

            # logging
            cur_hete_log_contents = []
            for k_dataset in mt_hete_results.keys():
                cur_content = f'{k_dataset},'
                cur_content += ','.join([f'{mt_hete_results[k_dataset][i_num]:.3f}' for i_num in range(self.args.heterogeneous_sampled_minnum, self.args.heterogeneous_sampled_maxnum + 1)])
                cur_hete_log_contents.append(cur_content)
            self.log_handle.log_per_epoch(epoch, cur_hete_log_contents, 'heterogeneous_sampled_acc.csv')

            self.log_handle.save_model(self.model, self.optimizer, epoch, f'{epoch}.pth')

            LearnwareDataset.__heterogeneous_prefetch_rank__ = None
            LearnwareDataset.__heterogeneous_sampled_fixnum__ = None
            LearnwareDataset.__heterogeneous_prefetch_rank__ = None

        # self.log2unified()
        # self.log_handle.log_pickle(self.best_state, 'unified_results.pkl')

    def log2unified(self):
        content = f'{self.args.time_str},{self.args.setting_str},'
        content += f'{self.args.lr},{self.args.optimizer},{self.args.weight_decay},{self.args.momentum},'
        for i_mdataset in self.best_state.keys():
            if len(self.best_state[i_mdataset]) == 0:
                continue
            for i_mt in self.best_state[i_mdataset].keys():
                # if 'num' in i_mt:
                #     continue
                cur_keys = sorted(self.best_state[i_mdataset][i_mt].keys())
                content += f'{i_mt},'
                for i_key in cur_keys:
                    content += f'{i_key}_{i_mdataset},{self.best_state[i_mdataset][i_mt][i_key]:.3f},'
        content = content[:-1] + '\n'

        with open('./ulearnware_unified_results.csv', 'a') as f:
            f.write(content)

    def test(self, epoch, control_saved=[]):
        stype = 'test'
        self.model.eval()
        with torch.no_grad():
            mt_results = {k: {} for k in self.args.test_dataset}
            mt_hete = {k: [] for k in self.args.test_dataset}
            mt_raw_results = {k: [] for k in self.args.test_dataset}
            mt_hete_results = {k: {i_num: -2 for i_num in range(self.args.heterogeneous_sampled_minnum, self.args.heterogeneous_sampled_maxnum + 1)} for k in self.args.test_dataset}
            for i, (inputs, labels, labels_dataset, pad_length) in enumerate(self.data_loader_test):
                x_uni, x_hete, prompt_id2hete_pad_length, cur_batch_heterogeneous_sampled_num = self.preprocess_hete_inputs(inputs)

                if x_hete is not None:
                    outputs = self.model(x_uni, x_hete, prompt_id2hete_pad_length, attn_mask_func=self.hete_attn_pad_func(pad_length))
                else:
                    pad_attn_mask = self.get_attn_pad_mask(pad_length)
                    outputs = self.model(x_uni, x_hete, pad_attn_mask)

                rankings = torch.zeros_like(outputs).long().scatter_(1, torch.sort(outputs, dim=-1)[1].to(torch.device('cuda')), torch.arange(outputs.shape[1]).repeat(outputs.shape[0], 1).to(torch.device('cuda')))  # Converting continuous numerical values into rankings, where smaller values receive lower rankings.

                # print(rankings.shape)
                for i_rank in range(rankings.shape[0]):
                    mt_raw_results[labels_dataset[i_rank]].append(outputs[i_rank])
                    cur_mt = measure_test(outputs=rankings[i_rank].detach().cpu().numpy(), labels=labels[i_rank].detach().cpu().numpy())
                    mt_hete[labels_dataset[i_rank]].append(cur_batch_heterogeneous_sampled_num)
                    for i_mt in cur_mt.keys():
                        if i_mt not in mt_results[labels_dataset[i_rank]].keys():
                            mt_results[labels_dataset[i_rank]][i_mt] = [cur_mt[i_mt]]
                        else:
                            mt_results[labels_dataset[i_rank]][i_mt].append(cur_mt[i_mt])

            for k_dataset in mt_results.keys():
                k_metric = 'weightedtau'
                # for keys in mt_hete_results, generate metric in mt_results[k_dataset][k_metric] and mt_hete[k_dataset]
                for k_mt_hete in set(mt_hete[k_dataset]):
                    cur_mt_result = torch.tensor(mt_results[k_dataset][k_metric])
                    cur_hete_num = torch.tensor(mt_hete[k_dataset])
                    mt_hete_results[k_dataset][k_mt_hete] = torch.mean(cur_mt_result[cur_hete_num == k_mt_hete]).item()

            # mt_hete_results is test results for each epoch
            cur_hete_log_contents = []
            for k_dataset in mt_results.keys():
                cur_content = f'{k_dataset},'
                cur_content += ','.join([f'{mt_hete_results[k_dataset][i_num]:.3f}' for i_num in range(self.args.heterogeneous_sampled_minnum, self.args.heterogeneous_sampled_maxnum + 1)])
                cur_hete_log_contents.append(cur_content)
            # self.log_handle.log_per_epoch(epoch, cur_hete_log_contents, 'heterogeneous_sampled_acc.csv')

            for i_mt_1 in mt_results.keys():
                for i_mt_2 in mt_results[i_mt_1].keys():
                    mt_results[i_mt_1][i_mt_2] = sum(mt_results[i_mt_1][i_mt_2]) / len(mt_results[i_mt_1][i_mt_2])

            for i_mdataset in mt_results.keys():
                if i_mdataset not in self.best_state.keys():
                    self.best_state[i_mdataset] = {}
                for i_mt in mt_results[i_mdataset].keys():
                    cur_mt_result = mt_results[i_mdataset]
                    cur_best_key = f'best_{stype}_{i_mt}'

                    if cur_best_key not in self.best_state[i_mdataset].keys() or (cur_best_key in self.best_state[i_mdataset].keys() and self.best_state[i_mdataset][cur_best_key][i_mt] < cur_mt_result[i_mt]):
                        logging.info(f'[{epoch}] {i_mdataset} {stype}\t{i_mt}\t{mt_results[i_mdataset]["weightedtau"]:.4f}\t{mt_results[i_mdataset]["pearsonr"]:.4f}')
                        self.best_state[i_mdataset][cur_best_key] = {
                            'epoch': epoch,
                            **mt_results[i_mdataset]
                            }

        for i_mdataset in mt_raw_results.keys():
            mt_raw_results[i_mdataset] = torch.stack(mt_raw_results[i_mdataset])
        return mt_raw_results, mt_hete_results


class LogHandle(object):
    def __init__(self, args):
        self.args = args
        self.save_path = Path(os.path.join(args.log_url, args.setting_str, args.time_str))
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.logger = Logger(args, self.save_path, 'INFO')

    def save_model(self, model, optimizer, epoch, save_file='best.pth'):
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'time_str': self.args.time_str
            }
        torch.save(state, os.path.join(self.save_path, save_file))

    def log_per_epoch(self, epoch, contents, save_file):
        with open(os.path.join(self.save_path, save_file), 'a') as f:
            for i in contents:
                f.write(f'{epoch},{i}\n')

    def add_scalar(self, key, value, counter):
        self.logger.add_scalar(key, value, counter)

    def log_pickle(self, data, file_name):
        save_pickle(os.path.join(self.save_path, file_name), data)


def main():
    parser = get_command_line_parser()
    parser = Trainer.parse_trainer_args(parser)
    args = parser.parse_args()

    set_gpu(args.gpu)
    set_seed(args.seed)
    torch.cuda.set_device(int(args.gpu))
    torch.autograd.set_detect_anomaly(True)

    trainer = Trainer(args)
    trainer.fit()
    # trainer.test()


if __name__ == "__main__":
    main()
