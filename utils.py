import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

import os
import time
import json
import random
import pprint
import pickle
import argparse
import scipy.stats
import numpy as np
import os.path as osp
from enum import Enum

import logging
from logging.config import dictConfig

from tensorboardX import SummaryWriter
from collections import defaultdict, OrderedDict

_utils_pp = pprint.PrettyPrinter()


def pprint(x):
    _utils_pp.pprint(x)


def get_command_line_parser():
    parser = argparse.ArgumentParser()
    # global config
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--time_str', type=str, default='')
    parser.add_argument('--log_url', type=str, default='/data/zhangyk/models/implclproto_logs')
    return parser


def save_pickle(file_name, data):
    with open(file_name, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.DEFAULT_PROTOCOL)


def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)


def pairwise_metric(x, y, matching_fn, temperature=1, is_distance=True):
    n_x = x.shape[0]
    n_y = y.shape[0]

    if matching_fn == 'euclidean':
        result_metric = -(
            x.unsqueeze(1).expand(n_x, n_y, *x.shape[1:]) -
            y.unsqueeze(0).expand(n_x, n_y, *x.shape[1:])
        ).pow(2).sum(dim=-1)
        if is_distance:
            result_metric = -result_metric

    elif matching_fn == 'cosine':
        EPSILON = 1e-8
        normalised_x = x / (x.pow(2).sum(dim=-1, keepdim=True).sqrt() + EPSILON)
        normalised_y = y / (y.pow(2).sum(dim=-1, keepdim=True).sqrt() + EPSILON)

        expanded_x = normalised_x.unsqueeze(1).expand(n_x, n_y, -1)
        expanded_y = normalised_y.unsqueeze(0).expand(n_x, n_y, -1)

        result_metric = (expanded_x * expanded_y).sum(dim=-1)
        if is_distance:
            result_metric = 1 - result_metric

    elif matching_fn == 'dot':
        expanded_x = x.unsqueeze(1).expand(n_x, n_y, -1)
        expanded_y = y.unsqueeze(0).expand(n_x, n_y, -1)

        result_metric = (expanded_x * expanded_y).sum(dim=2)
        if is_distance:
            result_metric = -result_metric

    else:
        raise ValueError('Unsupported similarity function')

    return result_metric / temperature


def gpu_state(gpu_id, get_return=False):
    qargs = ['index', 'gpu_name', 'memory.used', 'memory.total']
    cmd = 'nvidia-smi --query-gpu={} --format=csv,noheader'.format(','.join(qargs))

    results = os.popen(cmd).readlines()
    gpu_id_list = gpu_id.split(",")
    gpu_space_available = {}
    for cur_state in results:
        cur_state = cur_state.strip().split(", ")
        for i in gpu_id_list:
            if i == cur_state[0]:
                if not get_return:
                    print(f'GPU {i} {cur_state[1]}: Memory-Usage {cur_state[2]} / {cur_state[3]}.')
                else:
                    gpu_space_available[i] = int("".join(list(filter(str.isdigit, cur_state[3])))) - int("".join(list(filter(str.isdigit, cur_state[2]))))
    if get_return:
        return gpu_space_available


def set_gpu(x, space_hold=1000):
    assert torch.cuda.is_available()
    os.environ['CUDA_VISIBLE_DEVICES'] = x
    torch.backends.cudnn.benchmark = True
    gpu_available = 0
    while gpu_available < space_hold:
        gpu_space_available = gpu_state(x, get_return=True)
        for gpu_id, space in gpu_space_available.items():
            gpu_available += space
        if gpu_available < space_hold:
            gpu_available = 0
            time.sleep(1800)
    gpu_state(x)


def set_seed(seed):
    np.random.seed(seed=seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def nan_assert(x):
    assert torch.any(torch.isnan(x)) == False


class ProtoAverageMeter(object):
    def __init__(self):
        self.avg = None
        self.count = 0

    def update(self, val):
        if len(val) == 0:
            return
        if self.count == 0:
            self.avg = torch.mean(val, dim=0)
        else:
            self.avg = torch.sum(torch.cat([(self.avg.unsqueeze(0) * self.count), val]), dim=0) / (self.count + len(val))
        self.count += len(val)


class TestAugTransform:
    def __init__(self, transform, aug_times):
        self.transform = transform
        self.aug_times = aug_times

    def __call__(self, x):
        return [self.transform(x) for _ in range(self.aug_times)]


class ConfigEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, type):
            return {'$class': o.__module__ + "." + o.__name__}
        elif isinstance(o, Enum):
            return {
                '$enum': o.__module__ + "." + o.__class__.__name__ + '.' + o.name
            }
        elif callable(o):
            return {
                '$function': o.__module__ + "." + o.__name__
            }
        return json.JSONEncoder.default(self, o)


def prepare_parser(parser, temp_args):
    if 'swag' in temp_args.model:
        def add_model_specific_args(parser):
            parser.add_argument('--arch', type=str, default='regnety_16gf')
            parser.add_argument('--pretrained', type=str, default='/data/zhangyk/models')
            return parser
    elif 'esvit' in temp_args.model:
        from esvit.main_import import add_model_specific_args
    elif 'wtimm' in temp_args.model:
        def add_model_specific_args(parser):
            parser.add_argument('--arch', type=str, default='deit_small_patch16_224')
            parser.add_argument('--pretrained', type=str, default='/data/zhangyk/models')
            return parser
    elif temp_args.model in ['mobilenet_v2', 'mnasnet1_0', 'densenet121', 'densenet169', 'densenet201',
                             'resnet34', 'resnet50', 'resnet101', 'resnet152', 'googlenet', 'inception_v3']:
        def add_model_specific_args(parser):
            return parser
    else:
        assert False, f'Unkown model type {temp_args.model}'

    parser = add_model_specific_args(parser)
    return parser


def get_transform(crop_size, resize_transform, normalize_transform, testaug):
    if not testaug:
        return transforms.Compose(
            [
                resize_transform,
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
                normalize_transform
            ]
        )

    cur_transform = transforms.Compose(
        [
            resize_transform,
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            normalize_transform
        ]
    )
    cur_transform = TestAugTransform(cur_transform, 10)
    return cur_transform


def get_hub_transform(resize_transform, normalize_transform):
    return transforms.Compose(
        [
            resize_transform,
            transforms.ToTensor(),
            normalize_transform
        ]
    )


class AverageMeter(object):
    """computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def measure_test(outputs, labels, only_core=True):
    m_results = {}
    m_results['weightedtau'] = scipy.stats.weightedtau(labels, outputs, rank=None).correlation
    m_results['pearsonr'] = scipy.stats.pearsonr(outputs, labels)[0]

    if only_core:
        return m_results


class PrepareFunc(object):
    def __init__(self, args):
        self.args = args

    def prepare_optimizer(self, model):
        def set_optimizer(cur_type, cur_encoder):
            if cur_type == 'Adam':
                return optim.Adam(
                    cur_encoder.parameters(),
                    lr=self.args.lr,
                    # weight_decay=args.weight_decay, # do not use weight_decay here
                    )
            elif cur_type == 'SGD':
                return optim.SGD(
                    cur_encoder.parameters(),
                    lr=self.args.lr,
                    momentum=self.args.momentum,
                    weight_decay=self.args.weight_decay
                    )

        optimizer = set_optimizer(self.args.optimizer, model)

        def set_lr_scheduler(cur_type, optmz):
            if cur_type == 'step':
                return optim.lr_scheduler.StepLR(
                    optmz,
                    step_size=int(self.args.step_size),
                    gamma=self.args.gamma
                    )
            elif cur_type == 'multistep':
                return optim.lr_scheduler.MultiStepLR(
                    optmz,
                    milestones=[int(_) for _ in self.args.step_size.split(',')],
                    gamma=self.args.gamma,
                    )
            elif cur_type == 'cosine':
                return optim.lr_scheduler.CosineAnnealingLR(
                    optmz,
                    self.args.max_epoch,
                    eta_min=self.args.cosine_annealing_lr_eta_min   # a tuning parameter
                    )
            elif cur_type == 'plateau':
                return optim.lr_scheduler.ReduceLROnPlateau(
                    optmz,
                    mode='min',
                    factor=self.args.gamma,
                    patience=5
                    )
            else:
                raise ValueError('No Such Scheduler')

        lr_scheduler = set_lr_scheduler(self.args.lr_scheduler, optimizer)

        return optimizer, lr_scheduler


class Logger(object):
    def __init__(self, args, log_dir, level, **kwargs):
        self.logger_path = osp.join(log_dir, 'scalars.json')
        self.tb_logger = SummaryWriter(
                            logdir=osp.join(log_dir, 'tflogger'),
                            **kwargs,
                            )
        self.log_config(vars(args))

        self.scalars = defaultdict(OrderedDict)

        self.set_logging(level, log_dir)
        logging.info(f'Log at: {log_dir}')

    def add_scalar(self, key, value, counter):
        assert self.scalars[key].get(counter, None) is None, 'counter should be distinct'
        self.scalars[key][counter] = value
        self.tb_logger.add_scalar(key, value, counter)

    def log_config(self, variant_data):
        config_filepath = osp.join(osp.dirname(self.logger_path), 'configs.json')
        with open(config_filepath, "w") as fd:
            json.dump(variant_data, fd, indent=2, sort_keys=True, cls=ConfigEncoder)

    def dump(self):
        with open(self.logger_path, 'w') as fd:
            json.dump(self.scalars, fd, indent=2)

    def set_logging(self, level, work_dir):
        LOGGING = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "simple": {
                    "format": f"%(message)s"
                },
            },
            "handlers": {
                "console": {
                    "level": f"{level}",
                    "class": "logging.StreamHandler",
                    'formatter': 'simple',
                },
                'file': {
                    'level': f"{level}",
                    'formatter': 'simple',
                    'class': 'logging.FileHandler',
                    'filename': f'{work_dir if work_dir is not None else "."}/train.log',
                    'mode': 'a',
                },
            },
            "loggers": {
                "": {
                    "level": f"{level}",
                    "handlers": ["console", "file"] if work_dir is not None else ["console"],
                },
            },
        }
        dictConfig(LOGGING)
        logging.info(f"Log level set to: {level}")



class OnlineDict(object):
    def __init__(self, pkl_file_name):
        self.data = load_pickle(pkl_file_name) if os.path.isfile(pkl_file_name) else {}
        self.pkl_file_name = pkl_file_name

    def update(self, k, v):
        if k not in self.data.keys():
            self.data[k] = v

    def save(self):
        save_pickle(self.pkl_file_name, self.data)

    def add(self, v):
        k = f'c{len(self.data)}'
        self.update(k, v)
        return k

    def get(self, k):
        return self.data.get(k, None)

    def get_keys(self):
        return self.data.keys()
