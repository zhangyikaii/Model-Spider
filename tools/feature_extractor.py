import os
import argparse
import datetime
import numpy as np
from pathlib import Path
from tqdm import tqdm
from copy import deepcopy
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from trainer_utils import (
    get_command_line_parser,
    pprint,
    set_gpu,
    set_seed,
    log_config,
    prepare_parser,
    get_hub_transform,
    get_transform,
)
from learnware.learnware_info import DATA_PATHS, MODEL2FEAT_DIM, DATASET2NUM_CLASSES
from datasets.load_dataset import get_dataset


# Define a dictionary to map model names to the corresponding final classifier layer
MODEL_FC_LAYERS = {
    'mobilenet_v2': 'classifier[-1]',
    'mnasnet1_0': 'classifier[-1]',
    'densenet121': 'classifier',
    'densenet169': 'classifier',
    'densenet201': 'classifier',
    'resnet34': 'fc',
    'resnet50': 'fc',
    'resnet101': 'fc',
    'resnet152': 'fc',
    'googlenet': 'fc',
    'inception_v3': 'fc',
}

def get_model(args, **kwargs):
    transform_kwargs = {}
    transform_kwargs['normalize_transform'] = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    transform_kwargs['resize_transform'] = transforms.Resize(256)
    transform_kwargs['crop_size'] = 224

    fit_kwargs = {}

    if 'esvit' in args.model:
        assert args.uarch[args.uarch.find('_') + 1:] in args.pretrained_weights
        from esvit.main_import import get_model
        model, depths = get_model(args)
        model = model.to(torch.device('cuda'))
        fit_kwargs = {'n': args.n_last_blocks, 'avgpool': args.avgpool_patchtokens, 'depths': depths}

    elif args.model in ['mobilenet_v2', 'mnasnet1_0', 'densenet121', 'densenet169', 'densenet201',
                        'resnet34', 'resnet50', 'resnet101', 'resnet152', 'googlenet', 'inception_v3'] or 'FTD_' in args.model:
        import torchvision.models as models

        assert 'load_url_pretrained' in kwargs.keys() and 'load_manual_pretrained' in kwargs.keys()

        # FTD_resnet50_DTD
        cur_backbone = args.model[4: args.model.rfind('_')] if 'FTD_' in args.model else args.model
        num_classes = DATASET2NUM_CLASSES[args.dataset]
        load_url_pretrained = kwargs['load_url_pretrained']
        load_manual_pretrained = kwargs['load_manual_pretrained']  # True：only for resnet50 densenet201 inception_v3

        if load_url_pretrained and cur_backbone == 'inception_v3':
            model = models.__dict__[cur_backbone](pretrained=load_url_pretrained, num_classes=1000, aux_logits=False)
        elif cur_backbone != 'inception_v3':
            model = models.__dict__[cur_backbone](pretrained=load_url_pretrained, num_classes=1000)
        else:
            model = None  # inception_v3: if not load_url_pretrained, manual construct

        if load_manual_pretrained:
            # must random head, not for LogME
            cached_pretrained_weights = {
                'resnet50': os.path.join(args.pretrained, 'resnet50-0676ba61.pth'),
                'densenet201': os.path.join(args.pretrained, 'densenet201-c1103571.pth'),
                'inception_v3': os.path.join(args.pretrained, 'inception_v3_google-0cc3c7bd.pth'),
            }

            if cur_backbone == 'densenet201':
                import re
                pattern = re.compile(
                    r"^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$"
                )

                state_dict = torch.load(cached_pretrained_weights[cur_backbone], map_location=None)
                for key in list(state_dict.keys()):
                    res = pattern.match(key)
                    if res:
                        new_key = res.group(1) + res.group(2)
                        state_dict[new_key] = state_dict[key]
                        del state_dict[key]
                model.load_state_dict(state_dict)

                model.classifier = nn.Linear(MODEL2FEAT_DIM[cur_backbone], num_classes)
                model.classifier.weight.requires_grad = True
                model.classifier.bias.requires_grad = True
                nn.init.constant_(model.classifier.bias, 0)

            elif cur_backbone == 'inception_v3':
                from torchvision.models.inception import Inception3
                inception_kwargs = {'aux_logits': False}
                if "transform_input" not in inception_kwargs:
                    inception_kwargs["transform_input"] = True
                if "aux_logits" in inception_kwargs:
                    original_aux_logits = inception_kwargs["aux_logits"]
                    inception_kwargs["aux_logits"] = True
                else:
                    original_aux_logits = True
                inception_kwargs["init_weights"] = False  # we are loading weights from a pretrained model
                model = Inception3(**inception_kwargs)
                state_dict = torch.load(cached_pretrained_weights[cur_backbone], map_location=None)
                model.load_state_dict(state_dict)
                if not original_aux_logits:
                    model.aux_logits = False
                    model.AuxLogits = None

                model.fc = nn.Linear(MODEL2FEAT_DIM[cur_backbone], num_classes)
                model.fc.weight.requires_grad = True
                model.fc.bias.requires_grad = True
                stddev = float(model.fc.stddev) if hasattr(model.fc, "stddev") else 0.1  # type: ignore
                nn.init.trunc_normal_(model.fc.weight, mean=0.0, std=stddev, a=-2, b=2)

            elif cur_backbone == 'resnet50':
                model.load_state_dict(torch.load(cached_pretrained_weights[cur_backbone], map_location=None))

                model.fc = nn.Linear(MODEL2FEAT_DIM[cur_backbone], num_classes)
                model.fc.weight.requires_grad = True
                model.fc.bias.requires_grad = True
            else:
                raise Exception

            if 'FTD_' in args.model:
                state_dict = torch.load(args.init_weights, map_location=None)
                state_dict = {k: state_dict['model'][k] for k in state_dict['model'].keys() if 'fc' not in k and 'classifier' not in k}
                cur_state_dict = model.state_dict()
                cur_state_dict.update(state_dict)
                model.load_state_dict(cur_state_dict)

        model = model.to(torch.device('cuda'))

    else:
        assert False, f'Unkown model type {args.model}'

    total = sum([param.nelement() for param in model.parameters()])
    print("All parameters: %.2fM" % (total/1e6))
    return model, transform_kwargs, fit_kwargs


class FeatureExtractor(object):
    def parse_extrator_args(parser):
        # feature extractor config
        parser.add_argument('--model', type=str, default='None')
        parser.add_argument('--model_hub', nargs='+', default=['mobilenet_v2'])
        parser.add_argument('--dataset', type=str, default='ImageNet')
        parser.add_argument('--batch_size', type=int, default=128)
        parser.add_argument('--num_workers', type=int, default=8)

        parser.add_argument('--pretrained', type=str, default='/home/zhangyk/.cache/torch/hub/checkpoints')
        parser.add_argument('--save_url', type=str, default='.')

        parser.add_argument('--uarch', type=str, default='None')
        parser.add_argument('--rk_methods', nargs='+', default=['LogME'])
        parser.add_argument('--downstream', type=str, default='None')
        parser.add_argument('--way', type=int, default=100)
        parser.add_argument('--shot', type=int, default=100)
        parser.add_argument('--shot_transform', type=str, default='l2n')
        parser.add_argument('--testaug', action='store_true', default=False)
        return parser

    def __init__(self, args, data_path, save_dir, train_transform, val_transform):
        train_dataset, val_dataset, num_classes = \
            get_dataset(args.dataset, data_path, train_transform, val_transform)

        print(num_classes)

        self.data_loader_train = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            shuffle=False
        )

        self.model = args.model
        self.uarch = args.uarch
        self.dataset = args.dataset
        self.num_classes = num_classes
        self.save_dir = save_dir
        self.do_keys = args.rk_methods

        self.save_forwards_path = f'{args.save_url}/getforwards'
        self.target_path = f'{self.save_forwards_path}/{self.dataset}_targets.npy'

        Path(self.save_forwards_path).mkdir(parents=True, exist_ok=True)

    def fit(self, model, **kwargs):
        if self.model in ['esvit']:
            model.eval()
            judge_list = self.get_judge_list(model)
            features = []

            for (images, target) in tqdm(self.data_loader_train):
                images = images.cuda(non_blocking=True)
                emb = self.forward_feature_handle(images, model, judge_list, **kwargs)
                features.append(emb.detach().cpu())

            features = torch.cat(features)

        else:
            fc_layer_name = MODEL_FC_LAYERS.get(self.model, None)
            if fc_layer_name is None:
                raise NotImplementedError(f"Model '{self.model}' not supported")

            fc_layer = eval(f'model.{fc_layer_name}')

            features, outputs, targets = self.forward_pass(self.data_loader_train, model, fc_layer)

        if 'ZERO' in self.do_keys:
            # Save features
            if self.model not in ['esvit'] and not os.path.isfile(self.target_path):
                np.save(self.target_path, targets.detach().cpu().numpy())

            if 'OTCE' in self.do_keys:
                np.save(f'{self.save_forwards_path}/{self.model}_{self.dataset}5Shot_features.npy', features.detach().cpu().numpy())
                np.save(f'{self.save_forwards_path}/{self.dataset}5Shot_targets.npy', targets.detach().cpu().numpy())
            else:
                save_name = self.uarch if self.model in ['esvit'] else self.model
                np.save(f'{self.save_forwards_path}/{save_name}_{self.dataset}_features.npy', features.detach().cpu().numpy())
                np.save(f'{self.save_forwards_path}/{save_name}_{self.dataset}_outputs.npy', outputs.detach().cpu().numpy())

            print("Finish saving features!")
            return {}

        predictions = F.softmax(outputs, dim=1)
        results = {}
        for do_k in self.do_keys:
            if do_k == 'LogME':
                from mptms.LogME import LogME
                logme = LogME(regression=False)
                score = logme.fit(features.numpy(), targets.numpy())
            elif do_k == 'NCE':
                from mptms.NCE import NCE
                score = NCE(source_label=torch.argmax(predictions, dim=1).numpy(), target_label=targets.numpy())
            elif do_k == 'LEEP':
                from mptms.LEEP import LEEP
                score = LEEP(prob_np_all=predictions.numpy(), label_np_all=targets.numpy())
            elif do_k == 'H_Score':
                from mptms.H_Score import H_Score
                score = H_Score(features=features.numpy(), labels=targets.numpy())
            elif do_k == 'NLEEP':
                from mptms.LEEP import NLEEP
                score = NLEEP(features_np_all=features.numpy(), label_np_all=targets.numpy())
            elif do_k == 'OTCE':
                from mptms.OTCE import OTCE
                imagenet_features = np.load(f'{self.save_forwards_path}/{self.model}_ImageNet5Shot_features.npy')
                imagenet_targets = np.load(f'{self.save_forwards_path}/ImageNet5Shot_targets.npy')
                score = OTCE(src_x=torch.tensor(imagenet_features, dtype=torch.float), tar_x=features, src_y=imagenet_targets, tar_y=targets.numpy())
            elif do_k == 'PACTranDirichlet':
                from mptms.PACTran import PACTranDirichlet
                score = -PACTranDirichlet(prob_np_all=predictions.numpy(), label_np_all=targets.numpy(), alpha=1.)
            elif do_k == 'PACTranGamma':
                from mptms.PACTran import PACTranGamma
                score = -PACTranGamma(prob_np_all=predictions.numpy(), label_np_all=targets.numpy(), alpha=1.)
            elif do_k == 'GBC':
                from mptms.GBC import GBC
                score = GBC(features=features.numpy(), labels=targets.numpy())
            elif do_k == 'DEPARA':
                from mptms.DEPARA import DEPARA
                imagenet_features = np.load(f'{self.save_forwards_path}/{self.model}_ImageNet5Shot_features.npy')
                score = DEPARA(feature_p=imagenet_features, feature_q=features)
            elif do_k == 'LFC':
                from mptms.LFC import LFC
                score = LFC(features, targets)
            elif do_k == 'ZERO':
                score = 0
            else:
                raise Exception(f'Unknown {do_k}')
            results[do_k] = score

            print(f'{do_k} of {self.model} on {self.dataset}: {results[do_k]}\n')

        return results

    def get_judge_list(self, model):
        """
        Create a dictionary to judge the capabilities of a model.

        Args:
            model (torch.nn.Module): The PyTorch model.

        Returns:
            dict: A dictionary with keys indicating various model capabilities.
        """
        judge_list = {'module': hasattr(model, 'module')}

        if judge_list['module']:
            judge_list.update({
                'forward_features': hasattr(model.module, 'forward_features'),
                'get_intermediate_layers': hasattr(model.module, 'get_intermediate_layers'),
                'forward_return_n_last_blocks': hasattr(model.module, 'forward_return_n_last_blocks')
                })
        else:
            judge_list.update({
                'forward_features': hasattr(model, 'forward_features'),
                'get_intermediate_layers': hasattr(model, 'get_intermediate_layers'),
                'forward_return_n_last_blocks': hasattr(model, 'forward_return_n_last_blocks')
                })
        return judge_list

    def forward_intermediate_layers(self, cur, images, **kwargs):
        """
        Forward pass through intermediate layers of a model.

        Args:
            cur (torch.nn.Module): Current model.
            images (torch.Tensor): Input images.
            kwargs['avgpool'] (int): Type of average pooling to apply.

        Returns:
            torch.Tensor: Extracted features.
        """
        intermediate_output = cur.get_intermediate_layers(images, kwargs['n'])

        if kwargs['avgpool'] == 0:
            # norm(x[:, 0])
            output = [x[:, 0] for x in intermediate_output]
        elif kwargs['avgpool'] == 1:
            # x[:, 1:].mean(1)
            output = [torch.mean(intermediate_output[-1][:, 1:], dim=1)]
        elif kwargs['avgpool'] == 2:
            # norm(x[:, 0]) + norm(x[:, 1:]).mean(1)
            output = [x[:, 0] for x in intermediate_output] + [torch.mean(intermediate_output[-1][:, 1:], dim=1)]
        else:
            assert False, "Unkown avgpool type {}".format(kwargs['avgpool'])

        feats = torch.cat(output, dim=-1).clone()
        return feats

    def forward_feature_handle(self, images, model, judge_list, **kwargs):
        """
        Perform a forward pass through the model with various options.

        Args:
            images (torch.Tensor): Input images.
            model (torch.nn.Module): The PyTorch model.
            judge_list (dict): A dictionary indicating model capabilities.
            kwargs['avgpool'] (int): Type of average pooling to apply.
            kwargs['n'] (int): Value for 'n' parameter.
            kwargs['depths'] (list): List of depth values.

        Returns:
            torch.Tensor: Extracted features.
        """
        model.eval()

        if judge_list['module']:
            cur = model.module
        else:
            cur = model

        with torch.no_grad():
            if '_forward' in kwargs.keys():
                emb = kwargs['_forward'](cur, images)
            elif judge_list['get_intermediate_layers']:
                emb = self.forward_intermediate_layers(cur, images, **kwargs)
            elif judge_list['forward_return_n_last_blocks']:
                emb = cur.forward_return_n_last_blocks(images, kwargs['n'], kwargs['avgpool'], kwargs['depths'])
            elif judge_list['forward_features']:
                emb = cur.forward_features(images)
            else:
                emb = cur(images)
        return emb

    def forward_pass(self, score_loader, model, fc_layer):
        """
        Perform a forward pass on a target dataset.

        Args:
            score_loader (torch.utils.data.DataLoader): The dataloader for scoring transferability.
            model (torch.nn.Module): The model for scoring transferability.
            fc_layer (torch.nn.Module): The fc layer of the model, for registering hooks.

        Returns:
            torch.Tensor: Extracted features of the model.
            torch.Tensor: Outputs of the model.
            torch.Tensor: Ground-truth labels of the dataset.
        """
        features = []
        outputs = []
        targets = []

        def hook_fn_forward(module, input, output):
            features.append(input[0].detach().cpu())
            outputs.append(output.detach().cpu())

        forward_hook = fc_layer.register_forward_hook(hook_fn_forward)

        model.eval()
        with torch.no_grad():
            for _, (data, target) in enumerate(tqdm(score_loader)):
                targets.append(target)
                data = data.cuda()
                _ = model(data)

        forward_hook.remove()
        features = torch.cat([x for x in features])
        outputs = torch.cat([x for x in outputs])
        targets = torch.cat([x for x in targets])

        return features, outputs, targets


if __name__ == '__main__':
    parser = get_command_line_parser()
    parser = FeatureExtractor.parse_extrator_args(parser)

    temp_args, _ = parser.parse_known_args()
    parser = ArgumentParser(parents=[parser], add_help=False)
    if 'esvit' in temp_args.model:
        parser = prepare_parser(parser, temp_args)

    args = parser.parse_args()
    if args.time_str == '':
        args.time_str = datetime.datetime.now().strftime('%m%d-%H-%M-%S-%f')[:-3]

    set_gpu(args.gpu)
    torch.cuda.set_device(int(args.gpu))
    torch.autograd.set_detect_anomaly(True)

    args_namespace = argparse.Namespace()
    args_namespace.__dict__.update(dict(vars(args)))
    scores = {}
    for cur_model in args.model_hub:
        cur_namespace = deepcopy(args_namespace)
        cur_args = parser.parse_args(namespace=cur_namespace)
        cur_args.model = cur_model
        pprint(vars(cur_args))

        set_seed(args.seed)

        logits_save_path = Path(f'{args.save_url}/log/{args.time_str}/{cur_model.lower()}')
        logits_save_path.mkdir(parents=True, exist_ok=True)
        log_config(logits_save_path, vars(cur_args))

        model, transform_kwargs, fit_kwargs = get_model(cur_args, load_url_pretrained=True, load_manual_pretrained=False)

        if len(args.model_hub) > 1:
            resize_size = 299 if cur_model == 'inception_v3' else 224
            train_transform = get_hub_transform(transforms.Resize((resize_size, resize_size)), transform_kwargs['normalize_transform'])
            val_transform = get_hub_transform(transforms.Resize((resize_size, resize_size)), transform_kwargs['normalize_transform'])
        else:
            train_transform = get_transform(transform_kwargs['crop_size'], transform_kwargs['resize_transform'], transform_kwargs['normalize_transform'], testaug=False)
            val_transform = get_transform(transform_kwargs['crop_size'], transform_kwargs['resize_transform'], transform_kwargs['normalize_transform'], testaug=args.testaug)

        cur_handle = FeatureExtractor(cur_args, DATA_PATHS['ImageNet' if 'ImageNet' in args.dataset else args.dataset], logits_save_path, train_transform, val_transform)

        scores[cur_model] = cur_handle.fit(model, **fit_kwargs)

    for i_hub in args.model_hub:
        for i_method in args.rk_methods:
            ...
            # print(scores[i_hub][i_method], end='\t')
        # print('\n')
