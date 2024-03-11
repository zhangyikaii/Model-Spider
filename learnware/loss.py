import torch
import torch.nn.functional as F
from torch import nn
import numpy as np


class HierarchicalCE(nn.Module):
    def __init__(self, num_learnware, temperature=1):
        super().__init__()
        self.temperature = temperature
        self.num_learnware = num_learnware
        self.mask = []
        for i in range(0, self.num_learnware):
            self.mask.append([0 for _ in range(min(self.num_learnware - 1, self.num_learnware - i))] + [1 for _ in range(i - 1)])
        self.mask = torch.tensor(self.mask).to(torch.device('cuda'))

    def forward(self, logits, labels):
        """
        logits: [batch_size, num_learnware]
        labels: [batch_size, num_learnware], [3, 1, 4, 2, 0], larger number means higher rank
        """
        batch_size, _ = logits.shape
        logits = logits.repeat(1, self.num_learnware - 1).view(-1, self.num_learnware)  
        cur_mask = self.mask[labels.view(-1)].reshape(batch_size, self.num_learnware, self.num_learnware - 1).permute(1, 0, 2).reshape(self.num_learnware, -1).T 
        logits = logits.masked_fill(cur_mask == 1, -np.inf)
        return F.nll_loss(F.log_softmax(logits, 1), torch.sort(labels, descending=True, dim=-1)[1][:, :-1].reshape(-1)) 


class Top1CE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, labels):
        """
        ListNet: Learning to Rank
        :param true_rank_score : value higher, rank higher
        https://github.com/szdr/pytorch-listnet/blob/master/listnet.py

        logits: [batch_size, num_learnware]
        labels: [batch_size, num_learnware]
        """
        return -torch.sum(torch.mul(F.softmax(labels.float(), dim=1), F.log_softmax(logits, dim=1)))


class ListMLE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, labels, eps=1e-10, padded_value_indicator=-1):
        """
        ListMLE loss introduced in "Listwise Approach to Learning to Rank - Theory and Algorithm".
        :param eps: epsilon value, used for numerical stability
        :param padded_value_indicator: an indicator of the `labels` index containing a padded item, e.g. -1
        :return: loss value, a torch.Tensor
        https://github.com/allegro/allRank/blob/master/allrank/models/losses/listMLE.py
        """
        random_indices = torch.randperm(logits.shape[-1])
        y_pred_shuffled = logits[:, random_indices]
        y_true_shuffled = labels[:, random_indices]
        y_true_sorted, indices = y_true_shuffled.sort(descending=True, dim=-1)
        mask = y_true_sorted == padded_value_indicator
        preds_sorted_by_true = torch.gather(y_pred_shuffled, dim=1, index=indices)
        preds_sorted_by_true[mask] = float("-inf")
        max_pred_values, _ = preds_sorted_by_true.max(dim=1, keepdim=True)
        preds_sorted_by_true_minus_max = preds_sorted_by_true - max_pred_values
        cumsums = torch.cumsum(preds_sorted_by_true_minus_max.exp().flip(dims=[1]), dim=1).flip(dims=[1])
        observation_loss = torch.log(cumsums + eps) - preds_sorted_by_true_minus_max
        observation_loss[mask] = 0.0
        return torch.mean(torch.sum(observation_loss, dim=1))
