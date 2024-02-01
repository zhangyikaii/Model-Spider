import torch
import torch.nn.functional as F


DATA_LIMIT = 45000
def LFC(target_x, target_y):
    if target_x.shape[0] > DATA_LIMIT:
        sampled_index = torch.randperm(target_x.shape[0])[:DATA_LIMIT]
        target_x = target_x[sampled_index]
        target_y = target_y[sampled_index]

    y = F.one_hot(target_y)
    y = y @ y.T
    y[y == 0] = -1
    y = y.float()
    x = target_x @ target_x.T
    y = y - torch.mean(y).item()
    x = x - torch.mean(x).item()

    return torch.sum(torch.mul(x, y)).item() / (torch.sqrt(torch.sum(torch.mul(x, x))).item() * torch.sqrt(torch.sum(torch.mul(y, y))).item())