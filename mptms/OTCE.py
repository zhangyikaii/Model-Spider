import numpy as np
import ot
import geomloss
import torch
import math


def compute_coupling(X_src, X_tar):
    cost_function = lambda x, y: geomloss.utils.squared_distances(x, y)

    C = cost_function(X_src, X_tar)

    P = ot.emd(ot.unif(X_src.shape[0]), ot.unif(X_tar.shape[0]), C.numpy(), numItermax=100000)
    W = np.sum(P*np.array(C.numpy()))

    return P, W


def compute_CE(P, Y_src, Y_tar):
    src_label_set = set(sorted(list(Y_src.flatten())))
    tar_label_set = set(sorted(list(Y_tar.flatten())))

    # joint distribution of source and target label
    P_src_tar = np.zeros((np.max(Y_src)+1,np.max(Y_tar)+1))

    for y1 in src_label_set:
        y1_idx = np.where(Y_src==y1)
        for y2 in tar_label_set:
            y2_idx = np.where(Y_tar==y2)

            RR = y1_idx[0].repeat(y2_idx[0].shape[0])
            CC = np.tile(y2_idx[0], y1_idx[0].shape[0])

            P_src_tar[y1,y2] = np.sum(P[RR,CC])

    # marginal distribution of source label
    P_src = np.sum(P_src_tar,axis=1)

    ce = 0.0
    for y1 in src_label_set:
        P_y1 = P_src[y1]
        for y2 in tar_label_set:
            
            if P_src_tar[y1,y2] != 0:
                ce += -(P_src_tar[y1,y2] * math.log(P_src_tar[y1,y2] / P_y1))
    return ce


def OTCE(src_x, tar_x, src_y, tar_y):
    # https://github.com/tanyang1231/OTCE_Transferability_CVPR21/blob/main/compute_otce_demo.py
    # obtain the optimal coupling matrix P and the wasserstein distance W
    P, W = compute_coupling(src_x, tar_x)

    src_y = src_y[:,np.newaxis]
    tar_y = tar_y[:,np.newaxis]
    # compute the conditonal entropy (ce)
    ce = compute_CE(P, src_y, tar_y)

    return W + ce

def test():
    #-----------start: randomly generate the testing data-----------
    src_x_list = []
    src_y_list = []
    tar_x_list = []
    tar_y_list = []

    NUM_SAMPLE = 10000
    NUM_SAMPLE_ = 200

    # suppose the feature dimension is 512, and the label is in range [0,10].
    for i in range(NUM_SAMPLE):
        src_x_list.append(np.random.randn(2048))
        src_y_list.append(np.random.randint(0,10))
    
    for i in range(NUM_SAMPLE_):
        tar_x_list.append(np.random.randn(2048))
        tar_y_list.append(np.random.randint(0,10))
    
    # the shape of x is n*512, and the shape of y is n*1 
    src_x = torch.tensor(np.array(src_x_list), dtype=torch.float)
    tar_x = torch.tensor(np.array(tar_x_list), dtype=torch.float)
    src_y = np.array(src_y_list)[:,np.newaxis]
    tar_y = np.array(tar_y_list)[:,np.newaxis]
    #-----------end: randomly generate the testing data------------

    # print(src_x.shape)
    # print(tar_y.shape)
    # obtain the optimal coupling matrix P and the wasserstein distance W
    P,W = compute_coupling(src_x, tar_x)

    # compute the conditonal entropy (ce)
    ce = compute_CE(P, src_y, tar_y)

    print('Wasserstein distance:%.4f, Conditonal Entropy: %.4f'%(W, ce))


# if __name__ == '__main__':
#     test()