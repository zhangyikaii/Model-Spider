from scipy.spatial.distance import pdist, squareform
import numpy as np
import random
from copy import deepcopy


def spearman_correlation(matrix):
    spearman_corr = np.zeros((matrix.shape[0], matrix.shape[0]))
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[0]):
            if i is j:
                spearman_corr[i, j] = 1
            elif i < j:
                continue
            else:
                def rank(ind):
                    l = ind.shape[0]
                    r = np.zeros(l)
                    for i in range(l):
                        r[ind[i]] = i
                    return r
                ind_i = np.argsort(-matrix[i])
                ind_j = np.argsort(-matrix[j])
                rank_i = rank(ind_i)
                rank_j = rank(ind_j)
                print(rank_i, rank_j)
                spearman_corr[i, j] = 1 - 6.0 * np.sum(np.square(rank_i-rank_j)) / (matrix.shape[1]*(matrix.shape[1]**2-1))
                spearman_corr[j, i] = spearman_corr[i, j]
    return spearman_corr


def DEPARA(feature_p, feature_q, stype):
    # modified from https://github.com/zju-vipa/DEPARA
    # stype: 'cosine' or 'correlation'
    feature_all = []
    # feature_all_correlation = []

    if feature_q.shape[0] != feature_p.shape[0]:
        sampled_index = list(range(feature_q.shape[0]))
        random.shuffle(sampled_index)
        if feature_q.shape[0] < feature_p.shape[0]:
            feature_q_copy = deepcopy(feature_q)
            sampled_index = sampled_index[: abs(feature_p.shape[0] - feature_q.shape[0])]
            feature_q = np.concatenate([feature_q, feature_q_copy[sampled_index]])
        else:
            sampled_index = sampled_index[: feature_p.shape[0]]
            feature_q = feature_q[sampled_index]

    feature_list = [feature_p, feature_q]

    for i in range(len(feature_list)):
        feature = feature_list[i]
        cur_feature = feature - np.mean(feature, axis=0)
        feature_cosine = pdist(cur_feature, stype)
        feature_all.append(feature_cosine)
        # feature_correlation = pdist(cur_feature, 'correlation')
        # feature_all_correlation.append(feature_correlation)

    feature_all = np.stack(feature_all)
    # feature_all_correlation = np.stack(feature_all_correlation)

    spearman_2x2 = spearman_correlation(feature_all)
    # spearman_20x20_correlation = spearman_correlation(feature_all_correlation)
    return spearman_2x2[0, 1]
