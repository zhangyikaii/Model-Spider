import numpy as np
from sklearn.decomposition import PCA


def GBC(features, labels):
    pca = PCA(n_components=64)
    features = pca.fit_transform(features)
    means, covs = [], []
    for i in np.unique(labels):
        cur_indices = (labels == i)
        cur_mean = np.mean(features[cur_indices], axis=0)
        assert len(features[cur_indices]) != 1
        cur_cov = np.cov(features[cur_indices].T)
        means.append(cur_mean)
        covs.append(cur_cov)
    score = 0
    for i in range(len(means)):
        for j in range(len(means)):
            if i != j:
                mu_i, mu_j = means[i][:, np.newaxis], means[j][:, np.newaxis]
                cov_i, cov_j = covs[i], covs[j]
                cov_ij = 1 / 2 * (cov_i + cov_j)
                score -= np.exp(-1 / 8 * ((mu_i - mu_j).T @ np.linalg.pinv(cov_ij) @ (mu_i - mu_j))
                    - 1 / 2 * np.log(1e-5 + np.abs(np.linalg.det(cov_ij)) / (1e-5 + np.sqrt(1e-5 + np.linalg.det(cov_i) * np.linalg.det(cov_j)))))
    score = score.squeeze()
    assert not np.isnan(score) and np.isfinite(score)
    return score.item()
