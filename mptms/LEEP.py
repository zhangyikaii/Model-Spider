import numpy as np
import sklearn.decomposition
import sklearn.mixture


def one_hot(a):
    b = np.zeros((a.size, a.max()+1))
    b[np.arange(a.size), a] = 1.
    return b


def LEEP(prob_np_all, label_np_all):
    """Calculating the LEEP score."""

    label_np_all = one_hot(label_np_all)

    # compute p(y|z)
    pz = np.expand_dims(np.mean(prob_np_all, axis=0), axis=-1) + 1e-10  # [v, 1]
    pzy = np.mean(
        np.einsum("BF,BH->BFH", prob_np_all, label_np_all), axis=0)  # [v, v]
    pycz = pzy / pz  # p(y|z) [v, v]

    # compute p(y) = sum_z p(y|z)p(z)
    eep = np.matmul(prob_np_all, pycz) + 1e-10

    # leep = KL(label | eep)
    leep = np.mean(np.sum(
        label_np_all * (np.log(label_np_all + 1e-10) + np.log(eep)), axis=1))

    return leep


def gmm_estimator(features_np_all, label_np_all):
    """Estimate the GMM posterior assignment."""
    pca_model = sklearn.decomposition.PCA(n_components=0.8)
    pca_model.fit(features_np_all)
    features_lowdim_train = pca_model.transform(features_np_all)

    num_examples = label_np_all.shape[0]
    y_classes = max([min([label_np_all.max() + 1, int(num_examples * 0.2)]),
                    int(num_examples * 0.1)])
    clf = sklearn.mixture.GaussianMixture(n_components=y_classes)
    clf.fit(features_lowdim_train)
    prob_np_all_gmm = clf.predict_proba(features_lowdim_train)
    return prob_np_all_gmm, features_lowdim_train


def NLEEP(features_np_all, label_np_all):
    """Calculate LEEP with GMM classifier."""
    # https://github.com/google-research/pactran_metrics/blob/a5e49972ed0856bc79f74c5c198d1c25b6fb9025/compute_metrics.py#L58
    prob_np_all_gmm, _ = gmm_estimator(features_np_all, label_np_all)

    return LEEP(prob_np_all_gmm, label_np_all)