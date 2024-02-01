import numpy as np
import scipy.optimize
import scipy.special
import sklearn.svm


def one_hot(a):
    b = np.zeros((a.size, a.max()+1))
    b[np.arange(a.size), a] = 1.
    return b

def PACTranDirichlet(prob_np_all, label_np_all, alpha):
    """Compute the PACTran-Dirichlet estimator."""
    label_np_all = one_hot(label_np_all)  # [n, v]
    soft_targets_sum = np.sum(label_np_all, axis=0)  # [v]
    soft_targets_sum = np.expand_dims(soft_targets_sum, axis=1)  # [v, 1]
    a0 = alpha * soft_targets_sum / np.sum(soft_targets_sum) + 1e-10

    # initialize
    qz = prob_np_all  # [n, d]
    log_s = np.log(prob_np_all + 1e-10)  # [n, d]

    for _ in range(10):
        aw = a0 + np.sum(np.einsum("BY,BZ->BYZ", label_np_all, qz), axis=0)
        logits_qz = (log_s +
                    np.matmul(label_np_all, scipy.special.digamma(aw)) -
                    np.reshape(scipy.special.digamma(np.sum(aw, axis=0)), [1, -1]))
        log_qz = logits_qz - scipy.special.logsumexp(
            logits_qz, axis=-1, keepdims=True)
        qz = np.exp(log_qz)

    log_c0 = scipy.special.loggamma(np.sum(a0)) - np.sum(
        scipy.special.loggamma(a0))
    log_c = scipy.special.loggamma(np.sum(aw, axis=0)) - np.sum(
        scipy.special.loggamma(aw), axis=0)

    pac_dir = np.sum(
        log_c0 - log_c - np.sum(qz * (log_qz - log_s), axis=0))
    pac_dir = -pac_dir / label_np_all.size
    return pac_dir


def PACTranGamma(prob_np_all, label_np_all, alpha):
    """Compute the PAC-Gamma estimator."""
    label_np_all = one_hot(label_np_all)  # [n, v]
    soft_targets_sum = np.sum(label_np_all, axis=0)  # [v]
    soft_targets_sum = np.expand_dims(soft_targets_sum, axis=1)  # [v, 1]

    a0 = alpha * soft_targets_sum / np.sum(soft_targets_sum) + 1e-10
    beta = 1.

    # initialize
    qz = prob_np_all  # [n, d]
    s = prob_np_all  # [n, d]
    log_s = np.log(prob_np_all + 1e-10)  # [n, d]
    aw = a0
    bw = beta
    lw = np.sum(s, axis=-1, keepdims=True) * np.sum(aw / bw)  # [n, 1]

    for _ in range(10):
        aw = a0 + np.sum(np.einsum("BY,BZ->BYZ", label_np_all, qz),
                        axis=0)  # [v, d]
        lw = np.matmul(
            s, np.expand_dims(np.sum(aw / bw, axis=0), axis=1))  # [n, 1]
        logits_qz = (
            log_s + np.matmul(label_np_all, scipy.special.digamma(aw) - np.log(bw)))
        log_qz = logits_qz - scipy.special.logsumexp(
            logits_qz, axis=-1, keepdims=True)
        qz = np.exp(log_qz)  # [n, a, d]

    pac_gamma = (
        np.sum(scipy.special.loggamma(a0) - scipy.special.loggamma(aw) +
                aw * np.log(bw) - a0 * np.log(beta)) +
        np.sum(np.sum(qz * (log_qz - log_s), axis=-1) +
                np.log(np.squeeze(lw, axis=-1)) - 1.))
    pac_gamma /= label_np_all.size
    pac_gamma += 1.
    return pac_gamma
