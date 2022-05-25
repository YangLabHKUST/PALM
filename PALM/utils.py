import logging
import torch
import numpy as np
import scipy.stats as ss
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import KFold

class Logger:
    def __init__(self, path=None, clevel = logging.DEBUG, Flevel = logging.INFO):
        self.logger = logging.getLogger(path)
        self.logger.setLevel(logging.INFO)
        fmt = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', '%Y-%m-%d %H:%M:%S')
        # set cmd logging
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        sh.setLevel(clevel)
        self.logger.addHandler(sh)
        # set file logging
        if path is not None:
            fh = logging.FileHandler(path)
            fh.setFormatter(fmt)
            fh.setLevel(Flevel)
            self.logger.addHandler(fh)

    def debug(self, message):
        self.logger.debug(message)
        
    def info(self, message):
        self.logger.info(message)

    def warn(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)


Func_A = lambda x: -3
Func_B = lambda x: -3 + 1.5 * x[0] + 1.5 * x[1]
Func_C = lambda x: -4.25 + 2*x[0]**2 + 2*x[1]**2 - 2*x[0]*x[1]
Func_D = lambda x: -4 + 4*np.sin(np.pi*x[0]*x[1]) + 2*(x[2] - x[3])**2 + x[3] + 0.5*x[4]
Func_E = lambda x: np.where(x[1] == 0, 1 - 6*x[0]**2, -1 + 2*x[0] - 6*x[0]**2)

# Objective function for xgboost
def fobj(predt, dtrain):
    label = dtrain.get_label()
    sigma = 1 / (1 + np.exp(-predt))
    grad = sigma - label
    hess = sigma * (1 - sigma)
    return grad, hess

# Posterior to FDR
def post2FDR(post):
    M = len(post)
    fdr = 1 - post
    rank_fdr = (ss.rankdata(fdr) - 1).astype(int).tolist()
    sort_fdr = np.sort(fdr)
    cumsum_fdr = np.cumsum(sort_fdr)
    sort_FDR = cumsum_fdr / np.arange(1, M + 1)
    FDR = sort_FDR[rank_fdr]
    return FDR

# Association mapping
def Assoc_SNPs(post, FDRset=0.1, fdrcontrol=['global', 'local']):
    M = len(post)
    FDR = post2FDR(post)
    fdr = 1 - post
    est_Z = np.zeros(M)
    if fdrcontrol == 'global':
        est_Z[FDR <= FDRset] = 1
    elif fdrcontrol == 'local':
        est_Z[fdr <= FDRset] = 1
    return est_Z

# Performance evaluation
def Perf(true, est, fdr):
    t = confusion_matrix(true, est)
    if sum(est) == 0:
        fit_FDR = 0
        power = 0
    else:
        fit_FDR = t[0, 1] / (t[0, 1] + t[1, 1])
        power = t[1, 1] / (t[1, 0] + t[1, 1])
    AUC = roc_auc_score(true, 1 - fdr)
    pAUC = roc_auc_score(true, 1 - fdr, max_fpr=0.2)

    return {'FDR': fit_FDR, 'power': power, 'AUC': AUC, 'pAUC': pAUC}


# Generate data based on the p-value version two-groups model
def Generate_data_gen(M, D, true_alpha, Func):
    A = np.random.uniform(-1, 1, M * D).reshape(M, D)
    if Func == Func_E:
        A[:, 1] = np.random.binomial(1, 0.9, M)
    # prior
    true_FA = np.apply_along_axis(Func, 1, A)
    pi1 = 1 / (1 + np.exp(-true_FA))
    # latent status
    Z = np.zeros(M)
    risk_idx = np.random.uniform(0, 1, M) < pi1
    Z[risk_idx] = 1
    # p-values
    pvals = np.random.uniform(0, 1, M)
    pvals[risk_idx] = np.random.beta(true_alpha, 1, np.sum(risk_idx))

    return pvals, A, Z

## Gaussian mixture
def rnormmix(n, params):
    dummy = np.random.multinomial(1, params['pi'], n)
    index = np.where(dummy)[1].astype(int).tolist()
    y = np.zeros(n)
    for comp in np.unique(index):
        y[index == comp] = np.random.normal(params['mu'][comp], params['sigma'][comp], sum(index == comp))
    return y

## Means of zscores
def theta_dist(n, shape=['bimodal', 'spiky', 'skew', 'big-normal']):
    # parameter settings are referred to LSMM
    params = dict.fromkeys(['pi', 'mu', 'sigma'])
    if shape == 'bimodal':
        params['pi'] = [0.48, 0.04, 0.48]
        params['mu'] = [-2, 0, 2]
        params['sigma'] = [1, 4, 1]
    elif shape == 'spiky':
        params['pi'] = [0.4, 0.2, 0.2, 0.2]
        params['mu'] = [0, 0, 0, 0]
        params['sigma'] = [0.25, 0.5, 1, 2]
    elif shape == 'skew':
        params['pi'] = [0.25, 0.25, 1/3, 1/6]
        params['mu'] = [-2, -1, 0, 1]
        params['sigma'] = [2, 1.5, 1, 1]
    elif shape == 'big-normal':
        params['pi'] = [1]
        params['mu'] = [0]
        params['sigma'] = [4]
    x = rnormmix(n, params)
    return x

## Generate data in a discriminative way: p-values are converted from z-scores
def Generate_data_disc(M, D, Func, shape='bimodal'):
    A = np.random.uniform(-1, 1, M * D).reshape(M, D)
    if Func == Func_E:
        A[:, 1] = np.random.binomial(1, 0.9, M)
    # prior
    true_FA = np.apply_along_axis(Func, 1, A)
    pi1 = 1 / (1 + np.exp(-true_FA))
    # latent status
    Z = np.zeros(M)
    risk_idx = np.random.uniform(0, 1, M) < pi1
    Z[risk_idx] = 1
    # z-scores
    zs = np.random.randn(M)
    zs[risk_idx] = theta_dist(np.sum(risk_idx).astype(int), shape) + np.random.randn(sum(risk_idx).astype(int))
    # p-values
    pvals = 2 * ss.norm.cdf(-abs(zs))
    return zs, pvals, A, Z


## Calculate Lq_eval
def _eval_nn(net, pvals_eval, A_eval, alpha):

    M_eval = len(pvals_eval)
    # clamp pvals
    pvals_eval = torch.clamp(pvals_eval, 1e-16, 1 - 1e-16)

    # forward
    net.eval()
    FA_eval = net(A_eval)

    # Lq_eval
    pi_eval = FA_eval.softmax(1)  # (M_train, 2): (pi1, pi0)
    pi0_eval, pi1_eval = pi_eval[:, 0], pi_eval[:, 1]
    Lq_eval = torch.sum(torch.log(pi1_eval * alpha * pvals_eval ** (alpha - 1) + pi0_eval))
    Lq_eval /= M_eval

    # to number
    Lq_eval = float(Lq_eval.detach().cpu().numpy())

    return Lq_eval

## Permute one column of annotation matrix
def _permute_feature(A, j):

    A_permute = torch.clone(A)
    A_permute[:, j] = A_permute[:, j][torch.randperm(A_permute.size(0))]

    return A_permute


## Calculate variance importance: imp_j = s - sum_k{s_jk}/n
def get_nn_var_imp(net, pvals_eval, A_eval, alpha, nrep=50):

    D = A_eval.shape[1]
    var_imp = []

    # original Lq_eval (unpermuted)
    Lq_eval_orig = _eval_nn(net, pvals_eval, A_eval, alpha)

    for j in range(D):

        # Lq_eval on permuted A_eval
        Lq_eval_perms = np.array([_eval_nn(net, pvals_eval, _permute_feature(A_eval, j), alpha) for k in range(nrep)])
        Lq_eval_perm = np.mean(Lq_eval_perms)

        # difference between two scores
        var_imp.append(Lq_eval_orig - Lq_eval_perm)

        # print(f'Finish {j}-th variable')

    # relative importance
    var_imp = var_imp / sum(var_imp)

    return var_imp