import os, time
import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import Booster, DMatrix
from sklearn.model_selection import train_test_split, KFold
from sklearn.tree import DecisionTreeRegressor as DTR
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datetime import datetime
from utils import fobj, Assoc_SNPs, get_nn_var_imp

obj = fobj
criterion = nn.CrossEntropyLoss()

def _E_tree(pvals, FA, alpha):
    
    M = len(pvals)
    
    # Compute posterior
    pi1 = 1 / (1 + np.exp(-FA))
    pi0 = 1 - pi1
    comp1 = pi1 * alpha * pvals ** (alpha - 1)
    comp1 = np.clip(comp1, 1e-16, 1e16)
    z1 = comp1 / (comp1 + pi0) # posterior
    g = z1 - pi1 # gradient: for gbm
    Lq = np.sum(np.log(pi1 * alpha * pvals ** (alpha - 1) + pi0))
    Lq /= M
    
    return z1, g, Lq

def _M_tree(pvals, z1, g, dtrain, i, bst, module):
    
    # Update alpha
    alpha = - np.sum(z1) / np.dot(z1, np.log(pvals))
    
    # Grow a new tree to maximize the 2nd-order approximation of Q-function
    if module == 'xgb':
        dtrain.set_label(z1)
        bst.update(dtrain, i, obj)
    elif module == 'gbm':
        bst[i].fit(dtrain, g)
        
    return alpha, bst
        
def EM_tree(pvals, A, params, pvals_eval=None, A_eval=None, 
              alpha=0.4, num_boost_round=5000, print_every=100, verbose=False, module=['xgb', 'gbm']):
    
    '''
    EM algorithm for Tree-based PALM.
    
    :param pvals: training p-values of snps, np.ndarray (M,)
    :param A: training annotation matrix, np.ndarray (M, D)
    :param params: hyperparameters of xgboost
    :param pvals_eval: evaluation p-values of snps, np.ndarray (M_eval,)
    :param A_eval: evaluation annotation matrix, np.ndarray (M_eval, D)
    :param alpha: initial alpha (parameter of beta distribution)
    :param num_boost_round: maximum number of iterations
    :param print_every: print frequency
    :param verbose: verbose or not
    :param module: xgboost-type or gbm-type algorithm
    :return: a dictionary containing estimated parameters, likelihood, posterior and variable importance
    '''
    
    # Data size
    M, D = A.shape
    
    # Convert data into DMatrix object
    if module == 'xgb':
        
        dtrain = A if isinstance(A, DMatrix) else DMatrix(A)
        if A_eval is not None:
            M_eval = A_eval.shape[0]
            deval = A_eval if isinstance(A_eval, DMatrix) else DMatrix(A_eval)
        
        # Initialize booster, starting iteration and parameters
        start_iteration = 0
        num_boost_round = max(num_boost_round, 1)
    
        bst = Booster(params, [dtrain])
        FA = bst.predict(data=dtrain)
        
        # Record
        record = {'post': None, 'Lq': [], 'alpha': [], 'var_imp': None}
        if pvals_eval is not None:
            record = {'post': None, 'Lq': [], 'Lq_eval': [], 'alpha': [], 'var_imp': None}
            FA_eval = bst.predict(data=deval)
            
        # EM loop
        for i in range(start_iteration, num_boost_round):

            # E step
            z1, g, Lq = _E_tree(pvals, FA, alpha)
            record['Lq'].append(Lq)

            # Likelihood on validation data
            if pvals_eval is not None:
                _, _, Lq_eval = _E_tree(pvals_eval, FA_eval, alpha)
                record['Lq_eval'].append(Lq_eval)

            # M step
            alpha, bst = _M_tree(pvals, z1, g, dtrain, i, bst, module)
            record['alpha'].append(alpha)

            # Logit on training data
            FA = bst.predict(data=dtrain)

            # Logit on validation data
            if A_eval is not None:
                FA_eval = bst.predict(data=deval)

            # Verbose
            if verbose and i % print_every == 0:
                if pvals_eval is not None:
                    print(f'Iteration: {i:>5d}, Lq: {Lq:12.8f}, Lq_eval: {Lq_eval:12.8f}, alpha: {alpha:7.5f}')
                else:
                    print(f'Iteration: {i:>5d}, Lq: {Lq:12.8f}, alpha: {alpha:7.5f}')
                    
        # Final posterior
        record['post'] = z1

        # Variable importance
        var_imp = np.zeros(D)
        imp_scores = bst.get_score(importance_type='total_gain')
        for k, item in imp_scores.items():
            var_imp[int(k.replace('f', ''))] = item
        record['var_imp'] = var_imp
        
    elif module == 'gbm':
        
        dtrain = A
        if A_eval is not None:
            M_eval = A_eval.shape[0]
            deval = A_eval
            
        # Initialize booster, starting iteration and parameters
        start_iteration = 0
        num_boost_round = max(num_boost_round, 1)
        
        bst = {} # tree ensemble
        FA = params['base_score'] * np.ones(M)
    
        # Record
        record = {'post': None, 'Lq': [], 'alpha': [], 'var_imp': None}
        if pvals_eval is not None:
            record = {'post': None, 'Lq': [], 'Lq_eval': [], 'alpha': [], 'var_imp': None}
            FA_eval = params['base_score'] * np.ones(M_eval)
    
        # EM loop
        for i in range(start_iteration, num_boost_round):

            # E step
            z1, g, Lq = _E_tree(pvals, FA, alpha)
            record['Lq'].append(Lq)

            # Likelihood on validation data
            if pvals_eval is not None:
                _, _, Lq_eval = _E_tree(pvals_eval, FA_eval, alpha)
                record['Lq_eval'].append(Lq_eval)

            # M step
            bst[i] = DTR(max_depth=params['max_depth']) # initialize a new tree
            alpha, bst = _M_tree(pvals, z1, g, dtrain, i, bst, module)
            record['alpha'].append(alpha)

            # Logit on training data
            FA = FA + params['eta'] * bst[i].predict(X=dtrain)

            # Logit on validation data
            if A_eval is not None:
                FA_eval = FA_eval + params['eta'] * bst[i].predict(X=deval)

            # Verbose
            if verbose and i % print_every == 0:
                if pvals_eval is not None:
                    print(f'Iteration: {i:>5d}, Lq: {Lq:12.8f}, Lq_eval: {Lq_eval:12.8f}, alpha: {alpha:7.5f}')
                else:
                    print(f'Iteration: {i:>5d}, Lq: {Lq:12.8f}, alpha: {alpha:7.5f}')
            
        # Final posterior
        record['post'] = z1

        # Variable importance
        imp_scores = [f.feature_importances_ for f in bst.values()]
        var_imp = np.vstack(imp_scores).mean(axis=0)
        record['var_imp'] = var_imp

    return record


def palm_tree(pvals, A, params, inits, logger, K=5, max_iter=5000, print_every=1000, verbose=True, module='xgb', FDRset=0.1):
    
    '''
    Tree-based PALM.
    
    :param pvals: training p-values of snps, np.ndarray (M,)
    :param A: training annotation matrix, np.ndarray (M, D)
    :param params: hyperparameters of xgboost, {'max_depth', 'eta', 'gamma', 'lambda', 'base_score', 'nthread'}
    :param inits: initial values, {'alpha, 'f0'}
    :param logger: Logger object
    :param max_iter: maximum number of iterations
    :param print_every: print frequency
    :param verbose: verbose or not
    :param module: xgboost-type or gbm-type algorithm
    :param FDRset: FDR control level
    :return: result dictionary of the (K+1)-th model, likelihood on test folds, times for cv and fitting the (K+1)-th model
    '''

    # print(f"[{datetime.today().strftime('%Y-%m-%d %H:%M:%S')}] ********** Begin PALM-Tree cross-validation **********")
    # print(f'Parameters of PALM-Tree: {params}')
    logger.info("Begin PALM-Tree cross-validation")
    logger.info(f'Parameters of PALM-Tree: {params}')

    
    M, D = A.shape
    
    # Initial values
    alpha = inits['alpha']
    f0 = inits['f0']
    params['base_score'] = f0
    
    # Select best number of iterations via cross-validation
    idx = np.arange(M)
    kf = KFold(n_splits=K)
    cv_scores = np.zeros((max_iter, K))
    times = pd.DataFrame(columns=['stage', 'time'])
    times['stage'] = [f'fold {k}' for k in range(K)] + ['final']
    for k, (train, test) in enumerate(kf.split(idx)):
        pvals_train, A_train, pvals_eval, A_eval = pvals[train], A[train], pvals[test], A[test]
        t0 = time.time()
        ret = EM_tree(pvals_train, A_train, params, pvals_eval, A_eval,
                        alpha, max_iter, print_every, verbose, module)
        times.iloc[k, 1] = round(time.time() - t0, 2)
        cv_scores[:, k] = ret['Lq_eval']
        # print(f"[{datetime.today().strftime('%Y-%m-%d %H:%M:%S')}] Finish fold {k}.")
        logger.info(f"Finish fold {k}.")

    # Fit the (K+1)-th model
    cv_score = np.mean(cv_scores, axis=1)
    cv_opt_iter = np.argmax(cv_score[1:]) + 1
    t0 = time.time()
    ret_opt = EM_tree(pvals, A, params, None, None, alpha, cv_opt_iter, print_every, verbose, module)
    times.iloc[K, 1] = round(time.time() - t0, 2)
    # print(f"[{datetime.today().strftime('%Y-%m-%d %H:%M:%S')}] Finish fitting {K+1}-th model.")
    logger.info("Finish fitting the final model.")

    # SNP prioritization
    est_Z = Assoc_SNPs(ret_opt['post'], FDRset=FDRset, fdrcontrol='global')
    # print(f"[{datetime.today().strftime('%Y-%m-%d %H:%M:%S')}] Indentified risk SNPs.")
    logger.info("Identified risk SNPs.")

    # Result
    ret_opt['opt_iter'] = cv_opt_iter
    ret_opt['assoc'] = est_Z
    
    return ret_opt, cv_scores, times


# 3-layer MLP
class NN(nn.Module):
    def __init__(self, D, d, p=0):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(D, d)
        self.fc2 = nn.Linear(d, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    
def _E_nn(pvals, FA, alpha):
    
    M = len(pvals)
    pi = FA.softmax(1)  # (M, 2): (pi0, pi1)
    pi0, pi1 = pi[:, 0], pi[:, 1]
    comp1 = pi1 * alpha * pvals ** (alpha - 1)
    comp1 = torch.clamp(comp1, 1e-16, 1 - 1e-16)
    z1 = comp1 / (comp1 + pi0)
    Lq = torch.sum(torch.log(pi1 * alpha * pvals ** (alpha - 1) + pi0))
    Lq = Lq.item() / M
    
    return z1, Lq

def _M_nn(pvals, z1, FA, net, optimizer):
    
    # Update alpha
    alpha = - (torch.sum(z1[pvals > 0]) / torch.dot(z1[pvals > 0], torch.log(pvals[pvals > 0]))).item()

    # Update network parameters
    optimizer.zero_grad()
    label = torch.vstack((1 - z1, z1)).transpose(0, 1) # pred: FA (M, 2), label: (M, 2)
    loss = criterion(FA[pvals > 0], label[pvals > 0])
    loss.backward()
    optimizer.step()
    
    return alpha, net
    
           
def EM_nn(pvals, A, net, optimizer, pvals_eval=None, A_eval=None, 
            alpha=0.4, max_iter=1000, print_every=100, verbose=True, rank_var=False):
    
    '''
    EM algorithm for PALM.
    
    :param pvals: training p-values of snps, np.ndarray (M_train,)
    :param A: training annotation matrix, np.ndarray (M_train, D)
    :param net: neural network, torch.nn.Module
    :param optimizer: optimizer, torch.nn.optim
    :param pvals_eval: evaluation p-values of snps, np.ndarray (M_eval,)
    :param A_eval: evaluation annotation matrix, np.ndarray (M_eval, D)
    :param alpha: initial alpha (parameter of beta distribution)
    :param max_iter: maximum number of iterations
    :param print_every: print frequency
    :param verbose: verbose or not
    :param rank_var: rank variables or not
    :return: a dictionary containing estimated parameters, likelihood, posterior (and variable importance)
    '''
    
    # Data size
    M = len(pvals)
    
    # Clamp pvals
    pvals = torch.clamp(pvals, 1e-16, 1 - 1e-16)
    if pvals_eval is not None:
        pvals_eval = torch.clamp(pvals_eval, 1e-16, 1 - 1e-16)
        
    # Record
    record = {'post': None, 'Lq': [], 'alpha': []}
    if pvals_eval is not None:
        M_eval = len(pvals_eval)
        record = {'post': None, 'Lq': [], 'Lq_eval': [], 'alpha': []}
    net.train()
    FA = net(A)

    # EM loop
    for i in range(max_iter):
                
        # E step
        with torch.no_grad():
            
            # Likelihood on validation data
            if pvals_eval is not None:

                net.eval()
                FA_eval = net(A_eval)
                _, Lq_eval = _E_nn(pvals_eval, FA_eval, alpha)
                record['Lq_eval'].append(Lq_eval)
            
            z1, Lq = _E_nn(pvals, FA, alpha)
            record['Lq'].append(Lq)
            
        # M step
        alpha, net = _M_nn(pvals, z1, FA, net, optimizer)
        record['alpha'].append(alpha)
        
        # Update logit
        net.train()
        FA = net(A)
        
        # Verbose
        if verbose and i > 0 and (i % print_every == 0 or i == 1):
            if pvals_eval is not None:
                print(f'Iteration: {i:>5d}, Lq: {Lq:12.8f}, Lq_eval: {Lq_eval:12.8f}, alpha: {alpha:7.5f}')
            else:
                print(f'Iteration: {i:5d}, Lq: {Lq:12.8f}, alpha: {alpha:7.5f}')
    
    # Final posterior
    record['post'] = z1.cpu().detach().numpy()

    # Variable importance
    if rank_var:
        var_imp = get_nn_var_imp(net, pvals, A, alpha)
        record['var_imp'] = var_imp
        
    return record


def palm_nn(pvals, A, params, inits, logger, K=5, max_iter=1000, print_every=200, verbose=True, rank_var=False, FDRset=0.1):
    
    '''
    Network-based PALM.
    
    :param pvals: training p-values of snps, np.ndarray (M,)
    :param A: training annotation matrix, np.ndarray (M, D)
    :param params: hyperparameters of network and optimizer, {'d', 'lr', 'w', 'device'}
    :param inits: initial values, {'alpha, 'f0'}
    :param max_iter: maximum number of iterations
    :param print_every: print frequency
    :param verbose: verbose or not
    :param rank_var: rank variables or not
    :param FDRset: FDR control level
    :return: result dictionary of the (K+1)-th model, likelihood on test folds, times for cv and fitting the (K+1)-th model
    '''

    # print(f"[{datetime.today().strftime('%Y-%m-%d %H:%M:%S')}] ********** Begin PALM-NN cross-validation **********")
    # print(f'Parameters of PALM-NN: {params}')
    logger.info("Begin PALM-NN cross-validation")
    logger.info(f'Parameters of PALM-NN: {params}')

    
    # Number of annotations
    M, D = A.shape
    pvals, A = torch.Tensor(pvals), torch.Tensor(A)
    
    # Initial values
    alpha = inits['alpha']
    
    # Settings
    d, lr, w, device = params.values()
    
    # Select best number of iterations via cross-validation
    idx = np.arange(M)
    kf = KFold(n_splits=K)
    cv_scores = np.zeros((max_iter, K))
    times = pd.DataFrame(columns=['stage', 'time'])
    times['stage'] = [f'fold {k}' for k in range(K)] + ['final']
    for k, (train, test) in enumerate(kf.split(idx)):
        
        pvals_train, A_train, pvals_eval, A_eval = pvals[train].to(device), A[train].to(device), pvals[test].to(device), A[test].to(device)
        
        net = NN(D, d).to(device)
        optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=w)
        
        t0 = time.time()
        ret = EM_nn(pvals_train, A_train, net, optimizer, pvals_eval, A_eval,
                        alpha, max_iter, print_every, verbose)
        times.iloc[k, 1] = round(time.time() - t0, 2)
        
        cv_scores[:, k] = ret['Lq_eval']
        # print(f"[{datetime.today().strftime('%Y-%m-%d %H:%M:%S')}] Finish fold {k}.")
        logger.info(f"Finish fold {k}.")
    
    # Fit the (K+1)-th model
    pvals, A = pvals.to(device), A.to(device)
    cv_score = np.mean(cv_scores, axis=1)
    cv_opt_iter = np.argmax(cv_score[1:]) + 1
    net = NN(D, d).to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=w)
    
    t0 = time.time()
    ret_opt = EM_nn(pvals, A, net, optimizer, None, None, alpha, cv_opt_iter, print_every, verbose, rank_var)
    times.iloc[K, 1] = round(time.time() - t0, 2)

    # print(f"[{datetime.today().strftime('%Y-%m-%d %H:%M:%S')}] Finish fitting {K+1}-th model.")
    logger.info("Finish fitting the final model.")

    # SNP prioritization
    est_Z = Assoc_SNPs(ret_opt['post'], FDRset=FDRset, fdrcontrol='global')
    # print(f"[{datetime.today().strftime('%Y-%m-%d %H:%M:%S')}] Indentified risk SNPs.")
    logger.info("Indentified risk SNPs.")

    # Result
    ret_opt['opt_iter'] = cv_opt_iter
    ret_opt['assoc'] = est_Z
    
    return ret_opt, cv_scores, times

        