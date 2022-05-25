### Basic two-groups model
import os

nthread = '20'
os.environ["OMP_NUM_THREADS"] = nthread
os.environ["OPENBLAS_NUM_THREADS"] = nthread
os.environ["MKL_NUM_THREADS"] = nthread
os.environ["VECLIB_MAXIMUM_THREADS"] = nthread
os.environ["NUMEXPR_NUM_THREADS"] = nthread

import argparse
import traceback
import numpy as np
import pandas as pd
from utils import Assoc_SNPs

def TGM(pvals, inits, max_iter=1000, print_every=500, verbose=True):
    
    M = len(pvals)

    # Initial values
    alpha = inits['alpha']
    f0 = inits['f0']
    
    # Record
    record = {'post': None, 'Lq': [], 'alpha': [], 'f0': []}

    for i in range(max_iter):
        
        ## E step
        pi1 = 1 / (1 + np.exp(-f0))
        pi0 = 1 - pi1
        comp1 = pi1 * alpha * pvals ** (alpha - 1)
        comp1 = np.clip(comp1, 1e-16, 1e16)
        z1 = comp1 / (comp1 + pi0)
        Lq = np.sum(np.log(pi1 * alpha * pvals ** (alpha - 1) + pi0))
        Lq /= M

        ## M step
        alpha = - np.sum(z1) / np.dot(z1, np.log(pvals))
        f0 = np.log(np.sum(z1)) - np.log(np.sum(1 - z1))

        # record
        record['alpha'].append(alpha)
        record['f0'].append(f0)
        record['Lq'].append(Lq)

        # verbose
        if verbose and i % print_every == 0:
            print(f'Iteration: {i:>5d}, Lq: {Lq:12.8f}, alpha: {alpha:7.5f}')
            
    record['post'] = z1
    
    # SNP prioritization
    est_Z = Assoc_SNPs(record['post'], fdrcontrol='global')
    record['assoc'] = est_Z

    return record

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Basic two-groups model for risk variants prioritization')
    # Input/Output
    parser.add_argument('--out', type=str, help='Output file path', required=True)
    parser.add_argument('--sumstats', type=str, help='GWAS summary statistics file', required=True)
    # Initial values
    parser.add_argument('--alpha', type=float, default=0.4, help='Initial value of alpha (shape parameter of beta distribution)')
    parser.add_argument('--f0', type=float, default=-3, help='Initial value of f0 (constant in the prior model)')
    # Setting
    parser.add_argument('--max-iter', type=int, default=2000, help='Maximum number of iterations')
    parser.add_argument('--print-freq', type=int, default=500, help='Print frequency')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose inside algorithm')
    parser.add_argument('-q', '--quiet', action='store_true', help='Quiet inside algorithm')

    args = parser.parse_args()
        
    # GWAS and annotation data
    gwas = pd.read_csv(args.sumstats, sep='\t')
    pvals = gwas['pvalue'].to_numpy()
    
    # Parameters
    inits = {'alpha': args.alpha, 'f0': args.f0} # default initial values
    
    # Prioritize SNPs
    try:
        result = TGM(pvals, inits, max_iter=args.max_iter, print_every=args.print_freq, verbose=args.verbose)
    except Exception as e:
        print(traceback.format_exc())
        
    # Output result
    gwas['post'] = result['post']
    gwas['assoc'] = result['assoc']
    gwas.to_csv(f'{args.out}-TGM-prioritize.csv', sep='\t', index=False)

