import os
import sys
import time
import argparse
import traceback
import pandas as pd
from model import palm_tree, palm_nn
from utils import Logger

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='A powerful and adaptive latent model for risk variants prioritization')
    # Input/Output
    parser.add_argument('--out', type=str, help='Output file path', required=True)
    parser.add_argument('--sumstats', type=str, help='GWAS summary statistics file', required=True)
    parser.add_argument('--annotation', type=str, help='Annotation data file', required=True)
    parser.add_argument('--save-alpha', action='store_true', help='Save estimated alpha')
    parser.add_argument('--save-importance', action='store_true', help='Save evaluated variable importance')
    # Model choice
    parser.add_argument('--model', type=str, default='tree', choices=['tree', 'nn'], help='Tree-based PALM or network-based PALM')
    parser.add_argument('--module', type=str, default='xgb', choices=['xgb', 'gbm'], help='Tree-based PALM based on XGBoost-type or GBM-type algorithm')
    # PALM-Tree parameters
    parser.add_argument('--depth', type=int, default=2, help='Tree depth for tree-based PALM')
    parser.add_argument('--eta', type=float, default=0.1, help='Shrinkage parameter for tree-based PALM')
    parser.add_argument('--gam', type=float, default=0, help='XGBoost gamma, penalize number of leaves')
    parser.add_argument('--lam', type=float, default=0, help='XGBoost lambda, penalize leaf weights')
    # PALM-NN parameters
    parser.add_argument('--hidden', type=int, default=100, help='Number of hidden dimension for network-based PALM')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate for network-based PALM')
    parser.add_argument('--decay', type=float, default=0.002, help='Weight decay for network-based PALM')
    parser.add_argument('--device', type=str, default='cpu', help='Device for network-based PALM')
    # Initial values
    parser.add_argument('--alpha', type=float, default=0.4, help='Initial value of alpha (shape parameter of beta distribution)')
    parser.add_argument('--f0', type=float, default=-3, help='Initial value of f0 (constant in the prior model)')
    # Setting
    parser.add_argument('--max-iter', type=int, default=3000, help='Maximum number of iterations')
    parser.add_argument('--print-freq', type=int, default=1000, help='Print frequency')
    parser.add_argument('--nfold', type=int, default=5, help='Number of folds in cross-validation')
    parser.add_argument('--FDRset', type=float, default=0.1, help='FDR control level')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose inside algorithm')
    parser.add_argument('-q', '--quiet', action='store_true', help='Quiet inside algorithm')
    parser.add_argument('--nthread', type=int, default=20, help='Number of threads')
    
    args = parser.parse_args()
    
    os.environ["OMP_NUM_THREADS"] = str(args.nthread)
    os.environ["OPENBLAS_NUM_THREADS"] = str(args.nthread)
    os.environ["MKL_NUM_THREADS"] = str(args.nthread)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(args.nthread)
    os.environ["NUMEXPR_NUM_THREADS"] = str(args.nthread)
    
    # Log file
    log_file = f'{args.out}-PALM-{args.model}.log'
    open(log_file, 'w').close()
    logger = Logger(log_file)
    logger.info("\n\nProgram executed via:\n%s\n" % ' '.join(sys.argv).replace("--", " \\ \n\t--"))

    # GWAS and annotation data
    gwas = pd.read_csv(args.sumstats, sep='\t')
    anno = pd.read_csv(args.annotation, sep='\t')
    merge = pd.merge(gwas, anno, on='SNP', how='left') # should merge based on gwas
    # missing values: only xgboost can handle, sklearn decision tree and PALM-NN cannot
    if args.model == 'nn' or args.module == 'gbm':
        merge.fillna(0, inplace=True)
    pvals = merge['pvalue'].to_numpy()
    A = merge.iloc[:, gwas.shape[1]:].to_numpy()
    anno_names = list(anno.columns)
    anno_names.remove('SNP')
    print(f'Number of p-values: {len(pvals)}, shape of annotation: {A.shape}')
    
    # Parameters
    inits = {'alpha': args.alpha, 'f0': args.f0} # default initial values
    if args.model == 'tree':
        params = {'max_depth': args.depth, 'eta': args.eta, 'gamma': args.gam, 'lambda': args.lam, 'nthread': args.nthread}
    elif args.model == 'nn':
        params = {'hidden_dim': args.hidden, 'lr': args.lr, 'weight_decay': args.decay, 'device': args.device}
        
    # Prioritize SNPs
    try:
        if args.model == 'tree':
            result, cv_scores, times = palm_tree(pvals, A, params, inits, logger, \
                                                 K=args.nfold, max_iter=args.max_iter, print_every=args.print_freq, \
                                                 verbose=args.verbose, module=args.module, FDRset=args.FDRset)
        elif args.model == 'nn':
            result, cv_scores, times = palm_nn(pvals, A, params, inits, logger, \
                                               K=args.nfold, max_iter=args.max_iter, print_every=args.print_freq, \
                                               verbose=args.verbose, rank_var=args.save_importance)
    except Exception as e:
        logger.error(e)
        print(traceback.format_exc())
        
    # Save default outputs
    gwas['post'] = result['post']
    gwas['assoc'] = result['assoc']
    cv_df = pd.DataFrame(cv_scores, columns=[f'fold{k}' for k in range(args.nfold)])
    gwas.to_csv(f'{args.out}-PALM-{args.model}-prioritize.csv', sep='\t', index=False)
    cv_df.to_csv(f'{args.out}-PALM-{args.model}-cv.csv', sep='\t', index=False)
    times.to_csv(f'{args.out}-PALM-{args.model}-times.csv', sep='\t', index=False)

    # Save optional outputs
    if args.save_importance:
        imp_df = pd.DataFrame(columns=['annot', 'importance'])
        imp_df['annot'] = anno_names
        imp_df['importance'] = result['var_imp']
        imp_df.to_csv(f'{args.out}-PALM-{args.model}-importance.csv', sep='\t', index=False)
    if args.save_alpha:
        alpha_df = pd.DataFrame(columns=['step', 'alpha'])
        alpha_df['step'] = list(range(len(result['alpha'])))
        alpha_df['alpha'] = result['alpha']
        alpha_df.to_csv(f'{args.out}-PALM-{args.model}-alpha.csv', sep='\t', index=False) 
    
    