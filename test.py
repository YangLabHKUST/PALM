import os, sys

threads = '1'
os.environ["OMP_NUM_THREADS"] = threads
os.environ["OPENBLAS_NUM_THREADS"] = threads
os.environ["MKL_NUM_THREADS"] = threads
os.environ["VECLIB_MAXIMUM_THREADS"] = threads
os.environ["NUMEXPR_NUM_THREADS"] = threads

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
from pathlib import Path
from PALM.utils import *

def test_data(M, D, true_alpha, Func_name):
    gwas_file = os.path.join(data_dir, f'gwas-test.csv')
    anno_file = os.path.join(data_dir, f'anno-test.csv')
    if not os.path.exists(gwas_file) or not os.path.exists(anno_file):
        Func = eval(Func_name)
        pvals, A, Z = Generate_data_gen(M, D, true_alpha, Func)
        # save data
        gwas = pd.DataFrame(columns=['SNP', 'pvalue', 'Z'])
        anno = pd.DataFrame(columns=['SNP'] + [f'A{i}' for i in range(1, D + 1)])
        gwas['SNP'] = np.arange(M)
        gwas['pvalue'] = pvals
        gwas['Z'] = Z
        anno['SNP'] = np.arange(M)
        anno.iloc[:,1:] = A
        gwas.to_csv(gwas_file, sep='\t', index=False)
        anno.to_csv(anno_file, sep='\t', index=False)
        
def test_palm():
    
    command = f'python ./PALM/prioritize.py \
    --out {out_dir}/out-test \
    --sumstats {data_dir}/gwas-test.csv \
    --annotation {data_dir}/anno-test.csv \
    --model {model} --depth {J} --eta {v} --gam {gam} --lam {lam} \
    --max-iter {max_iter} --print-freq {print_every} --nfold {K} -v --nthread {nthread}'
    
    os.system(command)
    
def test_evaluate():
    
    command = f'python ./PALM/evaluate.py --result {out_dir}/out-test-PALM-{model}-prioritize.csv'
    
    os.system(command)
    
    
if __name__ == '__main__':
    
    # Setting
    M = 10000
    D = 10
    J = 1
    true_alpha = 0.2
    Func_name = 'Func_B'
    v = 0.1
    gam = 0
    lam = 0
    K = 2
    max_iter = 1000
    print_every = 200
    nthread = 20
    model = 'tree'
    out_dir = 'test'
    data_dir = 'test'
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    Path(data_dir).mkdir(parents=True, exist_ok=True)

    print('********** Begin testing PALM **********')
    
    # Generate data
    test_data(M, D, true_alpha, Func_name)

    # Run PALM-Tree
    test_palm()

    # Evaluate performance
    test_evaluate()

    print('********** Finish testing PALM **********')
    
    
    
