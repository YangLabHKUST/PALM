### Main simulation in paper: PALM-Tree1, PALM-Tree2, TGM-Pval
import os, sys

threads = '1'
os.environ["OMP_NUM_THREADS"] = threads
os.environ["OPENBLAS_NUM_THREADS"] = threads
os.environ["MKL_NUM_THREADS"] = threads
os.environ["VECLIB_MAXIMUM_THREADS"] = threads
os.environ["NUMEXPR_NUM_THREADS"] = threads

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pathlib import Path
import numpy as np
import pandas as pd
from multiprocessing import Pool
from PALM.utils import Generate_data_disc

# Generate and save data
def generate_data(M, D, Func_name, r, data_dir):
    gwas_file = os.path.join(data_dir, f'gwas-{Func_name}-r{r}.csv')
    anno_file = os.path.join(data_dir, f'anno-{Func_name}-r{r}.csv')
    if not os.path.exists(gwas_file) or not os.path.exists(anno_file):
        Func = eval(Func_name)
        zs, pvals, A, Z = Generate_data_disc(M, D, Func)
        # save data
        gwas = pd.DataFrame(columns=['SNP', 'zs', 'pvalue', 'Z'])
        anno = pd.DataFrame(columns=['SNP'] + [f'A{i}' for i in range(1, D + 1)])
        gwas['SNP'] = np.arange(M)
        gwas['zs'] = zs
        gwas['pvalue'] = pvals
        gwas['Z'] = Z
        anno['SNP'] = np.arange(M)
        anno.iloc[:,1:] = A
        gwas.to_csv(gwas_file, sep='\t', index=False)
        anno.to_csv(anno_file, sep='\t', index=False)
        
# Simuation function: two-groups model, PALM-Tree 5-fold cv and PALM-Tree with optimal number of iterations
def simu_comparison(M, D, Func_name, J, v, gam, lam, r, data_dir, out_dir):
    
    # PALM-Tree
    command = f'python ./PALM/prioritize.py \
    --out {out_dir}/{Func_name}-J{J}-v{v}-gam{gam}-lam{lam}-r{r} \
    --sumstats {data_dir}/gwas-{Func_name}-r{r}.csv \
    --annotation {data_dir}/anno-{Func_name}-r{r}.csv \
    --model tree --depth {J} --eta {v} --gam {gam} --lam {lam} \
    --max-iter {max_iter} --print-freq {print_every} --nfold {K} -v --nthread {nthread}'
    
    os.system(command)
    
if __name__ == '__main__':
    
    # Simulation setting
    Ms = [20000, 50000, 100000]
    Ds = [50, 100]
    Js = [1, 2]
    Func_names = ['Func_A', 'Func_B', 'Func_C', 'Func_D', 'Func_E']
    v = 0.1
    gam = 0
    lam = 0
    K = 5
    nrep = 50
    max_iter = 5000
    print_every = 1000
    nthread = 1
    out_root = './simu/tree'
    data_root = './simu/tree'
    Path(out_root).mkdir(parents=True, exist_ok=True)
    Path(data_root).mkdir(parents=True, exist_ok=True)
    out_dir = os.path.join(out_root, f'comp_M{M}_D{D}_K{K}_xgb')
    data_dir = os.path.join(data_root, f'comp_M{M}_D{D}_K{K}_xgb')
    
    # Generate data
    for Func_name in Func_names:
        for r in range(1, nrep+1):
            generate_data(M, D, Func_name, r, data_dir)

    # Run simulation in parallel
    for Func_name in Func_names:
        for J in Js:
            inp_args = []
            for r in range(1, nrep+1):
                inp_args.append((M, D, Func_name, J, v, gam, lam, r, data_dir, out_dir))

            try:
                with Pool(nrep) as pool:
                    pool.starmap(simu, inp_args)
                    pool.close()
                    pool.join()
            except Exception as e:
                print(e)
    
    