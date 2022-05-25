import os, sys

os.environ["OMP_NUM_THREADS"] = '1'
os.environ["OPENBLAS_NUM_THREADS"] = '1'
os.environ["MKL_NUM_THREADS"] = '1'
os.environ["VECLIB_MAXIMUM_THREADS"] = '1'
os.environ["NUMEXPR_NUM_THREADS"] = '1'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
from multiprocessing import Pool
from PALM.utils import Generate_data_disc

# Generate and save data
def generate_data(M, D, Func_name, r, data_dir):
    gwas_file = os.path.join(data_dir, f'gwas-M{M}-D{D}-{Func_name}-r{r}.csv')
    anno_file = os.path.join(data_dir, f'anno-M{M}-D{D}-{Func_name}-r{r}.csv')
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
        
def simu_compu_time(M, D, Func_name, J, v, gam, lam, r, data_dir, out_dir):
    
    # PALM-Tree
    command = f'python ./PALM/prioritize.py \
    --out {out_dir}/M{M}-D{D}-{Func_name}-J{J}-v{v}-gam{gam}-lam{lam}-r{r} \
    --sumstats {data_dir}/gwas-M{M}-D{D}-{Func_name}-r{r}.csv \
    --annotation {data_dir}/anno-M{M}-D{D}-{Func_name}-r{r}.csv \
    --model tree --depth {J} --eta {v} --gam {gam} --lam {lam} \
    --max-iter {max_iter} --print-freq {print_every} --nfold {K} -v --nthread {nthread}'
    os.system(command)
    
if __name__ == '__main__':
    
    # Simulation setting
    Ms = [5000, 10000, 15000]
    Ds = [10, 30, 50]
    Js = [1, 2]
    Func_name = 'Func_D'
    v = 0.1
    gam = 0
    lam = 0
    K = 2
    nrep = 30
    max_iter = 5000
    print_every = 1000
    nthread = 1
    out_root = './simu/tree'
    data_root = './simu/tree'

    # Multiprocess
    out_dir = os.path.join(out_root, f'time_K{K}')
    data_dir = os.path.join(data_root, f'time_K{K}')
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # generate data
    for M in Ms:
        for D in Ds:
            for r in range(1, nrep+1):
                generate_data(M, D, Func_name, r, data_dir)

    for M in Ms:
        for D in Ds:
            inp_args = []
            for J in Js:
                for r in range(1, nrep+1):
                    inp_args.append((M, D, Func_name, J, v, gam, lam, r, data_dir, out_dir))

            try:
                with Pool(len(Js)*nrep) as pool:
                    pool.starmap(simu_compu_time, inp_args)
                    pool.close()
                    pool.join()
            except Exception as e:
                print(e)
