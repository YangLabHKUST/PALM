import os, sys

os.environ["OMP_NUM_THREADS"] = '1'
os.environ["OPENBLAS_NUM_THREADS"] = '1'
os.environ["MKL_NUM_THREADS"] = '1'
os.environ["VECLIB_MAXIMUM_THREADS"] = '1'
os.environ["NUMEXPR_NUM_THREADS"] = '1'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from multiprocessing import Pool

# Simuation function: two-groups model, PALM-Tree 5-fold cv and PALM-Tree with optimal number of iterations
def simu_module(M, D, Func_name, J, v, gam, lam, r, data_dir, out_dir):
    
    # PALM-Tree
    command = f'python ./PALM/prioritize.py \
    --out {out_dir}/{Func_name}-J{J}-v{v}-gam{gam}-lam{lam}-r{r} \
    --sumstats {data_dir}/gwas-{Func_name}-r{r}.csv \
    --annotation {data_dir}/anno-{Func_name}-r{r}.csv \
    --model tree --depth {J} --eta {v} --gam {gam} --lam {lam} \
    --max-iter {max_iter} --print-freq {print_every} --nfold {K} -v --nthread {nthread} \
    --module gbm'
    
    os.system(command)
    
if __name__ == '__main__':
    
    # Simulation setting
    M = 20000
    D = 50
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
    
    # Multiprocess
    out_dir = os.path.join(out_root, f'comp_M{M}_D{D}_K{K}_gbm')
    data_dir = os.path.join(data_root, f'comp_M{M}_D{D}_K{K}_xgb')
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
        
    for Func_name in Func_names:
        for J in Js:
            inp_args = []
            for r in range(1, nrep+1):
                inp_args.append((M, D, Func_name, J, v, gam, lam, r, data_dir, out_dir))

            try:
                with Pool(nrep) as pool:
                    pool.starmap(simu_module, inp_args)
                    pool.close()
                    pool.join()
            except Exception as e:
                print(e)
                
                