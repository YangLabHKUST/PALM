import os, sys

os.environ["OMP_NUM_THREADS"] = '5'
os.environ["OPENBLAS_NUM_THREADS"] = '5'
os.environ["MKL_NUM_THREADS"] = '5'
os.environ["VECLIB_MAXIMUM_THREADS"] = '5'
os.environ["NUMEXPR_NUM_THREADS"] = '5'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import glob
from multiprocessing import Pool
from pathlib import Path

def run_real(trait, J, v):
    
    # PALM-Tree
    command = f'python ./PALM/prioritize.py \
    --out {out_dir}/{trait}-J{J}-v{v} \
    --sumstats {gwas_dir}/{trait}.csv \
    --annotation {anno_dir}/{anno}.csv \
    --model tree --depth {J} --eta {v} --gam {gam} --lam {lam} \
    --max-iter {max_iter} --print-freq {print_every} --nfold {K} -v'
    
    os.system(command)

if __name__ == '__main__':
    
    # Folders
    gwas_dir = './real/gwas30_noMHC'
    anno_dir = './real/data'
    out_dir = './real/palm_results'
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # All traits
    traits = [file.split('/')[-1].split('.')[0] for file in glob.glob(f"{gwas_dir}/*.csv")]
    
    # Annotation name
    anno = 'region9tissue127'
    
    # Settings
    Js = [1, 2]
    vs = [0.1, 0.5, 1.0]
    gam = 0
    lam = 0
    K = 5
    max_iter = 5000
    print_every = 1000
    nthread = 5
    
    # PALM-Tree1 and PALM-Tree2
    for trait in traits:
        inp_args = []
        for J in Js:
            for v in vs:
                inp_args.append((trait, J, v))
                
        try:
            with Pool(len(Js)*len(vs)) as pool:
                pool.starmap(run_real, inp_args)
                pool.close()
                pool.join()
        except Exception as e:
            print(e)


