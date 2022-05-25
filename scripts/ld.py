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
from pathlib import Path
from PALM.utils import Generate_data_disc

def palm_real_geno(J, K, v, gam, lam, r, data_dir, out_dir):
    
    # PALM-Tree
    command = f'python ./PALM/prioritize.py \
    --out {out_dir}/J{J}-K{K}-v{v}-gam{gam}-lam{lam}-r{r} \
    --sumstats {data_dir}/gwas-r{r}.csv \
    --annotation {data_dir}/anno.csv \
    --model tree --depth {J} --eta {v} --gam {gam} --lam {lam} \
    --max-iter {max_iter} --print-freq {print_every} --nfold {K} -v --nthread {nthread} \
    --save-alpha'
    
    os.system(command)
    
def tgm_real_geno(r, data_dir, out_dir):
    
    command = f'python ./PALM/tgm.py \
    --out {out_dir}/r{r} \
    --sumstats {data_dir}/gwas-r{r}.csv \
    --max-iter {max_iter} --print-freq {print_every} -v'
    
    os.system(command)
    
    
if __name__ == '__main__':
    
    # Simulation setting
    Js = [1, 2]
    v = 0.1
    gam = 0
    lam = 0
    K = 5
    nrep = 50
    max_iter = 5000
    print_every = 1000
    nthread = 1
    out_root = './simu/real_geno'
    Path(out_root).mkdir(parents=True, exist_ok=True)
    out_dir = out_root
    data_dir = out_dir
    
    # Generate data
    D = 100
    nrep = 50
    out_root = './simu/real_geno'
    # Prepare genotype data
    raw_bfile = '/home/share/WTCCC/ctrl/ctrl'
    qc_bfile = './simu/real_geno/ctrl'
    plink1 = f'/home/share/xiaojs/software/plink --bfile {raw_bfile} --geno 0.01 --hwe 0.001 --maf 0.05 --make-bed --mind 0.05 --out {qc_bfile}'
    os.system(plink1)
    print('Prepared genotype.')

    # Generate causal SNPs: every 1000 SNPs, one causal SNP (on average)
    ref = pd.read_csv('/home/share/UKB/1kg_ref/maf_0.05/1000G.EUR.QC.hm3.ind.bim', sep='\t', header=None)
    bim = pd.read_csv('./simu/wtccc/ctrl.bim', sep='\t', header=None)
    chr1 = bim.loc[(bim[0]==1) & (bim[1].str.contains('rs'))] 
    ref.columns = ['CHR', 'SNP', 'FRQ', 'BP', 'A1', 'A2']
    chr1.columns = ['CHR', 'SNP', 'POS', 'BP', 'A1', 'A2']
    merge = pd.merge(ref, chr1[['SNP']], how='inner')
    merge = merge.sort_values(by='BP').reset_index(drop=True)
    M = len(merge)

    # Write all snps to file
    all_snps = list(merge['SNP'].values)
    allsnp_file = './simu/real_geno/ctrl-allsnp.txt'
    with open(allsnp_file, 'w') as f:
        f.writelines('\n'.join(all_snps))
        f.write('\n')
        
    # Write causal snps to file
    cau_idx = np.sort(np.random.choice(np.arange(M), round(M/1000), replace=False))
    cau_snps = list(merge['SNP'].values[cau_idx])

    cau_file = f'./simu/real_geno/ctrl-causal.txt'
    with open(cau_file, 'w') as f:
        f.writelines('\n'.join(cau_snps))
        f.write('\n')
    print('Wrote causal SNPs.')
    
    # 10% elements of A are 1
    A = np.random.binomial(1, 0.1, M * D).reshape(M, D)
    
    # Relevant annotation columns
    relev_prop = 0.2
    relev_cols = np.random.choice(np.arange(D), int(D*relev_prop), replace=False)
    
    # SNPs within 1MB of causal SNPs
    anno_idx = []
    n = 0
    for i, row in merge.iterrows():
        if n < len(cau_idx) - 1:
            if abs(row['BP'] - merge.iloc[cau_idx[n]]['BP']) <= 1000000:
                anno_idx.append(i)
            else:
                if merge.iloc[cau_idx[n+1]]['BP'] - row['BP'] <= 1000000 and row['BP'] < merge.iloc[cau_idx[n+1]]['BP']:
                    anno_idx.append(i)
                    n += 1
        else:
            if abs(row['BP'] - merge.iloc[cau_idx[n], 1]) <= 1000000:
                anno_idx.append(i)
    anno_idx = list(set(anno_idx))
    
    # 20% A's columns of these SNPs: 60% elements are 1
    A[anno_idx, :][:, relev_cols] = np.random.binomial(1, 0.6, len(anno_idx) * int(D*relev_prop)).reshape(len(anno_idx), int(D*relev_prop))
    anno = pd.DataFrame(columns=['SNP'] + [f'A{i}' for i in range(1, D + 1)])
    anno['SNP'] = merge['SNP'].values
    anno.iloc[:,1:] = A
    
    # Save annotation file
    anno_file = os.path.join(out_root, f'anno.csv')
    anno.to_csv(anno_file, sep='\t', index=False)

    for r in range(1, nrep+1):

        # Generate phenotype by GCTA
        gcta_out = f'./simu/real_geno/ctrl-r{r}'
        gcta = f'/home/share/xiaojs/software/gcta_1.93.0beta/gcta64 --bfile {qc_bfile} --simu-qt  --simu-causal-loci {cau_file}  --simu-hsq 0.05 --simu-rep 1  --out {gcta_out}'
        os.system(gcta)
        # Calculate p-values by plink
        pheno_file = f'{gcta_out}.phen'
        plink2 = f'/home/share/xiaojs/software/plink --bfile {qc_bfile} --pheno {pheno_file} --out {gcta_out} --linear --extract {allsnp_file}'
        os.system(plink2)

        # Create gwas for PALM
        gwas_file = f'{gcta_out}.assoc.linear'
        gwas = pd.read_csv(gwas_file, delim_whitespace=True)
        # get BP by merging with ref
        gwas = pd.merge(gwas.drop(columns=['BP']), ref[['SNP', 'BP']], on='SNP', how='inner')
        gwas = gwas.sort_values(by='BP').reset_index(drop=True)
        gwas_palm = gwas[['SNP', 'BP', 'P']].rename(columns={'P': 'pvalue'})
        gwas_palm['Z'] = np.where(gwas_palm['SNP'].isin(cau_snps), 1, 0)
        gwas_palm.head()

        # Save gwas
        gwas_file = os.path.join(out_root, f'gwas-r{r}.csv')
        gwas_palm.to_csv(gwas_file, sep='\t', index=False)

        print(r)
    
    # PALM-Tree
    inp_args = []
    for J in Js:
        for r in range(1, nrep+1):
            inp_args.append((J, K, v, gam, lam, r, data_dir, out_dir))

    try:
        with Pool(nrep) as pool:
            pool.starmap(palm_real_geno, inp_args)
            pool.close()
            pool.join()
    except Exception as e:
        print(e)
        
    # TGM
    max_iter = 2000
    print_every = 1000
    nthread = 1
    nrep = 50
    data_dir = './simu/real_geno'
    out_dir = './simu/real_geno'
    
    inp_args = []
    for r in range(1, nrep+1):
        inp_args.append((r, data_dir, out_dir))

    try:
        with Pool(nrep) as pool:
            pool.starmap(tgm_real_geno, inp_args)
            pool.close()
            pool.join()
    except Exception as e:
        print(e)
        
