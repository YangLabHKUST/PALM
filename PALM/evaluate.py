import os
import argparse
import pandas as pd
from utils import Perf

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Evaluate performance of PALM if ground-truth is available.')
    # Input
    parser.add_argument('--result', type=str, help='PALM Prioritization result including ground-true latent variables', required=True)

    args = parser.parse_args()
    
    ret_opt = pd.read_csv(args.result, sep='\t')
    est_Z = ret_opt['assoc'] # risk SNPs have been identified
    perf = Perf(ret_opt['Z'], est_Z, 1 - ret_opt['post']) # FDR, power, AUC, pAUC
    
    # Print performance
    print(f"\nPerformance:\n"
          f"-- FDR: {perf['FDR']}\n"
          f"-- power: {perf['power']}\n"
          f"-- AUC: {perf['AUC']}\n"
          f"-- pAUC: {perf['pAUC']}\n")
    