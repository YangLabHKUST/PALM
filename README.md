# PALM: Powerful and Adaptive Latent Model
This repository holds the code for the paper "A powerful and adaptive latent model for prioritizing risk variants with functional annotations".

## Description
Many complex phenotypes are highly polygenic, making a large number of associated genetic variants fail to be discovered in GWAS. Evidence show that associated variants are enriched in some functional annotations, which offers us an opportunity to increase the statistical power of large-scale multiple testing by properly leveraging these auxiliary information. Existing methods for integrating functional annotations with GWAS results have limitations: (1) most of the existing methods assume a linear model for risk variant identification; (2) few of existing methods are scalable to handle a large number of annotations while maintaining good interpretability.

In this work, we proposed a novel statistical method to better utilize functional annotations in risk variants prioritization. Specifially, we assume the p-values from GWAS summary statistics follow a two-groups model with SNP-specific priors; we model the SNP-specific priors in the logit scale using boosted trees. An EM algorithm was developed to simultaneously optimize model parametes and add trees in a stagewise fashion such that annotations are adaptively incorporated into the model. The merits of boosted trees transplant to PALM: not only does it allow complex relationship between functional annotations and SNP association status but also automatically selects important annotations which provides some biological insights. Simulation studies and real data analysis illustrated PALM's superior performance on risk variants prioritization compared with existing representative methods and its effectiveness in identifying enriched cell-type/tissue specific annotations.

## Installation

1. Git clone the repository and install the package.

``` shell
$ git clone https://github.com/xinyiyu/PALM.git
$ cd PALM
$ conda env create -f environment.yml
$ conda activate palm
```

2. Check the installation status:
    
``` shell
$ python PALM/prioritize.py -h
```

3. Test if PALM runs successfully:

``` shell
$ python test.py
```
    
## Quick start

### Data preparation

Input files of PALM include:

- GWAS summary statistics file
- Annotation file

The GWAS summary statistics file should contain a `SNP` column and a `pvalue` column, e.g,

``` shell
$ head Bipolar_Disorder.csv
SNP	CHR	BP	MAF	A1	A2	pvalue
rs10907175	1	1130727	0.084493	A	C	0.351999999999945
rs2887286	1	1156131	0.175944	T	C	0.390399999999947
rs6685064	1	1211292	0.0666004	C	T	0.943799999999981
rs1571150	1	1474304	0.333002	C	A	0.667700000000157
rs7290	1	1477244	0.306163	T	C	0.847799999999996
rs3766180	1	1478153	0.306163	T	C	0.852900000000085
rs3766178	1	1478180	0.306163	T	C	0.801899999999967
rs7533	1	1479333	0.306163	A	G	0.85810000000001
rs7517401	1	1483010	0.349901	G	A	0.700699999999828
```

The annotation file has a `SNP` column for merging with the GWAS file and the other columns are different annotations, e.g,

``` shell
$ head region9.csv
downstream	exonic	intergenic	intronic	ncRNA_exonic	ncRNA_intronic	upstream	UTR3	UTR5	SNP
0	0	1	0	0	0	0	0	0	rs1000000
0	0	0	1	0	0	0	0	0	rs10000010
0	0	0	1	0	0	0	0	0	rs10000023
0	0	1	0	0	0	0	0	0	rs1000003
0	0	1	0	0	0	0	0	0	rs10000033
0	0	0	1	0	0	0	0	0	rs10000037
0	0	0	0	0	1	0	0	0	rs10000041
0	0	1	0	0	0	0	0	0	rs1000007
0	0	1	0	0	0	0	0	0	rs10000075
```

### Usage
PALM will integrate the GWAS summary statistics with annotations to prioritize risk SNPs. Specify `<data_dir>` containing the input files and `<out_dir>` for saving output files. The default method in the prior model is boosted trees with tree depth = 2 and shrinkage parameter = 0.1. By default, PALM will perform 5-fold cross-validation with maximum number of iterations 3000 to select the optimal number of iterations for the final model.

``` shell
python ./PALM/prioritize.py \
    --out <out_dir>/Bipolar_Disorder \
    --sumstats <data_dir>/Bipolar_Disorder.csv \
    --annotation <data_dir>/region9.csv \
    --model tree --depth 2 --eta 0.1 \
    --max-iter 3000 --nfold 5 -v
```

### Output
The main script `prioritize.py` will add two columns `post` and `assoc` to the input GWAS file and save as the new dataframe as the prioritization result. `post` represents the posterior probability of one SNP being associated with the phenotype given its p-value and annotations; `assoc` represents the SNP association status under a certain FDR control level (by default, 0.1). Besides, a file recording the log-likelihoods on test folds and a file recording computational times will be saved. And if specified, the estimated &alpha; in the two-groups model and the evaluated variable importance will be saved as well.

An example of the major output file:

``` shell
$ head Bipolar_Disorder-J2-v0.1-PALM-tree-prioritize.csv
SNP	CHR	BP	MAF	A1	A2	pvalue	post	assoc
rs10907175	1	1130727	0.084493	A	C	0.351999999999945	0.2459430394857876	0.0
rs2887286	1	1156131	0.175944	T	C	0.390399999999947	0.21100387604316348	0.0
rs6685064	1	1211292	0.0666004	C	T	0.943799999999981	0.10466973830102834	0.0
rs1571150	1	1474304	0.333002	C	A	0.667700000000157	0.312529201925508	0.0
rs7290	1	1477244	0.306163	T	C	0.847799999999996	0.22741445464179644	0.0
rs3766180	1	1478153	0.306163	T	C	0.852900000000085	0.2170770906036481	0.0
rs3766178	1	1478180	0.306163	T	C	0.801899999999967	0.2209031523383583	0.0
rs7533	1	1479333	0.306163	A	G	0.85810000000001	0.1511410932793709	0.0
rs7517401	1	1483010	0.349901	G	A	0.700699999999828	0.21465550994232505	0.0
```

## Reproducibility
We provide the source codes for reproducing the experimental results of PALM. The 30 GWAS summary statistics and functional annotations in real data analysis can be downloaded [here](https://drive.google.com/file/d/15btr71PD1lI6oqrOtf_T-i8aZM0YCRaP/view?usp=sharing).
+ [Simulation script](https://github.com/xinyiyu/PALM/blob/main/scripts/comparison_palm.py) and [simulation results](https://github.com/xinyiyu/PALM/blob/main/demos/simu_results.ipynb)
+ [Supplemental simulation scripts](https://github.com/xinyiyu/PALM/blob/main/scripts/) and [supplemental simulation results](https://github.com/xinyiyu/PALM/blob/main/demos/suppl_results.ipynb)
+ [Real data format](https://github.com/xinyiyu/PALM/blob/main/demos/real_data.ipynb) and [an example on real data](https://github.com/xinyiyu/PALM/blob/main/demos/real_example.ipynb)
+ [Real data analysis script](https://github.com/xinyiyu/PALM/blob/main/scripts/real_palm.py) and [real data analysis results](https://github.com/xinyiyu/PALM/blob/main/demos/real_results.ipynb)

## Extension
The prior model is a general framework which allows us to apply any appropriate method to fit the logit-scale prior probabilities of the SNP association status. In particular, the boosted trees can be replaced with a neural network. We also provide the implementation of network-based PALM with a 3-layer fully connected network. A demo on simulated data with network-based PALM can be found [here](https://github.com/xinyiyu/PALM/blob/main/demos/demo_nn.ipynb). 

We have demonstrated with simulations in the paper that the performance of network-based PALM is sensitive to the network design and hyperparameters tuning (at least for the very simple multi-layer perceptron). Since network-based PALM is not robust in FDR control, we haven't apply it to real data. For risk variants prioritization, currently we suggest tree-based PALM. Further research is needed for the potential usage of neural network in genomic integrative analysis. 

## Contact information

Please contact Xinyi Yu (xyubl@connect.ust.hk) and Prof. Can Yang (macyang@ust.hk) if any enquiry.
