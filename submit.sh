#!/bin/bash
#SBATCH -J python_test
#SBATCH -p node
#SBATCH -n 2
#ulimit -u unlimited
export PATH="/share/apps/anaconda3/bin:$PATH"
source activate survival_analysis
date
python RNASeq_miRNA_FGCNSurv.py
datenull
