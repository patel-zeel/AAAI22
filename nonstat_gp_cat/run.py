import sys
import os

m_name = sys.argv[1]
optim_name = sys.argv[2]
c_fold = sys.argv[3]
node = sys.argv[4]
nsgp_iters = sys.argv[5]
gp_iters = sys.argv[6]
restarts = sys.argv[7]
div = sys.argv[8]
sampling = sys.argv[9]
Xcols = sys.argv[10]
kernel = sys.argv[11]
time_kernel = sys.argv[12]

with open(m_name+c_fold+'.sh', 'w') as f:
	f.write('#!/bin/sh \n')
	f.write('#SBATCH --partition=gpu --nodelist='+node+'\n\n')
	f.write(' '.join(['python gp_train.py', m_name, optim_name, c_fold, 
	nsgp_iters, gp_iters, restarts, div, sampling, Xcols, kernel, time_kernel, '\n']))
	f.write(' '.join(['python gp_test.py', m_name, c_fold, sampling, Xcols, kernel, time_kernel, '\n']))

os.system('sbatch '+ m_name +c_fold+'.sh')