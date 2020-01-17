# %% CT Denoise
import os
import numpy as np
from skimage import io, data
from skimage.color import rgb2gray
from skimage.transform import downscale_local_mean
from skimage.measure import compare_mse as mse, compare_psnr as psnr, compare_ssim as ssim
import requests
import json
import matplotlib.pyplot as plt
from sklearn.datasets import load_sample_images
import time
import re

font = {'weight' : 'normal',
        'size'   : 16}

plt.rc('font', **font)


td = 50
textSize = 22

folder = '\\synthetic_data'

path = os.getcwd() + folder
files = os.listdir(path)

errors = {}
stats = {}

for file in files:
    with open(path + '\\' + file + '\\errors.json') as json_file:
        errors[file] = json.load(json_file)
        
for file in files:
    with open(path + '\\' + file + '\\stats.json') as json_file:
        stats[file] = json.load(json_file)


# Calculate average time for OMP and KSVD
averageTime = {}
for key in stats:
    averageTime[key] = {}
    averageTime[key]['timePerIteration'] = 0.0
    averageTime[key]['timePerOMP'] = 0.0
    for i, stat in enumerate(stats[key]):
        timePerIteration = ' '.join(re.split('[\n\0\[\] ]+', stats[key][i]['timePerIteration'])).split()
        timePerIteration = np.array(timePerIteration).astype(float)
        averageTime[key]['timePerIteration'] += sum(timePerIteration)
        timePerOMP = ' '.join(re.split('[\n\0\[\] ]+', stats[key][i]['timePerOMP'])).split()
        timePerOMP = np.array(timePerOMP).astype(float)
        averageTime[key]['timePerOMP'] += sum(timePerOMP)
    averageTime[key]['timePerIteration'] /= (len(stats[key]) * int(stats[key][0]['cloudKsvdIterations']))
    averageTime[key]['timePerOMP'] /= (len(stats[key]) * int(stats[key][0]['cloudKsvdIterations']))
    averageTime[key]['timePerKSVD'] = averageTime[key]['timePerIteration'] - averageTime[key]['timePerOMP']


with open(os.getcwd() + '\\results\\syntheticData\\averageTimes.json','w') as json_file:
    json_file.write(json.dumps(averageTime))
    
iterations = range(0, td+1)
iterations2 = range(0, 21)

# Figure 1 - Average MSE of D per iteration Cloud K-SVD
plt.figure(1, figsize=(20, 6.33))
plt.subplot(1, 3, 1)
plt.title('Cloud K-SVD\n$P = 4$', size=textSize, fontweight="bold")
plt.yscale('log')
l1, l2, l3, l4, l5 = plt.plot(iterations, errors['P4_C1_Q200']['MSE_D_AVG'],
                              iterations, errors['P4_C1_Q800']['MSE_D_AVG'],
                              iterations, errors['P4_C1_Q2000']['MSE_D_AVG'],
                              iterations, errors['P4_C1_Q8000']['MSE_D_AVG'],
                              iterations2, errors['P4_C1_Q20000']['MSE_D_AVG'])
plt.ylabel('MSE(D)', size=textSize)
plt.xlabel('Iteration', size=textSize)
plt.grid(True,which="both",ls="-")
plt.legend((l1, l2, l3, l4, l5), ('$Q = 200$', '$Q = 800$', '$Q = 2000$', '$Q = 8000$', '$Q = 20000$'), prop={'size': textSize})

plt.subplot(1, 3, 2)
plt.title('Cloud K-SVD\n$P = 8$', size=textSize, fontweight="bold")
plt.yscale('log')
l1, l2, l3, l4, l5 = plt.plot(iterations, errors['P8_C1_Q200']['MSE_D_AVG'],
                              iterations, errors['P8_C1_Q800']['MSE_D_AVG'],
                              iterations, errors['P8_C1_Q2000']['MSE_D_AVG'],
                              iterations, errors['P8_C1_Q8000']['MSE_D_AVG'],
                              iterations2, errors['P8_C1_Q20000']['MSE_D_AVG'])
plt.ylabel('MSE(D)', size=textSize)
plt.xlabel('Iteration', size=textSize)
plt.grid(True,which="both",ls="-")
plt.legend((l1, l2, l3, l4, l5), ('$Q = 200$', '$Q = 800$', '$Q = 2000$', '$Q = 8000$', '$Q = 20000$'), prop={'size': textSize})

plt.subplot(1, 3, 3)
plt.title('Cloud K-SVD\n$P = 16$', size=textSize, fontweight="bold")
plt.yscale('log')
l1, l2, l3, l4, l5 = plt.plot(iterations, errors['P16_C1_Q200']['MSE_D_AVG'],
                              iterations, errors['P16_C1_Q800']['MSE_D_AVG'],
                              iterations, errors['P16_C1_Q2000']['MSE_D_AVG'],
                              iterations, errors['P16_C1_Q8000']['MSE_D_AVG'],
                              iterations2, errors['P16_C1_Q20000']['MSE_D_AVG'])
plt.ylabel('MSE(D)', size=textSize)
plt.xlabel('Iteration', size=textSize)
plt.grid(True,which="both",ls="-")
plt.legend((l1, l2, l3, l4, l5), ('$Q = 200$', '$Q = 800$', '$Q = 2000$', '$Q = 8000$', '$Q = 20000$'), prop={'size': textSize})

plt.subplots_adjust(None, None, None, None, 0.25, None)
plt.savefig(os.getcwd() + '\\results\\syntheticData\\Cloud_MSE_D_AVG.pdf')



# Figure 2 - Average MSE of D per iteration Local K-SVD
plt.figure(2, figsize=(20, 6.33))
plt.subplot(1, 3, 1)
plt.title('Local K-SVD\n$P = 4$', size=textSize, fontweight="bold")
plt.yscale('log')
l1, l2, l3, l4, l5 = plt.plot(iterations, errors['P4_C0_Q200']['MSE_D_AVG'],
                              iterations, errors['P4_C0_Q800']['MSE_D_AVG'],
                              iterations, errors['P4_C0_Q2000']['MSE_D_AVG'],
                              iterations, errors['P4_C0_Q8000']['MSE_D_AVG'],
                              iterations2, errors['P4_C0_Q20000']['MSE_D_AVG'])
plt.ylabel('MSE(D)', size=textSize)
plt.xlabel('Iteration', size=textSize)
plt.grid(True,which="both",ls="-")
plt.legend((l1, l2, l3, l4, l5), ('$Q = 200$', '$Q = 800$', '$Q = 2000$', '$Q = 8000$', '$Q = 20000$'), prop={'size': textSize})

plt.subplot(1, 3, 2)
plt.title('Local K-SVD\n$P = 8$', size=textSize, fontweight="bold")
plt.yscale('log')
l1, l2, l3, l4, l5 = plt.plot(iterations, errors['P8_C0_Q200']['MSE_D_AVG'],
                              iterations, errors['P8_C0_Q800']['MSE_D_AVG'],
                              iterations, errors['P8_C0_Q2000']['MSE_D_AVG'],
                              iterations, errors['P8_C0_Q8000']['MSE_D_AVG'],
                              iterations2, errors['P8_C0_Q20000']['MSE_D_AVG'])
plt.ylabel('MSE(D)', size=textSize)
plt.xlabel('Iteration', size=textSize)
plt.grid(True,which="both",ls="-")
plt.legend((l1, l2, l3, l4, l5), ('$Q = 200$', '$Q = 800$', '$Q = 2000$', '$Q = 8000$', '$Q = 20000$'), prop={'size': textSize})

plt.subplot(1, 3, 3)
plt.title('Local K-SVD\n$P = 16$', size=textSize, fontweight="bold")
plt.yscale('log')
l1, l2, l3, l4, l5 = plt.plot(iterations, errors['P16_C0_Q200']['MSE_D_AVG'],
                              iterations, errors['P16_C0_Q800']['MSE_D_AVG'],
                              iterations, errors['P16_C0_Q2000']['MSE_D_AVG'],
                              iterations, errors['P16_C0_Q8000']['MSE_D_AVG'],
                              iterations2, errors['P16_C0_Q20000']['MSE_D_AVG'])
plt.ylabel('MSE(D)', size=textSize)
plt.xlabel('Iteration', size=textSize)
plt.grid(True,which="both",ls="-")
plt.legend((l1, l2, l3, l4, l5), ('$Q = 200$', '$Q = 800$', '$Q = 2000$', '$Q = 8000$', '$Q = 20000$'), prop={'size': textSize})

plt.subplots_adjust(None, None, None, None, 0.35, None)
plt.savefig(os.getcwd() + '\\results\\syntheticData\\Local_MSE_D_AVG.pdf')


# Figure 3 - Average l2-norm of D per iteration Cloud K-SVD
plt.figure(3, figsize=(20, 6.33))
plt.subplot(1, 3, 1)
plt.title('Cloud K-SVD\n$P = 4$', size=textSize, fontweight="bold")
l1, l2, l3, l4, l5 = plt.plot(iterations, errors['P4_C1_Q200']['NORM_D_AVG'],
                              iterations, errors['P4_C1_Q800']['NORM_D_AVG'],
                              iterations, errors['P4_C1_Q2000']['NORM_D_AVG'],
                              iterations, errors['P4_C1_Q8000']['NORM_D_AVG'],
                              iterations2, errors['P4_C1_Q20000']['NORM_D_AVG'])
plt.ylabel('$\ell_{2}(D)$', size=textSize)
plt.xlabel('Iteration', size=textSize)
plt.grid(True,which="both",ls="-")
plt.legend((l1, l2, l3, l4, l5), ('$Q = 200$', '$Q = 800$', '$Q = 2000$', '$Q = 8000$', '$Q = 20000$'), prop={'size': textSize})

plt.subplot(1, 3, 2)
plt.title('Cloud K-SVD\n$P = 8$', size=textSize, fontweight="bold")
l1, l2, l3, l4, l5 = plt.plot(iterations, errors['P8_C1_Q200']['NORM_D_AVG'],
                              iterations, errors['P8_C1_Q800']['NORM_D_AVG'],
                              iterations, errors['P8_C1_Q2000']['NORM_D_AVG'],
                              iterations, errors['P8_C1_Q8000']['NORM_D_AVG'],
                              iterations2, errors['P8_C1_Q20000']['NORM_D_AVG'])
plt.ylabel('$\ell_{2}(D)$', size=textSize)
plt.xlabel('Iteration', size=textSize)
plt.grid(True,which="both",ls="-")
plt.legend((l1, l2, l3, l4, l5), ('$Q = 200$', '$Q = 800$', '$Q = 2000$', '$Q = 8000$', '$Q = 20000$'), prop={'size': textSize})

plt.subplot(1, 3, 3)
plt.title('Cloud K-SVD\n$P = 16$', size=textSize, fontweight="bold")
l1, l2, l3, l4, l5 = plt.plot(iterations, errors['P16_C1_Q200']['NORM_D_AVG'],
                              iterations, errors['P16_C1_Q800']['NORM_D_AVG'],
                              iterations, errors['P16_C1_Q2000']['NORM_D_AVG'],
                              iterations, errors['P16_C1_Q8000']['NORM_D_AVG'],
                              iterations2, errors['P16_C1_Q20000']['NORM_D_AVG'])
plt.ylabel('$\ell_{2}(D)$', size=textSize)
plt.xlabel('Iteration', size=textSize)
plt.grid(True,which="both",ls="-")
plt.legend((l1, l2, l3, l4, l5), ('$Q = 200$', '$Q = 800$', '$Q = 2000$', '$Q = 8000$', '$Q = 20000$'), prop={'size': textSize})

plt.subplots_adjust(None, None, None, None, 0.2, None)
plt.savefig(os.getcwd() + '\\results\\syntheticData\\Cloud_l2_D_AVG.pdf')


# Figure 4 - Average l2-norm of D per iteration Local K-SVD
plt.figure(4, figsize=(20, 6.33))
plt.subplot(1, 3, 1)
plt.title('Local K-SVD\n$P = 4$', size=textSize, fontweight="bold")
l1, l2, l3, l4, l5 = plt.plot(iterations, errors['P4_C0_Q200']['NORM_D_AVG'],
                              iterations, errors['P4_C0_Q800']['NORM_D_AVG'],
                              iterations, errors['P4_C0_Q2000']['NORM_D_AVG'],
                              iterations, errors['P4_C0_Q8000']['NORM_D_AVG'],
                              iterations2, errors['P4_C0_Q20000']['NORM_D_AVG'])
plt.ylabel('$\ell_{2}(D)$', size=textSize)
plt.xlabel('Iteration', size=textSize)
plt.grid(True,which="both",ls="-")
plt.legend((l1, l2, l3, l4, l5), ('$Q = 200$', '$Q = 800$', '$Q = 2000$', '$Q = 8000$', '$Q = 20000$'), prop={'size': textSize})

plt.subplot(1, 3, 2)
plt.title('Local K-SVD\n$P = 8$', size=textSize, fontweight="bold")
l1, l2, l3, l4, l5 = plt.plot(iterations, errors['P8_C0_Q200']['NORM_D_AVG'],
                              iterations, errors['P8_C0_Q800']['NORM_D_AVG'],
                              iterations, errors['P8_C0_Q2000']['NORM_D_AVG'],
                              iterations, errors['P8_C0_Q8000']['NORM_D_AVG'],
                              iterations2, errors['P8_C0_Q20000']['NORM_D_AVG'])
plt.ylabel('$\ell_{2}(D)$', size=textSize)
plt.xlabel('Iteration', size=textSize)
plt.grid(True,which="both",ls="-")
plt.legend((l1, l2, l3, l4, l5), ('$Q = 200$', '$Q = 800$', '$Q = 2000$', '$Q = 8000$', '$Q = 20000$'), prop={'size': textSize})

plt.subplot(1, 3, 3)
plt.title('Local K-SVD\n$P = 16$', size=textSize, fontweight="bold")
l1, l2, l3, l4, l5 = plt.plot(iterations, errors['P16_C0_Q200']['NORM_D_AVG'],
                              iterations, errors['P16_C0_Q800']['NORM_D_AVG'],
                              iterations, errors['P16_C0_Q2000']['NORM_D_AVG'],
                              iterations, errors['P16_C0_Q8000']['NORM_D_AVG'],
                              iterations2, errors['P16_C0_Q20000']['NORM_D_AVG'])
plt.ylabel('$\ell_{2}(D)$', size=textSize)
plt.xlabel('Iteration', size=textSize)
plt.grid(True,which="both",ls="-")
plt.legend((l1, l2, l3, l4, l5), ('$Q = 200$', '$Q = 800$', '$Q = 2000$', '$Q = 8000$', '$Q = 20000$'), prop={'size': textSize})

plt.subplots_adjust(None, None, None, None, 0.2, None)
plt.savefig(os.getcwd() + '\\results\\syntheticData\\Local_l2_D_AVG.pdf')


# Figure 5 - MSE of Y per iteration Cloud K-SVD
plt.figure(5, figsize=(20, 6.33))
plt.subplot(1, 3, 1)
plt.title('Cloud K-SVD\n$P = 4$', size=textSize, fontweight="bold")
plt.yscale('log')
l1, l2, l3, l4, l5 = plt.plot(iterations, errors['P4_C1_Q200']['MSE_Y'],
                              iterations, errors['P4_C1_Q800']['MSE_Y'],
                              iterations, errors['P4_C1_Q2000']['MSE_Y'],
                              iterations, errors['P4_C1_Q8000']['MSE_Y'],
                              iterations2, errors['P4_C1_Q20000']['MSE_Y'])
plt.ylabel('MSE(Y)', size=textSize)
plt.xlabel('Iteration', size=textSize)
plt.grid(True,which="both",ls="-")
plt.legend((l1, l2, l3, l4, l5), ('$Q = 200$', '$Q = 800$', '$Q = 2000$', '$Q = 8000$', '$Q = 20000$'), prop={'size': textSize})

plt.subplot(1, 3, 2)
plt.title('Cloud K-SVD\n$P = 8$', size=textSize, fontweight="bold")
plt.yscale('log')
l1, l2, l3, l4, l5 = plt.plot(iterations, errors['P8_C1_Q200']['MSE_Y'],
                              iterations, errors['P8_C1_Q800']['MSE_Y'],
                              iterations, errors['P8_C1_Q2000']['MSE_Y'],
                              iterations, errors['P8_C1_Q8000']['MSE_Y'],
                              iterations2, errors['P8_C1_Q20000']['MSE_Y'])
plt.ylabel('MSE(Y)', size=textSize)
plt.xlabel('Iteration', size=textSize)
plt.grid(True,which="both",ls="-")
plt.legend((l1, l2, l3, l4, l5), ('$Q = 200$', '$Q = 800$', '$Q = 2000$', '$Q = 8000$', '$Q = 20000$'), prop={'size': textSize})

plt.subplot(1, 3, 3)
plt.title('Cloud K-SVD\n$P = 16$', size=textSize, fontweight="bold")
plt.yscale('log')
l1, l2, l3, l4, l5 = plt.plot(iterations, errors['P16_C1_Q200']['MSE_Y'],
                              iterations, errors['P16_C1_Q800']['MSE_Y'],
                              iterations, errors['P16_C1_Q2000']['MSE_Y'],
                              iterations, errors['P16_C1_Q8000']['MSE_Y'],
                              iterations2, errors['P16_C1_Q20000']['MSE_Y'])
plt.ylabel('MSE(Y)', size=textSize)
plt.xlabel('Iteration', size=textSize)
plt.grid(True,which="both",ls="-")
plt.legend((l1, l2, l3, l4, l5), ('$Q = 200$', '$Q = 800$', '$Q = 2000$', '$Q = 8000$', '$Q = 20000$'), prop={'size': textSize})

plt.subplots_adjust(None, None, None, None, 0.25, None)
plt.savefig(os.getcwd() + '\\results\\syntheticData\\Cloud_MSE_Y.pdf')



# Figure 6 - MSE of Y per iteration Local K-SVD
plt.figure(6, figsize=(20, 6.33))
plt.subplot(1, 3, 1)
plt.title('Local K-SVD\n$P = 4$', size=textSize, fontweight="bold")
plt.yscale('log')
l1, l2, l3, l4, l5 = plt.plot(iterations, errors['P4_C0_Q200']['MSE_Y'],
                              iterations, errors['P4_C0_Q800']['MSE_Y'],
                              iterations, errors['P4_C0_Q2000']['MSE_Y'],
                              iterations, errors['P4_C0_Q8000']['MSE_Y'],
                              iterations2, errors['P4_C0_Q20000']['MSE_Y'])
plt.ylabel('MSE(Y)', size=textSize)
plt.xlabel('Iteration', size=textSize)
plt.grid(True,which="both",ls="-")
plt.legend((l1, l2, l3, l4, l5), ('$Q = 200$', '$Q = 800$', '$Q = 2000$', '$Q = 8000$', '$Q = 20000$'), prop={'size': textSize})

plt.subplot(1, 3, 2)
plt.title('Local K-SVD\n$P = 8$', size=textSize, fontweight="bold")
plt.yscale('log')
l1, l2, l3, l4, l5 = plt.plot(iterations, errors['P8_C0_Q200']['MSE_Y'],
                              iterations, errors['P8_C0_Q800']['MSE_Y'],
                              iterations, errors['P8_C0_Q2000']['MSE_Y'],
                              iterations, errors['P8_C0_Q8000']['MSE_Y'],
                              iterations2, errors['P8_C0_Q20000']['MSE_Y'])
plt.ylabel('MSE(Y)', size=textSize)
plt.xlabel('Iteration', size=textSize)
plt.grid(True,which="both",ls="-")
plt.legend((l1, l2, l3, l4, l5), ('$Q = 200$', '$Q = 800$', '$Q = 2000$', '$Q = 8000$', '$Q = 20000$'), prop={'size': textSize})

plt.subplot(1, 3, 3)
plt.title('Local K-SVD\n$P = 16$', size=textSize, fontweight="bold")
plt.yscale('log')
l1, l2, l3, l4, l5 = plt.plot(iterations, errors['P16_C0_Q200']['MSE_Y'],
                              iterations, errors['P16_C0_Q800']['MSE_Y'],
                              iterations, errors['P16_C0_Q2000']['MSE_Y'],
                              iterations, errors['P16_C0_Q8000']['MSE_Y'],
                              iterations2, errors['P16_C0_Q20000']['MSE_Y'])
plt.ylabel('MSE(Y)', size=textSize)
plt.xlabel('Iteration', size=textSize)
plt.grid(True,which="both",ls="-")
plt.legend((l1, l2, l3, l4, l5), ('$Q = 200$', '$Q = 800$', '$Q = 2000$', '$Q = 8000$', '$Q = 20000$'), prop={'size': textSize})

plt.subplots_adjust(None, None, None, None, 0.25, None)
plt.savefig(os.getcwd() + '\\results\\syntheticData\\Local_MSE_Y.pdf')


# Figure 7 - l2-norm of Y per iteration Cloud K-SVD
plt.figure(7, figsize=(20, 6.33))
plt.subplot(1, 3, 1)
plt.title('Cloud K-SVD\n$P = 4$', size=textSize, fontweight="bold")
l1, l2, l3, l4, l5 = plt.plot(iterations, errors['P4_C1_Q200']['NORM_Y'],
                              iterations, errors['P4_C1_Q800']['NORM_Y'],
                              iterations, errors['P4_C1_Q2000']['NORM_Y'],
                              iterations, errors['P4_C1_Q8000']['NORM_Y'],
                              iterations2, errors['P4_C1_Q20000']['NORM_Y'])
plt.ylabel('$\ell_{2}(Y)$', size=textSize)
plt.xlabel('Iteration', size=textSize)
plt.grid(True,which="both",ls="-")
plt.legend((l1, l2, l3, l4, l5), ('$Q = 200$', '$Q = 800$', '$Q = 2000$', '$Q = 8000$', '$Q = 20000$'), prop={'size': textSize})

plt.subplot(1, 3, 2)
plt.title('Cloud K-SVD\n$P = 8$', size=textSize, fontweight="bold")
l1, l2, l3, l4, l5 = plt.plot(iterations, errors['P8_C1_Q200']['NORM_Y'],
                              iterations, errors['P8_C1_Q800']['NORM_Y'],
                              iterations, errors['P8_C1_Q2000']['NORM_Y'],
                              iterations, errors['P8_C1_Q8000']['NORM_Y'],
                              iterations2, errors['P8_C1_Q20000']['NORM_Y'])
plt.ylabel('$\ell_{2}(Y)$', size=textSize)
plt.xlabel('Iteration', size=textSize)
plt.grid(True,which="both",ls="-")
plt.legend((l1, l2, l3, l4, l5), ('$Q = 200$', '$Q = 800$', '$Q = 2000$', '$Q = 8000$', '$Q = 20000$'), prop={'size': textSize})

plt.subplot(1, 3, 3)
plt.title('Cloud K-SVD\n$P = 16$', size=textSize, fontweight="bold")
l1, l2, l3, l4, l5 = plt.plot(iterations, errors['P16_C1_Q200']['NORM_Y'],
                              iterations, errors['P16_C1_Q800']['NORM_Y'],
                              iterations, errors['P16_C1_Q2000']['NORM_Y'],
                              iterations, errors['P16_C1_Q8000']['NORM_Y'],
                              iterations2, errors['P16_C1_Q20000']['NORM_Y'])
plt.ylabel('$\ell_{2}(Y)$', size=textSize)
plt.xlabel('Iteration', size=textSize)
plt.grid(True,which="both",ls="-")
plt.legend((l1, l2, l3, l4, l5), ('$Q = 200$', '$Q = 800$', '$Q = 2000$', '$Q = 8000$', '$Q = 20000$'), prop={'size': textSize})

plt.subplots_adjust(None, None, None, None, 0.25, None)
plt.savefig(os.getcwd() + '\\results\\syntheticData\\Cloud_l2_Y.pdf')


# Figure 8 - l2-norm of Y per iteration Local K-SVD
plt.figure(8, figsize=(20, 6.33))
plt.subplot(1, 3, 1)
plt.title('Local K-SVD\n$P = 4$', size=textSize, fontweight="bold")
l1, l2, l3, l4, l5 = plt.plot(iterations, errors['P4_C0_Q200']['NORM_Y'],
                              iterations, errors['P4_C0_Q800']['NORM_Y'],
                              iterations, errors['P4_C0_Q2000']['NORM_Y'],
                              iterations, errors['P4_C0_Q8000']['NORM_Y'],
                              iterations2, errors['P4_C0_Q20000']['NORM_Y'])
plt.ylabel('$\ell_{2}(Y)$', size=textSize)
plt.xlabel('Iteration', size=textSize)
plt.grid(True,which="both",ls="-")
plt.legend((l1, l2, l3, l4, l5), ('$Q = 200$', '$Q = 800$', '$Q = 2000$', '$Q = 8000$', '$Q = 20000$'), prop={'size': textSize})

plt.subplot(1, 3, 2)
plt.title('Local K-SVD\n$P = 8$', size=textSize, fontweight="bold")
l1, l2, l3, l4, l5 = plt.plot(iterations, errors['P8_C0_Q200']['NORM_Y'],
                              iterations, errors['P8_C0_Q800']['NORM_Y'],
                              iterations, errors['P8_C0_Q2000']['NORM_Y'],
                              iterations, errors['P8_C0_Q8000']['NORM_Y'],
                              iterations2, errors['P8_C0_Q20000']['NORM_Y'])
plt.ylabel('$\ell_{2}(Y)$', size=textSize)
plt.xlabel('Iteration', size=textSize)
plt.grid(True,which="both",ls="-")
plt.legend((l1, l2, l3, l4, l5), ('$Q = 200$', '$Q = 800$', '$Q = 2000$', '$Q = 8000$', '$Q = 20000$'), prop={'size': textSize})

plt.subplot(1, 3, 3)
plt.title('Local K-SVD\n$P = 16$', size=textSize, fontweight="bold")
l1, l2, l3, l4, l5 = plt.plot(iterations, errors['P16_C0_Q200']['NORM_Y'],
                              iterations, errors['P16_C0_Q800']['NORM_Y'],
                              iterations, errors['P16_C0_Q2000']['NORM_Y'],
                              iterations, errors['P16_C0_Q8000']['NORM_Y'],
                              iterations2, errors['P16_C0_Q20000']['NORM_Y'])
plt.ylabel('$\ell_{2}(Y)$', size=textSize)
plt.xlabel('Iteration', size=textSize)
plt.grid(True,which="both",ls="-")
plt.legend((l1, l2, l3, l4, l5), ('$Q = 200$', '$Q = 800$', '$Q = 2000$', '$Q = 8000$', '$Q = 20000$'), prop={'size': textSize})

plt.subplots_adjust(None, None, None, None, 0.25, None)
plt.savefig(os.getcwd() + '\\results\\syntheticData\\Local_l2_Y.pdf')


# Figure 9 - PSNR of Y per iteration Cloud K-SVD
plt.figure(9, figsize=(20, 6.33))
plt.subplot(1, 3, 1)
plt.title('Cloud K-SVD\n$P = 4$', size=textSize, fontweight="bold")
l1, l2, l3, l4, l5 = plt.plot(iterations, errors['P4_C1_Q200']['PSNR_Y'],
                              iterations, errors['P4_C1_Q800']['PSNR_Y'],
                              iterations, errors['P4_C1_Q2000']['PSNR_Y'],
                              iterations, errors['P4_C1_Q8000']['PSNR_Y'],
                              iterations2, errors['P4_C1_Q20000']['PSNR_Y'])
plt.ylabel('PSNR(Y)', size=textSize)
plt.xlabel('Iteration', size=textSize)
plt.grid(True,which="both",ls="-")
plt.legend((l1, l2, l3, l4, l5), ('$Q = 200$', '$Q = 800$', '$Q = 2000$', '$Q = 8000$', '$Q = 20000$'), prop={'size': textSize})

plt.subplot(1, 3, 2)
plt.title('Cloud K-SVD\n$P = 8$', size=textSize, fontweight="bold")
l1, l2, l3, l4, l5 = plt.plot(iterations, errors['P8_C1_Q200']['PSNR_Y'],
                              iterations, errors['P8_C1_Q800']['PSNR_Y'],
                              iterations, errors['P8_C1_Q2000']['PSNR_Y'],
                              iterations, errors['P8_C1_Q8000']['PSNR_Y'],
                              iterations2, errors['P8_C1_Q20000']['PSNR_Y'])
plt.ylabel('PSNR(Y)', size=textSize)
plt.xlabel('Iteration', size=textSize)
plt.grid(True,which="both",ls="-")
plt.legend((l1, l2, l3, l4, l5), ('$Q = 200$', '$Q = 800$', '$Q = 2000$', '$Q = 8000$', '$Q = 20000$'), prop={'size': textSize})

plt.subplot(1, 3, 3)
plt.title('Cloud K-SVD\n$P = 16$', size=textSize, fontweight="bold")
l1, l2, l3, l4, l5 = plt.plot(iterations, errors['P16_C1_Q200']['PSNR_Y'],
                              iterations, errors['P16_C1_Q800']['PSNR_Y'],
                              iterations, errors['P16_C1_Q2000']['PSNR_Y'],
                              iterations, errors['P16_C1_Q8000']['PSNR_Y'],
                              iterations2, errors['P16_C1_Q20000']['PSNR_Y'])
plt.ylabel('PSNR(Y)', size=textSize)
plt.xlabel('Iteration', size=textSize)
plt.grid(True,which="both",ls="-")
plt.legend((l1, l2, l3, l4, l5), ('$Q = 200$', '$Q = 800$', '$Q = 2000$', '$Q = 8000$', '$Q = 20000$'), prop={'size': textSize})

plt.subplots_adjust(None, None, None, None, 0.18, None)
plt.savefig(os.getcwd() + '\\results\\syntheticData\\Cloud_PSNR_Y.pdf')


# Figure 10 - PSNR of Y per iteration Local K-SVD
plt.figure(10, figsize=(20, 6.33))
plt.subplot(1, 3, 1)
plt.title('Local K-SVD\n$P = 4$', size=textSize, fontweight="bold")
l1, l2, l3, l4, l5 = plt.plot(iterations, errors['P4_C0_Q200']['PSNR_Y'],
                              iterations, errors['P4_C0_Q800']['PSNR_Y'],
                              iterations, errors['P4_C0_Q2000']['PSNR_Y'],
                              iterations, errors['P4_C0_Q8000']['PSNR_Y'],
                              iterations2, errors['P4_C0_Q20000']['PSNR_Y'])
plt.ylabel('PSNR(Y)', size=textSize)
plt.xlabel('Iteration', size=textSize)
plt.grid(True,which="both",ls="-")
plt.legend((l1, l2, l3, l4, l5), ('$Q = 200$', '$Q = 800$', '$Q = 2000$', '$Q = 8000$', '$Q = 20000$'), prop={'size': textSize})

plt.subplot(1, 3, 2)
plt.title('Local K-SVD\n$P = 8$', size=textSize, fontweight="bold")
l1, l2, l3, l4, l5 = plt.plot(iterations, errors['P8_C0_Q200']['PSNR_Y'],
                              iterations, errors['P8_C0_Q800']['PSNR_Y'],
                              iterations, errors['P8_C0_Q2000']['PSNR_Y'],
                              iterations, errors['P8_C0_Q8000']['PSNR_Y'],
                              iterations2, errors['P8_C0_Q20000']['PSNR_Y'])
plt.ylabel('PSNR(Y)', size=textSize)
plt.xlabel('Iteration', size=textSize)
plt.grid(True,which="both",ls="-")
plt.legend((l1, l2, l3, l4, l5), ('$Q = 200$', '$Q = 800$', '$Q = 2000$', '$Q = 8000$', '$Q = 20000$'), prop={'size': textSize})

plt.subplot(1, 3, 3)
plt.title('Local K-SVD\n$P = 16$', size=textSize, fontweight="bold")
l1, l2, l3, l4, l5 = plt.plot(iterations, errors['P16_C0_Q200']['PSNR_Y'],
                              iterations, errors['P16_C0_Q800']['PSNR_Y'],
                              iterations, errors['P16_C0_Q2000']['PSNR_Y'],
                              iterations, errors['P16_C0_Q8000']['PSNR_Y'],
                              iterations2, errors['P16_C0_Q20000']['PSNR_Y'])
plt.ylabel('PSNR(Y)', size=textSize)
plt.xlabel('Iteration', size=textSize)
plt.grid(True,which="both",ls="-")
plt.legend((l1, l2, l3, l4, l5), ('$Q = 200$', '$Q = 800$', '$Q = 2000$', '$Q = 8000$', '$Q = 20000$'), prop={'size': textSize})

plt.subplots_adjust(None, None, None, None, 0.18, None)
plt.savefig(os.getcwd() + '\\results\\syntheticData\\Local_PSNR_Y.pdf')


# Figure 11 - SSIM of Y per iteration Cloud K-SVD
plt.figure(11, figsize=(20, 6.33))
plt.subplot(1, 3, 1)
plt.title('Cloud K-SVD\n$P = 4$', size=textSize, fontweight="bold")
l1, l2, l3, l4, l5 = plt.plot(iterations, errors['P4_C1_Q200']['SSIM_Y'],
                              iterations, errors['P4_C1_Q800']['SSIM_Y'],
                              iterations, errors['P4_C1_Q2000']['SSIM_Y'],
                              iterations, errors['P4_C1_Q8000']['SSIM_Y'],
                              iterations2, errors['P4_C1_Q20000']['SSIM_Y'])
plt.ylabel('SSIM(Y)', size=textSize)
plt.xlabel('Iteration', size=textSize)
plt.grid(True,which="both",ls="-")
plt.legend((l1, l2, l3, l4, l5), ('$Q = 200$', '$Q = 800$', '$Q = 2000$', '$Q = 8000$', '$Q = 20000$'), prop={'size': textSize})

plt.subplot(1, 3, 2)
plt.title('Cloud K-SVD\n$P = 8$', size=textSize, fontweight="bold")
l1, l2, l3, l4, l5 = plt.plot(iterations, errors['P8_C1_Q200']['SSIM_Y'],
                              iterations, errors['P8_C1_Q800']['SSIM_Y'],
                              iterations, errors['P8_C1_Q2000']['SSIM_Y'],
                              iterations, errors['P8_C1_Q8000']['SSIM_Y'],
                              iterations2, errors['P8_C1_Q20000']['SSIM_Y'])
plt.ylabel('SSIM(Y)', size=textSize)
plt.xlabel('Iteration', size=textSize)
plt.grid(True,which="both",ls="-")
plt.legend((l1, l2, l3, l4, l5), ('$Q = 200$', '$Q = 800$', '$Q = 2000$', '$Q = 8000$', '$Q = 20000$'), prop={'size': textSize})

plt.subplot(1, 3, 3)
plt.title('Cloud K-SVD\n$P = 16$', size=textSize, fontweight="bold")
l1, l2, l3, l4, l5 = plt.plot(iterations, errors['P16_C1_Q200']['SSIM_Y'],
                              iterations, errors['P16_C1_Q800']['SSIM_Y'],
                              iterations, errors['P16_C1_Q2000']['SSIM_Y'],
                              iterations, errors['P16_C1_Q8000']['SSIM_Y'],
                              iterations2, errors['P16_C1_Q20000']['SSIM_Y'])
plt.ylabel('SSIM(Y)', size=textSize)
plt.xlabel('Iteration', size=textSize)
plt.grid(True,which="both",ls="-")
plt.legend((l1, l2, l3, l4, l5), ('$Q = 200$', '$Q = 800$', '$Q = 2000$', '$Q = 8000$', '$Q = 20000$'), prop={'size': textSize})

plt.subplots_adjust(None, None, None, None, 0.20, None)
plt.savefig(os.getcwd() + '\\results\\syntheticData\\Cloud_SSIM_Y.pdf')


# Figure 12 - l2-norm of Y per iteration Local K-SVD
plt.figure(12, figsize=(20, 6.33))
plt.subplot(1, 3, 1)
plt.title('Local K-SVD\n$P = 4$', size=textSize, fontweight="bold")
l1, l2, l3, l4, l5 = plt.plot(iterations, errors['P4_C0_Q200']['SSIM_Y'],
                              iterations, errors['P4_C0_Q800']['SSIM_Y'],
                              iterations, errors['P4_C0_Q2000']['SSIM_Y'],
                              iterations, errors['P4_C0_Q8000']['SSIM_Y'],
                              iterations2, errors['P4_C0_Q20000']['SSIM_Y'])
plt.ylabel('SSIM(Y)', size=textSize)
plt.xlabel('Iteration', size=textSize)
plt.grid(True,which="both",ls="-")
plt.legend((l1, l2, l3, l4, l5), ('$Q = 200$', '$Q = 800$', '$Q = 2000$', '$Q = 8000$', '$Q = 20000$'), prop={'size': textSize})

plt.subplot(1, 3, 2)
plt.title('Local K-SVD\n$P = 8$', size=textSize, fontweight="bold")
l1, l2, l3, l4, l5 = plt.plot(iterations, errors['P8_C0_Q200']['SSIM_Y'],
                              iterations, errors['P8_C0_Q800']['SSIM_Y'],
                              iterations, errors['P8_C0_Q2000']['SSIM_Y'],
                              iterations, errors['P8_C0_Q8000']['SSIM_Y'],
                              iterations2, errors['P8_C0_Q20000']['SSIM_Y'])
plt.ylabel('SSIM(Y)', size=textSize)
plt.xlabel('Iteration', size=textSize)
plt.grid(True,which="both",ls="-")
plt.legend((l1, l2, l3, l4, l5), ('$Q = 200$', '$Q = 800$', '$Q = 2000$', '$Q = 8000$', '$Q = 20000$'), prop={'size': textSize})

plt.subplot(1, 3, 3)
plt.title('Local K-SVD\n$P = 16$', size=textSize, fontweight="bold")
l1, l2, l3, l4, l5 = plt.plot(iterations, errors['P16_C0_Q200']['SSIM_Y'],
                              iterations, errors['P16_C0_Q800']['SSIM_Y'],
                              iterations, errors['P16_C0_Q2000']['SSIM_Y'],
                              iterations, errors['P16_C0_Q8000']['SSIM_Y'],
                              iterations2, errors['P16_C0_Q20000']['SSIM_Y'])
plt.ylabel('SSIM(Y)', size=textSize)
plt.xlabel('Iteration', size=textSize)
plt.grid(True,which="both",ls="-")
plt.legend((l1, l2, l3, l4, l5), ('$Q = 200$', '$Q = 800$', '$Q = 2000$', '$Q = 8000$', '$Q = 20000$'), prop={'size': textSize})

plt.subplots_adjust(None, None, None, None, 0.20, None)
plt.savefig(os.getcwd() + '\\results\\syntheticData\\Local_SSIM_Y.pdf')

# Figure 13 -  MSE  and l2-norm of D per iteration Centralized K-SVD
plt.figure(13, figsize=(20, 20))
plt.subplot(2, 2, 1)
plt.title('Centralized K-SVD\n$P = 1$', size=textSize, fontweight="bold")
plt.yscale('log')
l1, l2, l3, l4, l5 = plt.plot(iterations, errors['P1_C0_Q200']['MSE_Y'],
                              iterations, errors['P1_C0_Q800']['MSE_Y'],
                              iterations, errors['P1_C0_Q2000']['MSE_Y'],
                              iterations, errors['P1_C0_Q8000']['MSE_Y'],
                              iterations2, errors['P1_C0_Q20000']['MSE_Y'])
plt.ylabel('MSE(Y)', size=textSize)
plt.xlabel('Iteration', size=textSize)
plt.grid(True,which="both",ls="-")
plt.legend((l1, l2, l3, l4, l5), ('$Q = 200$', '$Q = 800$', '$Q = 2000$', '$Q = 8000$', '$Q = 20000$'), prop={'size': textSize})

plt.subplot(2, 2, 2)
plt.title('Centralized K-SVD\n$P = 1$', size=textSize, fontweight="bold")
l1, l2, l3, l4, l5 = plt.plot(iterations, errors['P1_C0_Q200']['NORM_Y'],
                              iterations, errors['P1_C0_Q800']['NORM_Y'],
                              iterations, errors['P1_C0_Q2000']['NORM_Y'],
                              iterations, errors['P1_C0_Q8000']['NORM_Y'],
                              iterations2, errors['P1_C0_Q20000']['NORM_Y'])
plt.ylabel('$\ell_{2}(Y)$', size=textSize)
plt.xlabel('Iteration', size=textSize)
plt.grid(True,which="both",ls="-")
plt.legend((l1, l2, l3, l4, l5), ('$Q = 200$', '$Q = 800$', '$Q = 2000$', '$Q = 8000$', '$Q = 20000$'), prop={'size': textSize})

plt.subplot(2, 2, 3)
plt.title('Centralized K-SVD\n$P = 1$', size=textSize, fontweight="bold")
l1, l2, l3, l4, l5 = plt.plot(iterations, errors['P1_C0_Q200']['PSNR_Y'],
                              iterations, errors['P1_C0_Q800']['PSNR_Y'],
                              iterations, errors['P1_C0_Q2000']['PSNR_Y'],
                              iterations, errors['P1_C0_Q8000']['PSNR_Y'],
                              iterations2, errors['P1_C0_Q20000']['PSNR_Y'])
plt.ylabel('PSNR(Y)', size=textSize)
plt.xlabel('Iteration', size=textSize)
plt.grid(True,which="both",ls="-")
plt.legend((l1, l2, l3, l4, l5), ('$Q = 200$', '$Q = 800$', '$Q = 2000$', '$Q = 8000$', '$Q = 20000$'), prop={'size': textSize})

plt.subplot(2, 2, 4)
plt.title('Centralized K-SVD\n$P = 1$', size=textSize, fontweight="bold")
l1, l2, l3, l4, l5 = plt.plot(iterations, errors['P1_C0_Q200']['SSIM_Y'],
                              iterations, errors['P1_C0_Q800']['SSIM_Y'],
                              iterations, errors['P1_C0_Q2000']['SSIM_Y'],
                              iterations, errors['P1_C0_Q8000']['SSIM_Y'],
                              iterations2, errors['P1_C0_Q20000']['SSIM_Y'])
plt.ylabel('SSIM(Y)', size=textSize)
plt.xlabel('Iteration', size=textSize)
plt.grid(True,which="both",ls="-")
plt.legend((l1, l2, l3, l4, l5), ('$Q = 200$', '$Q = 800$', '$Q = 2000$', '$Q = 8000$', '$Q = 20000$'), prop={'size': textSize})

plt.subplots_adjust(None, None, None, None, None, 0.25)
plt.savefig(os.getcwd() + '\\results\\syntheticData\\Centralized_ERROR_Y.pdf')


