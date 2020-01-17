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

folder = '\\synthetic_consensus'

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


with open(os.getcwd() + '\\results\\syntheticConsensus\\averageTimes.json','w') as json_file:
    json_file.write(json.dumps(averageTime))
    
iterations = range(0, td+1)

# Figure 1 - Average MSE of D per iteration
plt.figure(1, figsize=(20, 6.33))
plt.subplot(1, 3, 1)
plt.title('$T_{c} = 1$', size=textSize)
plt.yscale('log')
l11, l12, l13, l14 = plt.plot(iterations, errors['tp2_tc1']['MSE_D_AVG'],
                              iterations, errors['tp3_tc1']['MSE_D_AVG'],
                              iterations, errors['tp4_tc1']['MSE_D_AVG'],
                              iterations, errors['tp5_tc1']['MSE_D_AVG'])
plt.ylabel('MSE(D)', size=textSize)
plt.xlabel('Iteration', size=textSize)
plt.grid(True,which="both",ls="-")
plt.legend((l11, l12, l13, l14), ('$T_{p} = 2$', '$T_{p} = 3$', '$T_{p} = 4$', '$T_{p} = 5$'), prop={'size': textSize})

plt.subplot(1, 3, 2)
plt.title('$T_{c} = 5$', size=textSize)
plt.yscale('log')
l11, l12, l13, l14 = plt.plot(iterations, errors['tp2_tc5']['MSE_D_AVG'],
                              iterations, errors['tp3_tc5']['MSE_D_AVG'],
                              iterations, errors['tp4_tc5']['MSE_D_AVG'],
                              iterations, errors['tp5_tc5']['MSE_D_AVG'])
plt.ylabel('MSE(D)', size=textSize)
plt.xlabel('Iteration', size=textSize)
plt.grid(True,which="both",ls="-")
plt.legend((l11, l12, l13, l14), ('$T_{p} = 2$', '$T_{p} = 3$', '$T_{p} = 4$', '$T_{p} = 5$'), prop={'size': textSize})

plt.subplot(1, 3, 3)
plt.title('$T_{c} = 10$', size=textSize)
plt.yscale('log')
l11, l12, l13, l14 = plt.plot(iterations, errors['tp2_tc10']['MSE_D_AVG'],
                              iterations, errors['tp3_tc10']['MSE_D_AVG'],
                              iterations, errors['tp4_tc10']['MSE_D_AVG'],
                              iterations, errors['tp5_tc10']['MSE_D_AVG'])
plt.ylabel('MSE(D)', size=textSize)
plt.xlabel('Iteration', size=textSize)
plt.grid(True,which="both",ls="-")
plt.legend((l11, l12, l13, l14), ('$T_{p} = 2$', '$T_{p} = 3$', '$T_{p} = 4$', '$T_{p} = 5$'), prop={'size': textSize})

plt.subplots_adjust(None, None, None, None, 0.25, None)
plt.savefig(os.getcwd() + '\\results\\syntheticConsensus\\MSE_D_AVG.pdf')


# Figure 2 - Average l2-norm of D per iteration
plt.figure(2, figsize=(20, 6.33))
plt.subplot(1, 3, 1)
plt.title('$T_{c} = 1$', size=textSize)
l11, l12, l13, l14 = plt.plot(iterations, errors['tp2_tc1']['NORM_D_AVG'],
                              iterations, errors['tp3_tc1']['NORM_D_AVG'],
                              iterations, errors['tp4_tc1']['NORM_D_AVG'],
                              iterations, errors['tp5_tc1']['NORM_D_AVG'])
plt.ylabel('$\ell_{2}(D)$', size=textSize)
plt.xlabel('Iteration', size=textSize)
plt.grid(True,which="both",ls="-")
plt.legend((l11, l12, l13, l14), ('$T_{p} = 2$', '$T_{p} = 3$', '$T_{p} = 4$', '$T_{p} = 5$'), prop={'size': textSize})

plt.subplot(1, 3, 2)
plt.title('$T_{c} = 5$', size=textSize)
l11, l12, l13, l14 = plt.plot(iterations, errors['tp2_tc5']['NORM_D_AVG'],
                              iterations, errors['tp3_tc5']['NORM_D_AVG'],
                              iterations, errors['tp4_tc5']['NORM_D_AVG'],
                              iterations, errors['tp5_tc5']['NORM_D_AVG'])
plt.ylabel('$\ell_{2}(D)$', size=textSize)
plt.xlabel('Iteration', size=textSize)
plt.grid(True,which="both",ls="-")
plt.legend((l11, l12, l13, l14), ('$T_{p} = 2$', '$T_{p} = 3$', '$T_{p} = 4$', '$T_{p} = 5$'), prop={'size': textSize})

plt.subplot(1, 3, 3)
plt.title('$T_{c} = 10$', size=textSize)
l11, l12, l13, l14 = plt.plot(iterations, errors['tp2_tc10']['NORM_D_AVG'],
                              iterations, errors['tp3_tc10']['NORM_D_AVG'],
                              iterations, errors['tp4_tc10']['NORM_D_AVG'],
                              iterations, errors['tp5_tc10']['NORM_D_AVG'])
plt.ylabel('$\ell_{2}(D)$', size=textSize)
plt.xlabel('Iteration', size=textSize)
plt.grid(True,which="both",ls="-")
plt.legend((l11, l12, l13, l14), ('$T_{p} = 2$', '$T_{p} = 3$', '$T_{p} = 4$', '$T_{p} = 5$'), prop={'size': textSize})

plt.subplots_adjust(None, None, None, None, 0.2, None)
plt.savefig(os.getcwd() + '\\results\\syntheticConsensus\\l2_D_AVG.pdf')

# Figure 3 - MSE of Y per iteration
plt.figure(3, figsize=(20, 6.33))
plt.subplot(1, 3, 1)
plt.title('$T_{c} = 1$', size=textSize)
plt.yscale('log')
l11, l12, l13, l14 = plt.plot(iterations, errors['tp2_tc1']['MSE_Y'],
                              iterations, errors['tp3_tc1']['MSE_Y'],
                              iterations, errors['tp4_tc1']['MSE_Y'],
                              iterations, errors['tp5_tc1']['MSE_Y'])
plt.ylabel('MSE(Y)', size=textSize)
plt.xlabel('Iteration', size=textSize)
plt.grid(True,which="both",ls="-")
plt.legend((l11, l12, l13, l14), ('$T_{p} = 2$', '$T_{p} = 3$', '$T_{p} = 4$', '$T_{p} = 5$'), prop={'size': textSize})

plt.subplot(1, 3, 2)
plt.title('$T_{c} = 5$', size=textSize)
plt.yscale('log')
l11, l12, l13, l14 = plt.plot(iterations, errors['tp2_tc5']['MSE_Y'],
                              iterations, errors['tp3_tc5']['MSE_Y'],
                              iterations, errors['tp4_tc5']['MSE_Y'],
                              iterations, errors['tp5_tc5']['MSE_Y'])
plt.ylabel('MSE(Y)', size=textSize)
plt.xlabel('Iteration', size=textSize)
plt.grid(True,which="both",ls="-")
plt.legend((l11, l12, l13, l14), ('$T_{p} = 2$', '$T_{p} = 3$', '$T_{p} = 4$', '$T_{p} = 5$'), prop={'size': textSize})

plt.subplot(1, 3, 3)
plt.title('$T_{c} = 10$', size=textSize)
plt.yscale('log')
l11, l12, l13, l14 = plt.plot(iterations, errors['tp2_tc10']['MSE_Y'],
                              iterations, errors['tp3_tc10']['MSE_Y'],
                              iterations, errors['tp4_tc10']['MSE_Y'],
                              iterations, errors['tp5_tc10']['MSE_Y'],)
plt.ylabel('MSE(Y)', size=textSize)
plt.xlabel('Iteration', size=textSize)
plt.grid(True,which="both",ls="-")
plt.legend((l11, l12, l13, l14), ('$T_{p} = 2$', '$T_{p} = 3$', '$T_{p} = 4$', '$T_{p} = 5$'), prop={'size': textSize})

plt.subplots_adjust(None, None, None, None, 0.35, None)
plt.savefig(os.getcwd() + '\\results\\syntheticConsensus\\MSE_Y.pdf')


# Figure 4 - l2-norm of Y per iteration
plt.figure(4, figsize=(20, 6.33))
plt.subplot(1, 3, 1)
plt.title('$T_{c} = 1$', size=textSize)
l11, l12, l13, l14 = plt.plot(iterations, errors['tp2_tc1']['NORM_Y'],
                              iterations, errors['tp3_tc1']['NORM_Y'],
                              iterations, errors['tp4_tc1']['NORM_Y'],
                              iterations, errors['tp5_tc1']['NORM_Y'])
plt.ylabel('$\ell_{2}(Y)$', size=textSize)
plt.xlabel('Iteration', size=textSize)
plt.grid(True,which="both",ls="-")
plt.legend((l11, l12, l13, l14), ('$T_{p} = 2$', '$T_{p} = 3$', '$T_{p} = 4$', '$T_{p} = 5$'), prop={'size': textSize})

plt.subplot(1, 3, 2)
plt.title('$T_{c} = 5$', size=textSize)
l11, l12, l13, l14 = plt.plot(iterations, errors['tp2_tc5']['NORM_Y'],
                              iterations, errors['tp3_tc5']['NORM_Y'],
                              iterations, errors['tp4_tc5']['NORM_Y'],
                              iterations, errors['tp5_tc5']['NORM_Y'])
plt.ylabel('$\ell_{2}(Y)$', size=textSize)
plt.xlabel('Iteration', size=textSize)
plt.grid(True,which="both",ls="-")
plt.legend((l11, l12, l13, l14), ('$T_{p} = 2$', '$T_{p} = 3$', '$T_{p} = 4$', '$T_{p} = 5$'), prop={'size': textSize})

plt.subplot(1, 3, 3)
plt.title('$T_{c} = 10$', size=textSize)
l11, l12, l13, l14 = plt.plot(iterations, errors['tp2_tc10']['NORM_Y'],
                              iterations, errors['tp3_tc10']['NORM_Y'],
                              iterations, errors['tp4_tc10']['NORM_Y'],
                              iterations, errors['tp5_tc10']['NORM_Y'])
plt.ylabel('$\ell_{2}(Y)$', size=textSize)
plt.xlabel('Iteration', size=textSize)
plt.grid(True,which="both",ls="-")
plt.legend((l11, l12, l13, l14), ('$T_{p} = 2$', '$T_{p} = 3$', '$T_{p} = 4$', '$T_{p} = 5$'), prop={'size': textSize})

plt.subplots_adjust(None, None, None, None, 0.2, None)
plt.savefig(os.getcwd() + '\\results\\syntheticConsensus\\l2_Y.pdf')


# Figure 5 - l2-norm of Y per iteration
plt.figure(5, figsize=(20, 6.33))
plt.subplot(1, 3, 1)
plt.title('$T_{c} = 1$', size=textSize)
l11, l12, l13, l14 = plt.plot(iterations, errors['tp2_tc1']['PSNR_Y'],
                              iterations, errors['tp3_tc1']['PSNR_Y'],
                              iterations, errors['tp4_tc1']['PSNR_Y'],
                              iterations, errors['tp5_tc1']['PSNR_Y'])
plt.ylabel('PSNR(Y)', size=textSize)
plt.xlabel('Iteration', size=textSize)
plt.grid(True,which="both",ls="-")
plt.legend((l11, l12, l13, l14), ('$T_{p} = 2$', '$T_{p} = 3$', '$T_{p} = 4$', '$T_{p} = 5$'), prop={'size': textSize})

plt.subplot(1, 3, 2)
plt.title('$T_{c} = 5$', size=textSize)
l11, l12, l13, l14 = plt.plot(iterations, errors['tp2_tc5']['PSNR_Y'],
                              iterations, errors['tp3_tc5']['PSNR_Y'],
                              iterations, errors['tp4_tc5']['PSNR_Y'],
                              iterations, errors['tp5_tc5']['PSNR_Y'])
plt.ylabel('PSNR(Y)', size=textSize)
plt.xlabel('Iteration', size=textSize)
plt.grid(True,which="both",ls="-")
plt.legend((l11, l12, l13, l14), ('$T_{p} = 2$', '$T_{p} = 3$', '$T_{p} = 4$', '$T_{p} = 5$'), prop={'size': textSize})

plt.subplot(1, 3, 3)
plt.title('$T_{c} = 10$', size=textSize)
l11, l12, l13, l14 = plt.plot(iterations, errors['tp2_tc10']['PSNR_Y'],
                              iterations, errors['tp3_tc10']['PSNR_Y'],
                              iterations, errors['tp4_tc10']['PSNR_Y'],
                              iterations, errors['tp5_tc10']['PSNR_Y'])
plt.ylabel('PSNR(Y)', size=textSize)
plt.xlabel('Iteration', size=textSize)
plt.grid(True,which="both",ls="-")
plt.legend((l11, l12, l13, l14), ('$T_{p} = 2$', '$T_{p} = 3$', '$T_{p} = 4$', '$T_{p} = 5$'), prop={'size': textSize})

plt.subplots_adjust(None, None, None, None, None, None)
plt.savefig(os.getcwd() + '\\results\\syntheticConsensus\\PSNR_Y.pdf')


# Figure 6 - l2-norm of Y per iteration
plt.figure(6, figsize=(20, 6.33))
plt.subplot(1, 3, 1)
plt.title('$T_{c} = 1$', size=textSize)
l11, l12, l13, l14 = plt.plot(iterations, errors['tp2_tc1']['SSIM_Y'],
                              iterations, errors['tp3_tc1']['SSIM_Y'],
                              iterations, errors['tp4_tc1']['SSIM_Y'],
                              iterations, errors['tp5_tc1']['SSIM_Y'])
plt.ylabel('SSIM(Y)', size=textSize)
plt.xlabel('Iteration', size=textSize)
plt.grid(True,which="both",ls="-")
plt.legend((l11, l12, l13, l14), ('$T_{p} = 2$', '$T_{p} = 3$', '$T_{p} = 4$', '$T_{p} = 5$'), prop={'size': textSize})

plt.subplot(1, 3, 2)
plt.title('$T_{c} = 5$', size=textSize)
l11, l12, l13, l14 = plt.plot(iterations, errors['tp2_tc5']['SSIM_Y'],
                              iterations, errors['tp3_tc5']['SSIM_Y'],
                              iterations, errors['tp4_tc5']['SSIM_Y'],
                              iterations, errors['tp5_tc5']['SSIM_Y'])
plt.ylabel('SSIM(Y)', size=textSize)
plt.xlabel('Iteration', size=textSize)
plt.grid(True,which="both",ls="-")
plt.legend((l11, l12, l13, l14), ('$T_{p} = 2$', '$T_{p} = 3$', '$T_{p} = 4$', '$T_{p} = 5$'), prop={'size': textSize})

plt.subplot(1, 3, 3)
plt.title('$T_{c} = 10$', size=textSize)
l11, l12, l13, l14 = plt.plot(iterations, errors['tp2_tc10']['SSIM_Y'],
                              iterations, errors['tp3_tc10']['SSIM_Y'],
                              iterations, errors['tp4_tc10']['SSIM_Y'],
                              iterations, errors['tp5_tc10']['SSIM_Y'])
plt.ylabel('SSIM(Y)', size=textSize)
plt.xlabel('Iteration', size=textSize)
plt.grid(True,which="both",ls="-")
plt.legend((l11, l12, l13, l14), ('$T_{p} = 2$', '$T_{p} = 3$', '$T_{p} = 4$', '$T_{p} = 5$'), prop={'size': textSize})

plt.subplots_adjust(None, None, None, None, 0.25, None)
plt.savefig(os.getcwd() + '\\results\\syntheticConsensus\\SSIM_Y.pdf')
