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


td = 10
textSize = 22

folder = '\\ct_denoise'

path = os.getcwd() + folder
files = os.listdir(path)

errors = {}
stats = {}
imgIter = {}

for file in files:
    with open(path + '\\' + file + '\\errors.json') as json_file:
        errors[file] = json.load(json_file)
        
for file in files:
    with open(path + '\\' + file + '\\stats.json') as json_file:
        stats[file] = json.load(json_file)
        
for file in files:
    with open(path + '\\' + file + '\\imgIter.json') as json_file:
        imgIter[file] = np.matrix(json.load(json_file)[td])
        print(file + ' has been loaded')



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


with open(os.getcwd() + '\\results\\ctDenoise\\averageTimes.json','w') as json_file:
    json_file.write(json.dumps(averageTime))
    
iterations = range(0, td+1)

# Figure 1 - Error of original Y per iteration
plt.figure(1, figsize=(20, 20))
plt.subplot(2, 2, 1)
plt.yscale('log')
plt.title('CT Scan after Cloud K-SVD\n$MSE$', size=textSize, fontweight="bold")
l1, l2, l3, l4, l5, l6, l7, l8, l9 = plt.plot(iterations, errors['img30_PS5x5_K3']['MSE_Y'],
                                              iterations, errors['img30_PS5x5_K5']['MSE_Y'],
                                              iterations, errors['img30_PS5x5_K7']['MSE_Y'],
                                              iterations, errors['img30_PS7x7_K3']['MSE_Y'],
                                              iterations, errors['img30_PS7x7_K5']['MSE_Y'],
                                              iterations, errors['img30_PS7x7_K7']['MSE_Y'],
                                              iterations, errors['img30_PS9x9_K3']['MSE_Y'],
                                              iterations, errors['img30_PS9x9_K5']['MSE_Y'],
                                              iterations, errors['img30_PS9x9_K7']['MSE_Y'])
plt.ylabel('MSE(Y)', size=textSize)
plt.xlabel('Iteration', size=textSize)
plt.grid(True,which="both",ls="-")



plt.subplot(2, 2, 3)
plt.title('CT Scan after Cloud K-SVD\n$PSNR$', size=textSize, fontweight="bold")
l1, l2, l3, l4, l5, l6, l7, l8, l9 = plt.plot(iterations, errors['img30_PS5x5_K3']['PSNR_Y'],
                                              iterations, errors['img30_PS5x5_K5']['PSNR_Y'],
                                              iterations, errors['img30_PS5x5_K7']['PSNR_Y'],
                                              iterations, errors['img30_PS7x7_K3']['PSNR_Y'],
                                              iterations, errors['img30_PS7x7_K5']['PSNR_Y'],
                                              iterations, errors['img30_PS7x7_K7']['PSNR_Y'],
                                              iterations, errors['img30_PS9x9_K3']['PSNR_Y'],
                                              iterations, errors['img30_PS9x9_K5']['PSNR_Y'],
                                              iterations, errors['img30_PS9x9_K7']['PSNR_Y'])
plt.ylabel('PSNR(Y)', size=textSize)
plt.xlabel('Iteration', size=textSize)
plt.grid(True,which="both",ls="-")


plt.subplot(2, 2, 4)
plt.title('CT Scan after Cloud K-SVD\n$SSIM$', size=textSize, fontweight="bold")
l1, l2, l3, l4, l5, l6, l7, l8, l9 = plt.plot(iterations, errors['img30_PS5x5_K3']['SSIM_Y'],
                                              iterations, errors['img30_PS5x5_K5']['SSIM_Y'],
                                              iterations, errors['img30_PS5x5_K7']['SSIM_Y'],
                                              iterations, errors['img30_PS7x7_K3']['SSIM_Y'],
                                              iterations, errors['img30_PS7x7_K5']['SSIM_Y'],
                                              iterations, errors['img30_PS7x7_K7']['SSIM_Y'],
                                              iterations, errors['img30_PS9x9_K3']['SSIM_Y'],
                                              iterations, errors['img30_PS9x9_K5']['SSIM_Y'],
                                              iterations, errors['img30_PS9x9_K7']['SSIM_Y'])
plt.ylabel('SSIM(Y)', size=textSize)
plt.xlabel('Iteration', size=textSize)
plt.grid(True,which="both",ls="-")
plt.subplot(2, 2, 2)
plt.axis('off')
plt.legend((l1, l2, l3, l4, l5, l6, l7, l8, l9), 
           ('$PS=(5x5), K=3, N=50$', '$PS=(5x5), K=5, N=50$', '$PS=(5x5), K=7, N=50$',
            '$PS=(7x7), K=3, N=100$', '$PS=(7x7), K=5, N=100$', '$PS=(7x7), K=7, N=100$',
            '$PS=(9x9), K=3, N=200$', '$PS=(9x9), K=5, N=200$', '$PS=(9x9), K=7, N=200$'), prop={'size': textSize}, loc="center")

plt.subplots_adjust(None, None, None, None, None, 0.25)
plt.savefig(os.getcwd() + '\\results\\ctDenoise\\img30_ERROR_Y.pdf')



# Figure 2 - Image 30
plt.figure(2, figsize=(20, 20))

plt.subplot(3, 3, 1)
plt.imshow(imgIter['img30_PS5x5_K3'], cmap=plt.cm.bone)
plt.xticks(())
plt.yticks(())
plt.ylabel('$PS=(5x5), N=50$', size=textSize)

plt.subplot(3, 3, 2)
plt.imshow(imgIter['img30_PS5x5_K5'], cmap=plt.cm.bone)
plt.xticks(())
plt.yticks(())

plt.subplot(3, 3, 3)
plt.imshow(imgIter['img30_PS5x5_K7'], cmap=plt.cm.bone)
plt.xticks(())
plt.yticks(())

plt.subplot(3, 3, 4)
plt.imshow(imgIter['img30_PS7x7_K3'], cmap=plt.cm.bone)
plt.xticks(())
plt.yticks(())
plt.ylabel('$PS=(7x7), N=100$', size=textSize)

plt.subplot(3, 3, 5)
plt.imshow(imgIter['img30_PS7x7_K5'], cmap=plt.cm.bone)
plt.xticks(())
plt.yticks(())

plt.subplot(3, 3, 6)
plt.imshow(imgIter['img30_PS7x7_K7'], cmap=plt.cm.bone)
plt.xticks(())
plt.yticks(())

plt.subplot(3, 3, 7)
plt.imshow(imgIter['img30_PS9x9_K3'], cmap=plt.cm.bone)
plt.xticks(())
plt.yticks(())
plt.ylabel('$PS=(9x9), N=200$', size=textSize)
plt.xlabel('$K=3$', size=textSize)

plt.subplot(3, 3, 8)
plt.imshow(imgIter['img30_PS9x9_K5'], cmap=plt.cm.bone)
plt.xticks(())
plt.yticks(())
plt.xlabel('$K=5$', size=textSize)

plt.subplot(3, 3, 9)
plt.imshow(imgIter['img30_PS9x9_K7'], cmap=plt.cm.bone)
plt.xticks(())
plt.yticks(())
plt.xlabel('$K=7$', size=textSize)

plt.subplots_adjust(None, None, None, None, 0.1, 0.1)
plt.savefig(os.getcwd() + '\\results\\ctDenoise\\img30_ctAfterPatches.pdf')


