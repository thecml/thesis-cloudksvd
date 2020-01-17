# %% Patch Denoise
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

folder = '\\patch_denoise'

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

dataset = load_sample_images()   
images = []
images.append(rgb2gray(io.imread('images\\castle.jpg')))
images.append(rgb2gray(io.imread('images\\lenna.jpg')))
images.append(rgb2gray(dataset.images[0]))
images.append(rgb2gray(dataset.images[1]))
images.append(rgb2gray(data.chelsea()))
images.append(rgb2gray(data.camera()))
images.append(rgb2gray(data.astronaut()))
images.append(rgb2gray(data.astronaut()[30:180, 150:300]))


# Convert images to values between 0 and 1
for i, s in enumerate(images):
    if np.max(images[i]) > 1:
        images[i] = np.clip((images[i] / 255),0,1)

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

with open(os.getcwd() + '\\results\\patchDenoise\\averageTimes.json','w') as json_file:
    json_file.write(json.dumps(averageTime))
    
iterations = range(0, td+1)

# Figure 1 - Error of Y per iteration
plt.figure(1, figsize=(20, 20))
plt.subplot(2, 2, 1)
plt.yscale('log')
plt.title('Face after Cloud K-SVD\n$MSE, \sigma^{2}=0.01$', size=textSize, fontweight="bold")
l1, l2, l3, l4, l5, l6, l7, l8, l9 = plt.plot(iterations, errors['face_K3_5x5_C1_sig001']['MSE_Y'],
                                              iterations, errors['face_K5_5x5_C1_sig001']['MSE_Y'],
                                              iterations, errors['face_K7_5x5_C1_sig001']['MSE_Y'],
                                              iterations, errors['face_K3_7x7_C1_sig001']['MSE_Y'],
                                              iterations, errors['face_K5_7x7_C1_sig001']['MSE_Y'],
                                              iterations, errors['face_K7_7x7_C1_sig001']['MSE_Y'],
                                              iterations, errors['face_K3_9x9_C1_sig001']['MSE_Y'],
                                              iterations, errors['face_K5_9x9_C1_sig001']['MSE_Y'],
                                              iterations, errors['face_K7_9x9_C1_sig001']['MSE_Y'])
plt.ylabel('MSE(Y)', size=textSize)
plt.xlabel('Iteration', size=textSize)
plt.grid(True,which="both",ls="-")



plt.subplot(2, 2, 3)
plt.title('Face after Cloud K-SVD\n$PSNR, \sigma^{2}=0.01$', size=textSize, fontweight="bold")
l1, l2, l3, l4, l5, l6, l7, l8, l9 = plt.plot(iterations, errors['face_K3_5x5_C1_sig001']['PSNR_Y'],
                                              iterations, errors['face_K5_5x5_C1_sig001']['PSNR_Y'],
                                              iterations, errors['face_K7_5x5_C1_sig001']['PSNR_Y'],
                                              iterations, errors['face_K3_7x7_C1_sig001']['PSNR_Y'],
                                              iterations, errors['face_K5_7x7_C1_sig001']['PSNR_Y'],
                                              iterations, errors['face_K7_7x7_C1_sig001']['PSNR_Y'],
                                              iterations, errors['face_K3_9x9_C1_sig001']['PSNR_Y'],
                                              iterations, errors['face_K5_9x9_C1_sig001']['PSNR_Y'],
                                              iterations, errors['face_K7_9x9_C1_sig001']['PSNR_Y'])
plt.ylabel('PSNR(Y)', size=textSize)
plt.xlabel('Iteration', size=textSize)
plt.grid(True,which="both",ls="-")


plt.subplot(2, 2, 4)
plt.title('Face after Cloud K-SVD\n$SSIM, \sigma^{2}=0.01$', size=textSize, fontweight="bold")
l1, l2, l3, l4, l5, l6, l7, l8, l9 = plt.plot(iterations, errors['face_K3_5x5_C1_sig001']['SSIM_Y'],
                                              iterations, errors['face_K5_5x5_C1_sig001']['SSIM_Y'],
                                              iterations, errors['face_K7_5x5_C1_sig001']['SSIM_Y'],
                                              iterations, errors['face_K3_7x7_C1_sig001']['SSIM_Y'],
                                              iterations, errors['face_K5_7x7_C1_sig001']['SSIM_Y'],
                                              iterations, errors['face_K7_7x7_C1_sig001']['SSIM_Y'],
                                              iterations, errors['face_K3_9x9_C1_sig001']['SSIM_Y'],
                                              iterations, errors['face_K5_9x9_C1_sig001']['SSIM_Y'],
                                              iterations, errors['face_K7_9x9_C1_sig001']['SSIM_Y'])
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
plt.savefig(os.getcwd() + '\\results\\patchDenoise\\ERROR_Y_sig001.pdf')


# Figure 2 - Error of Y per iteration
plt.figure(2, figsize=(20, 20))
plt.subplot(2, 2, 1)
plt.yscale('log')
plt.title('Face after Cloud K-SVD\n$MSE, \sigma^{2}=0.001$', size=textSize, fontweight="bold")
l1, l2, l3, l4, l5, l6, l7, l8, l9 = plt.plot(iterations, errors['face_K3_5x5_C1_sig0001']['MSE_Y'],
                                              iterations, errors['face_K5_5x5_C1_sig0001']['MSE_Y'],
                                              iterations, errors['face_K7_5x5_C1_sig0001']['MSE_Y'],
                                              iterations, errors['face_K3_7x7_C1_sig0001']['MSE_Y'],
                                              iterations, errors['face_K5_7x7_C1_sig0001']['MSE_Y'],
                                              iterations, errors['face_K7_7x7_C1_sig0001']['MSE_Y'],
                                              iterations, errors['face_K3_9x9_C1_sig0001']['MSE_Y'],
                                              iterations, errors['face_K5_9x9_C1_sig0001']['MSE_Y'],
                                              iterations, errors['face_K7_9x9_C1_sig0001']['MSE_Y'])
plt.ylabel('MSE(Y)', size=textSize)
plt.xlabel('Iteration', size=textSize)
plt.grid(True,which="both",ls="-")



plt.subplot(2, 2, 3)
plt.title('Face after Cloud K-SVD\n$PSNR, \sigma^{2}=0.001$', size=textSize, fontweight="bold")
l1, l2, l3, l4, l5, l6, l7, l8, l9 = plt.plot(iterations, errors['face_K3_5x5_C1_sig0001']['PSNR_Y'],
                                              iterations, errors['face_K5_5x5_C1_sig0001']['PSNR_Y'],
                                              iterations, errors['face_K7_5x5_C1_sig0001']['PSNR_Y'],
                                              iterations, errors['face_K3_7x7_C1_sig0001']['PSNR_Y'],
                                              iterations, errors['face_K5_7x7_C1_sig0001']['PSNR_Y'],
                                              iterations, errors['face_K7_7x7_C1_sig0001']['PSNR_Y'],
                                              iterations, errors['face_K3_9x9_C1_sig0001']['PSNR_Y'],
                                              iterations, errors['face_K5_9x9_C1_sig0001']['PSNR_Y'],
                                              iterations, errors['face_K7_9x9_C1_sig0001']['PSNR_Y'])
plt.ylabel('PSNR(Y)', size=textSize)
plt.xlabel('Iteration', size=textSize)
plt.grid(True,which="both",ls="-")


plt.subplot(2, 2, 4)
plt.title('Face after Cloud K-SVD\n$SSIM, \sigma^{2}=0.001$', size=textSize, fontweight="bold")
l1, l2, l3, l4, l5, l6, l7, l8, l9 = plt.plot(iterations, errors['face_K3_5x5_C1_sig0001']['SSIM_Y'],
                                              iterations, errors['face_K5_5x5_C1_sig0001']['SSIM_Y'],
                                              iterations, errors['face_K7_5x5_C1_sig0001']['SSIM_Y'],
                                              iterations, errors['face_K3_7x7_C1_sig0001']['SSIM_Y'],
                                              iterations, errors['face_K5_7x7_C1_sig0001']['SSIM_Y'],
                                              iterations, errors['face_K7_7x7_C1_sig0001']['SSIM_Y'],
                                              iterations, errors['face_K3_9x9_C1_sig0001']['SSIM_Y'],
                                              iterations, errors['face_K5_9x9_C1_sig0001']['SSIM_Y'],
                                              iterations, errors['face_K7_9x9_C1_sig0001']['SSIM_Y'])
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
plt.savefig(os.getcwd() + '\\results\\patchDenoise\\ERROR_Y_sig0001.pdf')



# Figure 3 - Error of Y per iteration
plt.figure(3, figsize=(20, 20))
plt.subplot(2, 2, 1)
plt.yscale('log')
plt.title('Face after Cloud K-SVD\n$MSE, \sigma^{2}=0.005$', size=textSize, fontweight="bold")
l1, l2, l3, l4, l5, l6, l7, l8, l9 = plt.plot(iterations, errors['face_K3_5x5_C1_sig0005']['MSE_Y'],
                                              iterations, errors['face_K5_5x5_C1_sig0005']['MSE_Y'],
                                              iterations, errors['face_K7_5x5_C1_sig0005']['MSE_Y'],
                                              iterations, errors['face_K3_7x7_C1_sig0005']['MSE_Y'],
                                              iterations, errors['face_K5_7x7_C1_sig0005']['MSE_Y'],
                                              iterations, errors['face_K7_7x7_C1_sig0005']['MSE_Y'],
                                              iterations, errors['face_K3_9x9_C1_sig0005']['MSE_Y'],
                                              iterations, errors['face_K5_9x9_C1_sig0005']['MSE_Y'],
                                              iterations, errors['face_K7_9x9_C1_sig0005']['MSE_Y'])
plt.ylabel('MSE(Y)', size=textSize)
plt.xlabel('Iteration', size=textSize)
plt.grid(True,which="both",ls="-")



plt.subplot(2, 2, 3)
plt.title('Face after Cloud K-SVD\n$PSNR, \sigma^{2}=0.005$', size=textSize, fontweight="bold")
l1, l2, l3, l4, l5, l6, l7, l8, l9 = plt.plot(iterations, errors['face_K3_5x5_C1_sig0005']['PSNR_Y'],
                                              iterations, errors['face_K5_5x5_C1_sig0005']['PSNR_Y'],
                                              iterations, errors['face_K7_5x5_C1_sig0005']['PSNR_Y'],
                                              iterations, errors['face_K3_7x7_C1_sig0005']['PSNR_Y'],
                                              iterations, errors['face_K5_7x7_C1_sig0005']['PSNR_Y'],
                                              iterations, errors['face_K7_7x7_C1_sig0005']['PSNR_Y'],
                                              iterations, errors['face_K3_9x9_C1_sig0005']['PSNR_Y'],
                                              iterations, errors['face_K5_9x9_C1_sig0005']['PSNR_Y'],
                                              iterations, errors['face_K7_9x9_C1_sig0005']['PSNR_Y'])
plt.ylabel('PSNR(Y)', size=textSize)
plt.xlabel('Iteration', size=textSize)
plt.grid(True,which="both",ls="-")


plt.subplot(2, 2, 4)
plt.title('Face after Cloud K-SVD\n$SSIM, \sigma^{2}=0.005$', size=textSize, fontweight="bold")
l1, l2, l3, l4, l5, l6, l7, l8, l9 = plt.plot(iterations, errors['face_K3_5x5_C1_sig0005']['SSIM_Y'],
                                              iterations, errors['face_K5_5x5_C1_sig0005']['SSIM_Y'],
                                              iterations, errors['face_K7_5x5_C1_sig0005']['SSIM_Y'],
                                              iterations, errors['face_K3_7x7_C1_sig0005']['SSIM_Y'],
                                              iterations, errors['face_K5_7x7_C1_sig0005']['SSIM_Y'],
                                              iterations, errors['face_K7_7x7_C1_sig0005']['SSIM_Y'],
                                              iterations, errors['face_K3_9x9_C1_sig0005']['SSIM_Y'],
                                              iterations, errors['face_K5_9x9_C1_sig0005']['SSIM_Y'],
                                              iterations, errors['face_K7_9x9_C1_sig0005']['SSIM_Y'])
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
plt.savefig(os.getcwd() + '\\results\\patchDenoise\\ERROR_Y_sig0005.pdf')


# Figure 4 - Faces with sig^2=0.01
plt.figure(4, figsize=(20, 20))

plt.subplot(3, 3, 1)
#plt.title('Face After Cloud-KSVD\n$PS=(5x5), K=3, N=50, \sigma^{2}=0.01$', size=textSize, fontweight="bold")
plt.imshow(imgIter['face_K3_5x5_C1_sig001'], cmap=plt.cm.gray)
plt.xticks(())
plt.yticks(())
plt.ylabel('$PS=(5x5), N=50$', size=textSize)

plt.subplot(3, 3, 2)
#plt.title('Face After Cloud-KSVD\n$PS=(5x5), K=5, N=50, \sigma^{2}=0.01$', size=textSize, fontweight="bold")
plt.imshow(imgIter['face_K5_5x5_C1_sig001'], cmap=plt.cm.gray)
plt.xticks(())
plt.yticks(())

plt.subplot(3, 3, 3)
#plt.title('Face After Cloud-KSVD\n$PS=(5x5), K=7, N=50, \sigma^{2}=0.01$', size=textSize, fontweight="bold")
plt.imshow(imgIter['face_K7_5x5_C1_sig001'], cmap=plt.cm.gray)
plt.xticks(())
plt.yticks(())

plt.subplot(3, 3, 4)
#plt.title('Face After Cloud-KSVD\n$PS=(7x7), K=3, N=100, \sigma^{2}=0.01$', size=textSize, fontweight="bold")
plt.imshow(imgIter['face_K3_7x7_C1_sig001'], cmap=plt.cm.gray)
plt.xticks(())
plt.yticks(())
plt.ylabel('$PS=(7x7), N=100$', size=textSize)

plt.subplot(3, 3, 5)
#plt.title('Face After Cloud-KSVD\n$PS=(7x7), K=5, N=100, \sigma^{2}=0.01$', size=textSize, fontweight="bold")
plt.imshow(imgIter['face_K5_7x7_C1_sig001'], cmap=plt.cm.gray)
plt.xticks(())
plt.yticks(())

plt.subplot(3, 3, 6)
#plt.title('Face After Cloud-KSVD\n$PS=(7x7), K=7, N=100, \sigma^{2}=0.01$', size=textSize, fontweight="bold")
plt.imshow(imgIter['face_K7_7x7_C1_sig001'], cmap=plt.cm.gray)
plt.xticks(())
plt.yticks(())

plt.subplot(3, 3, 7)
#plt.title('Face After Cloud-KSVD\n$PS=(9x9), K=3, N=200, \sigma^{2}=0.01$', size=textSize, fontweight="bold")
plt.imshow(imgIter['face_K3_9x9_C1_sig001'], cmap=plt.cm.gray)
plt.xticks(())
plt.yticks(())
plt.ylabel('$PS=(9x9), N=200$', size=textSize)
plt.xlabel('$K=3$', size=textSize)

plt.subplot(3, 3, 8)
#plt.title('Face After Cloud-KSVD\n$PS=(9x9), K=5, N=200, \sigma^{2}=0.01$', size=textSize, fontweight="bold")
plt.imshow(imgIter['face_K5_9x9_C1_sig001'], cmap=plt.cm.gray)
plt.xticks(())
plt.yticks(())
plt.xlabel('$K=5$', size=textSize)

plt.subplot(3, 3, 9)
#plt.title('Face After Cloud-KSVD\n$PS=(9x9), K=7, N=200, \sigma^{2}=0.01$', size=textSize, fontweight="bold")
plt.imshow(imgIter['face_K7_9x9_C1_sig001'], cmap=plt.cm.gray)
plt.xticks(())
plt.yticks(())
plt.xlabel('$K=7$', size=textSize)

plt.suptitle('Noise level: $\sigma^{2}=0.01$', y=0.91, fontsize=textSize)
plt.subplots_adjust(None, None, None, None, 0.1, 0.1)
plt.savefig(os.getcwd() + '\\results\\patchDenoise\\facesAfterPatchesSig001.pdf')


# Figure 5 - Faces with sig^2=0.005
plt.figure(5, figsize=(20, 20))

plt.subplot(3, 3, 1)
#plt.title('Face After Cloud-KSVD\n$PS=(5x5), K=3, N=50, \sigma^{2}=0.005$', size=textSize, fontweight="bold")
plt.imshow(imgIter['face_K3_5x5_C1_sig0005'], cmap=plt.cm.gray)
plt.xticks(())
plt.yticks(())
plt.ylabel('$PS=(5x5), N=50$', size=textSize)

plt.subplot(3, 3, 2)
#plt.title('Face After Cloud-KSVD\n$PS=(5x5), K=5, N=50, \sigma^{2}=0.005$', size=textSize, fontweight="bold")
plt.imshow(imgIter['face_K5_5x5_C1_sig0005'], cmap=plt.cm.gray)
plt.xticks(())
plt.yticks(())

plt.subplot(3, 3, 3)
#plt.title('Face After Cloud-KSVD\n$PS=(5x5), K=7, N=50, \sigma^{2}=0.005$', size=textSize, fontweight="bold")
plt.imshow(imgIter['face_K7_5x5_C1_sig0005'], cmap=plt.cm.gray)
plt.xticks(())
plt.yticks(())

plt.subplot(3, 3, 4)
#plt.title('Face After Cloud-KSVD\n$PS=(7x7), K=3, N=100, \sigma^{2}=0.005$', size=textSize, fontweight="bold")
plt.imshow(imgIter['face_K3_7x7_C1_sig0005'], cmap=plt.cm.gray)
plt.xticks(())
plt.yticks(())
plt.ylabel('$PS=(7x7), N=100$', size=textSize)

plt.subplot(3, 3, 5)
#plt.title('Face After Cloud-KSVD\n$PS=(7x7), K=5, N=100, \sigma^{2}=0.005$', size=textSize, fontweight="bold")
plt.imshow(imgIter['face_K5_7x7_C1_sig0005'], cmap=plt.cm.gray)
plt.xticks(())
plt.yticks(())

plt.subplot(3, 3, 6)
#plt.title('Face After Cloud-KSVD\n$PS=(7x7), K=7, N=100, \sigma^{2}=0.005$', size=textSize, fontweight="bold")
plt.imshow(imgIter['face_K7_7x7_C1_sig0005'], cmap=plt.cm.gray)
plt.xticks(())
plt.yticks(())

plt.subplot(3, 3, 7)
#plt.title('Face After Cloud-KSVD\n$PS=(9x9), K=3, N=200, \sigma^{2}=0.005$', size=textSize, fontweight="bold")
plt.imshow(imgIter['face_K3_9x9_C1_sig0005'], cmap=plt.cm.gray)
plt.xticks(())
plt.yticks(())
plt.ylabel('$PS=(9x9), N=200$', size=textSize)
plt.xlabel('$K=3$', size=textSize)

plt.subplot(3, 3, 8)
#plt.title('Face After Cloud-KSVD\n$PS=(9x9), K=5, N=200, \sigma^{2}=0.005$', size=textSize, fontweight="bold")
plt.imshow(imgIter['face_K5_9x9_C1_sig0005'], cmap=plt.cm.gray)
plt.xticks(())
plt.yticks(())
plt.xlabel('$K=5$', size=textSize)

plt.subplot(3, 3, 9)
#plt.title('Face After Cloud-KSVD\n$PS=(9x9), K=7, N=200, \sigma^{2}=0.005$', size=textSize, fontweight="bold")
plt.imshow(imgIter['face_K7_9x9_C1_sig0005'], cmap=plt.cm.gray)
plt.xticks(())
plt.yticks(())
plt.xlabel('$K=7$', size=textSize)

plt.suptitle('Noise level: $\sigma^{2}=0.005$', y=0.91, fontsize=textSize)
plt.subplots_adjust(None, None, None, None, 0.1, 0.1)
plt.savefig(os.getcwd() + '\\results\\patchDenoise\\facesAfterPatchesSig0005.pdf')


# Figure 6 - Faces with sig^2=0.001
plt.figure(6, figsize=(20, 20))

plt.subplot(3, 3, 1)
#plt.title('Face After Cloud-KSVD\n$PS=(5x5), K=3, N=50, \sigma^{2}=0.001$', size=textSize, fontweight="bold")
plt.imshow(imgIter['face_K3_5x5_C1_sig0001'], cmap=plt.cm.gray)
plt.xticks(())
plt.yticks(())
plt.ylabel('$PS=(5x5), N=50$', size=textSize)

plt.subplot(3, 3, 2)
#plt.title('Face After Cloud-KSVD\n$PS=(5x5), K=5, N=50, \sigma^{2}=0.001$', size=textSize, fontweight="bold")
plt.imshow(imgIter['face_K5_5x5_C1_sig0001'], cmap=plt.cm.gray)
plt.xticks(())
plt.yticks(())

plt.subplot(3, 3, 3)
#plt.title('Face After Cloud-KSVD\n$PS=(5x5), K=7, N=50, \sigma^{2}=0.001$', size=textSize, fontweight="bold")
plt.imshow(imgIter['face_K7_5x5_C1_sig0001'], cmap=plt.cm.gray)
plt.xticks(())
plt.yticks(())

plt.subplot(3, 3, 4)
#plt.title('Face After Cloud-KSVD\n$PS=(7x7), K=3, N=100, \sigma^{2}=0.001$', size=textSize, fontweight="bold")
plt.imshow(imgIter['face_K3_7x7_C1_sig0001'], cmap=plt.cm.gray)
plt.xticks(())
plt.yticks(())
plt.ylabel('$PS=(7x7), N=100$', size=textSize)

plt.subplot(3, 3, 5)
#plt.title('Face After Cloud-KSVD\n$PS=(7x7), K=5, N=100, \sigma^{2}=0.001$', size=textSize, fontweight="bold")
plt.imshow(imgIter['face_K5_7x7_C1_sig0001'], cmap=plt.cm.gray)
plt.xticks(())
plt.yticks(())

plt.subplot(3, 3, 6)
#plt.title('Face After Cloud-KSVD\n$PS=(7x7), K=7, N=100, \sigma^{2}=0.001$', size=textSize, fontweight="bold")
plt.imshow(imgIter['face_K7_7x7_C1_sig0001'], cmap=plt.cm.gray)
plt.xticks(())
plt.yticks(())

plt.subplot(3, 3, 7)
#plt.title('Face After Cloud-KSVD\n$PS=(9x9), K=3, N=200, \sigma^{2}=0.001$', size=textSize, fontweight="bold")
plt.imshow(imgIter['face_K3_9x9_C1_sig0001'], cmap=plt.cm.gray)
plt.xticks(())
plt.yticks(())
plt.ylabel('$PS=(9x9), N=200$', size=textSize)
plt.xlabel('$K=3$', size=textSize)

plt.subplot(3, 3, 8)
#plt.title('Face After Cloud-KSVD\n$PS=(9x9), K=5, N=200, \sigma^{2}=0.001$', size=textSize, fontweight="bold")
plt.imshow(imgIter['face_K5_9x9_C1_sig0001'], cmap=plt.cm.gray)
plt.xticks(())
plt.yticks(())
plt.xlabel('$K=5$', size=textSize)

plt.subplot(3, 3, 9)
#plt.title('Face After Cloud-KSVD\n$PS=(9x9), K=7, N=200, \sigma^{2}=0.001$', size=textSize, fontweight="bold")
plt.imshow(imgIter['face_K7_9x9_C1_sig0001'], cmap=plt.cm.gray)
plt.xticks(())
plt.yticks(())
plt.xlabel('$K=7$', size=textSize)

plt.suptitle('Noise level: $\sigma^{2}=0.001$', y=0.91, fontsize=textSize)
plt.subplots_adjust(None, None, None, None, 0.1, 0.1)
plt.savefig(os.getcwd() + '\\results\\patchDenoise\\facesAfterPatchesSig0001.pdf')
