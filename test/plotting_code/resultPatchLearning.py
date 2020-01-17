# %% Patch Learning
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

folder = '\\patch_learning'

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

with open(os.getcwd() + '\\results\\patchLearning\\averageTimes.json','w') as json_file:
    json_file.write(json.dumps(averageTime))
    
iterations = range(0, td+1)

# Figure 1 - Average MSE of D per iteration
plt.figure(1, figsize=(20, 10))
plt.subplot(1, 2, 1)
plt.title('Local K-SVD', size=textSize, fontweight="bold")
plt.yscale('log')
l1, l2, l3, l4, l5, l6, l7, l8= plt.plot(iterations, errors['astronaut_a2_P4_C0']['MSE_D_AVG'],
                                         iterations, errors['camera_a2_P4_C0']['MSE_D_AVG'],
                                         iterations, errors['castle_a1_P4_C0']['MSE_D_AVG'],
                                         iterations, errors['chelsea_a2_P4_C0']['MSE_D_AVG'],
                                         iterations, errors['china_a2_P4_C0']['MSE_D_AVG'],
                                         iterations, errors['face_a1_P4_C0']['MSE_D_AVG'],
                                         iterations, errors['flower_a2_P4_C0']['MSE_D_AVG'],
                                         iterations, errors['lenna_a2_P4_C0']['MSE_D_AVG'])
plt.ylabel('MSE(D)', size=textSize)
plt.xlabel('Iteration', size=textSize)
plt.grid(True,which="both",ls="-")
plt.legend((l1, l2, l3, l4, l5, l6, l7, l8), ('Astronaut', 'Camera', 'Castle', 'Chelsea', 'China', 'Face', 'Flower', 'Lenna'), prop={'size': textSize})

plt.subplot(1, 2, 2)
plt.title('Cloud K-SVD', size=textSize, fontweight="bold")
plt.yscale('log')
l1, l2, l3, l4, l5, l6, l7, l8= plt.plot(iterations, errors['astronaut_a2_P4_C1']['MSE_D_AVG'],
                                         iterations, errors['camera_a2_P4_C1']['MSE_D_AVG'],
                                         iterations, errors['castle_a1_P4_C1']['MSE_D_AVG'],
                                         iterations, errors['chelsea_a2_P4_C1']['MSE_D_AVG'],
                                         iterations, errors['china_a2_P4_C1']['MSE_D_AVG'],
                                         iterations, errors['face_a1_P4_C1']['MSE_D_AVG'],
                                         iterations, errors['flower_a2_P4_C1']['MSE_D_AVG'],
                                         iterations, errors['lenna_a2_P4_C1']['MSE_D_AVG'])
plt.ylabel('MSE(D)', size=textSize)
plt.xlabel('Iteration', size=textSize)
plt.grid(True,which="both",ls="-")
plt.legend((l1, l2, l3, l4, l5, l6, l7, l8), ('Astronaut', 'Camera', 'Castle', 'Chelsea', 'China', 'Face', 'Flower', 'Lenna'), prop={'size': textSize})
plt.savefig(os.getcwd() + '\\results\\patchLearning\\MSE_D_AVG.pdf')


# Figure 2 - Average l2 of D per iteration
plt.figure(2, figsize=(20, 10))
plt.subplot(1, 2, 1)
plt.title('Local K-SVD', size=textSize, fontweight="bold")
l1, l2, l3, l4, l5, l6, l7, l8= plt.plot(iterations, errors['astronaut_a2_P4_C0']['NORM_D_AVG'],
                                         iterations, errors['camera_a2_P4_C0']['NORM_D_AVG'],
                                         iterations, errors['castle_a1_P4_C0']['NORM_D_AVG'],
                                         iterations, errors['chelsea_a2_P4_C0']['NORM_D_AVG'],
                                         iterations, errors['china_a2_P4_C0']['NORM_D_AVG'],
                                         iterations, errors['face_a1_P4_C0']['NORM_D_AVG'],
                                         iterations, errors['flower_a2_P4_C0']['NORM_D_AVG'],
                                         iterations, errors['lenna_a2_P4_C0']['NORM_D_AVG'])
plt.ylabel('$\ell_{2}(D)$', size=textSize)
plt.xlabel('Iteration', size=textSize)
plt.grid(True,which="both",ls="-")
plt.legend((l1, l2, l3, l4, l5, l6, l7, l8), ('Astronaut', 'Camera', 'Castle', 'Chelsea', 'China', 'Face', 'Flower', 'Lenna'), prop={'size': textSize})

plt.subplot(1, 2, 2)
plt.title('Cloud K-SVD', size=textSize, fontweight="bold")
l1, l2, l3, l4, l5, l6, l7, l8= plt.plot(iterations, errors['astronaut_a2_P4_C1']['NORM_D_AVG'],
                                         iterations, errors['camera_a2_P4_C1']['NORM_D_AVG'],
                                         iterations, errors['castle_a1_P4_C1']['NORM_D_AVG'],
                                         iterations, errors['chelsea_a2_P4_C1']['NORM_D_AVG'],
                                         iterations, errors['china_a2_P4_C1']['NORM_D_AVG'],
                                         iterations, errors['face_a1_P4_C1']['NORM_D_AVG'],
                                         iterations, errors['flower_a2_P4_C1']['NORM_D_AVG'],
                                         iterations, errors['lenna_a2_P4_C1']['NORM_D_AVG'])
plt.ylabel('$\ell_{2}(D)$', size=textSize)
plt.xlabel('Iteration', size=textSize)
plt.grid(True,which="both",ls="-")
plt.legend((l1, l2, l3, l4, l5, l6, l7, l8), ('Astronaut', 'Camera', 'Castle', 'Chelsea', 'China', 'Face', 'Flower', 'Lenna'), prop={'size': textSize})
plt.savefig(os.getcwd() + '\\results\\patchLearning\\NORM_D_AVG.pdf')



# Figure 3 - MSE of Y per iteration
plt.figure(3, figsize=(20, 10))
plt.subplot(1, 2, 1)
plt.title('Local K-SVD', size=textSize, fontweight="bold")
plt.yscale('log')
l1, l2, l3, l4, l5, l6, l7, l8= plt.plot(iterations, errors['astronaut_a2_P4_C0']['MSE_Y'],
                                         iterations, errors['camera_a2_P4_C0']['MSE_Y'],
                                         iterations, errors['castle_a1_P4_C0']['MSE_Y'],
                                         iterations, errors['chelsea_a2_P4_C0']['MSE_Y'],
                                         iterations, errors['china_a2_P4_C0']['MSE_Y'],
                                         iterations, errors['face_a1_P4_C0']['MSE_Y'],
                                         iterations, errors['flower_a2_P4_C0']['MSE_Y'],
                                         iterations, errors['lenna_a2_P4_C0']['MSE_Y'])
plt.ylabel('MSE(Y)', size=textSize)
plt.xlabel('Iteration', size=textSize)
plt.grid(True,which="both",ls="-")
plt.legend((l1, l2, l3, l4, l5, l6, l7, l8), ('Astronaut', 'Camera', 'Castle', 'Chelsea', 'China', 'Face', 'Flower', 'Lenna'), prop={'size': textSize})

plt.subplot(1, 2, 2)
plt.title('Cloud K-SVD', size=textSize, fontweight="bold")
plt.yscale('log')
l1, l2, l3, l4, l5, l6, l7, l8= plt.plot(iterations, errors['astronaut_a2_P4_C1']['MSE_Y'],
                                         iterations, errors['camera_a2_P4_C1']['MSE_Y'],
                                         iterations, errors['castle_a1_P4_C1']['MSE_Y'],
                                         iterations, errors['chelsea_a2_P4_C1']['MSE_Y'],
                                         iterations, errors['china_a2_P4_C1']['MSE_Y'],
                                         iterations, errors['face_a1_P4_C1']['MSE_Y'],
                                         iterations, errors['flower_a2_P4_C1']['MSE_Y'],
                                         iterations, errors['lenna_a2_P4_C1']['MSE_Y'])
plt.ylabel('MSE(Y)', size=textSize)
plt.xlabel('Iteration', size=textSize)
plt.grid(True,which="both",ls="-")
plt.legend((l1, l2, l3, l4, l5, l6, l7, l8), ('Astronaut', 'Camera', 'Castle', 'Chelsea', 'China', 'Face', 'Flower', 'Lenna'), prop={'size': textSize})
plt.savefig(os.getcwd() + '\\results\\patchLearning\\MSE_Y.pdf')


# Figure 4 - SSIM of Y per iteration
plt.figure(4, figsize=(20, 10))
plt.subplot(1, 2, 1)
plt.title('Local K-SVD', size=textSize, fontweight="bold")
l1, l2, l3, l4, l5, l6, l7, l8= plt.plot(iterations, errors['astronaut_a2_P4_C0']['SSIM_Y'],
                                         iterations, errors['camera_a2_P4_C0']['SSIM_Y'],
                                         iterations, errors['castle_a1_P4_C0']['SSIM_Y'],
                                         iterations, errors['chelsea_a2_P4_C0']['SSIM_Y'],
                                         iterations, errors['china_a2_P4_C0']['SSIM_Y'],
                                         iterations, errors['face_a1_P4_C0']['SSIM_Y'],
                                         iterations, errors['flower_a2_P4_C0']['SSIM_Y'],
                                         iterations, errors['lenna_a2_P4_C0']['SSIM_Y'])
plt.ylabel('SSIM(Y)', size=textSize)
plt.xlabel('Iteration', size=textSize)
plt.grid(True,which="both",ls="-")
plt.legend((l1, l2, l3, l4, l5, l6, l7, l8), ('Astronaut', 'Camera', 'Castle', 'Chelsea', 'China', 'Face', 'Flower', 'Lenna'), prop={'size': textSize})

plt.subplot(1, 2, 2)
plt.title('Cloud K-SVD', size=textSize, fontweight="bold")
l1, l2, l3, l4, l5, l6, l7, l8= plt.plot(iterations, errors['astronaut_a2_P4_C1']['SSIM_Y'],
                                         iterations, errors['camera_a2_P4_C1']['SSIM_Y'],
                                         iterations, errors['castle_a1_P4_C1']['SSIM_Y'],
                                         iterations, errors['chelsea_a2_P4_C1']['SSIM_Y'],
                                         iterations, errors['china_a2_P4_C1']['SSIM_Y'],
                                         iterations, errors['face_a1_P4_C1']['SSIM_Y'],
                                         iterations, errors['flower_a2_P4_C1']['SSIM_Y'],
                                         iterations, errors['lenna_a2_P4_C1']['SSIM_Y'])
plt.ylabel('SSIM(Y)', size=textSize)
plt.xlabel('Iteration', size=textSize)
plt.grid(True,which="both",ls="-")
plt.legend((l1, l2, l3, l4, l5, l6, l7, l8), ('Astronaut', 'Camera', 'Castle', 'Chelsea', 'China', 'Face', 'Flower', 'Lenna'), prop={'size': textSize})
plt.savefig(os.getcwd() + '\\results\\patchLearning\\SSIM_Y.pdf')


# Figure 5 - Face
plt.figure(5, figsize=(20, 7))

plt.subplot(1, 3, 1)
plt.title('Face Original\n', size=textSize, fontweight="bold")
plt.imshow(images[7], cmap=plt.cm.gray)
plt.xticks(())
plt.yticks(())

plt.subplot(1, 3, 2)
plt.title('Face after Local K-SVD\n', size=textSize, fontweight="bold")
plt.imshow(imgIter['face_a1_P4_C0'], cmap=plt.cm.gray)
plt.xticks(())
plt.yticks(())

plt.subplot(1, 3, 3)
plt.title('Face after Cloud K-SVD\n', size=textSize, fontweight="bold")
plt.imshow(imgIter['face_a1_P4_C1'], cmap=plt.cm.gray)
plt.xticks(())
plt.yticks(())

plt.savefig(os.getcwd() + '\\results\\patchLearning\\faceAfterPatches.pdf')


# Figure 6 - Castle
plt.figure(6, figsize=(20, 10))

plt.subplot(1, 3, 1)
plt.title('Castle Original\n', size=textSize, fontweight="bold")
plt.imshow(images[0], cmap=plt.cm.gray)
plt.xticks(())
plt.yticks(())

plt.subplot(1, 3, 2)
plt.title('Castle after Local K-SVD\n', size=textSize, fontweight="bold")
plt.imshow(imgIter['castle_a1_P4_C0'], cmap=plt.cm.gray)
plt.xticks(())
plt.yticks(())

plt.subplot(1, 3, 3)
plt.title('Castle after Cloud K-SVD\n', size=textSize, fontweight="bold")
plt.imshow(imgIter['castle_a1_P4_C1'], cmap=plt.cm.gray)
plt.xticks(())
plt.yticks(())

plt.savefig(os.getcwd() + '\\results\\patchLearning\\castleAfterPatches.pdf')


# Figure 7 - K = 3, K = 5, K = 7
plt.figure(7, figsize=(20, 10))

plt.subplot(1, 4, 1)
plt.title('Castle Original\n', size=textSize, fontweight="bold")
plt.imshow(images[0], cmap=plt.cm.gray)
plt.xticks(())
plt.yticks(())

plt.subplot(1, 4, 2)
plt.title('Castle Cloud K-SVD\n$K=3$', size=textSize, fontweight="bold")
plt.imshow(imgIter['castle_a1_P4_C1'], cmap=plt.cm.gray)
plt.xticks(())
plt.yticks(())

plt.subplot(1, 4, 3)
plt.title('Castle Cloud K-SVD\n$K=5$', size=textSize, fontweight="bold")
plt.imshow(imgIter['castle_a1_P4_C1_K5'], cmap=plt.cm.gray)
plt.xticks(())
plt.yticks(())

plt.subplot(1, 4, 4)
plt.title('Castle Cloud K-SVD\n$K=7$', size=textSize, fontweight="bold")
plt.imshow(imgIter['castle_a1_P4_C1_K7'], cmap=plt.cm.gray)
plt.xticks(())
plt.yticks(())

plt.savefig(os.getcwd() + '\\results\\patchLearning\\castleDifferentKC1.pdf')


# Figure 8  - K = 3, K = 5, K = 7 error of Y per iteration
plt.figure(8, figsize=(20, 15))

plt.subplot(2, 2, 2)
plt.title('$\ell_{2}$-norm of Y', size=textSize, fontweight="bold")
l1, l2, l3 = plt.plot(iterations, errors['castle_a1_P4_C1']['NORM_Y'],
                      iterations, errors['castle_a1_P4_C1_K5']['NORM_Y'],
                      iterations, errors['castle_a1_P4_C1_K7']['NORM_Y'])
plt.ylabel('$\ell_{2}(Y)$', size=textSize)
plt.xlabel('Iteration', size=textSize)
plt.grid(True,which="both",ls="-")
plt.legend((l1, l2, l3), ('$K=3$', '$K=5$', '$K=7$'), prop={'size': textSize})

plt.subplot(2, 2, 1)
plt.yscale('log')
plt.title('MSE of Y', size=textSize, fontweight="bold")
l1, l2, l3 = plt.plot(iterations, errors['castle_a1_P4_C1']['MSE_Y'],
                      iterations, errors['castle_a1_P4_C1_K5']['MSE_Y'],
                      iterations, errors['castle_a1_P4_C1_K7']['MSE_Y'])
plt.ylabel('MSE(Y)', size=textSize)
plt.xlabel('Iteration', size=textSize)
plt.grid(True,which="both",ls="-")
plt.legend((l1, l2, l3), ('$K=3$', '$K=5$', '$K=7$'), prop={'size': textSize})

plt.subplot(2, 2, 3)
plt.title('PSNR of Y', size=textSize, fontweight="bold")
l1, l2, l3 = plt.plot(iterations, errors['castle_a1_P4_C1']['PSNR_Y'],
                      iterations, errors['castle_a1_P4_C1_K5']['PSNR_Y'],
                      iterations, errors['castle_a1_P4_C1_K7']['PSNR_Y'])
plt.ylabel('PSNR(Y)', size=textSize)
plt.xlabel('Iteration', size=textSize)
plt.grid(True,which="both",ls="-")
plt.legend((l1, l2, l3), ('$K=3$', '$K=5$', '$K=7$'), prop={'size': textSize})

plt.subplot(2, 2, 4)
plt.title('SSIM of Y', size=textSize, fontweight="bold")
l1, l2, l3 = plt.plot(iterations, errors['castle_a1_P4_C1']['SSIM_Y'],
                      iterations, errors['castle_a1_P4_C1_K5']['SSIM_Y'],
                      iterations, errors['castle_a1_P4_C1_K7']['SSIM_Y'])
plt.ylabel('SSIM(Y)', size=textSize)
plt.xlabel('Iteration', size=textSize)
plt.grid(True,which="both",ls="-")
plt.legend((l1, l2, l3), ('$K=3$', '$K=5$', '$K=7$'), prop={'size': textSize})
plt.subplots_adjust(None, None, None, None, None, 0.25)
plt.savefig(os.getcwd() + '\\results\\patchLearning\\errorYCastleDifferentKC1.pdf')


# Figure 9 - K = 3, K = 5, K = 7
plt.figure(9, figsize=(20, 14))

plt.subplot(2, 3, 1)
plt.title('Face Original\n', size=(textSize-5), fontweight="bold")
plt.imshow(images[7], cmap=plt.cm.gray)
plt.xticks(())
plt.yticks(())

plt.subplot(2, 3, 2)
plt.title('Face after Cloud K-SVD\n$Patch Size=(5x5), N=50$', size=(textSize-5), fontweight="bold")
plt.imshow(imgIter['face_a1_P4_C1'], cmap=plt.cm.gray)
plt.xticks(())
plt.yticks(())

plt.subplot(2, 3, 3)
plt.title('Face after Cloud K-SVD\n$Patch Size=(6x6), N=100$', size=(textSize-5), fontweight="bold")
plt.imshow(imgIter['face_a1_P4_C1_6x6'], cmap=plt.cm.gray)
plt.xticks(())
plt.yticks(())

plt.subplot(2, 3, 4)
plt.title('Face after Cloud K-SVD\n$Patch Size=(7x7), N=100$', size=(textSize-5), fontweight="bold")
plt.imshow(imgIter['face_a1_P4_C1_7x7'], cmap=plt.cm.gray)
plt.xticks(())
plt.yticks(())

plt.subplot(2, 3, 5)
plt.title('Face after Cloud K-SVD\n$Patch Size=(8x8), N=150$', size=(textSize-5), fontweight="bold")
plt.imshow(imgIter['face_a1_P4_C1_8x8'], cmap=plt.cm.gray)
plt.xticks(())
plt.yticks(())

plt.subplot(2, 3, 6)
plt.title('Face after Cloud K-SVD\n$Patch Size=(9x9), N=200$', size=(textSize-5), fontweight="bold")
plt.imshow(imgIter['face_a1_P4_C1_9x9'], cmap=plt.cm.gray)
plt.xticks(())
plt.yticks(())
plt.savefig(os.getcwd() + '\\results\\patchLearning\\faceDifferentPatchSizeC1.pdf')


# Figure 10  - K = 3, K = 5, K = 7 error of Y per iteration
plt.figure(10, figsize=(20, 15))

plt.subplot(2, 2, 2)
plt.title('$\ell_{2}$-norm of Y', size=textSize, fontweight="bold")
l1, l2, l3, l4, l5 = plt.plot(iterations, errors['face_a1_P4_C1']['NORM_Y'],
                              iterations, errors['face_a1_P4_C1_6x6']['NORM_Y'],
                              iterations, errors['face_a1_P4_C1_7x7']['NORM_Y'],
                              iterations, errors['face_a1_P4_C1_8x8']['NORM_Y'],
                              iterations, errors['face_a1_P4_C1_9x9']['NORM_Y'])
plt.ylabel('$\ell_{2}(Y)$', size=textSize)
plt.xlabel('Iteration', size=textSize)
plt.grid(True,which="both",ls="-")
plt.legend((l1, l2, l3, l4, l5), ('$PS=(5x5)$', '$PS=(6x6)$', '$PS=(7x7)$', '$PS=(8x8)$', '$PS=(9x9)$'), prop={'size': textSize})

plt.subplot(2, 2, 1)
plt.yscale('log')
plt.title('MSE of Y', size=textSize, fontweight="bold")
l1, l2, l3, l4, l5 = plt.plot(iterations, errors['face_a1_P4_C1']['MSE_Y'],
                              iterations, errors['face_a1_P4_C1_6x6']['MSE_Y'],
                              iterations, errors['face_a1_P4_C1_7x7']['MSE_Y'],
                              iterations, errors['face_a1_P4_C1_8x8']['MSE_Y'],
                              iterations, errors['face_a1_P4_C1_9x9']['MSE_Y'])
plt.ylabel('MSE(Y)', size=textSize)
plt.xlabel('Iteration', size=textSize)
plt.grid(True,which="both",ls="-")
plt.legend((l1, l2, l3, l4, l5), ('$PS=(5x5)$', '$PS=(6x6)$', '$PS=(7x7)$', '$PS=(8x8)$', '$PS=(9x9)$'), prop={'size': textSize})

plt.subplot(2, 2, 3)
plt.title('PSNR of Y', size=textSize, fontweight="bold")
l1, l2, l3, l4, l5 = plt.plot(iterations, errors['face_a1_P4_C1']['PSNR_Y'],
                              iterations, errors['face_a1_P4_C1_6x6']['PSNR_Y'],
                              iterations, errors['face_a1_P4_C1_7x7']['PSNR_Y'],
                              iterations, errors['face_a1_P4_C1_8x8']['PSNR_Y'],
                              iterations, errors['face_a1_P4_C1_9x9']['PSNR_Y'])
plt.ylabel('PSNR(Y)', size=textSize)
plt.xlabel('Iteration', size=textSize)
plt.grid(True,which="both",ls="-")
plt.legend((l1, l2, l3, l4, l5), ('$PS=(5x5)$', '$PS=(6x6)$', '$PS=(7x7)$', '$PS=(8x8)$', '$PS=(9x9)$'), prop={'size': textSize})

plt.subplot(2, 2, 4)
plt.title('SSIM of Y', size=textSize, fontweight="bold")
l1, l2, l3, l4, l5 = plt.plot(iterations, errors['face_a1_P4_C1']['SSIM_Y'],
                              iterations, errors['face_a1_P4_C1_6x6']['SSIM_Y'],
                              iterations, errors['face_a1_P4_C1_7x7']['SSIM_Y'],
                              iterations, errors['face_a1_P4_C1_8x8']['SSIM_Y'],
                              iterations, errors['face_a1_P4_C1_9x9']['SSIM_Y'])
plt.ylabel('SSIM(Y)', size=textSize)
plt.xlabel('Iteration', size=textSize)
plt.grid(True,which="both",ls="-")
plt.legend((l1, l2, l3, l4, l5), ('$PS=(5x5)$', '$PS=(6x6)$', '$PS=(7x7)$', '$PS=(8x8)$', '$PS=(9x9)$'), prop={'size': textSize})
plt.subplots_adjust(None, None, None, None, None, 0.25)
plt.savefig(os.getcwd() + '\\results\\patchLearning\\errorYfaceDifferentPatchSizeC1.pdf')
