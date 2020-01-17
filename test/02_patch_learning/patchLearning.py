# %% 1: Setup
import os
import numpy as np
from skimage import io, data
from skimage.color import rgb2gray
from sklearn.feature_extraction import image
from skimage.transform import downscale_local_mean
from skimage.measure import compare_mse as mse, compare_psnr as psnr, compare_ssim as ssim
import requests
import json
import matplotlib.pyplot as plt
from sklearn.datasets import load_sample_images
import time

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

'''
images = all test images from sklearn
imgDims = dimensions of all test images
imgIndex = the chosen image for the test, 0 - (len(images)-1)
N = number of atoms in the dictionary
podQuantity= number of pods in Kubernetes during test
patch_size = dimensions of each patch, m = patch_width x patch_height

consensusEnabled = whether or not consensus step is enabled
td = Cloud K-SVD iterations
K = Sparsity
tp = Power iterations
tc = Consensus iterations
wR = Weight Ratio - less than 1 and more than 0
'''

N = 200
podQuantity = 4
patchSize = (9, 9)
imgIndex = 7
downscaleFactor = (1, 1)

consensusEnabled = 1 # Value 0 or 1
K = 3 # Sparsity
td = 10 # Cloud K-SVD iterations
tp = 3 # Power iterations
tc = 5 # Consensus iterations

if (tc*podQuantity) > 1:
    wR = 1 / (tc*podQuantity) # Weight Ratio - less than 1 and more than 0
else:
    wR = 0.5

# Load 10 real images gray-scaled
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

# Downsample images by a factor along each axis.
for i, s in enumerate(images):
   images[i] = np.clip(downscale_local_mean(images[i], downscaleFactor),0,1)


# Save dimensions
imgDims = []
for img in images:
    height, width = img.shape
    imgDims.append((height, width))

noiselessImages = []

for img in images:
    noiselessImages.append(img.astype(np.float16))

# Show original image before patches
plt.imshow(noiselessImages[imgIndex].astype(np.float64), cmap=plt.cm.gray)

# Patches
allPatches = image.extract_patches_2d(noiselessImages[imgIndex][:, :], patchSize)

patchDims = allPatches.shape

allPatches = allPatches.reshape(allPatches.shape[0], -1)

allPatches = np.transpose(allPatches)

splitSize = int(np.floor((patchDims[0]/podQuantity)))
Y = []

for i in range(0, podQuantity):
    Y.append(allPatches[:, range(splitSize*i, splitSize*(i+1))])


# Make D with random data
D = []
for i in range(0, podQuantity):
    D.append(np.matrix(np.random.rand(np.shape(Y[i])[0],N)))

# %% 2: Convert to JSON and send data

# Make HTTP post
#url = 'http://192.168.1.111:30470/load_data/'
url = 'http://192.168.1.151:31839/load_data/'

payloadD = json.dumps(D, cls=NumpyEncoder)
payloadY = json.dumps(Y, cls=NumpyEncoder)

payload = {'D': payloadD,'Y': payloadY, 'CONSENSUS_ENABLED': str(consensusEnabled),
           'CLOUD_KSVD_ITERATIONS': str(td), 'SPARSITY': str(K),
           'POWER_ITERATIONS': str(tp), 'CONSENSUS_ITERATIONS': str(tc),
           'WEIGHT_RATIO': str(wR)}

response = requests.post(url,json=payload)
print(response)


# %% 3: Start work

#url = 'http://192.168.1.111:30470/start_work/'
url = 'http://192.168.1.151:31839/start_work/'
response = requests.post(url)
print(response)


# %% 4: Get results

#response = requests.get('http://192.168.1.111:31664/get_results/')
response = requests.get('http://192.168.1.151:30372/get_results/')
print(response)


# %% 5: Convert response

stats = []
dIter = []
xIter = []
yIter = []

for i, res in enumerate(response.json()):
    stats.append(res['STATS'])
    dIter.append([])
    xIter.append([])
    for d in json.loads(res['DITER']):
        dIter[i].append(np.array(d))
    for x in json.loads(res['XITER']):
        xIter[i].append(np.array(x))


# %% 6: Convert data

# Convert data for all iterations
yIter = [None] * (td+1)
imgIter = [None] * (td+1)

for i in range(0, (td+1)):
    for p in range(0, len(stats)):
        for num, stat in enumerate(stats):
            if int(stat['correlationId']) == p :
                if p > 0 :
                    yIter[i] = np.concatenate((yIter[i], np.dot(dIter[num][i], xIter[num][i])), axis=1)
                elif p == 0 :
                    yIter[i] = np.dot(dIter[num][i], xIter[num][i])
    yIter[i] = np.transpose(yIter[i])
    newPatches = yIter[i].reshape(np.size(yIter[i],0), *patchSize)
    imgIter[i] = image.reconstruct_from_patches_2d(newPatches, imgDims[imgIndex])


# %% 7: Error per iteration

normD = []
mseD = []

# The dictionaries
for i in range(0, (td+1)): # iteration
    normD.append([])
    mseD.append([])
    for p1 in range(0, (len(dIter)-1)): # First pod
        for p2 in range((p1+1), len(dIter)): # Second pod
            normD[i].append(np.sqrt(np.sum((np.array(dIter[p1][i]) - np.array(dIter[p2][i])) ** 2)))
            mseD[i].append(mse(np.array(dIter[p1][i]), np.array(dIter[p2][i])))

normDAvg = []
mseDAvg = []

# Average error
for i in range(0, (td+1)):
    normDAvg.append(sum(normD[i]) / len(normD[i]))
    mseDAvg.append(sum(mseD[i]) / len(mseD[i]))

normY = []
mseY = []
psnrY = []
ssimY = []

# The image
for img in imgIter:
    normY.append(np.sqrt(np.sum((img-images[imgIndex]) ** 2)))
    mseY.append(mse(img, images[imgIndex]))
    psnrY.append(psnr(images[imgIndex], img))
    ssimY.append(ssim(img, images[imgIndex]))

errors = {}

errors['NORM_D'] = normD
errors['MSE_D'] = mseD
errors['NORM_D_AVG'] = normDAvg
errors['MSE_D_AVG'] = mseDAvg
errors['NORM_Y'] = normY
errors['MSE_Y'] = mseY
errors['PSNR_Y'] = psnrY
errors['SSIM_Y'] = ssimY


# %% 8: Save and show data

# Create object for saving
setup = {}
setup['N'] = N
setup['podQuantity'] = podQuantity
setup['patchSize'] = patchSize
setup['imgIndex'] = imgIndex
setup['downscaleFactor'] = downscaleFactor
setup['consensusEnabled'] = consensusEnabled
setup['K'] = K
setup['td'] = td
setup['tp'] = tp
setup['tc'] = tc
setup['wR'] = wR

timestr = time.strftime('%Y%m%d-%H%M%S') # Create timestamp for current date and time
os.mkdir('results\\' + timestr) # Create new folder using timestamp

with open('results\\' + timestr + '\\stats.json','w') as json_file:
    json_file.write(json.dumps(stats))
with open('results\\' + timestr + '\\dIter.json','w') as json_file:
    json_file.write(json.dumps(dIter, cls=NumpyEncoder))
with open('results\\' + timestr + '\\xIter.json','w') as json_file:
    json_file.write(json.dumps(xIter, cls=NumpyEncoder))
with open('results\\' + timestr + '\\yIter.json','w') as json_file:
    json_file.write(json.dumps(yIter, cls=NumpyEncoder))
with open('results\\' + timestr + '\\imgIter.json','w') as json_file:
    json_file.write(json.dumps(imgIter, cls=NumpyEncoder))
with open('results\\' + timestr + '\\errors.json','w') as json_file:
    json_file.write(json.dumps(errors))
with open('results\\' + timestr + '\\setup.json','w') as json_file:
    json_file.write(json.dumps(setup))


def show_with_diff(image, reference, title):
    # The image
    image_normY = np.sqrt(np.sum((image-reference) ** 2))
    image_mseY = mse(image, reference)
    image_psnrY = psnr(reference, image)
    image_ssimY = ssim(image, reference)
    """Helper function to display denoising"""
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title('Original\n')
    plt.imshow(reference, vmin=0, vmax=1, cmap=plt.cm.gray, 
               interpolation='nearest')
    plt.xticks(())
    plt.yticks(())
    plt.subplot(1, 3, 2)
    difference = image - reference
    plt.title('Difference \n$\ell_2$: {0:.2f}, $MSE$: {1:.4f}, $PSNR$: {2:.2f}, $SSIM$: {3:.2f}'
              .format(image_normY, image_mseY, image_psnrY, image_ssimY))
    plt.imshow(difference, vmin=-0.5, vmax=0.5, cmap=plt.cm.PuOr,
               interpolation='nearest')
    plt.xticks(())
    plt.yticks(())
    plt.subplot(1, 3, 3)
    plt.title('Reconstruction\n')
    plt.imshow(image, vmin=0, vmax=1, cmap=plt.cm.gray,
               interpolation='nearest')
    plt.xticks(())
    plt.yticks(())
    plt.suptitle(title, size=16)
    plt.subplots_adjust(0.02, 0.02, 0.98, 0.82, 0.02, 0.1)
    plt.savefig('results\\' + timestr + '\\' + title + '.pdf')
    plt.savefig('results\\' + timestr + '\\' + title + '.png')


show_with_diff(imgIter[(len(imgIter)-1)], noiselessImages[imgIndex].astype(np.float64), 'Reconstructed image')    

# %% Plot all images

textSize = 14

plt.figure(1, figsize=(15, 5))
plt.subplot(2, 4, 1)
plt.title('Castle', size=textSize)
plt.imshow(images[0], cmap=plt.cm.gray)
plt.xticks(())
plt.yticks(())

plt.subplot(2, 4, 2)
plt.title('Lenna', size=textSize)
plt.imshow(images[1], cmap=plt.cm.gray)
plt.xticks(())
plt.yticks(())

plt.subplot(2, 4, 3)
plt.title('China', size=textSize)
plt.imshow(images[2], cmap=plt.cm.gray)
plt.xticks(())
plt.yticks(())

plt.subplot(2, 4, 4)
plt.title('Flower', size=textSize)
plt.imshow(images[3], cmap=plt.cm.gray)
plt.xticks(())
plt.yticks(())

plt.subplot(2, 4, 5)
plt.title('Chelsea', size=textSize)
plt.imshow(images[4], cmap=plt.cm.gray)
plt.xticks(())
plt.yticks(())

plt.subplot(2, 4, 6)
plt.title('Camera', size=textSize)
plt.imshow(images[5], cmap=plt.cm.gray)
plt.xticks(())
plt.yticks(())

plt.subplot(2, 4, 7)
plt.title('Astronaut', size=textSize)
plt.imshow(images[6], cmap=plt.cm.gray)
plt.xticks(())
plt.yticks(())

plt.subplot(2, 4, 8)
plt.title('Face', size=textSize)
plt.imshow(images[7], cmap=plt.cm.gray)
plt.xticks(())
plt.yticks(())

#plt.subplots_adjust(0.05, 0.05, 0.95, 0.8, 0.05, 0.2)

plt.savefig(os.getcwd() + '\\10Images.pdf')
plt.savefig(os.getcwd() + '\\10Images.png')
