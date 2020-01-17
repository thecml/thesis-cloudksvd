# %% 1: Setup
import os
import time
import numpy as np
from sklearn.datasets import make_sparse_coded_signal
from skimage.measure import compare_mse as mse, compare_psnr as psnr, compare_ssim as ssim
import requests
import json

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)



'''
N = number of atoms in the dictionary
podQuantity= number of pods in Kubernetes during test

consensusEnabled = whether or not consensus step is enabled
td = Cloud K-SVD iterations
K = Sparsity
tp = Power iterations
tc = Consensus iterations
wR = Weight Ratio - less than 1 and more than 0
'''

podQuantity = 4

Q = 2000
N = 50
M = 20

consensusEnabled = 1 # value 0 or 1
td = 50
K = 3
tp = 3
tc = 5

if tc > 1:
    wR = 1 / (tc*podQuantity) # Weight Ratio - less than 1 and more than 0
else:
    wR = 0.5


'''
# Generate new data
Y, D2, X2 = make_sparse_coded_signal(n_samples=Q,
                                   n_components=N,
                                   n_features=M,
                                   n_nonzero_coefs=K,
                                   random_state=0)
'''
# Load saved data
D = []
Y = []
with open('Y' + str(Q) +'.json') as json_file:
    jsonData = json.load(json_file)
    Y = np.array(jsonData)

with open('D' + str(podQuantity) + '.json') as json_file:
    jsonData = json.load(json_file)
    for elementD in jsonData:
        D.append(np.matrix(elementD))

'''
with open('consensusData.json') as json_file:
    jsonData = json.load(json_file)
    for elementD in jsonData['D']:
        D.append(np.matrix(elementD))
    Y = np.array(jsonData['Y'])
'''

# Split Y into parts for each pod
data = []
split_size = int(np.floor(Q/podQuantity))
for i in range(0, podQuantity):
    data.append(Y[:, range(split_size*i, split_size*(i+1))])

'''
# Make D with random data
D = []
for i in range(0, podQuantity):
    D.append(np.matrix(np.random.rand(M,N)))
'''

'''
# Save data
consensusData = {}
consensusData['Y'] = Y
consensusData['D'] = D

with open('consensusData.json','w') as json_file:
    json_file.write(json.dumps(consensusData, cls=NumpyEncoder))
'''

# %% 2: Send data

#Make HTTP post
#url = 'http://192.168.1.111:30470/load_data/'
url = 'http://192.168.1.151:31839/load_data/'

payloadD = json.dumps(D, cls=NumpyEncoder)
payloadY = json.dumps(data, cls=NumpyEncoder)

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

for i in range(0, (td+1)):
    for p in range(0, len(stats)):
        for num, stat in enumerate(stats):
            if int(stat['correlationId']) == p :
                if p > 0 :
                    yIter[i] = np.concatenate((yIter[i], np.dot(dIter[num][i], xIter[num][i])), axis=1)
                elif p == 0 :
                    yIter[i] = np.dot(dIter[num][i], xIter[num][i])


# %% 7: Error per iteration

normD = []
mseD = []

# The dictionaries
for i in range(0, (td+1)): # iteration
    normD.append([])
    mseD.append([])
    for p1 in range(0, (len(dIter)-1)): # First pod
        for p2 in range((p1+1), len(dIter)): # Second pod
            normD[i].append(np.sqrt(np.sum((dIter[p1][i] - dIter[p2][i]) ** 2)))
            mseD[i].append(mse(dIter[p1][i], dIter[p2][i]))

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
for yI in yIter:
    normY.append(np.sqrt(np.sum((yI-Y) ** 2)))
    mseY.append(mse(yI, Y))
    psnrY.append(psnr(Y, yI, data_range=(np.max(Y)-np.min(Y))))
    ssimY.append(ssim(yI, Y))

errors = {}
errors['NORM_D'] = normD
errors['MSE_D'] = mseD
errors['NORM_D_AVG'] = normDAvg
errors['MSE_D_AVG'] = mseDAvg
errors['NORM_Y'] = normY
errors['MSE_Y'] = mseY
errors['PSNR_Y'] = psnrY
errors['SSIM_Y'] = ssimY

# %% 8: Save data

# Create object for saving
setup = {}
setup['N'] = N
setup['podQuantity'] = podQuantity
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
with open('results\\' + timestr + '\\errors.json','w') as json_file:
    json_file.write(json.dumps(errors))
with open('results\\' + timestr + '\\setup.json','w') as json_file:
    json_file.write(json.dumps(setup))

