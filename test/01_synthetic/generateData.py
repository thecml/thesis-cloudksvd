# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 17:21:31 2019

@author: Bagger
"""
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



podQuantity = 16

Q = 20000
N = 50
M = 20

consensusEnabled = 1 # value 0 or 1
td = 50
K = 3
tp = 2
tc = 5

if tc > 1:
    wR = 1 / (tc*podQuantity) # Weight Ratio - less than 1 and more than 0
else:
    wR = 0.5



# Generate new data
Y20000, D2, X2 = make_sparse_coded_signal(n_samples=Q,
                                   n_components=N,
                                   n_features=M,
                                   n_nonzero_coefs=K,
                                   random_state=0)
'''
# Load saved data
D = []
Y = []

with open('consensusData.json') as json_file:
    jsonData = json.load(json_file)
    for elementD in jsonData['D']:
        D.append(np.matrix(elementD))
    Y = np.array(jsonData['Y'])
'''
'''
# Split Y into parts for each pod
data = []
split_size = int(np.floor(Q/podQuantity))
for i in range(0, podQuantity):
    data.append(Y[:, range(split_size*i, split_size*(i+1))])
'''



# Make D with random data
D16 = []
for i in range(0, podQuantity):
    D16.append(np.matrix(np.random.rand(M,N)))


'''
with open('Y20000.json','w') as json_file:
    json_file.write(json.dumps(Y20000, cls=NumpyEncoder))
'''

with open('D16.json','w') as json_file:
    json_file.write(json.dumps(D16, cls=NumpyEncoder))