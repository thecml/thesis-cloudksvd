# Courtesy of https://github.com/ZacBlanco/cloud-ksvd
# cython: language_level=3
import numpy as np 
from numpy import linalg as LA

def SOMP(D,Y,L):
	K = np.shape(D)[1]
	S = np.shape(Y)[1]
	x = np.matrix(np.zeros((K,S)))
	for s in range(0,S):
		x[:,s] = OMP(D,Y[:,s],L)
	return x

def OMP(D,Y,L):
	#K sparsity
	#Occasionally has a convergence issue with pinv function
	N = D.shape[0]
	K = D.shape[1]
	P = Y.shape[1]
	A = np.matrix('')

	if(N != Y.shape[0]):
		print("Feature-size does not match!")
		return

	for k in range(0,P):
		a = []
		x = Y[:,k]
		residual = x
		indx = [0]*L

		for j in range(0,L):
			proj = np.dot(np.transpose(D),residual)
			k_hat = np.argmax(np.absolute(proj))
			indx[j] = k_hat
			t1 = D[:,indx[0:j+1]]
			a = np.dot(np.linalg.pinv(t1),x)
			residual = x - np.dot(D[:,indx[0:j+1]],a) 
			if(np.sum(np.square(residual)) < 1e-6):    #1e-6 = magic number to quit pursuit
				break
		temp = np.zeros((K,1))
		temp[indx[0:j+1]] = a

		if (A.size == 0):
			A = temp
		else:
			A = np.column_stack((A,temp))
	return A