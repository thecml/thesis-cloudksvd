import numpy as np 
import aioredis
from numpy import linalg as LA
import utils
import time
import aiohttp
import asyncio
import concurrent.futures
import somp

async def CloudKSVD(D,Y,refvec,td,t0,tc,tp,weights,timeOut,\
	redisUrl="",getConsensus=False,podList=[],webPort=8080,debugMode=False):
	#run CloudKSVD
	ddim = np.shape(D)[0]
	K = np.shape(D)[1]
	S = np.shape(Y)[1]
	x = np.matrix(np.zeros((K,S)))
	rerror = np.zeros(td)
	iterTime = np.zeros(td)
	ompTime = np.zeros(td)
	overallStartTime = time.time()
	dIter = []
	xIter = []

	#open client session
	timeout = aiohttp.ClientTimeout(total=timeOut)
	async with aiohttp.ClientSession(timeout=timeout) as session:
		for t in range(0,td): #iterations of KSVD
		
			print("=================Iteration %d=================" % (t+1))
			print("Completed: %d procent" % (((t/td)*100)))
			startTime = time.time()
			
			#Sparse coding
			loop = asyncio.get_event_loop()
			x = await loop.run_in_executor(None,somp.SOMP,D,Y,t0)
			
			#Track D and x
			dIter.append(np.array(D, copy=True).astype(np.float16))
			xIter.append(np.array(x, copy=True).astype(np.float16))
			
			#Track OMP time
			ompTime[t] = str('%.3f'%(time.time() - startTime))
			
			#Line 49-51, 55-59 and 63-70: Courtesy of https://github.com/ZacBlanco/cloud-ksvd
			for k in range(0,K):
				if(debugMode): print("Making error matrix for atom %d" % (k))
				#Error matrix
				wk = [i for i,a in enumerate((np.array(x[k,:])).ravel()) if a!=0]
				Ek = (Y-np.dot(D,x)) + (D[:,k]*x[k,:])
				ERk = Ek[:,wk]
				
				#Power Method
				if(debugMode): print("Running power method for atom %d" % (k))
				if np.size(wk) == 0: #if empty
					M = np.matrix(np.zeros((ddim,ddim)))
				else:
					M = ERk*ERk.transpose()
				q = await powerMethod(M,tc,tp,k,weights,redisUrl,podList,webPort,debugMode,session,getConsensus)

				#Codebook Update
				if(debugMode): print("Doing codebook update for atom %d" % (k))
				if np.size(wk) != 0: #if not empty
					refdirection = np.sign(np.array(q*refvec)[0][0])
					if LA.norm(q) != 0:
						D[:,k] = (refdirection*(q/(LA.norm(q)))).reshape(ddim,1)
					else:
						D[:,k] = q.reshape(ddim,1)
					x[k,:] = 0
					x[k,wk] = np.array(D[:,k].transpose()*ERk).ravel()
		
			
			#If last iteration, run a final OMP
			if(t == (td-1)):
				loop = asyncio.get_event_loop()
				x = await loop.run_in_executor(None,somp.SOMP,D,Y,t0)
				dIter.append(np.array(D.astype(np.float16), copy=True).astype(np.float16))
				xIter.append(np.array(x.astype(np.float16), copy=True).astype(np.float16))
			
			#Error Data
			rerror[t] = np.linalg.norm(Y-np.dot(D,x))

			#Iteration time
			iterTime[t] = str('%.3f'%(time.time() - startTime))

	await session.close()

	#track time
	overallEndTime = time.time()
	overallTotalTime = overallEndTime - overallStartTime 

	return D,x,rerror,iterTime,ompTime,overallTotalTime,xIter,dIter

async def powerMethod(M,tc,tp,k,weights,redisUrl,podList,webPort,debugMode,session,getConsensus):
	#Calls on a Consensus method to find the top eigenvector
	datadim = M.shape[0]
	qnew = np.matrix(np.ones((datadim,1)))
	if(getConsensus):
		for powerIteration in range(0,tp,1):
			if(debugMode): print("Power iteration: %d" % (powerIteration))
			qnew = (M*qnew)
			qnew = await averagingConsensus(qnew,tc,k,weights,redisUrl,podList,webPort,debugMode,session)
			if LA.norm(qnew) != 0: qnew /= LA.norm(qnew) #normalize
	else:
		qnew = (M*qnew)
	return qnew.reshape(datadim) #returns eigenvector in 1d

async def averagingConsensus(z,tc,k,weights,redisUrl,podList,webPort,debugMode,session):
	'''Run a version of averaging consensus'''
	q,qnew = (z),(z)
	for consensusIteration in range(0,tc):
		#Standard consensus. Remember qnew is from last iteration
		if(debugMode): print("Consensus iteration: %d" % (consensusIteration))
		redis = await aioredis.create_redis(redisUrl)
		await utils.save_to_redis(redis, qnew, 'latestQ' + str(k))
		await utils.close_redis(redis)
		getDataIters = [getData(ip,webPort,'/qresidual/',k,debugMode,session) for ip in podList]
		receivedData = await asyncio.gather(*getDataIters)
		if(len(receivedData) > 0):
			sum = 0 #sum for all collected data
			for num, data in enumerate(receivedData):
				if(data is not None): #if actual data was received
					difference = data-q
					sum += weights[num]*(difference) #'sum' added to itself
				else:
					if(debugMode): print("data was none, did nothing")
			qnew = q + sum #update local data
		else:
			if(debugMode): print("receivedData did not contain any elements")
	return qnew #q(t) after processing

async def getData(ip,webPort,url,k,debugMode,session):
	params = {'k': str(k)}
	podUrl = 'http://' + str(ip) + ':' + str(webPort) + str(url)
	try:
		async with session.get(podUrl,params=params) as resp:
			if(resp.status == 200):
				return await resp.json() #read all into memory
		return None
	except Exception:
		return None
