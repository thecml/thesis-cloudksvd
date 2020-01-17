import time
import json
import asyncio
import nest_asyncio
import aioredis
import numpy as np
import methods
import kubernetes.client
import os
import utils
from aiohttp import web
from kubernetes.client.rest import ApiException
from kubernetes import client, config
import kubernetes.client
import aiohttp
import concurrent.futures

global redisUrl
global configuration

#ENV variables
podIp = str(os.environ['POD_IP'])
webPort = int(os.environ['WEB_PORT'])
timeOut = int(os.environ['TIME_OUT'])
workerApiLabel = str(os.environ['WORKER_API_LABEL'])
postprocWebPort = int(os.environ['POSTPROC_WEB_PORT'])
postprocApiLabel = str(os.environ['POSTPROC_API_LABEL'])
debugMode = os.environ['DEBUG_MODE'] != "0"

class NumpyEncoder(json.JSONEncoder):
	def default(self, obj):
		if isinstance(obj, np.ndarray):
			return obj.tolist()
		return json.JSONEncoder.default(self, obj)

async def setup_redis():
	redisUrl = 'redis://localhost/0'
	redis = await aioredis.create_redis(redisUrl)
	try:
		await redis.flushdb()
		await redis.set('state', 'ready')
		print("Redis setup completed")
		return redisUrl
	finally:
		await utils.close_redis(redis)

async def setup_k8s():
	config.load_incluster_config()
	configuration = kubernetes.client.Configuration()
	configuration.api_key['authorization'] = utils.API_KEY
	configuration.api_key_prefix['authorization'] = 'Bearer'
	print("K8S setup completed")
	return configuration

async def do_work():
	#get pod ips
	podList = []
	api_instance = kubernetes.client.CoreV1Api(kubernetes.client.ApiClient(configuration))
	api_response = api_instance.list_pod_for_all_namespaces(watch=False,label_selector=workerApiLabel)
	for pod in api_response.items:
		ip = pod.status.pod_ip
		if(ip is None): continue
		podList.append(str(ip))
		if(debugMode): print("Inserted pod ip in podList: %s" % (str(podList)))
	
	#remove own ip from podList
	podList.remove(podIp)
	if(debugMode): print("Got %d pod ip's" % (len(podList)))

	#load training data
	redis = await aioredis.create_redis(redisUrl)
	D = np.asmatrix(await utils.load_from_redis(redis, 'trainingDictionary'))
	Y = np.asmatrix(await utils.load_from_redis(redis, 'trainingData'))
	await utils.close_redis(redis)

	#load spec
	redis = await aioredis.create_redis(redisUrl)
	spec = await redis.hgetall('spec', encoding='utf-8')
	consensusEnabled = (str(spec['CONSENSUS_ENABLED']) != "0")
	td = int(spec['CLOUD_KSVD_ITERATIONS'])
	t0 = int(spec['SPARSITY'])
	tp = int(spec['POWER_ITERATIONS'])
	tc = int(spec['CONSENSUS_ITERATIONS'])
	weightRatio = float(spec['WEIGHT_RATIO'])

	print("Running with this configuration: ")
	print("consensusEnabled: %d" % (int(consensusEnabled)))
	print("td: %d" % (td)) 
	print("t0: %d" % (t0)) 
	print("tp: %d" % (tp)) 
	print("tc: %d" % (tc)) 
	print("weightRatio: %8.2f" % (weightRatio))

	#set dimensions
	ddim = np.shape(D)[0]
	numOtherPods = len(podList)
	refvec = np.matrix(np.ones((ddim,1))) #Q_init for power method, sets direction of result
	weights = weightRatio*np.ones(numOtherPods)

	#run cloud-ksvd
	print("Starting Cloud K-SVD with %d pods" % (numOtherPods+1))
	print("dRows: %s" % (str(np.size(D,0))))
	print("dColumns: %s" % (str(np.size(D,1))))
	print("yRows: %s" % (str(np.size(Y,0))))
	print("yColumns: %s" % (str(np.size(Y,1))))

	loop = asyncio.get_event_loop()
	D,X,rerror,itertime,omptime,overalltotaltime,xiter,diter = loop.run_until_complete \
	(methods.CloudKSVD(D,Y,refvec,td,t0,tc,tp,weights,timeOut,redisUrl, \
	consensusEnabled,podList,webPort,debugMode))

	#finished with cloud-ksvd, stop time
	print("Finished Cloud K-SVD in %d seconds" % (overalltotaltime))

	#get correlation id
	redis = await aioredis.create_redis(redisUrl)
	id = await redis.get('correlationId', encoding='utf-8')
	await utils.close_redis(redis)

	#send results to postprocessor
	print("Saving stats to postproc")
	resultStats = {
		'totalTime': str('%.3f'%(overalltotaltime)),
		'cloudKsvdIterations': str(td),
		'sparsity': str(t0),
		'powerIterations': str(tp),
		'consensusIterations': str(tc),
		'errorPerIteration': np.array_str(rerror),
		'timePerIteration': np.array_str(itertime),
		'timePerOMP': np.array_str(omptime),
		'dRows': str(np.size(D,0)),
		'dColumns': str(np.size(D,1)),
		'yRows': str(np.size(Y,0)),
		'yColumns': str(np.size(Y,1)),
		'xRows': str(np.size(X,0)),
		'xColumns': str(np.size(X,1)),
		'numberOfPods': str(numOtherPods+1),
		'getConsensus': str(consensusEnabled),
		'timeOut': str(timeOut),
		'podIp': str(podIp),
		'correlationId': int(id)
	}
	resultObj = {
		'resultDictionary': D.tolist(),
		'resultSignal': X.tolist(),
		'resultListX': json.dumps(xiter, cls=NumpyEncoder),
		'resultListD': json.dumps(diter, cls=NumpyEncoder),
		'resultStats': resultStats,
	}
	api_instance = kubernetes.client.CoreV1Api(kubernetes.client.ApiClient(configuration))
	api_response = api_instance.list_pod_for_all_namespaces(watch=False,label_selector=postprocApiLabel)
	for pod in api_response.items:
		ip = pod.status.pod_ip
		if(ip is None): print("No postproc found!")
		async with aiohttp.ClientSession() as session:
			endpoint = '/save_results/'
			url = 'http://' + str(ip) + ':' + str(postprocWebPort) + endpoint
			async with session.post(url, json=resultObj) as resp:
				if(resp.status == 200 and debugMode):
					print("Data saved to pod: " + str(ip))
		await session.close()

	#ready again
	redis = await aioredis.create_redis(redisUrl)
	await redis.set('state', 'ready')
	await utils.close_redis(redis)
	print("Ready again")

#GET /qresidual
async def qresidual(request):
	#Read params
	k = request.rel_url.query['k']
	redis = await aioredis.create_redis(redisUrl)
	try:
		latestQ = await utils.load_from_redis(redis, 'latestQ' + str(k))
		if len(latestQ) == 0:
			if(debugMode): print("Did not find latest q, returning 404")
			return web.HTTPNotFound(text='Latest q not found')
		return web.json_response(latestQ.tolist())
	finally:
		await utils.close_redis(redis)

#GET /status
async def status(request):
	redis = await aioredis.create_redis(redisUrl)
	try:
		status = dict()
		if await redis.get('state', encoding='utf-8') == 'running':
			status['state'] = 'running'
		else:
			status['state'] = 'not running'
		return web.json_response(status)
	finally:
		await utils.close_redis(redis)

#POST /load_data
async def load_data(request):
	if(debugMode): print("Pod requested to load work")
	redis = await aioredis.create_redis(redisUrl)
	try:
		if await redis.get('state', encoding='utf-8') == 'ready':
			data = await request.json()
			D = np.asmatrix(data['D'])
			Y = np.asmatrix(data['Y'])
			id = int((data['correlationId']))
			spec = data['spec']
			if (debugMode): print("Loading dRows: %s" % (str(np.size(D,0))))
			if (debugMode): print("Loading dColumns: %s" % (str(np.size(D,1))))
			if (debugMode): print("Loading yRows: %s" % (str(np.size(Y,0))))
			if (debugMode): print("Loading yColumns: %s" % (str(np.size(Y,1))))
			await redis.flushdb()
			await redis.hmset_dict('spec', spec)
			await redis.set('correlationId', id)
			await utils.save_to_redis(redis, D, 'trainingDictionary')
			await utils.save_to_redis(redis, Y, 'trainingData')
			await redis.set('state', 'ready')
			if(debugMode): print("Pod done loading work")
			return web.HTTPOk()
		else:
			print("Work was not loaded")
			return web.HTTPInternalServerError()
	finally:
		await utils.close_redis(redis)

#POST /start_work
async def start_work(request):
	if(debugMode): print("Pod requested to start work")
	redis = await aioredis.create_redis(redisUrl)
	try:
		loop = asyncio.get_event_loop()
		if await redis.get('state', encoding='utf-8') == 'ready':
			await redis.set('state', 'running')
			loop.create_task(do_work())
			print("Work started in pod")
			return web.HTTPOk()
		else:
			print("Work was not started")
			return web.HTTPInternalServerError()
	finally:
		await utils.close_redis(redis)

#GET /training_data
async def training_data(request):
	redis = await aioredis.create_redis(redisUrl)
	try:
		trainingData = await utils.load_from_redis(redis, 'trainingData')
		if len(trainingData) == 0:
			return web.HTTPNotFound(text='Training data not found')
		return web.json_response(trainingData.tolist())
	finally:
		await utils.close_redis(redis)

if __name__ == "__main__":
	nest_asyncio.apply()
	loop = asyncio.new_event_loop()
	redisUrl = loop.run_until_complete(setup_redis())
	configuration = loop.run_until_complete(setup_k8s())
	app = web.Application(client_max_size=32000000)
	app.router.add_post('/load_data/', load_data)
	app.router.add_post('/start_work/', start_work)
	app.router.add_get('/status/', status)
	app.router.add_get('/qresidual/', qresidual)
	app.router.add_get('/training_data/', training_data)
	web.run_app(app, host='0.0.0.0', port=webPort)
