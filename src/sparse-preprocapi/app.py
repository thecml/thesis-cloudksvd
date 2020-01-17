import time
import json
import asyncio
import nest_asyncio
import os
import utils
import aiohttp
from aiohttp import web
from kubernetes.client.rest import ApiException
from kubernetes import client, config
import kubernetes.client
import numpy as np

global configuration

preprocWebPort = os.environ['PREPROC_WEB_PORT']
workerWebPort = os.environ['WORKER_WEB_PORT']
workerApiLabel = os.environ['WORKER_API_LABEL']
debugMode = os.environ['DEBUG_MODE'] != "0"

async def setup_k8s():
    config.load_incluster_config()
    configuration = kubernetes.client.Configuration()
    configuration.api_key['authorization'] = utils.API_KEY
    configuration.api_key_prefix['authorization'] = 'Bearer'
    print("K8S setup completed")
    return configuration

async def prepare_data(data):
    Y = []
    D = []
    for element in json.loads(data['Y']):
        Y.append(np.asmatrix(element))
    for element in json.loads(data['D']):
        D.append(np.asmatrix(element))
    spec = {
        'CONSENSUS_ENABLED': str(json.loads(data['CONSENSUS_ENABLED'])),
        'CLOUD_KSVD_ITERATIONS': str(json.loads(data['CLOUD_KSVD_ITERATIONS'])),
        'SPARSITY': str(json.loads(data['SPARSITY'])),
        'POWER_ITERATIONS': str(json.loads(data['POWER_ITERATIONS'])),
        'CONSENSUS_ITERATIONS': str(json.loads(data['CONSENSUS_ITERATIONS'])),
        'WEIGHT_RATIO': str(json.loads(data['WEIGHT_RATIO']))
    }
    return D, Y, spec

#POST /load_data/
async def load_data(request):
    #Load data to workers
    data = await request.json()
    D, Y, spec = await prepare_data(data)
    endpoint = '/load_data/'
    failed = False
    api_instance = kubernetes.client.CoreV1Api(kubernetes.client.ApiClient(configuration))
    api_response = api_instance.list_pod_for_all_namespaces(watch=False,label_selector=workerApiLabel)
    if(debugMode): print("Creating session to load data to workers")
    async with aiohttp.ClientSession() as session:
        for i, pod in enumerate(api_response.items):
            ip = pod.status.pod_ip
            dataObj = {'D': D[i].tolist(),'Y':Y[i].tolist(), 'correlationId':i, 'spec':spec}
            if(ip is None): continue
            if(debugMode): print("Loading data to pod: " + str(ip))
            url = 'http://' + str(ip) + ':' + str(workerWebPort) + endpoint
            async with session.post(url, json=dataObj) as resp:
                if(resp.status == 200):
                    print("Data loaded ok for pod: " + str(ip))
                else:
                    print("Data not loaded for pod: " + str(ip))
                    failed = True
        await session.close()
        if(failed): return web.HTTPInternalServerError
        return web.HTTPOk()

#POST /start_work/
async def start_work(request):
    #Start workers
    endpoint = '/start_work/'
    failed = False
    api_instance = kubernetes.client.CoreV1Api(kubernetes.client.ApiClient(configuration))
    api_response = api_instance.list_pod_for_all_namespaces(watch=False,label_selector=workerApiLabel)
    if(debugMode): print("Creating session to start workers")
    async with aiohttp.ClientSession() as session:
        for pod in api_response.items:
            ip = pod.status.pod_ip
            if(ip is None): continue
            if(debugMode): print("Starting work for pod: " + str(ip))
            url = 'http://' + str(ip) + ':' + str(workerWebPort) + endpoint
            async with session.post(url) as resp:
                if(resp.status == 200):
                    print("Work started ok for pod: " + str(ip))
                else:
                    print("Work not started for pod: " + str(ip))
                    failed = True
        await session.close()
        if(failed): return web.HTTPInternalServerError
        return web.HTTPOk()

if __name__ == "__main__":
    nest_asyncio.apply()
    loop = asyncio.new_event_loop()
    configuration = loop.run_until_complete(setup_k8s())
    app = web.Application(client_max_size=32000000)
    app.router.add_post('/load_data/', load_data)
    app.router.add_post('/start_work/', start_work)
    web.run_app(app, host='0.0.0.0', port=preprocWebPort)