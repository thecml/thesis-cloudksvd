import json
import asyncio
import nest_asyncio
import aioredis
import os
import numpy as np
import utils
from aiohttp import web

global redisUrl

#    for element in json.loads(data['Y']):
        #Y.append(np.asmatrix(element))

#ENV variables
webPort = int(os.environ['WEB_PORT'])
debugMode = os.environ['DEBUG_MODE'] != "0"

#Mutex
mutex = asyncio.Lock()

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

async def setup_redis():
    redisUrl = 'redis://localhost/0'
    redis = await aioredis.create_redis(redisUrl)
    await redis.flushdb()
    redis.close()
    await redis.wait_closed()
    return redisUrl

#POST /save_results
async def save_results(request):
    
    await mutex.acquire()
    
    if(debugMode): print("Pod requested to save results")
    
    #Read JSON result data
    data = await request.json()
    D = np.asmatrix(data['resultDictionary'])
    X = np.asmatrix(data['resultSignal'])

    xIter = json.loads(data['resultListX'])
    dIter = json.loads(data['resultListD'])

    S = data['resultStats']
    podIp = S['podIp']

    #Save results for pod
    redis = await aioredis.create_redis(redisUrl)
    redis.lpush('resultIps', podIp)

    if (debugMode): print("Length of xiter: %d" % (len(xIter)))
    if (debugMode): print("Length of diter: %d" % (len(dIter)))

    for num, data in enumerate(xIter):
        await utils.save_to_redis(redis, np.array(data), podIp + 'resultListX' + str(num))

    for num, data in enumerate(dIter):
        await utils.save_to_redis(redis, np.array(data), podIp + 'resultListD' + str(num))

    await utils.save_to_redis(redis, D, podIp + 'resultDictionary')
    await utils.save_to_redis(redis, X, podIp + 'resultSignal')
    await redis.hmset_dict(podIp + 'resultStats', S)
    await utils.close_redis(redis)

    if(debugMode): print("Pod done saving results")
    
    mutex.release()
    
    return web.HTTPOk()

#GET /get_results
async def get_results(request):
    if(debugMode): print("Pod requested to get results")
    redis = await aioredis.create_redis(redisUrl)
    
    podIpList = []
    while(await redis.llen('resultIps')!=0):
        ip = await redis.rpop('resultIps', encoding='utf-8')
        podIpList.append(ip)

    if not podIpList: return web.HTTPNotFound()

    if(debugMode): print("Creating list with results")
    resultList = []
    for podIp in podIpList:
        D = await utils.load_from_redis(redis, podIp + 'resultDictionary')
        X = await utils.load_from_redis(redis, podIp + 'resultSignal')
        S = await redis.hgetall(podIp + 'resultStats', encoding='utf-8')
        xIter = []
        dIter = []
        td = int(S['cloudKsvdIterations'])
        for tdval in range(0,(td+1)):
            x = await utils.load_from_redis(redis, podIp + 'resultListX' + str(tdval))
            d = await utils.load_from_redis(redis, podIp + 'resultListD' + str(tdval))
            xIter.append(np.array(x, copy=True))
            dIter.append(np.array(d, copy=True))
        resultList.append({'D': D.tolist(),'X':X.tolist(), \
        'XITER': json.dumps(xIter, cls=NumpyEncoder), 'DITER':json.dumps(dIter, cls=NumpyEncoder),'STATS':S})
    await redis.flushdb()
    await utils.close_redis(redis)

    if(debugMode): print("Pod done getting results")
    return web.json_response(resultList)

if __name__ == "__main__":
    nest_asyncio.apply()
    loop = asyncio.new_event_loop()
    redisUrl = loop.run_until_complete(setup_redis())
    app = web.Application(client_max_size=128000000)
    app.router.add_post('/save_results/', save_results)
    app.router.add_get('/get_results/', get_results)
    web.run_app(app, host='0.0.0.0', port=webPort)