import numpy as np
import struct
import base64
import time
import asyncio

#API_KEY
API_KEY = 'eyJhbGciOiJSUzI1NiIsImtpZCI6Ilg0SGEwd1kzQU1BTXBKeGxlcGxBbE8zNHY1bWNENWd6TkhCbnFPakFYTzAifQ.eyJpc3MiOiJrdWJlcm5ldGVzL3NlcnZpY2VhY2NvdW50Iiwia3ViZXJuZXRlcy5pby9zZXJ2aWNlYWNjb3VudC9uYW1lc3BhY2UiOiJkZWZhdWx0Iiwia3ViZXJuZXRlcy5pby9zZXJ2aWNlYWNjb3VudC9zZWNyZXQubmFtZSI6ImRlZmF1bHQtdG9rZW4tc3ZkOWwiLCJrdWJlcm5ldGVzLmlvL3NlcnZpY2VhY2NvdW50L3NlcnZpY2UtYWNjb3VudC5uYW1lIjoiZGVmYXVsdCIsImt1YmVybmV0ZXMuaW8vc2VydmljZWFjY291bnQvc2VydmljZS1hY2NvdW50LnVpZCI6ImJjNWY1YTJjLWIzYjQtNGVjZi05MGI1LWQzNjA3OTZmMTRhYSIsInN1YiI6InN5c3RlbTpzZXJ2aWNlYWNjb3VudDpkZWZhdWx0OmRlZmF1bHQifQ.aCvgRQIkgYA7TQ9S2UTJVMVm-SIUCKTQ2KUrHYoK401pt4oH1Nw6qJFrf_neaQvizn4QxQ_PTzPxIJokSM6xSyRD3kKhF46LD9Nt3mVpQqmslkLnXfw4ADZS0A9ZpBWtqJUbE_6CE-vWvKLbhHOn_f6EunR6d5UmBTOWv-cXDXbc86CXY8W4ITmfd-CXVd3vEUKl0phGrgDX1ewQiC860jRCwlZdy2FS4L-cZu2Q170LFaSk3DE9yuEFYmU3eiTB4e6LE0P2prmLMV2zEaV55xBZKHDKpjQFmjuiH-41vJEfJCBpfxPgBdlAOOcGHuBrwRTW_0KzoxulL_KYvp5dcg'
# Raspberry Pi 3 token 'eyJhbGciOiJSUzI1NiIsImtpZCI6IiJ9.eyJpc3MiOiJrdWJlcm5ldGVzL3NlcnZpY2VhY2NvdW50Iiwia3ViZXJuZXRlcy5pby9zZXJ2aWNlYWNjb3VudC9uYW1lc3BhY2UiOiJkZWZhdWx0Iiwia3ViZXJuZXRlcy5pby9zZXJ2aWNlYWNjb3VudC9zZWNyZXQubmFtZSI6ImRlZmF1bHQtdG9rZW4tOHpuZnYiLCJrdWJlcm5ldGVzLmlvL3NlcnZpY2VhY2NvdW50L3NlcnZpY2UtYWNjb3VudC5uYW1lIjoiZGVmYXVsdCIsImt1YmVybmV0ZXMuaW8vc2VydmljZWFjY291bnQvc2VydmljZS1hY2NvdW50LnVpZCI6Ijk4YjYzYzZmLWYzMjQtNGZhNC1iZTljLTEwMDYwNTEwYzUwMCIsInN1YiI6InN5c3RlbTpzZXJ2aWNlYWNjb3VudDpkZWZhdWx0OmRlZmF1bHQifQ.kmlA54ZgCpVrFcKzbgNuJzHH0HSit2qysaYXWg5rcAyMvoLjJMwFM9DdyE89X3VMl_HeexAm-3U1prZs7IoBn2jv1l1-bho_DKYP5dond4wwOKhI-59Sm3ZNI8h90NkthrQmgu_1PtXs5dZYW6wd1ojU74maYiaFaewSSdsKDaaDyQJOlXiPi5Huh-DZ2wqDtLqn_6FF8c9PPMR2KqgbMfc2_SD4SrXfhTWHQMHmdJ4k1OaL26v1H9bl0ko4AUMZhMXjucCCO4xvw4vIpFdgIRMB5AUS3YonMTWXZFwbY5yp2DDTWaa1rS5hAI6R5V15IRxLAZBiCw8GjFKIgRudiw'

async def save_results(redis, startT, endT, totalT, td, t0, tp, tc, rerror, dataType, dRows, \
    dColumns, yRows, yColumns, xRows, xColumns, number_pods, getConsensus, timeOut, podIp):
    #Saves the result of a cloud-ksvd run
    result = {
        'startTime': startT, 'endTime': endT, 'totalTime': str('%.3f'%(totalT)),
        'cloudKsvdIterations': str(td), 'sparsity': str(t0), 'powerIterations': str(tc),
        'consensusIterations': str(tp), 'errorPerIteration': np.array_str(rerror),
        'dataType': dataType, 'dRows': str(dRows), 'dColumns': str(dColumns),
        'yRows': str(yRows), 'yColumns': str(yColumns), 'xRows': str(xRows),
        'xColumns': str(xColumns), 'numberOfPods': str(number_pods),
        'genConsensus': str(getConsensus),'timeOut': str(timeOut), 'podIp': str(podIp)
    }
    await redis.hmset_dict('resultStats', result)

async def close_redis(redis):
    redis.close()
    await redis.wait_closed()

async def save_to_redis(redis,A,key):
    #Takes an open redis conn and a matrix to save
    array_dtype = str(A.dtype)
    l, w = A.shape
    A = A.ravel().tostring()
    valkey = '{0}|{1}#{2}#{3}'.format(int(time.time_ns()), array_dtype, l, w)
    await redis.set(valkey, A)
    await redis.set(key, valkey)

async def load_from_redis(redis, key):
    #Takes an open redis conn and a key that identifices the valkey to the saved matrix
    #Returns an empty array when no data found
    valkey = await redis.get(key, encoding='utf-8')
    if not valkey:
        return []
    A = await redis.get(valkey)
    array_dtype, l, w = valkey.split('|')[1].split('#')
    return np.fromstring(A, dtype=array_dtype).reshape(int(l), int(w))

#Courtesy of https://github.com/fubel/sparselandtools
def random_dictionary(n, K, normalized=True, seed=None):
    """
    Build a random dictionary matrix with K = n
    Args:
        n: square of signal dimension
        K: square of desired dictionary atoms
        normalized: If true, columns will be l2-normalized
        seed: Random seed

    Returns:
        Random dictionary
    """
    if seed:
        np.random.seed(seed)
    H = np.random.rand(n, K) * 255
    if normalized:
        for k in range(K):
            H[:, k] *= 1 / np.linalg.norm(H[:, k])
    return np.kron(H, H)

def serialize(A):
    array_dtype = str(A.dtype)
    l, w = A.shape
    A = A.ravel().tostring()
    key = '{0}|{1}#{2}#{3}'.format(int(time.time()), array_dtype, l, w)
    return A, key

def deserialize(A, key):
    array_dtype, l, w = key.split('|')[1].split('#')
    return np.fromstring(A, dtype=array_dtype).reshape(int(l), int(w))

def encode_vector(ar):
    return base64.encodestring(ar.tobytes()).decode('ascii')

def decode_vector(ar):
    return np.fromstring(base64.decodestring(bytes(ar.decode('ascii'), 'ascii')), dtype='uint16')

def encodeRedis(decoded):
   h, w = decoded.shape
   shape = struct.pack('>II',h,w)
   encoded = shape + decoded.tobytes()
   return encoded

def decodeRedis(encoded):
   h, w = struct.unpack('>II',encoded[:8])
   a = np.frombuffer(encoded, dtype=np.uint16, offset=8).reshape(h,w)
   return a