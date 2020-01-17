import numpy as np
import struct
import base64
import time
import asyncio

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