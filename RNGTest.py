# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 23:25:51 2024

@author: tarun
"""

import multiprocessing as mlt
import numpy as np

def TaskFunc(c_id, burn, rng):
    single = np.zeros(1+c_id)
    multi = np.zeros((1+c_id, 3))
    #print(burn)
    for i in range(burn):
        print(rng.child.random())
    for i in range(1+c_id):
        single[i] = rng.child.random()
        burn += 1
    #for i in range(4*c_id):
    #    multi[i, :] = rng.child.multivariate_normal([0, 0, 0], np.diag([1, 1, 1]))
    #    rng.burn_count[c_id] += 1
    #print(rng.burn_count[c_id])
    return c_id, single, burn#, multi

class RNG(object):
    parent_rng = np.random.default_rng(12345)
    def __init__(self):
        self.child = self.parent_rng.spawn(1)[0]

# TODO: Test when object changed to TestObject
class MP(object):
    num_proc = 3
    num_runs = 1
    def Multiprocessing(self, rng_list, burn):
        res = []
        with mlt.Pool(min(self.num_proc, self.num_proc)) as pool:
            multi_results = [pool.apply_async(TaskFunc, args=(child_id, burn[child_id], rng_list[child_id])) for child_id in range(self.num_proc)]
            for r in multi_results:
                res.append(r.get())
        #print(res)
        for r in res:
            burn[r[0]] = r[2]
        return res

if __name__ == "__main__":
    optimizer = MP()
    rng_class = [RNG() for i in range(optimizer.num_proc)]
    burn_count = np.zeros(3, dtype=int)
    a = optimizer.Multiprocessing(rng_class, burn_count)
    print(burn_count)
    b = optimizer.Multiprocessing(rng_class, burn_count)
    #print()
    print(burn_count)
    burn_count.fill(0)
    c = optimizer.Multiprocessing(rng_class, burn_count)
    print(burn_count)
    d = optimizer.Multiprocessing(rng_class, burn_count)
    print(burn_count)