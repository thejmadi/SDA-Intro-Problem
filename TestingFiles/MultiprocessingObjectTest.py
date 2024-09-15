# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 00:09:26 2024

@author: tarun
"""

import multiprocessing as mlt
import numpy as np

class TestObject(object):
    att_arr = np.array([0, 0, 0, 0])
    att_bool = False
    att_list = [0]
    att_str = "False"
    att_int = 0
    att_float = 0.0
    
    def PrintAtt(self):
        print(self.att_arr)
        print(self.att_bool)
        print(self.att_list)
        print(self.att_str)
        print(self.att_int)
        print(self.att_float)

# TODO: Figure out why np array is different in task func vs outside
def TaskFunc(obj, id_num):
    print("During #" + str(id_num))
    print(obj.PrintAtt())
    print()
    obj.att_arr[id_num] = id_num
    obj.att_bool = True
    obj.att_list = [1]
    obj.att_str = "True"
    obj.att_int = 1
    obj.att_float = 1.0
    return obj

# TODO: Test when object changed to TestObject
class MP(object):
    num_proc = 6
    num_runs = 3
    res = []
    def Multiprocessing(self, obj_instance):
        with mlt.Pool(min(self.num_proc, self.num_runs)) as pool:
            multi_results = [pool.apply_async(TaskFunc, args=(obj_instance, child_id+1)) for child_id in range(self.num_runs)]
            for r in multi_results:
                self.res.append(r.get())
        return self.res

# TODO: Figure out where none comes from in print statement
if __name__ == "__main__":
    object_instance = TestObject()
    optimizer = MP()
    changed_obj = optimizer.Multiprocessing(object_instance)
    print("Original Object")
    object_instance.PrintAtt()
    print()
    for k in range(len(changed_obj)):
        print("Changed Object")
        print(changed_obj[k].PrintAtt())
        print()
    