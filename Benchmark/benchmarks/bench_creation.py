"""
 ASV benchmarks for creation.py
 """

import zarr
import numpy as np 


class creation:
    
    def setup(self):
        self.shape = ([3, 2])
        self.chunks = (1000, 1000)
        self.dtype = "i4"
        self.arr = np.array([])
        self.fill_value = 1 
        
    
    def time_create(self):
        creation.create(self.shape, self.chunks, self.dtype)
        
    def time_empty(self):
        creation.empty(self.shape)
        
    
    def time_full(self):
        creation.full(self.shape, self.fill_value)
