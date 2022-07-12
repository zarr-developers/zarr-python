"""
 ASV benchmarks for creation.py
 """
import zarr
import numpy as np

class creation:
    
    params = [[3, 3], [1000, 1000]]
    
    def time_full(self, *n):
        self.array = zarr.full(n, fill_value = 2)
    
    def time_create(self, *n):
        self.array = zarr.create(n)

    def time_empty(self, *n):
        self.array = zarr.empty(n)

    def time_zeros(self, *n):
        self.array = zarr.zeros(n)

    def time_ones(self, *n):
        self.array = zarr.ones(n)
        

if __name__ == "__main__":
    instance = creation()
    instance.time_create()
    print("array", instance.array[:])
    instance.time_empty()
    print("array", instance.array[:])
    instance.time_zeros()
    print("array", instance.array[:])
    instance.time_ones()
    print("array", instance.array[:])
