import numpy as np
import zarr


class ZarrManager:    
    def __init__(self, params):
        ref_start, ref_end = params['reference_period']
        self.path = f"{params['var']}_{ref_start}_{ref_end}_{str(params['aggregation'])}_aggrwindow_{params['aggregation_window']}_percboost_{params['perc_boosting_window']}_perc_{params['percentile']}.zarr"
        self.zarr_store = zarr.open(self.path, mode='a')
        self.num_arrays = 0
        self.chunk_size = 100
        self.elements_list = []
        
    def add(self, array):
        self.elements_list.append(array)
        if sum([len(x) for x in self.elements_list]) >= self.chunk_size:
            self.store()
        
    def store(self):
        data = np.concat(self.elements_list)
        self.zarr_store.create_dataset(f'array_{self.num_arrays}', data=data, chunks=(1, -1, -1), overwrite=True)
        self.num_arrays += 1
        self.elements_list = []