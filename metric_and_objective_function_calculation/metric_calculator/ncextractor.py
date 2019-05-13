#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 15:26:38 2018

@author: barnhark
"""
import pandas as pd
import numpy as np
#import netCDF4 as nc4

import xarray as xr
import dask

class NCExtractor(object):
    """NCExtractor class."""
    def __init__(self, 
                 file_name_dict, 
                 point_file):
        """Initialize NCExtractor with input values."""
        
        self.point_location_df = pd.read_csv(point_file)
        self.timesteps = np.sort(list(file_name_dict.keys()))
        #self.file_name_dict = file_name_dict
        self.file_names = [file_name_dict[ts] for ts in self.timesteps]
        
        # metric_order
        self.metric_order = []
        self.loc_order = np.sort(self.point_location_df.Point_Name)
        for loc_name in self.loc_order:
            mets = [loc_name+'.'+str(time) for time in self.timesteps]
            self.metric_order.extend(mets)
            
    def extract_values(self):
        """Extracdt values from netcdf files."""
        
        # open dataset
        ds = xr.open_mfdataset(self.file_names,
                               concat_dim='nt',
                               engine='netcdf4')
        
        # extract values
        extracted_values = ds['topographic__elevation'].values[:, self.point_location_df.Row_number.values, self.point_location_df.Column_number.values]
        
        # construct dataframe of extracted values
        extracted_df = pd.DataFrame(extracted_values, columns=self.point_location_df.Point_Name)
        
        # stack dataframe
        stacked_df = extracted_df.stack()
        
        # create new index that matches expected metric name
        stacked_df.index = [stacked_df.index.get_level_values(0), stacked_df.index.map('{0[1]}.{0[0]}'.format)]
        
        # drop first level of idenx
        stacked_df.index = stacked_df.index.droplevel(level=0)
        
        # order metrics in the correct order
        self.reordered_df = stacked_df.reindex(self.metric_order)

        # assign correctly ordered metrics to extracted values class attribute
        self.extracted_values = self.reordered_df.values