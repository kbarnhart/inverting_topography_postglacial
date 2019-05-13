#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 15:26:47 2017

@author: barnhark
"""

# -*- coding: utf-8 -*-
"""
metric_calculator.py: calculates and saves topographic metrics for West Valley
modeling project.

Created on Fri Jul 15 08:23:33 2016

@author: gtucker
"""
import os
import numpy as np
from landlab.io import read_esri_ascii
from landlab.io.netcdf import read_netcdf

class GroupedDifferences(object):
    """Calculator for topographic metrics used in sensitivity analysis and
    model evaluation."""
    
    def __init__(self, 
                 modeled_dem_name, 
                 modern_dem_name, 
                 outlet_id, 
                 category_file=None, 
                 category_values=None, 
                 weight_file=None,
                 weight_values=None):
        """Initialize GroupedDifferences with names of postglacial and modern
        DEMs."""

        # save dem names
        self.modern_dem_name = modern_dem_name
        self.modeled_dem_name = modeled_dem_name
        
        # Read and remember the modern DEM
        (self.grid, self.z) = self.read_topography(modern_dem_name)
        self.grid.set_watershed_boundary_condition_outlet_id(outlet_id,
                                                             self.z, 
                                                             nodata_value=-9999)
        # Read and remember the modeled DEM 
        (self.mgrid, self.mz) = self.read_topography(modeled_dem_name)
        self.mgrid.set_watershed_boundary_condition_outlet_id(outlet_id,
                                                              self.mz, 
                                                              nodata_value=-9999)
        if self.mz.size != self.z.size:
            raise ValueError(('Size of provided DEMS is different.'))
                    
        if category_file and category_values:
            raise ValueError(('Provide either an array-like structure of catetory ',
                             'values or a filename, not both.'))
        if weight_file and weight_values:
            raise ValueError(('Provide either an array-like structure of weight ',
                             'values or a filename, not both.'))
        if category_file:
            if os.path.exists(category_file):
                catagory_values = np.loadtxt(category_file)
                if catagory_values.size != self.z.size:
                    raise ValueError(('Size of catagory array is different than the ',
                                      'provided DEM.'))
        if weight_file:
            if os.path.exists(weight_file):
                weight_values = np.loadtxt(weight_file)
                if weight_values.size != self.z.size:
                    raise ValueError(('Size of weight array is different than the ',
                                      'provided DEM.'))
        try:
            np.asarray(weight_values).size == self.z.size 
        except TypeError:
            weight_values = np.ones_like(self.z)
               
        self.category_values = category_values
        self.weight_values = weight_values
        self.cat_vals = np.sort(np.unique(self.category_values[self.grid.core_nodes]))
        self.metric = {}
        
    def read_topography(self, topo_file_name):
        """Read and return topography from file, as a Landlab grid and field.
        
        Along the way, process the topography to identify the watershed."""
        try:
            (grid, z) = read_esri_ascii(topo_file_name,
                                        name='topographic__elevation',
                                        halo=1)
        except:
            grid = read_netcdf(topo_file_name)
            z = grid.at_node['topographic__elevation']
        
        return (grid, z)
    
    def calculate_metrics(self):
        """Calculate and store each metric."""
        
        for cv in self.cat_vals:
            cat_inds = np.where(self.category_values == cv)[0]
            weighted_difference = (self.z[cat_inds]-self.mz[cat_inds])/self.weight_values[cat_inds]
            resid = np.sqrt(np.sum(np.square(weighted_difference))/(cat_inds.size))
            self.metric[str(cv)] = resid
        
        return self.metric

    def save_metrics(self, filename='grouped_differences.txt'):
        """Write metrics to file."""
        outfile = open(filename, 'w')
        outfile.write('Model topography file : ' + self.modeled_dem_name + '\n')
        outfile.write('Modern topography file : ' + self.modern_dem_name + '\n')
        
        sort_keys = sorted(list(self.metric.keys()))
        
        for m in sort_keys:
            outfile.write(m + ': ' + str(self.metric[m])  + '\n')
        outfile.close()
    
    def dakota_bundle(self):
        """Make dakota bundled metric output."""
        
        sort_keys = sorted(list(self.metric.keys()))
        bundle = [self.metric[key] for key in sort_keys]
        return bundle          

def main():
    
    # Initialize
    gd = GroupedDifferences(modeled_dem_name, 
                            modern_dem_name, 
                            outlet_id, 
                            category_file=None,
                            category_values=None)

    # Process
    gd.calculate_metrics()

    # Clean up
    gd.save_metrics(output_filename)

if __name__ == '__main__':
    main()
