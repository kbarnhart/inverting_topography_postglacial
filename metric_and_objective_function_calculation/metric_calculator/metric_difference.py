#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculate metric differences.

Created on Mon Mar  6 09:49:36 2017

@author: katybarnhart
"""
from metric_calculator import MetricCalculator
import numpy as np
from yaml import load

class MetricDifference(object):
    """Calculator for difference between model and observation topographic 
    metrics used in sensitivity analysis and model evaluation."""
    
    def __init__(self, 
                 model_dem_name,
                 modern_dem_name=None,
                 outlet_id = None, 
                 modern_dem_metric_file=None,
                 modern_dem_chi_file=None,
                 chi_mask_dem_name=None,
                 output_file_name = 'metric_diff.txt'):
        """Initialize metric difference calculator."""
        if outlet_id is None:
            assert ValueError ('You must provide an outlet ID')
        self.modern_dem = modern_dem_name
        self.model = model_dem_name
        self.modern_metric = modern_dem_metric_file
        self.output_file_name = output_file_name
        
        if (self.modern_dem) is None and (self.modern_metric is None):
            assert ValueError('You must provide either a modern metric file'
                              ' or a modern dem file. This MetricDifference'
                              ' was initialized with neither')
        
        
        self.mc = MetricCalculator(model_dem_name,
                                   outlet_id,
                                   chi_mask_dem_name=chi_mask_dem_name)
        self.mc.calculate_metrics()


        
        if self.modern_dem is not None:
            self.mc0 = MetricCalculator(modern_dem_name,
                                        outlet_id,
                                        chi_mask_dem_name=chi_mask_dem_name)
            self.mc0.calculate_metrics()
        
        else:
            
            self.mc0 = MetricCalculator(modern_dem_name,
                                        outlet_id,
                                        chi_mask_dem_name=chi_mask_dem_name,
                                        from_file=self.modern_metric)
            
            with open(self.modern_metric, 'r') as f:
                metrics = load(f)
                
            self.modern_dem = metrics.pop('Topo file')

    def calc_metric_diffs(self):
        """Calculate metric value differences"""
        
        self.metric_diffs = {}
        
        for key in self.mc.metric.keys():
            self.metric_diffs[key] = self.mc.metric[key]-self.mc0.metric[key]
        
        chi_density_diff = self.mc.density_chi - self.mc0.density_chi
        
        self.metric_diffs['chi_density_sum_squares'] = np.sum(chi_density_diff**2)
        
    def save_metric_diffs(self):
        """ Save metrics value differences to a file."""
        
        outfile = open(self.output_file_name, 'w')
        outfile.write('Modern topography file : ' + self.modern_dem + '\n')
        outfile.write('Model topography file : ' + self.model + '\n')
        
        sort_keys = sorted(list(self.metric_diffs.keys()))
        for m in sort_keys:
            outfile.write(m + ' : ' + str(self.metric_diffs[m]) + '\n')
        outfile.close()

    def run(self):
        """Run and save metric differences."""
        
        self.calc_metric_diffs()
        self.save_metric_diffs()
    
    def dakota_bundle(self):
        """Make dakota bundled metric output."""
        
        sort_keys = sorted(list(self.metric_diffs.keys()))
        bundle = [self.metric_diffs[key] for key in sort_keys]
        return bundle            
