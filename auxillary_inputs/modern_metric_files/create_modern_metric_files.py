#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 13:29:26 2017

@author: barnhark
"""

# create modern metric files 

from metric_calculator import  MetricCalculator

# sew modern metrics
modern_dem_name = '../dems/sew/modern/dem24fil_ext.txt'
chi_mask_dem_name = '../chi_mask/sew/chi_mask.txt'
outlet_id = 178579

mc0 = MetricCalculator(modern_dem_name,
                       outlet_id,
                       chi_mask_dem_name=chi_mask_dem_name)
mc0.calculate_metrics()
mc0.save_metrics(modern_dem_name, filename='dem24fil_ext.metrics.txt')


# gully modern metrics
modern_dem_name = '../dems/gully/modern/gdem3r1f.txt'
outlet_id = 397031

mc0 = MetricCalculator(modern_dem_name,
                       outlet_id)
mc0.calculate_metrics()
mc0.save_metrics(modern_dem_name, filename='gdem3r1f.metrics.txt')