# -*- coding: utf-8 -*-
"""
metric_calculator.py: calculates and saves topographic metrics for West Valley
modeling project.

Created on Fri Jul 15 08:23:33 2016

@author: gtucker
"""

import numpy as np
from landlab.io import read_esri_ascii
from landlab.io.netcdf import read_netcdf
from landlab.components import (ChiFinder, FlowRouter,
                                DepressionFinderAndRouter)
from yaml import load

class MetricCalculator(object):
    """Calculator for topographic metrics used in sensitivity analysis and
    model evaluation."""
    
    def __init__(self, modern_dem_name, outlet_id, chi_mask_dem_name=None, from_file=None):
        """Initialize MetricCalculator with names of postglacial and modern
        DEMs."""


        if from_file is None:

            # Read and remember the modern DEM (whether data or model)
            (self.grid, self.z) = self.read_topography(modern_dem_name)
            #print self.grid.x_of_node
    
            self.grid.set_watershed_boundary_condition_outlet_id(outlet_id,
                                                                 self.z, nodata_value=-9999)
    
            # Instantiate and run a FlowRouter and lake filler, so we get
            # drainage area for cumulative-area statistic, and also fields for chi.
            fr = FlowRouter(self.grid)
            dfr = DepressionFinderAndRouter(self.grid)
            fr.route_flow()
            dfr.map_depressions()
    
            # Remember modern drainage area grid
            self.area = self.grid.at_node['drainage_area']
    
            # Instantiate a ChiFinder for chi-index
            self.chi_finder = ChiFinder(self.grid, min_drainage_area=10000.,
                                        reference_concavity=0.5)
            
            core_nodes = np.zeros(self.area.shape, dtype=bool)
            core_nodes[self.grid.core_nodes] = True
            # Read and remember the MASK, if provided
            if chi_mask_dem_name is None:
                 self.mask = (self.area>1e5)
                 self.till_mask = np.zeros(self.mask.shape, dtype=bool) 
                 self.till_mask[self.grid.core_nodes] = 1
            else:
                (self.mask_grid, zmask) = self.read_topography(chi_mask_dem_name)
                mask = (zmask>0)*1
                self.mask = (self.area>1e5)*(mask==1)
                
                mask_bool = (zmask>0)
                self.till_mask = np.zeros(self.mask.shape, dtype=bool) 
                self.till_mask[mask_bool*core_nodes] = 1
            # 
            
            # Create dictionary to contain metrics
            self.metric = {}
        
        else:
            with open(from_file, 'r') as f:
                metrics = load(f)
                self.modern_dem_name = metrics.pop('Topo file')

                self.metric = metrics
    
            fn_split = from_file.split('.')
            fn_split[-1] = 'chi'
            fn_split.append('txt')
            chi_filename = '.'.join(fn_split)
            self.density_chi = np.loadtxt(chi_filename)

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

    def calc_hyps_integral(self):
        """Calculate and return hypsometric integral of modern topo."""

        # Get just those elevation values that are within the watershed
        wshed_elevs = self.z[self.grid.core_nodes]

        # Get min and max
        min_elev = np.amin(wshed_elevs)
        max_elev = np.amax(wshed_elevs)

        # Calc and return hyps int
        return np.mean(wshed_elevs - min_elev) / (max_elev - min_elev)

    def calc_mean_and_var_gradient(self):
        """Calculate and return the mean and variance of gradients.
        """
        # Calculate the gradient at links
        grad = self.grid.calc_grad_at_link(self.z)
        
        # Find IDs of nodes that have four active links
        active_links_at_node = (self.grid.links_at_node * 
                                np.abs(self.grid.active_link_dirs_at_node))
        nodes = np.where(np.amin(active_links_at_node, axis=1) > 0)
        
        # Get the x and y components of slope at each of these.
        # We're assuming we're dealing with raster grids, in which columns
        # 0 and 2 of the links_at_node array contain east-west links, and
        # columns 1 and 3 contain north-south links. We take the average in
        # each direction.
        grad_x = 0.5 * (grad[self.grid.links_at_node[nodes, 0]] +
                        grad[self.grid.links_at_node[nodes, 2]])
        grad_y = 0.5 * (grad[self.grid.links_at_node[nodes, 1]] +
                        grad[self.grid.links_at_node[nodes, 3]])
        
        # Combine them to get the slope magnitude
        slope = np.sqrt(grad_x**2 + grad_y**2)
        
        # Return the mean and variance
        return np.mean(slope), np.var(slope)
        
    def calc_mean_and_var_gradient_chi_area(self):
        """Calculate and return the mean and variance of gradients in chi area.
        """
        # Calculate the gradient at links
        grad = self.grid.calc_grad_at_link(self.z)
        
        # Find IDs of nodes that have four active links
        active_links_at_node = (self.grid.links_at_node * 
                                np.abs(self.grid.active_link_dirs_at_node))
        
        nodes = np.where((np.amin(active_links_at_node, axis=1) > 0) & self.till_mask)
        
        # Get the x and y components of slope at each of these.
        # We're assuming we're dealing with raster grids, in which columns
        # 0 and 2 of the links_at_node array contain east-west links, and
        # columns 1 and 3 contain north-south links. We take the average in
        # each direction.
        grad_x = 0.5 * (grad[self.grid.links_at_node[nodes, 0]] +
                        grad[self.grid.links_at_node[nodes, 2]])
        grad_y = 0.5 * (grad[self.grid.links_at_node[nodes, 1]] +
                        grad[self.grid.links_at_node[nodes, 3]])
        
        # Combine them to get the slope magnitude
        slope = np.sqrt(grad_x**2 + grad_y**2)
        
        # Return the mean and variance
        return np.mean(slope), np.var(slope)
        
    def calc_chi_index(self):
        """Calculate and return coefficients of chi plot."""
        self.chi_finder.calculate_chi()
        return self.chi_finder.best_fit_chi_elevation_gradient_and_intercept()

    def calculate_channel_chi_distribution(self):
        """Calculate and return Chi distribution."""
        # set the x and y bin edges for consistency. 
        xedges = np.arange(1160, 2000, 5)
        yedges = np.arange(0, 6.2, 0.2)
        
        #calculate on DEM
        self.density_chi, xedges, yedges = np.histogram2d(self.z[self.mask], 
                                                     self.grid.at_node['channel__chi_index'][self.mask], 
                                                     bins=(xedges,yedges), normed=True)
        return self.density_chi
    
    def calc_number_source_nodes(self, factor=1.0):
        """Calculate and return number of nodes below a given threshold.
        
        Parameters
            factor, float, default
            
        Threshold given by cell area x factor
        """
        single_cell_area = factor*(self.grid.dx**2.0)
        
        source_nodes = np.sum(self.area[self.grid.core_nodes] <= single_cell_area)
        
        return source_nodes                                                
    
    def calculate_metrics(self):
        """Calculate and store each metric."""
        
        # characterize the lowest portions of the area distribution
        self.metric['one_cell_nodes'] = self.calc_number_source_nodes(factor=1.0)
        self.metric['two_cell_nodes'] = self.calc_number_source_nodes(factor=2.0)
        self.metric['three_cell_nodes'] = self.calc_number_source_nodes(factor=3.0)
        self.metric['four_cell_nodes'] = self.calc_number_source_nodes(factor=4.0)
    
        # uppermost portion of the distribution
        self.metric['cumarea95'] = np.percentile(self.area[self.grid.core_nodes], 95)
        self.metric['cumarea96'] = np.percentile(self.area[self.grid.core_nodes], 96)
        self.metric['cumarea97'] = np.percentile(self.area[self.grid.core_nodes], 97)
        self.metric['cumarea98'] = np.percentile(self.area[self.grid.core_nodes], 98)
        self.metric['cumarea99'] = np.percentile(self.area[self.grid.core_nodes], 99)
        
        # hypsometric integral
        self.metric['hypsometric_integral'] = self.calc_hyps_integral()
        
        # mean and variance of elevation 
        self.metric['mean_elevation'] = np.mean(self.z[self.grid.core_nodes])
        self.metric['var_elevation'] = np.var(self.z[self.grid.core_nodes])
        
        self.metric['mean_elevation_chi_area'] = np.mean(self.z[self.till_mask])
        self.metric['var_elevation_chi_area'] = np.var(self.z[self.till_mask])
        
        # mean and variance of slope
        (mean_slope, var_slope) = self.calc_mean_and_var_gradient()
        self.metric['mean_gradient'] = mean_slope
        self.metric['var_gradient'] = var_slope
        
        (mean_slope_chi, var_slope_chi) = self.calc_mean_and_var_gradient_chi_area()
        self.metric['mean_gradient_chi_area'] = mean_slope_chi
        self.metric['var_gradient_chi_area'] = var_slope_chi
        
        # distribution of elevations,
        # these percentiles chosen to characterize the shape of the modern distribution
        self.metric['elev02'] = np.percentile(self.z[self.grid.core_nodes], 2)
        self.metric['elev08'] = np.percentile(self.z[self.grid.core_nodes], 8)
        self.metric['elev23'] = np.percentile(self.z[self.grid.core_nodes], 23)
        self.metric['elev30'] = np.percentile(self.z[self.grid.core_nodes], 30)
        self.metric['elev36'] = np.percentile(self.z[self.grid.core_nodes], 36)
        self.metric['elev50'] = np.percentile(self.z[self.grid.core_nodes], 50)
        self.metric['elev75'] = np.percentile(self.z[self.grid.core_nodes], 75)
        self.metric['elev85'] = np.percentile(self.z[self.grid.core_nodes], 85)
        self.metric['elev90'] = np.percentile(self.z[self.grid.core_nodes], 90)
        self.metric['elev96'] = np.percentile(self.z[self.grid.core_nodes], 96)
        self.metric['elev100'] = np.percentile(self.z[self.grid.core_nodes], 100)
        
        # Chi-related metrics
        (chi_grad, chi_intercept) = self.calc_chi_index()
        self.metric['chi_gradient'] = chi_grad
        self.metric['chi_intercept'] = chi_intercept
        self.density_chi = self.calculate_channel_chi_distribution()

        return self.metric


    def save_metrics(self, topofile, filename='metrics.txt'):
        """Write metrics to file."""
        outfile = open(filename, 'w')
        outfile.write('Topo file : ' + topofile + '\n')
        
        sort_keys = sorted(list(self.metric.keys()))
        
        for m in sort_keys:
            outfile.write(m + ': ' + str(self.metric[m])  + '\n')
        outfile.close()
        
        if (filename=='metrics.txt'):
            chi_filename = 'metrics.chi.txt'
        else:
            fn_split = filename.split('.')
            fn_split[-1] = 'chi'
            fn_split.append('txt')
            chi_filename = '.'.join(fn_split)
    
        np.savetxt(chi_filename, self.density_chi,fmt='%f')


def main():
    
    # Initialize
    mc = MetricCalculator(modern_dem_name, outlet_id, chi_mask_dem_name=None)

    # Process
    mc.calculate_metrics()

    # Clean up
    mc.save_metrics(modern_dem_name)
    print(mc.metric)


if __name__ == '__main__':
    main()
