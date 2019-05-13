# -*- coding: utf-8 -*-
"""
effective_drainage_area_model.py: calculates drainage area for an input DEM,
as well as effective area, defined as: Aeff = A exp( -bS/A), where b is a
parameter.

This is a very simple hydrologic model that inherits from the ErosionModel 
class. It calculates drainage area using the standard "D8" approach (assuming
the input grid is a raster; "DN" if not), then modifies it by running a
lake-filling component. Finally, it computes effective drainage area,
defined as

$A_{eff} = A \exp ( -b S / A )$

where the parameter $b$ is defined as

$b = T \Delta x / R_m$

with $T$ being transmissivity, $\Delta x$ face width, and $R_m$ mean recharge
rate.

Landlab components used: FlowRouter, DepressionFinderAndRouter

@author: gtucker
"""

from erosion_model.erosion_model import _ErosionModel
from landlab.components import FlowRouter, DepressionFinderAndRouter
import numpy as np


class EffectiveDrainageAreaModel(_ErosionModel):
    """
    A DrainageAreaModel simply computes drainage area on a raster-grid DEM.
    """
    
    def __init__(self, input_file=None, params=None):
        """Initialize the LinearDiffusionModel."""
        
        # Call ErosionModel's init
        super(EffectiveDrainageAreaModel, self).__init__(input_file=input_file,
                                                         params=params)

        # Instantiate a FlowRouter and DepressionFinderAndRouter components
        self.flow_router = FlowRouter(self.grid, **self.params)
        self.lake_filler = DepressionFinderAndRouter(self.grid, **self.params)
        
        # Add a field for effective drainage area
        self.eff_area = self.grid.add_zeros('node', 'effective_drainage_area')
        
        # Get the effective-area parameter
        self.sat_param = self.params['saturation_area_scale']


    def run_one_step(self, dt):
        """
        Advance model for one time-step of duration dt.
        """
        
        # Run flow routing and lake filler
        self.flow_router.run_one_step()
        self.lake_filler.map_depressions()

        # Calculate effective area
        area = self.grid.at_node['drainage_area']
        slope = self.grid.at_node['topographic__steepest_slope']
        cores = self.grid.core_nodes
        self.eff_area[cores] = (area[cores] * 
                                np.exp(-self.sat_param * slope[cores] /
                                       area[cores]))

        # Lower outlet
        self.z[self.outlet_node] -= self.outlet_lowering_rate * dt
        print(('lowering outlet to ', self.z[self.outlet_node]))


def main():
    """Executes model."""
    import sys

    try:
        infile = sys.argv[1]
    except IndexError:
        print('Must include input file name on command line')
        sys.exit(1)

    dm = EffectiveDrainageAreaModel(input_file=infile)
    dm.run()


if __name__ == '__main__':
    main()
