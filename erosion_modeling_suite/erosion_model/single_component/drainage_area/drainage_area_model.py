# -*- coding: utf-8 -*-
"""
drainage_area_model.py: calculates drainage area for an input DEM.

This is a very simple hydrologic model that inherits from the ErosionModel 
class. It calculates drainage area using the standard "D8" approach (assuming
the input grid is a raster), then modifies it by running a lake-filling
component.

Landlab components used: FlowRouter, DepressionFinderAndRouter

Created on Fri Jan 15 07:50:09 2016

@author: gtucker
"""

from erosion_model.erosion_model import _ErosionModel
from landlab.components import FlowRouter, DepressionFinderAndRouter


class DrainageAreaModel(_ErosionModel):
    """
    A DrainageAreaModel simply computes drainage area on a raster-grid DEM.
    """
    
    def __init__(self, input_file=None, params=None):
        """Initialize the LinearDiffusionModel."""
        
        # Call ErosionModel's init
        super(DrainageAreaModel, self).__init__(input_file=input_file,
                                                params=params)

        # Instantiate a FlowRouter and DepressionFinderAndRouter components
        self.flow_router = FlowRouter(self.grid, **self.params)
        self.lake_filler = DepressionFinderAndRouter(self.grid, **self.params)


    def run_one_step(self, dt):
        """
        Advance model for one time-step of duration dt.
        """
        self.flow_router.run_one_step()
        
        # for debug
        from landlab.io import write_esri_ascii
        import numpy
        
        write_esri_ascii('test_dr_area_before_temp.txt', self.grid, names='drainage_area', clobber=True)        
        loga = self.grid.add_zeros('node', 'loga')
        loga[:] = numpy.log10(self.grid.at_node['drainage_area'] + 1)
        write_esri_ascii('test_logdr_area_before_temp.txt', self.grid, names='loga', clobber=True)        
        write_esri_ascii('test_sink_flag_before_temp.txt', self.grid, names='flow__sink_flag', clobber=True)        

        self.lake_filler.map_depressions()


def main():
    """Executes model."""
    import sys

    try:
        infile = sys.argv[1]
    except IndexError:
        print('Must include input file name on command line')
        sys.exit(1)

    dm = DrainageAreaModel(input_file=infile)
    dm.run()


if __name__ == '__main__':
    main()
