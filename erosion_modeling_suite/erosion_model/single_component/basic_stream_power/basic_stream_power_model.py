# -*- coding: utf-8 -*-
"""
basic_stream_power_model.py: calculates water erosion using the most basic form
of the unit stream power model.

This is a very simple hydrologic model that inherits from the ErosionModel 
class. It calculates drainage area using the standard "D8" approach (assuming
the input grid is a raster), then modifies it by running a lake-filling
component. Erosion at each core node is that calculated using the basic stream
power formula:

$E = K A^{1/2} S$

where $E$ is vertical incision rate, $A$ is drainage area, $S$ is gradient in 
the steepest down-slope direction, and $K$ is a parameter with dimensions of
$T^{-1}$.

Landlab components used: FlowRouter, DepressionFinderAndRouter,
                         FastscapeStreamPower

@author: gtucker
"""

from erosion_model.erosion_model import _ErosionModel
from landlab.components import (FlowRouter, DepressionFinderAndRouter,
                                FastscapeEroder)
import numpy as np


class BasicStreamPowerErosionModel(_ErosionModel):
    """
    A BasicStreamPowerErosionModel computes erosion using the simplest form of
    the unit stream power model.
    """
    
    def __init__(self, input_file=None, params=None):
        """Initialize the BasicStreamPowerErosionModel."""
        
        # Call ErosionModel's init
        super(BasicStreamPowerErosionModel, self).__init__(input_file=input_file,
                                                params=params)

        # Instantiate a FlowRouter and DepressionFinderAndRouter components
        self.flow_router = FlowRouter(self.grid, **self.params)
        self.lake_filler = DepressionFinderAndRouter(self.grid, **self.params)

        # Instantiate a FastscapeEroder component
        self.eroder = FastscapeEroder(self.grid, K_sp=self.params['K_sp'],
                                                 m_sp=self.params['m_sp'],
                                                 n_sp=self.params['n_sp'])

    def run_one_step(self, dt):
        """
        Advance model for one time-step of duration dt.
        """

        # Route flow
        self.flow_router.run_one_step()
        self.lake_filler.map_depressions()

        # Get IDs of flooded nodes, if any
        flooded = np.where(self.lake_filler.flood_status==3)[0]

        # Do some erosion (but not on the flooded nodes)
        self.eroder.run_one_step(dt, flooded_nodes=flooded)


def main():
    """Executes model."""
    import sys

    try:
        infile = sys.argv[1]
    except IndexError:
        print('Must include input file name on command line')
        sys.exit(1)

    dm = BasicStreamPowerErosionModel(input_file=infile)
    dm.run()


if __name__ == '__main__':
    main()
