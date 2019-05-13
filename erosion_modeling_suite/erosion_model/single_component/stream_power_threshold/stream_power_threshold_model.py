# -*- coding: utf-8 -*-
"""
stream_power_threshold_model.py: calculates water erosion using the 
unit stream power model with a "smoothed" threshold.

This is a very simple hydrologic model that inherits from the ErosionModel 
class. It calculates drainage area using the standard "D8" approach (assuming
the input grid is a raster), then modifies it by running a lake-filling
component. Erosion at each core node is that calculated using the stream
power formula with a smooth threshold:

$E = \omega - \omega_c [1 - \exp ( -\omega / \omega_c )]$

$\omega = K A^{1/2} S$

where $E$ is vertical incision rate, $A$ is drainage area, $S$ is gradient in 
the steepest down-slope direction,  $K$ is a parameter with dimensions of
$T^{-1}$, and $\omega_c$ is a threshold.

Landlab components used: FlowRouter, DepressionFinderAndRouter,
                         SmoothThresholdStreamPowerEroder

@author: gtucker
"""

from erosion_model.erosion_model import _ErosionModel
from landlab.components import (FlowRouter, DepressionFinderAndRouter,
                                StreamPowerSmoothThresholdEroder)
import numpy as np


class StreamPowerThresholdModel(_ErosionModel):
    """
    A StreamPowerThresholdModel computes erosion using a form of the unit
    stream power model that represents a threshold using an exponential term.
    """
    
    def __init__(self, input_file=None, params=None):
        """Initialize the StreamPowerThresholdModel."""
        
        # Call ErosionModel's init
        super(StreamPowerThresholdModel, self).__init__(input_file=input_file,
                                                params=params)

        # Instantiate a FlowRouter and DepressionFinderAndRouter components
        self.flow_router = FlowRouter(self.grid, **self.params)
        self.lake_filler = DepressionFinderAndRouter(self.grid, **self.params)

        # Instantiate a FastscapeEroder component
        self.eroder = StreamPowerSmoothThresholdEroder(
            self.grid,
            K_sp=self.params['K_sp'],
            threshold_sp=self.params['threshold_sp'])


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

    spt = StreamPowerThresholdModel(input_file=infile)
    spt.run()


if __name__ == '__main__':
    main()
