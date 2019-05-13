# -*- coding: utf-8 -*-
"""
stream_power_var_threshold_model.py: calculates water erosion using the 
unit stream power model with a "smoothed" threshold. In this case, the
threshold varies with depth beneath a defined "initial" surface.

This is a very simple hydrologic model that inherits from the ErosionModel 
class. It calculates drainage area using the standard "D8" approach (assuming
the input grid is a raster), then modifies it by running a lake-filling
component. Erosion at each core node is that calculated using the stream
power formula with a smooth threshold:

$E = \omega - \omega_c [1 - \exp ( -\omega / \omega_c )]$

$\omega = K A^{1/2} S$

where $E$ is vertical incision rate, $A$ is drainage area, $S$ is gradient in 
the steepest down-slope direction,  $K$ is a parameter with dimensions of
$T^{-1}$, and $\omega_c$ is a threshold. The threshold is:

$\omega_c = \omega_{c0} + R_T * (\eta_0 (x,y) - \eta (x,y) )$

where $\eta_0$ is height of an initial reference surface at a given $(x,y)$
location, $\eta$ is the current height at that location, and $R_T$ is the rate
at which $\omega_c$ increases with incision depth.

Landlab components used: FlowRouter, DepressionFinderAndRouter,
                         SmoothThresholdStreamPowerEroder

@author: gtucker
"""

from erosion_model.erosion_model import _ErosionModel
from landlab.components import (FlowRouter, DepressionFinderAndRouter,
                                StreamPowerSmoothThresholdEroder)
import numpy as np


class StreamPowerVarThresholdModel(_ErosionModel):
    """
    A StreamPowerVarThresholdModel computes erosion using a form of the unit
    stream power model that represents a threshold using an exponential term.
    The threshold value itself depends on incision depth below an initial
    reference surface. This is meant to mimic coarsening of sediment in the
    channel with progressive incision, similar to the model of 
    Gran et al. (2013)
    """
    
    def __init__(self, input_file=None, params=None):
        """Initialize the StreamPowerVarThresholdModel."""

        # Call ErosionModel's init
        super(StreamPowerVarThresholdModel, self).__init__(input_file=input_file,
                                                params=params)

        # Instantiate a FlowRouter and DepressionFinderAndRouter components
        self.flow_router = FlowRouter(self.grid, **self.params)
        self.lake_filler = DepressionFinderAndRouter(self.grid, **self.params)
        
        # Create a field for the (initial) erosion threshold
        self.threshold = self.grid.add_zeros('node', 'erosion__threshold')
        self.threshold[:] = self.params['threshold_sp']
        
        # Instantiate a FastscapeEroder component
        self.eroder = StreamPowerSmoothThresholdEroder(
            self.grid,
            K_sp=self.params['K_sp'],
            threshold_sp=self.threshold)

        # Get the parameter for rate of threshold increase with erosion depth
        self.thresh_change_per_depth = self.params['thresh_change_per_depth']


    def run_one_step(self, dt):
        """
        Advance model for one time-step of duration dt.
        """

        # Route flow
        self.flow_router.run_one_step()
        self.lake_filler.map_depressions()

        # Get IDs of flooded nodes, if any
        flooded = np.where(self.lake_filler.flood_status==3)[0]

        # Set the erosion threshold.
        #
        # Note that a minus sign is used because cum ero depth is negative for
        # erosion, positive for deposition.
        # The second line handles the case where there is growth, in which case
        # we want the threshold to stay at its initial value rather than 
        # getting smaller.
        cum_ero = self.grid.at_node['cumulative_erosion__depth']
        cum_ero[:] = (self.z
                      - self.grid.at_node['initial_topographic__elevation'])
        self.threshold[:] = (self.params['threshold_sp']
                             - (self.thresh_change_per_depth
                                * cum_ero))
        self.threshold[self.threshold < self.params['threshold_sp']] = \
            self.params['threshold_sp']

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
