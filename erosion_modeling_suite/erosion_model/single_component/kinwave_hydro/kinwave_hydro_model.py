# -*- coding: utf-8 -*-
"""
kinwave_hydro_model.py: calculates time evolution of water depth, given an
input DEM, runoff generation rate, storm duration, and other parameters.

Landlab components used: KinwaveImplicitOverlandFlow (which itself uses 
FlowAccumulator)

@author: gtucker
"""

from erosion_model.erosion_model import _ErosionModel
from landlab.components import KinwaveImplicitOverlandFlow


class KinwaveModel(_ErosionModel):
    """
    A DrainageAreaModel simply computes drainage area on a raster-grid DEM.
    """
    
    def __init__(self, input_file=None, params=None):
        """Initialize the LinearDiffusionModel."""
        
        # Call ErosionModel's init
        super(KinwaveModel, self).__init__(input_file=input_file,
                                           params=params)

        # Get input parameters: duration for a runoff (i.e., storm or snowmelt)
        # event, interval of time from the end of one event to the next, and
        # rate of runoff generation (in mm/hr) during an event.
        self.runoff_duration = self.params['runoff__duration']
        self.interstorm_duration = self.params['interstorm__duration']
        self.runoff_rate = self.params['runoff__rate'] / 3.6e6
        
        # A "cycle" is a runoff event plus the interlude before the next event.
        self.cycle_duration = self.runoff_duration + self.interstorm_duration
        
        # Runoff continues for some time after runoff generation ceases. Here
        # we set the amount of time we actively calculate flow. After this
        # time period, we ignore flow for the remainder of the cycle. But if
        # the user-specified hydrograph duration is longer than our cycle, we
        # truncate at the cycle duration, because we'll start computing flow
        # again at the start of the next cycle.
        self.hydrograph_duration = min(self.cycle_duration,
            self.params['hydrograph__maximum_duration'])

        # This variable keeps track of how far we are in one cycle.
        self.current_time_in_cycle = 0.0

        # Instantiate a KinwaveImplicitOverlandFlow
        self.water_router = KinwaveImplicitOverlandFlow(self.grid)


    def run_one_step(self, dt):
        """
        Advance model for one time-step of duration dt.
        """
        self.water_router.run_one_step(dt)


    def run_for(self, dt, runtime):
        """
        Run model without interruption for a specified time period. This
        involves 
        """

        remaining_time = runtime
        remaining_time_in_cycle = (self.cycle_duration
                                   - self.current_time_in_cycle)
        
        while remaining_time > 0.0:

            # If we're in the hydrograph part of the cycle, run a hydrograph.
            if self.current_time_in_cycle < self.hydrograph_duration:
            
                # Set the component's runoff rate according to whether it's
                # still raining or not.
                if self.current_time_in_cycle < self.runoff_duration:
                    self.water_router.runoff_rate = self.runoff_rate
                else:
                    self.water_router.runoff_rate = 0.0

                # Calculate time-step size: we adjust downward from dt if
                # needed to make sure we don't exceed the runoff duration or
                # the runtime.
                delt = min(dt, remaining_time)
                delt = min(delt, self.runoff_duration)

                # Run some water flow
                self.run_one_step(delt)
                
                # Advance time
                remaining_time -= delt
                remaining_time_in_cycle -= delt
                self.current_time_in_cycle += delt

            # If we're in the "dry" part of a cycle, simply advance to either
            # the beginning of the next cycle or the end of the "runtime"
            # period, whichever is shorter.
            else:

                dry_time = min(remaining_time, remaining_time_in_cycle)
                remaining_time -= dry_time
                remaining_time_in_cycle -= dry_time
                if remaining_time_in_cycle > 0.0:
                    self.current_time_in_cycle += dry_time
                else:
                    self.current_time_in_cycle = 0.0  # start a new cycle
                    remaining_time_in_cycle = self.cycle_duration


def main():
    """Executes model."""
    import sys

    try:
        infile = sys.argv[1]
    except IndexError:
        print('Must include input file name on command line')
        sys.exit(1)

    kw = KinwaveModel(input_file=infile)
    kw.run()


if __name__ == '__main__':
    main()
