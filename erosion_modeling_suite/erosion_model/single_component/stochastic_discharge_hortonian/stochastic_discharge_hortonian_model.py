# -*- coding: utf-8 -*-
"""
stochastic_discharge_hortonia_model.py: models discharge across a topographic
surface assuming (1) stochastic Poisson storm arrivals, (2) single-direction
flow routing, and (3) Hortonian infiltration model.

This is a very simple hydrologic model that inherits from the ErosionModel 
class. It calculates drainage area using the standard "D8" approach (assuming
the input grid is a raster; "DN" if not), then modifies it by running a
lake-filling component. It then iterates through a sequence of storm and
interstorm periods. Storm depth is drawn at random from a gamma distribution,
and storm duration from an exponential distribution; storm intensity is then
depth divided by duration. Given a storm precipitation intensity $P$, the
runoff production rate $R$ [L/T] is calculated using:

$R = P - I (1 - \exp ( -P / I ))$

where $I$ is the soil infiltration capacity. At the sub-grid scale, soil
infiltration capacity is assumed to have an exponential distribution of which
$I$ is the mean. Hence, there are always some spots within any given grid cell
that will generate runoff. This approach yields a smooth transition from 
near-zero runoff (when $I>>P$) to $R \approx P$ (when $P>>I$), without a 
"hard threshold."

Landlab components used: FlowRouter, DepressionFinderAndRouter,
PrecipitationDistribution

@author: gtucker
"""

from erosion_model.erosion_model import _ErosionModel
from landlab.components import (FlowRouter, DepressionFinderAndRouter,
                                PrecipitationDistribution)
import numpy as np


class StochasticDischargeHortonianModel(_ErosionModel):
    """
    A StochasticDischargeHortonianModel generates a random sequency of
    runoff events across a topographic surface, calculating the resulting
    water discharge at each node.
    """
    
    def __init__(self, input_file=None, params=None):
        """Initialize the StochasticDischargeHortonianModel."""

        # Call ErosionModel's init
        super(StochasticDischargeHortonianModel,
              self).__init__(input_file=input_file, params=params)

        # Instantiate components
        self.flow_router = FlowRouter(self.grid, **self.params)

        self.lake_filler = DepressionFinderAndRouter(self.grid, **self.params)

        self.rain_generator = \
            PrecipitationDistribution(delta_t=self.params['dt'],
                                      total_time=self.params['run_duration'],
                                      **self.params)

        # Add a field for discharge
        if 'surface_water__discharge' not in self.grid.at_node:
            self.grid.add_zeros('node', 'surface_water__discharge')
        self.discharge = self.grid.at_node['surface_water__discharge']                                    

        # Get the infiltration-capacity parameter
        self.infilt = self.params['infiltration_capacity']

        # Run flow routing and lake filler (only once, because we are not
        # not changing topography)
        self.flow_router.run_one_step()
        self.lake_filler.map_depressions()


    def reset_random_seed(self):
        """Re-set the random number generation sequence."""
        try:
            seed = self.params['random_seed']
        except KeyError:
            seed = 0
        self.rain_generator.seed_generator(seedval=seed)


    def run_one_step(self, dt):
        """
        Advance model for one time-step of duration dt.
        """

        # Calculate discharge field
        area = self.grid.at_node['drainage_area']
        if self.infilt > 0.0:
            runoff = self.rain_rate - (self.infilt * 
                                       (1.0 - 
                                        np.exp(-self.rain_rate / self.infilt)))
        else:
            runoff = self.rain_rate
        self.discharge[:] = runoff * area


    def run_for(self, dt, runtime):
        """
        Run model without interruption for a specified time period.
        """
        self.rain_generator.delta_t = dt
        self.rain_generator.run_time = runtime
        for (tr, p) in self.rain_generator.yield_storm_interstorm_duration_intensity(): 
            self.rain_rate = p
            self.run_one_step(tr)


    def write_storm_sequence_to_file(self, filename=None):
        """
        Write event duration and intensity to a formatted text file.
        """

        # Re-seed the random number generator, so we get the same sequence.
        self.reset_random_seed()
        
        # Generate a set of event parameters. This is needed because the
        # PrecipitationDistribution component automatically generates a
        # parameter set on initialization. Therefore, to get to the same
        # starting point that we used in the sequence-through-time, we need
        # to regenerate these.
        self.rain_generator.get_precipitation_event_duration()
        self.rain_generator.get_interstorm_event_duration()
        self.rain_generator.get_storm_depth()
        self.rain_generator.get_storm_intensity()
        
        # Open a file for writing
        if filename is None:
            filename = 'event_sequence.txt'
        stormfile = open(filename, 'w')

        # Set the generator's time step and run time to the full duration of
        # the run. This ensures that we get a sequence that is as long as the
        # model run, and does not split events by time step (unlike the actual
        # model run)
        self.rain_generator.delta_t = self.params['run_duration']
        self.rain_generator.run_time = self.params['run_duration']
        tt = 0.0
        for (tr, p) in self.rain_generator.yield_storm_interstorm_duration_intensity():        
            runoff = p - self.infilt * (1.0 - np.exp(-p / self.infilt))
            stormfile.write(str(tt) + ',' + str(p) + ',' + str(runoff) + '\n')
            tt += tr
        stormfile.write(str(tt) + ',' + str(p) + ',' + str(runoff) + '\n')

        # Close the file
        stormfile.close()


def main():
    """Executes model."""
    import sys

    try:
        infile = sys.argv[1]
    except IndexError:
        print('Must include input file name on command line')
        sys.exit(1)

    dm = StochasticDischargeHortonianModel(input_file=infile)
    dm.run()


if __name__ == '__main__':
    main()
