# -*- coding: utf-8 -*-
"""
stochastic_discharge_vsa_model.py: models discharge across a topographic
surface assuming (1) stochastic Poisson storm arrivals, (2) single-direction
flow routing, and (3) a variable-source-area (VSA) runoff-generation model.

This is a very simple hydrologic model that inherits from the ErosionModel 
class. It calculates drainage area using the standard "D8" approach (assuming
the input grid is a raster; "DN" if not), then modifies it by running a
lake-filling component. It then iterates through a sequence of storm and
interstorm periods. Storm depth is drawn at random from a gamma distribution,
and storm duration from an exponential distribution; storm intensity is then
depth divided by duration. Given a storm precipitation intensity $P$, the
discharge $Q$ [L$^3$/T] is calculated using:

$Q = PA - T\lambda S [1 - \exp (-PA/T\lambda S) ]$

where $T$ is the soil transmissivity and $\lambda$ is cell width.

Landlab components used: FlowRouter, DepressionFinderAndRouter,
PrecipitationDistribution

@author: gtucker
"""

from erosion_model.erosion_model import _ErosionModel
from landlab.components import (FlowRouter, DepressionFinderAndRouter,
                                PrecipitationDistribution)
import numpy as np


class StochasticDischargeVSAModel(_ErosionModel):
    """
    A StochasticDischargeHortonianModel generates a random sequency of
    runoff events across a topographic surface, calculating the resulting
    water discharge at each node.
    """
    
    def __init__(self, input_file=None, params=None):
        """Initialize the StochasticDischargeHortonianModel."""

        # Call ErosionModel's init
        super(StochasticDischargeVSAModel,
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

        # Add a field for subsurface discharge                                 
        if 'subsurface_water__discharge' not in self.grid.at_node:
            self.grid.add_zeros('node', 'subsurface_water__discharge')
        self.qss = self.grid.at_node['subsurface_water__discharge']  

        # Get the transmissivity parameter
        self.trans = self.params['soil_transmissivity']
        assert (self.trans > 0.0), 'Transmissivity must be > 0'
        self.tlam = self.trans * self.grid._dx  # assumes raster

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

        # Get hold of references to area and slope
        area = self.grid.at_node['drainage_area']
        slope = self.grid.at_node['topographic__steepest_slope']

        # Here's the total (surface + subsurface) discharge
        pa = self.rain_rate * area

        # Transmissivity x lambda x slope = subsurface discharge capacity
        tls = self.tlam * slope[np.where(slope > 0.0)[0]]

        # Subsurface discharge: zero where slope is flat
        self.qss[np.where(slope <= 0.0)[0]] = 0.0
        self.qss[np.where(slope > 0.0)[0]] = tls * (1.0 - np.exp(-pa[np.where(slope > 0.0)[0]] / tls))

        # Surface discharge = total minus subsurface
        self.discharge[:] = pa - self.qss


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
            stormfile.write(str(tt) + ',' + str(p) + '\n')
            tt += tr
        stormfile.write(str(tt) + ',' + str(p) + '\n')

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

    dm = StochasticDischargeVSAModel(input_file=infile)
    dm.run()


if __name__ == '__main__':
    main()
