# -*- coding: utf-8 -*-
"""
depth_dep_linear_diffusion_model.py is a linear diffusion model that operates
on a layer of regolith, with the transport rate shrinking smoothly to zero
as the regolith thickness vanishes. The model inherits from the
ErosionModel class. Implements the DepthDependentDiffuser and .

Created on Fri Jan 15 07:50:09 2016

@author: gtucker
"""

from erosion_model.erosion_model import _ErosionModel
from landlab.components import DepthDependentDiffuser, ExponentialWeatherer


class DepthDependentDiffusionModel(_ErosionModel):
    """
    A DepthDependentDiffusionModel is a single-component model that uses 
    a finite-volume solution to the 2D linear diffusion equation to compute
    erosion.
    """

    def __init__(self, input_file=None, params=None):
        """Initialize the LinearDiffusionModel."""

        # Call ErosionModel's init
        super(DepthDependentDiffusionModel,
              self).__init__(input_file=input_file, params=params)

        # Create soil thickness (a.k.a. depth) field
        if 'soil__depth' in self.grid.at_node:
            soil_thickness = self.grid.at_node['soil__depth']
        else:
            soil_thickness = self.grid.add_zeros('node', 'soil__depth')
        
        # Create bedrock elevation field
        if 'bedrock__elevation' in self.grid.at_node:
            bedrock_elev = self.grid.at_node['bedrock__elevation']
        else:
            bedrock_elev = self.grid.add_zeros('node', 'bedrock__elevation')

        # Set soil thickness and bedrock elevation
        try:        
            initial_soil_thickness = self.params['initial_soil_thickness']
        except KeyError:
            initial_soil_thickness = 1.0  # default value
        soil_thickness[:] = initial_soil_thickness
        bedrock_elev[:] = self.z - initial_soil_thickness

        # Instantiate a LinearDiffuser component
        self.diffuser = DepthDependentDiffuser(self.grid, **self.params)
        self.weatherer = ExponentialWeatherer(self.grid, **self.params)


    def run_one_step(self, dt):
        """
        Advance model for one time-step of duration dt.
        """
        
        # Calculate regolith-production rate
        self.weatherer.calc_soil_prod_rate()
        
        # Generate and move soil around
        self.diffuser.run_one_step(dt)


def main():
    """Executes model."""
    import sys

    try:
        infile = sys.argv[1]
    except IndexError:
        print('Must include input file name on command line')
        sys.exit(1)
        
    ldm = DepthDependentDiffusionModel(input_file=infile)
    ldm.run()


if __name__ == '__main__':
    main()
