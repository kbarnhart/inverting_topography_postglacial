# -*- coding: utf-8 -*-
"""
linear_diffusion_model.py is a linear diffusion model that inherits from the
ErosionModel class. Implements a single component: the LinearDiffuser.

Created on Fri Jan 15 07:50:09 2016

@author: gtucker
"""

from erosion_model.erosion_model import _ErosionModel
from landlab.components.diffusion import LinearDiffuser


class LinearDiffusionModel(_ErosionModel):
    """
    A LinearDiffusionModel is a single-component model that uses a finite-
    volume solution to the 2D linear diffusion equation to compute erosion.
    """
    
    def __init__(self, input_file=None, params=None):
        """Initialize the LinearDiffusionModel."""
        
        # Call ErosionModel's init
        super(LinearDiffusionModel, self).__init__(input_file=input_file,
                                                   params=params)

        # Instantiate a LinearDiffuser component
        self.diffuser = LinearDiffuser(self.grid, **self.params)


    def run_one_step(self, dt):
        """
        Advance model for one time-step of duration dt.
        """
        self.diffuser.run_one_step(dt)


def main():
    """Executes model."""
    import sys

    try:
        infile = sys.argv[1]
    except IndexError:
        print('Must include input file name on command line')
        sys.exit(1)
        
    ldm = LinearDiffusionModel(input_file=infile)
    ldm.run()


if __name__ == '__main__':
    main()
