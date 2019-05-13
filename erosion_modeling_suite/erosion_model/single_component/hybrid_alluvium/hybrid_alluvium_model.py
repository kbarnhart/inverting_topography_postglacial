# -*- coding: utf-8 -*-
"""
hybrid_alluvium_model.py: calculates water erosion using the 
hybrid alluvium model.

Landlab components used: FlowRouter, DepressionFinderAndRouter,
                         HybridAlluvium

@author: Charlie Shobe
5 April 2017
"""
import numpy as np
from erosion_model.erosion_model import _ErosionModel
from landlab.components import (FlowRouter, DepressionFinderAndRouter,
                                Space)
                                
class HybridAlluviumModel(_ErosionModel):
    """
    A HybridAlluviumModel computes erosion of sediment and bedrock
    using dual mass conservation on the bed and in the water column. It
    applies exponential entrainment rules to account for bed cover.
    """
    
    def __init__(self, input_file=None, params=None):
        """Initialize the HybridAlluviumModel."""
      
        # Call ErosionModel's init
        super(HybridAlluviumModel, self).__init__(input_file=input_file,
                                                params=params)

        # Instantiate a FlowRouter and DepressionFinderAndRouter components
        self.flow_router = FlowRouter(self.grid, **self.params)
        self.lake_filler = DepressionFinderAndRouter(self.grid, **self.params)

        #make area_field and/or discharge_field depending on discharge_method
        if self.params['discharge_method'] is not None:
            if self.params['discharge_method'] == 'area_field':
                area_field = self.grid.at_node['drainage_area']
                discharge_field = None
            elif self.params['discharge_method'] == 'discharge_field':
                discharge_field = self.grid.at_node['surface_water__discharge']
                area_field = None
            else:
                raise(KeyError)
        else:
            area_field = None
            discharge_field = None

        # Instantiate a HybridAlluvium component
        self.eroder = Space(self.grid,
                            K_sed=self.params['K_sed'],
                            K_br=self.params['K_br'],
                            F_f=self.params['F_f'],
                            phi=self.params['phi'],
                            H_star=self.params['H_star'],
                            v_s=self.params['v_s'],
                            m_sp=self.params['m_sp'],
                            n_sp=self.params['n_sp'],
                            sp_crit_sed=self.params['sp_crit_sed'],
                            sp_crit_br=self.params['sp_crit_br'],
                            method=self.params['method'],
                            discharge_method=self.params['discharge_method'],
                            area_field=area_field,
                            discharge_field=discharge_field)

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

    ha = HybridAlluviumModel(input_file=infile)
    ha.run()


if __name__ == '__main__':
    main()
