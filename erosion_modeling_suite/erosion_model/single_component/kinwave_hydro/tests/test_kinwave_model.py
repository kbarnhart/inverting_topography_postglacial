#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
test_kinwave_model.py: run unit tests for the KinwaveModel.

Created on Mon Apr 10 13:47:13 2017

@author: gtucker
"""

from erosion_model import KinwaveModel


def test_timer():
    """Test the timing: cycle length, hydrograph duration, time step, etc."""

    params = {}

    params['runoff__duration'] = 2.0
    params['interstorm__duration'] = 3.0 
    params['runoff__rate'] = 10.0
    params['hydrograph__maximum_duration'] = 10.0

    kw = KinwaveModel(params=params)
    assert (kw.hydrograph_duration == 5.0), 'error in hydrograph duration'

    kw.run_for(dt=1.0, runtime=4.0)
    assert (kw.current_time_in_cycle == 4.0), 'error in timing'
    assert (kw.water_router.runoff_rate == 0.0), 'error in runoff rate'

    kw.run_for(dt=1.0, runtime=4.0)
    assert (kw.current_time_in_cycle == 3.0), 'error in timing'
    assert (kw.water_router.runoff_rate == 0.0), 'error in runoff rate'

    kw.run_for(dt=1.0, runtime=4.0)
    assert (kw.current_time_in_cycle == 2.0), 'error in timing'
    assert (kw.water_router.runoff_rate == 10.0/3.6e6), 'error in runoff rate'

    kw.run_for(dt=1.0, runtime=4.0)
    assert (kw.current_time_in_cycle == 1.0), 'error in timing'
    assert (kw.water_router.runoff_rate == 10.0/3.6e6), 'error in runoff rate'

if __name__ == '__main__':
    test_timer()    
