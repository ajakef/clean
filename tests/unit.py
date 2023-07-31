import scipy
import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sqrt, sin, cos, exp
import obspy
import importlib
import cleanbf
import subprocess
import os
data_dir = os.path.join(os.path.dirname(cleanbf.__file__), 'data')

def test_data_available():
    print(data_dir)
    assert os.path.exists(os.path.join(data_dir, 'aftershock_beginning.mseed'))
    assert os.path.exists(os.path.join(data_dir, 'noise.mseed'))
    assert os.path.exists(os.path.join(data_dir, 'XP_PARK_inventory.xml'))
    assert not os.path.exists(os.path.join(data_dir, 'non-existent-file'))

def test_add_inv_coords():
    eq_stream = obspy.read(os.path.join(data_dir, 'aftershock_beginning.mseed'))
    inv = obspy.read_inventory(os.path.join(data_dir, 'XP_PARK_inventory.xml')) # includes coordinates
    cleanbf.add_inv_coords(eq_stream, inv) # store the coordinates in the stream

def test_calc_station_pair_distance():
    eq_stream = obspy.read(os.path.join(data_dir, 'aftershock_beginning.mseed'))
    inv = obspy.read_inventory(os.path.join(data_dir, 'XP_PARK_inventory.xml')) # includes coordinates
    cleanbf.add_inv_coords(eq_stream, inv) # store the coordinates in the stream

    distances = cleanbf.calc_station_pair_distance(eq_stream)


# this will fail if calc_station_pair_distance fails
def test_calc_station_pair_distance():
    eq_stream = obspy.read(os.path.join(data_dir, 'aftershock_beginning.mseed'))
    inv = obspy.read_inventory(os.path.join(data_dir, 'XP_PARK_inventory.xml')) # includes coordinates
    cleanbf.add_inv_coords(eq_stream, inv) # store the coordinates in the stream

    cleanbf.plot_distances(eq_stream, 0) # plot the array (or sub-array) geometry

