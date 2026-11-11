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


def test_nnls_beamform_plane_wave():
    np.random.seed(0)
    s_list = np.arange(-4, 4, 0.25)
    st = cleanbf.make_synth_stream(Nt = 800, Nx = 4, Ny = 4, sx = [1], sy = [-2], amp = [1], fc = [6])
    for smoothing in [0, 0.01]:
        result = cleanbf.nnls_beamform(st, sxList = s_list, syList = s_list, smoothing = smoothing, verbose = False)
        spec = result['cleanSpec'].sum(0)
        sx, sy = np.meshgrid(s_list, s_list, indexing = 'ij')
        near_true = (np.abs(sx - 1) <= 0.25) & (np.abs(sy + 2) <= 0.25) # within one grid cell of the true slowness
        assert spec[near_true].sum() / np.einsum('iik->', result['originalCrossSpec']).real > 0.9
        assert cleanbf.check_output_power(result)
    ## pure noise: little or no power should be assigned to waves
    st = cleanbf.make_synth_stream(Nt = 800, Nx = 4, Ny = 4, sx = [0], sy = [0], amp = [0], fc = [6], uncorrelatedNoiseAmp = 2)
    result = cleanbf.nnls_beamform(st, sxList = s_list, syList = s_list, verbose = False)
    assert result['cleanSpec'].sum() / np.einsum('iik->', result['originalCrossSpec']).real < 0.1
