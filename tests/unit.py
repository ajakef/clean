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


def test_cross_spectrum_batched_matches_loop():
    ## the default (batched) path must agree with the per-window path used when a criterion_function is given
    st = cleanbf.make_synth_stream(Nt = 1200, Nx = 3, Ny = 2, sx = [1], sy = [-2], amp = [1], fc = [6], uncorrelatedNoiseAmp = 0.3)
    all_ok = lambda s: np.ones((len(s), len(s)))
    for kw in [{}, {'prewhiten':True}, {'raw':True}, {'freq_bin_width':3}]:
        a = cleanbf.calc_cross_spectrum(st, **kw)
        b = cleanbf.calc_cross_spectrum(st, criterion_function = all_ok, **kw)
        for x, y in zip(a[:3], b[:3]):
            assert np.allclose(x, y, rtol = 1e-10, atol = 1e-14 * np.abs(y).max())
        assert a[3:] == b[3:]


def test_clean_loop_reuses_weights():
    np.random.seed(2)
    s_list = np.arange(-3, 3, 0.5)
    st = cleanbf.make_synth_stream(Nt = 1000, Nx = 3, Ny = 2, sx = [1], sy = [-2], amp = [1], fc = [6], uncorrelatedNoiseAmp = 0.3)
    out = cleanbf.core.clean_loop(st, loop_step = 3, loop_width = 3, sxList = s_list, syList = s_list, verbose = False)
    single = cleanbf.clean(st.slice(st[0].stats.starttime + 3, st[0].stats.starttime + 6), sxList = s_list, syList = s_list, verbose = False)
    assert np.allclose(out['clean'][1], single['cleanSpec'])
    assert len(out['all_data']) == len(out['t'])
    assert np.allclose(out['original_sx'], 1) and np.allclose(out['original_sy'], -2)
