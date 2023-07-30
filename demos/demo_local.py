import sys
sys.path.append('/home/jake/Work/Aftershocks/lib/clean')
import numpy as np
import matplotlib.pyplot as plt
import obspy, scipy, importlib
import cleanbf
try:
    importlib.reload(cleanbf)
    print('reloaded')
except:
    pass

def approx_equal(x, y, p = 0.01):
    return np.abs((x-y)/x) < p

def approx_equal_all(x, p = 0.01):
    return all([approx_equal(x[0], y, p) for y in x])
#steeringVectors = clean.make_steering_vectors_local(stream, src_xyz, freqs = freqs)
#result = clean.clean(stream, verbose = True, phi=0.1, win_length_sec=4, 
#                     steering_vectors = steeringVectors, freq_max = np.inf)#, cross_spec_mask = station_distances < 0.25)

#%% Simple test with 1 active and 2 quiet source locations. Check that amplitudes are right for spherical spreading
src_xyz = np.array([[-0.1, -0.1, 0],[0.1,-0.1,0], [0,-0.1,0]])
src_amp = [1, 0, 0]
fl = [2,2,2] 
fh = [8,8,8]
# peaks that get chopped off at the ends of the trace lead to spurious failures for short traces
stream = cleanbf.make_synth_stream_local(src_xyz = src_xyz, amp = src_amp, fl = fl, fh = fh, Nt = 3000)
sta_xyz = np.array([[d for d in tr.stats.coordinates.values()] for tr in stream])

distance = np.sqrt(np.einsum('ij->i', (src_xyz[0,:] - sta_xyz)**2))
assert approx_equal_all(distance * stream.std())

#%% test the steeringVectors function
freqs = np.arange(5)
crossSpec, FT, freqs, dfN, dfD = cleanbf.calc_cross_spectrum(stream, win_length_sec = 4, freq_bin_width = 1)
steeringVectors = cleanbf.make_steering_vectors_local(stream, src_xyz, freqs = freqs)

distance = np.sqrt(np.sum((sta_xyz[0,:] - src_xyz[0,:])**2))
#assert approx_equal(steeringVectors[1,0,0,0], 1/distance * np.exp(2j*np.pi * freqs[1] * distance / 0.340))
assert steeringVectors.shape[0] == len(freqs)
assert steeringVectors.shape[1] == src_xyz.shape[0]
assert steeringVectors.shape[2] == 1
assert steeringVectors.shape[3] == sta_xyz.shape[0]
#%% simulate monitoring a dam
N = 50
src_xyz = np.zeros((N, 3))
src_xyz[:,0] = 0.05 * np.arange(N)/N - 0.025
src_xyz[:,1] = 0.05
#src_amp = np.array([0+(i in [10, 40]) for i in range(50)])
#src_amp = np.ones(50)
src_amp = 0*src_xyz[:,0]
src_amp[np.argmin(np.abs(src_xyz[:,0]))] = 1
plt.plot(src_xyz[:,0], src_amp)
#%% dam: make stream and steering vectors
sta_i = np.arange(50)
fl = 10 + np.zeros(50)
fh = 30 + np.zeros(50)
sta_x = np.concatenate([np.arange(5) * 0.02 - 0.04, np.zeros(4) - 0.035])
sta_y = np.concatenate([np.zeros(5), np.arange(1,5)*0.005])
stream = cleanbf.make_synth_stream_local(src_xyz = src_xyz, amp = src_amp, fl = fl, fh = fh, Nt = 6000, uncorrelatedNoiseAmp=.30,
                                       #Nx = 8, dx = 0.01, Ny = 1, dy = 0.03)
                                       x = sta_x, y = sta_y, c = 0.34)
freq_max = 40
crossSpec, FT, freqList, dfN, dfD = cleanbf.calc_cross_spectrum(stream, win_length_sec = 2, freq_bin_width = 1)
        ## drop freqs outside the user-defined range
w = np.where((freqList < freq_max) & (freqList > 0))[0]
crossSpec = crossSpec[:,:,w]
freqList = freqList[w]
steeringVectors = cleanbf.make_steering_vectors_local(stream, src_xyz, freqs = freqList, c = 0.34)
station_distances = cleanbf.calc_station_pair_distance(stream)

#%% run the cleanbf analysis
result = cleanbf.clean(stream, verbose = True, phi=0.1, win_length_sec=2, freq_bin_width=1,
                     steering_vectors = steeringVectors/np.abs(steeringVectors), prewhiten = True, freq_max = freq_max)#, cross_spec_mask = station_distances < 0.25)
#%%
plt.close('all')
plt.figure()
plt.subplot(1,2,1)
plt.plot(np.einsum('ijk->j', result['cleanSpec']))
plt.plot(np.einsum('ijk->j', result['originalSpec']))
plt.subplot(1,2,2)
plt.plot(src_xyz[src_amp != 1,0], src_xyz[src_amp != 1,1], 'k*')
plt.plot(src_xyz[src_amp == 1,0], src_xyz[src_amp == 1,1], 'r*')
sta_xyz = np.array([[d for d in tr.stats.coordinates.values()] for tr in stream])
plt.plot(sta_xyz[:,0], sta_xyz[:,1], 'bv')
stream_geometry = cleanbf.get_geometry(stream) ## check to make sure the coordinates are correct
plt.plot(stream_geometry[:,0], stream_geometry[:,1], 'gv')
plt.axis('equal')

#%%

plt.figure(2)
plt.plot(np.abs(np.fft.fft(stream[0].data)))
result = cleanbf.clean(stream, verbose = True, phi=0.2, win_length_sec=2)
plt.subplot(1,2,1)
cleanbf.plot_freq_slow_spec(result, 'fx', fRange = [0,10])
plt.subplot(1,2,2)
cleanbf.plot_freq_slow_spec(result, 'fx', type = 'original', fRange=[0,10])
plt.tight_layout()

## power considerations:
original_power = np.einsum('iij ->', result['originalCrossSpec']) # total power in uncleaned cross spectrum
clean_power = np.sum(result['cleanSpec'])
1 - clean_power / original_power # this is equal to the remaining power ratio printed at the end of clean

