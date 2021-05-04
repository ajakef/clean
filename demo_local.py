import numpy as np
import matplotlib.pyplot as plt
import obspy, scipy, importlib
import clean
try:
    importlib.reload(clean)
    print('reloaded')
except:
    pass
    
def approx_equal(x, y, p = 0.01):
    return np.abs((x-y)/x) < p

def approx_equal_all(x, p = 0.01):
    return all([approx_equal(x[0], y, p) for y in x])
        
#%% Simple test with 1 active and 2 quiet source locations. Check that amplitudes are right for spherical spreading
src_xyz = np.array([[-0.1, -0.1, 0],[0.1,-0.1,0], [0,-0.1,0]])
src_amp = [1, 0, 0]
fl = [2,2,2]
fh = [8,8,8]
stream = clean.make_synth_stream_local(src_xyz = src_xyz, amp = src_amp, fl = fl, fh = fh)
sta_xyz = np.array([[d for d in tr.stats.coordinates.values()] for tr in stream])

distance = np.sqrt(np.einsum('ij->i', (src_xyz[0,:] - sta_xyz)**2))
assert approx_equal_all(distance * stream.std())

#%% test the steeringVectors function
freqs = np.arange(5)
steeringVectors = clean.make_steering_vectors_local(stream, src_xyz, freqs = freqs)

distance = np.sqrt(np.sum((sta_xyz[0,:] - src_xyz[0,:])**2))
assert approx_equal(steeringVectors[0,0,0,0], 1/distance * np.exp(2j*np.pi * freqs[1] * distance / 0.340))
assert steeringVectors.shape[0] == len(freqs)-1
assert steeringVectors.shape[1] == src_xyz.shape[0]
assert steeringVectors.shape[2] == 1
assert steeringVectors.shape[3] == sta_xyz.shape[0]
#%%

#%%
plt.figure(2)
plt.plot(np.abs(np.fft.fft(stream[0].data)))
result = clean.clean(stream, verbose = True, phi=0.2, win_length_sec=2)
plt.subplot(1,2,1)
clean.plot_freq_slow_spec(result, 'fx', fRange = [0,10])
plt.subplot(1,2,2)
clean.plot_freq_slow_spec(result, 'fx', type = 'original', fRange=[0,10])
plt.tight_layout()

## power considerations:
original_power = np.einsum('iij ->', result['originalCrossSpec']) # total power in uncleaned cross spectrum
clean_power = np.sum(result['cleanSpec'])
1 - clean_power / original_power # this is equal to the remaining power ratio printed at the end of clean

