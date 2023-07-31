import numpy as np
import matplotlib.pyplot as plt
import obspy#, scipy
import sys, os
import cleanbf
try:
    import importlib
    importlib.reload(cleanbf)
    print('reloaded')
except:
    pass

#%% Load EQ Data
## Load an earthquake recording. This includes preliminary background noise,
## primary infrasound (simple wavefield), and secondary infrasound (diffuse wavefield)
data_dir = os.path.join(os.path.dirname(cleanbf.__file__), 'data')

eq_stream = obspy.read(os.path.join(data_dir, 'aftershock_beginning.mseed'))
eq_stream.filter('highpass', freq=2)
inv = obspy.read_inventory(os.path.join(data_dir, 'XP_PARK_inventory.xml')) # includes coordinates
cleanbf.add_inv_coords(eq_stream, inv) # store the coordinates in the stream

## define significant times in the data
t1 = eq_stream.traces[0].stats.starttime # start of trace
t2 = eq_stream[0].stats.endtime # end of trace
t_trans = obspy.UTCDateTime('2020-04-14T03:27:08.9') # transition between primary-secondary sound

## define slowness grid to search
s_list = np.arange(-4, 4, 0.1)

#%% Process Primary Infrasound
## Aftershock primary infrasound (seismic-acoustic conversion at the array)
## This moves at seismic wave speeds and should have very low slowness values.
## Consequently, the Clean spectrum is mostly concentrated around the bullseye.

st = eq_stream.slice(t2 - 4, t2-0.2)

## Experiment with different subsets of sensors. Full array yields a compact bullseye at the origin (0.253 power ratio)
#st = st[17:] # smallest triangle 0.113 power ratio
#st = st[6:7] + st[9:11] # medium triangle 0.208 power ratio, mild aliasing
st = st[3:5] + st[16:17] # huge triangle 0.0452 power ratio, severe aliasing
plt.close(0)
plt.figure(0)
cleanbf.plot_distances(st, 0) # plot the array (or sub-array) geometry

distances = cleanbf.calc_station_pair_distance(eq_stream)

result = cleanbf.clean(st, verbose = True, phi = 0.01, separate_freqs = 0, win_length_sec = 0.5,
                              freq_bin_width = 1, freq_min = 4, freq_max = 8, 
                              sxList = s_list, syList = s_list, prewhiten = False)#,                               cross_spec_weights = distances < 0.05)

plt.close(1)
plt.figure(1)
plt.subplot(2,2,1)
cleanbf.plot_freq_slow_spec(result, 'xy', 'original')
plt.subplot(2,2,2)
cleanbf.plot_freq_slow_spec(result, 'xy')
plt.subplot(2,2,3)
cleanbf.polar_freq_slow_spec(result, 'fh')
plt.subplot(2,2,4)
cleanbf.polar_freq_slow_spec(result, 'fa')

plt.tight_layout()


#%% Process pre-earthquake background noise
## Background sounds before aftershock begins
## The clean spectrum should have little energy and should be concentrated around 3 s/km 

st = eq_stream.slice(t2-20, t2-10)

s_list = np.arange(-4, 4, 0.25)
result = cleanbf.clean(st, verbose = True, phi = 0.2, win_length_sec=1, 
                              freq_bin_width = 1, freq_min = 1, freq_max = 25, 
                              sxList = s_list, syList = s_list, prewhiten = False)
plt.figure(3)
plt.subplot(2,2,1)
cleanbf.plot_freq_slow_spec(result, 'xy', 'original')
plt.subplot(2,2,2)
cleanbf.plot_freq_slow_spec(result, 'xy')
plt.subplot(2,2,3)
cleanbf.polar_freq_slow_spec(result, 'fh')
plt.subplot(2,2,4)
cleanbf.polar_freq_slow_spec(result, 'fa')
plt.tight_layout()

#%% microbarom detection
st = obspy.read(os.path.join(data_dir, 'noise.mseed'))
st.filter('bandpass', freqmin=0.1, freqmax = 1)
inv = obspy.read_inventory(os.path.join(data_dir, 'XP_PARK_inventory.xml')) # includes coordinates
cleanbf.add_inv_coords(st, inv) # store the coordinates in the stream

## define slowness grid to search
s_list = np.arange(-4, 4, 0.25)
result = cleanbf.clean(st, phi = 0.2, win_length_sec = 180, 
                     freq_bin_width = 1, freq_min = 0, freq_max = 1, 
                     sxList = s_list, syList = s_list, prewhiten = False)
plt.figure(4)
plt.subplot(2,2,1)
cleanbf.plot_freq_slow_spec(result, 'xy', 'original')
plt.subplot(2,2,2)
cleanbf.plot_freq_slow_spec(result, 'xy')
plt.subplot(2,2,3)
cleanbf.polar_freq_slow_spec(result, 'fh')
plt.subplot(2,2,4)
cleanbf.polar_freq_slow_spec(result, 'fa')
plt.tight_layout()

