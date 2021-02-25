import numpy as np
import matplotlib.pyplot as plt
import obspy, scipy, importlib
try:
    importlib.reload(clean)
    print('reloaded')
except:
    import clean

#%%
## Load an earthquake recording. This includes preliminary background noise,
## primary infrasound (simple wavefield), and secondary infrasound (diffuse wavefield)

eq_stream = obspy.read('data/aftershock.mseed')
eq_stream.filter('highpass', freq=2)
inv = obspy.read_inventory('data/XP_PARK_inventory.xml') # includes coordinates
clean.add_inv_coords(eq_stream, inv) # store the coordinates in the stream

## define significant times in the data
t1 = eq_stream.traces[0].stats.starttime # start of trace
t2 = eq_stream[0].stats.endtime # end of trace
t_trans = obspy.UTCDateTime('2020-04-14T03:27:08.9') # transition between primary-secondary sound

## define slowness grid to search
s_list = np.arange(-4, 4, 0.25)

#%%
## Aftershock primary infrasound (seismic-acoustic conversion at the array)
## This moves at seismic wave speeds and should have very low slowness values.
## Consequently, the Clean spectrum is mostly concentrated around the bullseye.

st = eq_stream.slice(t_trans - 4, t_trans-0.2)

result = clean.clean(st, verbose = True, phi = 0.2, separateFreqs = 0, win_length_sec = 0.5,
                              freq_bin_width = 1, freq_min = 0, freq_max = 20, 
                              sxList = s_list, syList = s_list, prewhiten = False)
plt.figure(1)
plt.subplot(2,2,1)
clean.plot_freq_slow_spec(result, 'xy', 'original')
plt.subplot(2,2,2)
clean.plot_freq_slow_spec(result, 'xy')
plt.subplot(2,2,3)
clean.polar_freq_slow_spec(result, 'fh')
plt.subplot(2,2,4)
clean.polar_freq_slow_spec(result, 'fa')

plt.tight_layout()

#nWelch 8 gives best power (0.42)
#%%
## Aftershock secondary infrasound (seismic-to-acoustic conversion away from the array)
## Slowness should mostly be that of horizontally-propagating acoustic waves.
## Consequently, the energy should mainly be on the 3 s/km circle.

st = eq_stream.slice(t_trans+2, t2)

s_list = np.arange(-4, 4, 0.25)
result = clean.clean(st, verbose = True, phi = 0.2, separateFreqs = 0, win_length_sec = 1, 
                              freq_bin_width = 1, freq_min = 0, freq_max = 20, 
                              sxList = s_list, syList = s_list, prewhiten = False)
plt.figure(2)
plt.subplot(2,2,1)
clean.plot_freq_slow_spec(result, 'xy', 'original')
plt.subplot(2,2,2)
clean.plot_freq_slow_spec(result, 'xy')
plt.subplot(2,2,3)
clean.polar_freq_slow_spec(result, 'fh')
plt.subplot(2,2,4)
clean.polar_freq_slow_spec(result, 'fa', 'clean')

plt.tight_layout()


#%%
## Background sounds before aftershock begins
## The clean spectrum should have little energy and should be concentrated around the 3 s/km circle

st = eq_stream.slice(t1, t_trans-10)

s_list = np.arange(-4, 4, 0.25)
result = clean.clean(st, verbose = True, phi = 0.2, separateFreqs = 0,  
                              freq_bin_width = 1, freq_min = 0, freq_max = 20, 
                              sxList = s_list, syList = s_list, prewhiten = False)
plt.figure(3)
plt.subplot(2,2,1)
clean.plot_freq_slow_spec(result, 'xy', 'original')
plt.subplot(2,2,2)
clean.plot_freq_slow_spec(result, 'xy')
plt.subplot(2,2,3)
clean.polar_freq_slow_spec(result, 'fh')
plt.subplot(2,2,4)
clean.polar_freq_slow_spec(result, 'fa')
plt.tight_layout()