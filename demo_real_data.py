import numpy as np
import matplotlib.pyplot as plt
import obspy, scipy, importlib
try:
    importlib.reload(clean)
    print('reloaded')
except:
    import clean

#%%
## Aftershock primary infrasound
eq_stream = obspy.read('data/aftershock.mseed')
inv = obspy.read_inventory('data/XP_PARK_inventory.xml')
clean.add_inv_coords(eq_stream, inv)
eq_stream.filter('highpass', freq=2)

t1 = eq_stream.traces[0].stats.starttime
t2 = eq_stream[0].stats.endtime
t_trans = obspy.UTCDateTime('2020-04-14T03:27:08.9')
eq_stream.trim(t1, t_trans-0.2)

s_list = np.arange(-4, 4, 0.25)
result = clean.clean(eq_stream, verbose = True, phi = 0.2, separateFreqs = 0, nWelch = 16, 
                              p_value=0.0001, freq_bin_width = 1, freq_min = 0, freq_max = 20, 
                              sxList = s_list, syList = s_list, prewhiten = False)
plt.subplot(2,2,1)
clean.plot_freq_slow_spec(result, 'xy')
plt.subplot(2,2,2)
clean.plot_freq_slow_spec(result, 'xy', 'original')
plt.subplot(2,2,3)
clean.plot_freq_slow_spec(result, 'fx')
plt.subplot(2,2,4)
clean.plot_freq_slow_spec(result, 'fx', 'original')

#plt.show()
clean.check_output_power(result)

#%%
## Aftershock secondary infrasound
eq_stream = obspy.read('data/aftershock.mseed')
inv = obspy.read_inventory('data/XP_PARK_inventory.xml')
clean.add_inv_coords(eq_stream, inv)
eq_stream.filter('highpass', freq=2)

t1 = eq_stream.traces[0].stats.starttime
t2 = eq_stream[0].stats.endtime
t_trans = obspy.UTCDateTime('2020-04-14T03:27:08.9')
eq_stream.trim(t_trans+0.6, t2)

s_list = np.arange(-4, 4, 0.25)
result = clean.clean(eq_stream, verbose = True, phi = 0.2, separateFreqs = 0, nWelch = 32, 
                              p_value=0.0001, freq_bin_width = 1, freq_min = 0, freq_max = 20, 
                              sxList = s_list, syList = s_list, prewhiten = False)
plt.subplot(2,2,1)
clean.plot_freq_slow_spec(result, 'xy')
plt.subplot(2,2,2)
clean.plot_freq_slow_spec(result, 'xy', 'original')
plt.subplot(2,2,3)
clean.plot_freq_slow_spec(result, 'fx')
plt.subplot(2,2,4)
clean.plot_freq_slow_spec(result, 'fx', 'original')

#plt.show()
clean.check_output_power(result)


#%%
## Aftershock preliminary
eq_stream = obspy.read('data/aftershock.mseed')
inv = obspy.read_inventory('data/XP_PARK_inventory.xml')
clean.add_inv_coords(eq_stream, inv)
eq_stream.filter('highpass', freq=2)

t1 = eq_stream.traces[0].stats.starttime
t2 = eq_stream[0].stats.endtime
t_trans = obspy.UTCDateTime('2020-04-14T03:27:08.9')
eq_stream.trim(t1, t_trans-5)

s_list = np.arange(-4, 4, 0.25)
result = clean.clean(eq_stream, verbose = True, phi = 0.2, separateFreqs = 0, nWelch = 16, 
                              p_value=0.0001, freq_bin_width = 1, freq_min = 0, freq_max = 20, 
                              sxList = s_list, syList = s_list, prewhiten = False)
plt.subplot(2,2,1)
clean.plot_freq_slow_spec(result, 'xy')
plt.subplot(2,2,2)
clean.plot_freq_slow_spec(result, 'xy', 'original')
plt.subplot(2,2,3)
clean.plot_freq_slow_spec(result, 'fx')
plt.subplot(2,2,4)
clean.plot_freq_slow_spec(result, 'fx', 'original')

#plt.show()
clean.check_output_power(result)

