import numpy as np
import matplotlib.pyplot as plt
import obspy, scipy
import cleanbf
import sys
#sys.path.append('/home/jake/Work/Aftershocks/lib/clean')
try:
    import importlib
    importlib.reload(cleanbf)
    print('reloaded')
except:
    pass

#%%
## Load an earthquake recording. This includes preliminary background noise,
## primary infrasound (simple wavefield), and secondary infrasound (diffuse wavefield)

eq_stream = obspy.read('data/aftershock.mseed')
eq_stream.filter('highpass', freq=2)
inv = obspy.read_inventory('data/XP_PARK_inventory.xml') # includes coordinates
cleanbf.add_inv_coords(eq_stream, inv) # store the coordinates in the stream

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

result = cleanbf.clean(st, verbose = True, phi = 0.01, separate_freqs = 0, win_length_sec = 0.5,
                              freq_bin_width = 1, freq_min = 0, freq_max = 20, 
                              sxList = s_list, syList = s_list, prewhiten = False)

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

#%%
## Aftershock secondary infrasound (seismic-to-acoustic conversion away from the array)
## Slowness should mostly be that of horizontally-propagating acoustic waves.
## Consequently, the energy should mainly be on the 3 s/km circle.

st = eq_stream.slice(t_trans+2, t2)
#st.pop(4) # signals at index 4 are weirdly quiet
#st.pop(6)
s_list = np.arange(-4, 4, 0.25)
result = cleanbf.clean(st, verbose = True, phi = 0.05, separate_freqs = 0, win_length_sec = 0.5, 
                              freq_bin_width = 1, freq_min = 1, freq_max = 25, 
                              sxList = s_list, syList = s_list, p_value = 0.0001)
plt.close(23)
plt.figure(23)
plt.subplot(2,2,1)
cleanbf.plot_freq_slow_spec(result, 'xy', 'original')
plt.subplot(2,2,2)
cleanbf.plot_freq_slow_spec(result, 'xy')
plt.subplot(2,2,3)
cleanbf.polar_freq_slow_spec(result, 'fh')
plt.subplot(2,2,4)
cleanbf.polar_freq_slow_spec(result, 'fa', 'clean')

plt.tight_layout()


#%%
## Background sounds before aftershock begins
## The clean spectrum should have little energy and should be concentrated around 3 s/km 

st = eq_stream.slice(t_trans-20, t_trans-10)

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
st = obspy.read('data/noise.mseed')
st.filter('bandpass', freqmin=0.15, freqmax = 1)
inv = obspy.read_inventory('data/XP_PARK_inventory.xml') # includes coordinates
cleanbf.add_inv_coords(st, inv) # store the coordinates in the stream

## define slowness grid to search
s_list = np.arange(-4, 4, 0.25)
result = cleanbf.clean(st, phi = 0.2, win_length_sec = 180, 
                     freq_bin_width = 1, freq_min = 0, freq_max = 2, 
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


#%% loop through time
loop_start = t_trans - 10
loop_end = t_trans + 30
loop_step = 2
loop_width = 4
t1 = loop_start
z = np.array([])
specs = {'t':z, 'clean':[], 'original':[], 'result':[]}
clean_output = {'t':z, 'sx':z, 'sy':z, 'f':z, 'power':z, 'cleanRatio' : z}
dirty_output = {'t':z, 'sx':z, 'sy':z, 'power':z}

while (t1 + loop_width) <= loop_end:
    st = eq_stream.slice(t1, t1 + loop_width)
    halftime = t1 + loop_width/2 - loop_start
    print('%f of %f' % (halftime, loop_end - loop_start))
    result = cleanbf.clean(st, verbose = False, phi = 0.2, separate_freqs = 0, win_length_sec = 1,
                              freq_bin_width = 1, freq_min = 0, freq_max = 20, 
                              sxList = s_list, syList = s_list, prewhiten = False)
    t1 += loop_step
    
    i_f, j_sx, k_sy = np.where(result['cleanSpec'] > 0)
    clean_output['t'] = np.append(clean_output['t'], halftime + np.zeros(len(i_f)))
    clean_output['sx'] = np.append(clean_output['sx'], result['sx'][j_sx])
    clean_output['sy'] = np.append(clean_output['sy'], result['sy'][k_sy])
    clean_output['f'] = np.append(clean_output['f'], result['freq'][i_f])
    clean_output['power'] = np.append(clean_output['power'], result['cleanSpec'][i_f, j_sx, k_sy])
    clean_output['cleanRatio'] = np.append(clean_output['cleanRatio'], np.sum(result['cleanSpec']) / 
                                np.einsum('iik->', np.abs(result['originalCrossSpec'])) + np.zeros(len(i_f)))
    
    dirty_spec = np.sum(result['originalSpec'], 0) # sum over frequencies
    j_sx, k_sy = np.where(dirty_spec == dirty_spec.max())
    dirty_output['t'] = np.append(dirty_output['t'], halftime)
    dirty_output['sx'] = np.append(dirty_output['sx'], result['sx'][j_sx])
    dirty_output['sy'] = np.append(dirty_output['sy'], result['sy'][k_sy])
    dirty_output['power'] = np.append(dirty_output['power'], dirty_spec[j_sx, k_sy])
    
    specs['clean'].append(result['cleanSpec'])
    specs['original'].append(result['originalSpec'])
    specs['result'].append(result)
    specs['t'] = np.append(specs['t'], halftime)
