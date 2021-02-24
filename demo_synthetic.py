import numpy as np
import matplotlib.pyplot as plt
import obspy, scipy, importlib
try:
    importlib.reload(clean)
    print('reloaded')
except:
    import clean
    
#%% Pure noise, no correlated signal
stream = clean.make_synth_stream(sx = [0], sy = [0], amp = [0], uncorrelatedNoiseAmp=10) # x,y are built into stream
plt.plot(np.abs(np.fft.fft(stream[0].data)))
result = clean.clean(stream, verbose = True, phi=0.1, syList = [0], separateFreqs = 0, p_value = 0.01)
plt.subplot(1,2,1)
clean.plot_freq_slow_spec(result, 'fx')#, fRange = [0,10])
plt.subplot(1,2,2)
clean.plot_freq_slow_spec(result, 'fx', type = 'original')#, fRange=[0,10])

#%% Simple test with defaults (1 wave, 2.56 sec, sx=sy=0, signal freq 1-4, no noise)
stream = clean.make_synth_stream() # x,y are built into stream
plt.plot(np.abs(np.fft.fft(stream[0].data)))
result = clean.clean(stream, verbose = True, phi=0.2, nWelch = 4)
plt.subplot(1,2,1)
clean.plot_freq_slow_spec(result, 'fx', fRange = [0,10])
plt.subplot(1,2,2)
clean.plot_freq_slow_spec(result, 'fx', type = 'original', fRange=[0,10])

## power considerations:
original_power = np.einsum('iij ->', result['originalCrossSpec']) # total power in uncleaned cross spectrum
clean_power = np.sum(result['cleanSpec'])
1 - clean_power / original_power # this is equal to the remaining power ratio printed at the end of clean

#%% Large-N style test; 2 signal sources, and equal total signal and total noise power.
## separateFreqs=0 works pretty well given enough nWelch (~4
## improvement between nWelch=4 and multitaper is noticeable.
#!rm -f ~/Conferences/AGU2020/poster/plot_steps/*
nWelch = 1
freq_bin_width = 1
s_list = np.arange(-4, 4, 0.25)

if False:
    stream = clean.make_synth_stream(Nt = 256, sx = [1, -2], sy = [2, 0.5], Nx = 4, Ny = 4, 
                                            fc = [6, 6], uncorrelatedNoiseAmp = 2) 
else:
    for tr in stream:
        tr.stats['coordinates']['x'] += np.random.normal(0,0.002,1)
        tr.stats['coordinates']['y'] += np.random.normal(0,0.002,1)
print(stream[0].stats['coordinates']['x'])

#%%
cs1 = clean.calc_cross_spectrum(stream, taper = 'multitaper')
cs2 = clean.calc_cross_spectrum(stream, taper = None, nWelch=1)
plt.close(1)
plt.figure(1)
plt.subplot(2,1,1)
plt.plot(cs1[-3], np.abs(cs1[0][0,1,:]))
plt.plot(cs2[-3], np.abs(cs2[0][0,1,:]))
plt.subplot(2,1,2)
plt.plot(cs1[-3], np.angle(cs1[0][0,1,:]))
plt.plot(cs2[-3], np.angle(cs2[0][0,1,:]))


## perturb sensor locations by 1 m each after wavefield is created, before clean is run
    
result = clean.clean(stream, verbose = True, phi = 0.05, separateFreqs = 0, nWelch = nWelch, 
                              p_value=0.01, freq_bin_width = freq_bin_width, show_plots = False, sxList = s_list, syList = s_list, taper = 'multitaper', taper_param = 4)
plt.close(2)                              
plt.figure(2)
#imageAdj = lambda x: np.log(x + x.max()*1e-3)
imageAdj = lambda x:x * (x>(x.max()*0.01))
plt.subplot(2,2,1)
clean.plot_freq_slow_spec(result, 'xy')#, imageAdj = imageAdj)
plt.subplot(2,2,2)
clean.plot_freq_slow_spec(result, 'xy', 'original')
plt.subplot(2,2,3)
clean.plot_freq_slow_spec(result, 'fx')
plt.subplot(2,2,4)
clean.plot_freq_slow_spec(result, 'fx', 'original')

clean.check_output_power(result)

# F threshold: from den Ouden https://academic.oup.com/gji/article/221/1/305/5698307?casa_token=NH1vXHKRcxwAAAAA:uEHimusKc5szgSX-rr9_ekAXr3YxFO9MKUsfTZEiazj87zBGjO9BjzEvU7Rv5dBibAxrCwi_poy75A
# where Tf is nWelch + (num freqs in a bin)
# df1: 2 Tf
# df2: 2*Tf*(Nsta - 1)
#%% Large-N style test; pure noise: the goal is to see if we get false detections. looks like yes!
## separateFreqs=0 works pretty well given enough nWelch (~4)
nWelch = 4
freq_bin_width = 1
stream = clean.make_synth_stream(Nt = nWelch * 64, sx = [1, -2], amp = [0,0], sy = [2, 0], Nx = 4, Ny = 4, 
                                        fc = [6, 6], uncorrelatedNoiseAmp = 2) 
result = clean.clean(stream, verbose = True, phi = 0.5, separateFreqs = 0, nWelch = nWelch, 
                              p_value=0.0001, freq_bin_width = freq_bin_width, show_plots = False, sxList = s_list, syList = s_list)
                              
#imageAdj = lambda x: np.log(x + x.max()*1e-3)
imageAdj = lambda x:x * (x>(x.max()*0.01))
plt.subplot(2,2,1)
clean.plot_freq_slow_spec(result, 'xy')#, imageAdj = imageAdj)
plt.subplot(2,2,2)
clean.plot_freq_slow_spec(result, 'xy', 'original')
plt.subplot(2,2,3)
clean.plot_freq_slow_spec(result, 'fx')
plt.subplot(2,2,4)
clean.plot_freq_slow_spec(result, 'fx', 'original')

clean.check_output_power(result)

# F threshold: from den Ouden https://academic.oup.com/gji/article/221/1/305/5698307?casa_token=NH1vXHKRcxwAAAAA:uEHimusKc5szgSX-rr9_ekAXr3YxFO9MKUsfTZEiazj87zBGjO9BjzEvU7Rv5dBibAxrCwi_poy75A
# where Tf is nWelch + (num freqs in a bin)
# df1: 2 Tf
# df2: 2*Tf*(Nsta - 1)

#%% Large-N style test; linear source
## goal is to see how it handles sources close enough to fit within the array response
## it works passably but doesn't show the detail of the source well
s_list = np.arange(-4, 4, 0.25)

nWelch = 8
freq_bin_width = 1
sx = np.concatenate([np.arange(2, 3, 0.05), np.arange(-3, -2, 0.05)])
stream = clean.make_synth_stream(Nt = nWelch * 64, sx = sx, amp = sx*0+1, sy = 0*sx, Nx = 4, Ny = 4, 
                                        fc = 15 + 0*sx, uncorrelatedNoiseAmp = 0) 
result = clean.clean(stream, verbose = True, phi = 0.5, separateFreqs = 0, nWelch = nWelch, 
                              p_value=0.0001, freq_bin_width = freq_bin_width, show_plots = False, sxList = s_list, syList = s_list)
                              
#imageAdj = lambda x: np.log(x + x.max()*1e-3)
imageAdj = lambda x:x * (x>(x.max()*0.01))
plt.subplot(2,2,1)
clean.plot_freq_slow_spec(result, 'xy')#, imageAdj = imageAdj)
plt.subplot(2,2,2)
clean.plot_freq_slow_spec(result, 'xy', 'original')
plt.subplot(2,2,3)
clean.plot_freq_slow_spec(result, 'fx')
plt.subplot(2,2,4)
clean.plot_freq_slow_spec(result, 'fx', 'original')

clean.check_output_power(result)

#%% Small array test. This takes a lot more nWelch (~16) to detect anything.
nWelch = 16
freq_bin_width = 1
stream = clean.make_synth_stream(Nt = nWelch * freq_bin_width * 64, sx = [1, -2], sy = [2, 0], 
                                        Nx = 3, Ny = 1, fc = [6, 6], uncorrelatedNoiseAmp = 2) 
result = clean.clean(stream, verbose = True, phi = 0.1, separateFreqs = 0, nWelch = nWelch, 
                              p_value=0.9, freq_bin_width = freq_bin_width)
                              
#imageAdj = lambda x: np.log(x + x.max()*1e-3)
imageAdj = lambda x:x * (x>(x.max()*0.01))
plt.subplot(2,2,1)
clean.plot_freq_slow_spec(result, 'xy', imageAdj = imageAdj)
plt.subplot(2,2,2)
clean.plot_freq_slow_spec(result, 'xy', 'original')
plt.subplot(2,2,3)
clean.plot_freq_slow_spec(result, 'fx')
plt.subplot(2,2,4)
clean.plot_freq_slow_spec(result, 'fx', 'original')

