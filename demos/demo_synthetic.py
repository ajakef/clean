import numpy as np
import matplotlib.pyplot as plt
import obspy, scipy, importlib
import cleanbf
try:
    importlib.reload(cleanbf)
    print('reloaded')
except:
    pass
    
#%% Pure noise, no correlated signal
## In this case, we just plot the f-sx spectra (clean and original). 
## The clean spectrum is empty, and the original spectrum is random.

stream = cleanbf.make_synth_stream(sx = [0], sy = [0], amp = [0], uncorrelatedNoiseAmp=10) # x,y are built into stream
plt.plot(np.abs(np.fft.fft(stream[0].data)))
result = cleanbf.clean(stream, verbose = True, phi=0.1, syList = [0], separate_freqs = 0, p_value = 0.01)
plt.figure(1)
plt.subplot(1,2,1)
cleanbf.plot_freq_slow_spec(result, 'fx')#, fRange = [0,10])
plt.subplot(1,2,2)
cleanbf.plot_freq_slow_spec(result, 'fx', type = 'original')#, fRange=[0,10])
plt.tight_layout()

#%% Simple test with defaults (1 wave, 2.56 sec, sx=sy=0, signal freq 1-4, no noise)
## The clean spectrum is highly concentrated at sx=0, as expected. 
## With only one wave and zero noise, nearly all the power is recovered in the clean spectrum.
stream = cleanbf.make_synth_stream() # x,y are built into stream
plt.figure(2)
plt.plot(np.abs(np.fft.fft(stream[0].data)))
result = cleanbf.clean(stream, verbose = True, phi=0.2, win_length_sec=2)
fig, ax = plt.subplots(3, 1)
cleanbf.plot_freq_slow_spec(result, 'fx', fRange = [0,10], ax = ax[0])
cleanbf.plot_freq_slow_spec(result, 'fx', type = 'original', fRange=[0,10], ax = ax[1])
cleanbf.plot_freq_slow_spec(result, 'xy', type = 'original', fRange=[0,10], ax = ax[2])
plt.tight_layout()

## power considerations:
original_power = np.einsum('iij ->', result['originalCrossSpec']) # total power in uncleaned cross spectrum
clean_power = np.sum(result['cleanSpec'])
1 - clean_power / original_power # this is equal to the remaining power ratio printed at the end of clean


#%% Large-N style test; pure noise: the goal is to see if we get false detections. 
## Result depends on the p-value

num_windows = 8
win_length_sec = 1
freq_bin_width = 1
s_list = np.arange(-4, 4, 0.25)

stream = cleanbf.make_synth_stream(Nt = win_length_sec * num_windows * 100, sx = [0], amp = [0], 
                                 sy = [0], fc = [6, 6], uncorrelatedNoiseAmp = 2, Nx = 4, Ny = 4) 
result = cleanbf.clean(stream, verbose = True, phi = 0.5, separate_freqs = 0, win_length_sec = win_length_sec, 
                              p_value=0.0001, freq_bin_width = freq_bin_width, show_plots = False,
                              sxList = s_list, syList = s_list)

plt.close(3) 
#%%                             
fig, ax = plt.subplots(2,2)
cleanbf.plot_freq_slow_spec(result, 'xy', ax = ax[0,0])
cleanbf.plot_freq_slow_spec(result, 'xy', 'original', ax = ax[0,1])
cleanbf.plot_freq_slow_spec(result, 'fx', ax = ax[1,0])
cleanbf.plot_freq_slow_spec(result, 'fx', 'original', ax = ax[1,1])
plt.tight_layout()

cleanbf.check_output_power(result)

#%% Small array (2x2 square) test with two point sources and noise
s_list = np.arange(-4, 4, 0.25)

num_windows = 8 # the number of windows strongly affects the remaining power ratio
win_length_sec = 1
freq_bin_width = 1
Nt = num_windows * win_length_sec * 100
 
stream = cleanbf.make_synth_stream(Nt = Nt, sx = [1, -2], sy = [1, 2], amp = [1, 1], fc = [6, 6],
                                 Nx = 2, Ny = 2, uncorrelatedNoiseAmp = 2) 
result = cleanbf.clean(stream, verbose = True, phi = 0.1, separate_freqs = 0, win_length_sec = win_length_sec, 
                              p_value=0.1, freq_bin_width = freq_bin_width, show_plots = False,
                              sxList = s_list, syList = s_list)

#imageAdj = lambda x: np.log(x + x.max()*1e-3)
#imageAdj = lambda x:x * (x>(x.max()*0.01))
plt.close(4)
plt.figure(4)
plt.subplot(2,2,1)
cleanbf.plot_freq_slow_spec(result, 'xy')#, imageAdj = imageAdj)
plt.subplot(2,2,2)
cleanbf.plot_freq_slow_spec(result, 'xy', 'original')
plt.subplot(2,2,3)
cleanbf.polar_freq_slow_spec(result, 'fh')
plt.subplot(2,2,4)
cleanbf.polar_freq_slow_spec(result, 'fa')
plt.tight_layout()
#%% Large-N array (5x5 square) test with two point sources. 
## Notice the lower remaining power ratio vs the small array test, and the longer runtime. 
s_list = np.arange(-4, 4, 0.25)

num_windows = 8 # the number of windows strongly affects the clean power ratio
win_length_sec = 1
freq_bin_width = 1
Nt = num_windows * win_length_sec * 100
 
stream = cleanbf.make_synth_stream(Nt = Nt, sx = [1, -2], sy = [1, 2], amp = [1,1], fc = [6, 6], 
                                        Nx = 5, Ny = 5,uncorrelatedNoiseAmp = 2) 
result = cleanbf.clean(stream, verbose = True, phi = 0.1, separate_freqs = 0, win_length_sec = win_length_sec, 
                              p_value=0.1, freq_bin_width = freq_bin_width, show_plots = False,
                              sxList = s_list, syList = s_list)
plt.figure(5)
plt.subplot(2,2,1)
cleanbf.plot_freq_slow_spec(result, 'xy', 'original')
plt.subplot(2,2,2)
cleanbf.plot_freq_slow_spec(result, 'xy')
plt.subplot(2,2,3)
cleanbf.polar_freq_slow_spec(result, 'fh')
plt.subplot(2,2,4)
cleanbf.polar_freq_slow_spec(result, 'fa')
plt.tight_layout()

#%% Large-N style test; linear source
## goal is to see how it handles line sources (consisting of closely-spaced point sources)
## Two small lines: it resolves the two source regions but fails to show their dimensions
## One long line: it shows many point sources spaced haphazardly along the source region
s_list = np.arange(-4, 4, 0.25)

num_windows = 8 # the number of windows strongly affects the clean power ratio
win_length_sec = 1
freq_bin_width = 1
Nt = num_windows * win_length_sec * 100
#sx = np.concatenate([np.arange(2, 3, 0.05), np.arange(-3, -2, 0.05)]) # two short lines
sx = np.arange(-3, 3, 0.05) # one long line
stream = cleanbf.make_synth_stream(Nt = Nt, sx = sx, amp = sx*0+1, sy = 0*sx, Nx = 4, Ny = 4, 
                                        fc = 6 + 0*sx, uncorrelatedNoiseAmp = 0) 
result = cleanbf.clean(stream, verbose = True, phi = 0.1, separate_freqs = 0, win_length_sec = win_length_sec, 
                              p_value=0.0001, freq_bin_width = freq_bin_width, show_plots = False,
                              sxList = s_list, syList = s_list)
            
plt.figure(6)                  
plt.subplot(2,2,1)
cleanbf.plot_freq_slow_spec(result, 'xy', 'original')
plt.plot(sx, 0*sx, 'b.') # true wave slownesses in blue
plt.subplot(2,2,2)
cleanbf.plot_freq_slow_spec(result, 'xy')
plt.subplot(2,2,3)
cleanbf.polar_freq_slow_spec(result, 'fh')
plt.subplot(2,2,4)
cleanbf.polar_freq_slow_spec(result, 'fa')

plt.tight_layout()
