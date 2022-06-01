import numpy as np
import matplotlib.pyplot as plt
import obspy, scipy, importlib
import clean
try:
    importlib.reload(clean)
    print('reloaded')
except:
    pass
    
#%% Pure noise, no correlated signal
## In this case, we just plot the f-sx spectra (clean and original). 
## The clean spectrum is empty, and the original spectrum is random.

stream = clean.make_synth_stream(sx = [0], sy = [0], amp = [0], uncorrelatedNoiseAmp=10) # x,y are built into stream
plt.plot(np.abs(np.fft.fft(stream[0].data)))
result = clean.clean(stream, verbose = True, phi=0.1, syList = [0], separateFreqs = 0, p_value = 0.01)
fig, ax = plt.subplots(2, 1)
clean.plot_freq_slow_spec(result, 'fx', fig = fig, ax = ax[0])#, fRange = [0,10])
clean.plot_freq_slow_spec(result, 'fx', type = 'original', fig = fig, ax = ax[1])#, fRange=[0,10])
plt.tight_layout()

#%% Simple test with defaults (1 wave, 2.56 sec, sx=sy=0, signal freq 1-4, no noise)
## The clean spectrum is highly concentrated at sx=0, as expected. 
## With only one wave and zero noise, nearly all the power is recovered in the clean spectrum.
stream = clean.make_synth_stream() # x,y are built into stream
plt.figure(2)
plt.plot(np.abs(np.fft.fft(stream[0].data)))
result = clean.clean(stream, verbose = True, phi=0.2, win_length_sec=2)
fig, ax = plt.subplots(3, 1)
clean.plot_freq_slow_spec(result, 'fx', fRange = [0,10], fig = fig, ax = ax[0])
clean.plot_freq_slow_spec(result, 'fx', type = 'original', fRange=[0,10], fig = fig, ax = ax[1])
clean.plot_freq_slow_spec(result, 'xy', type = 'original', fRange=[0,10], fig = fig, ax = ax[2])
plt.tight_layout()

## power considerations:
original_power = np.einsum('iij ->', result['originalCrossSpec']) # total power in uncleaned cross spectrum
clean_power = np.sum(result['cleanSpec'])
1 - clean_power / original_power # this is equal to the remaining power ratio printed at the end of clean


#%% Large-N style test; pure noise: the goal is to see if we get false detections. 
## The result depends on the p-value: false detections often occur at p<0.01 but not p<0.0001.

num_windows = 8
win_length_sec = 1
freq_bin_width = 1
s_list = np.arange(-4, 4, 0.25)

stream = clean.make_synth_stream(Nt = win_length_sec * num_windows * 100, sx = [0], amp = [0], 
                                 sy = [0], fc = [6, 6], uncorrelatedNoiseAmp = 2, Nx = 4, Ny = 4) 
result = clean.clean(stream, verbose = True, phi = 0.5, separateFreqs = 0, win_length_sec = win_length_sec, 
                              p_value=0.0001, freq_bin_width = freq_bin_width, show_plots = False,
                              sxList = s_list, syList = s_list)

plt.close(3)                              
fig, ax = plt.subplots(2,2)
clean.plot_freq_slow_spec(result, 'xy', fig = fig, ax = ax[0,0])
clean.plot_freq_slow_spec(result, 'xy', 'original', fig = fig, ax = ax[0,1])
clean.plot_freq_slow_spec(result, 'fx', fig = fig, ax = ax[1,0])
clean.plot_freq_slow_spec(result, 'fx', 'original', fig = fig, ax = ax[1,1])
plt.tight_layout()

clean.check_output_power(result)

#%% Small array (2x2 square) test with two point sources and noise
s_list = np.arange(-4, 4, 0.25)

num_windows = 8 # the number of windows strongly affects the remaining power ratio
win_length_sec = 1
freq_bin_width = 1
Nt = num_windows * win_length_sec * 100
 
stream = clean.make_synth_stream(Nt = Nt, sx = [1, -2], sy = [1, 2], amp = [1, 1], fc = [6, 6],
                                 Nx = 2, Ny = 2, uncorrelatedNoiseAmp = 2) 
result = clean.clean(stream, verbose = True, phi = 0.1, separateFreqs = 0, win_length_sec = win_length_sec, 
                              p_value=0.1, freq_bin_width = freq_bin_width, show_plots = False,
                              sxList = s_list, syList = s_list)

#imageAdj = lambda x: np.log(x + x.max()*1e-3)
#imageAdj = lambda x:x * (x>(x.max()*0.01))
plt.close(4)
plt.figure(4)
plt.subplot(2,2,1)
clean.plot_freq_slow_spec(result, 'xy')#, imageAdj = imageAdj)
plt.subplot(2,2,2)
clean.plot_freq_slow_spec(result, 'xy', 'original')
plt.subplot(2,2,3)
clean.polar_freq_slow_spec(result, 'fh')
plt.subplot(2,2,4)
clean.polar_freq_slow_spec(result, 'fa')
plt.tight_layout()
#%% Large-N array (5x5 square) test with two point sources. 
## Notice the lower remaining power ratio vs the small array test, and the longer runtime. 
s_list = np.arange(-4, 4, 0.25)

num_windows = 8 # the number of windows strongly affects the clean power ratio
win_length_sec = 1
freq_bin_width = 1
Nt = num_windows * win_length_sec * 100
 
stream = clean.make_synth_stream(Nt = Nt, sx = [1, -2], sy = [1, 2], amp = [1,1], fc = [6, 6], 
                                        Nx = 5, Ny = 5,uncorrelatedNoiseAmp = 2) 
result = clean.clean(stream, verbose = True, phi = 0.1, separateFreqs = 0, win_length_sec = win_length_sec, 
                              p_value=0.1, freq_bin_width = freq_bin_width, show_plots = False,
                              sxList = s_list, syList = s_list)
plt.figure(5)
plt.subplot(2,2,1)
clean.plot_freq_slow_spec(result, 'xy', 'original')
plt.subplot(2,2,2)
clean.plot_freq_slow_spec(result, 'xy')
plt.subplot(2,2,3)
clean.polar_freq_slow_spec(result, 'fh')
plt.subplot(2,2,4)
clean.polar_freq_slow_spec(result, 'fa')
plt.tight_layout()

#%% Large-N style test; linear source
## goal is to see how it handles line-source (consisting of closely-spaced point sources)
## it resolves the two source regions but fails to show their dimensions
s_list = np.arange(-4, 4, 0.25)

num_windows = 8 # the number of windows strongly affects the clean power ratio
win_length_sec = 1
freq_bin_width = 1
Nt = num_windows * win_length_sec * 100
sx = np.concatenate([np.arange(2, 3, 0.05), np.arange(-3, -2, 0.05)])
stream = clean.make_synth_stream(Nt = Nt, sx = sx, amp = sx*0+1, sy = 0*sx, Nx = 4, Ny = 4, 
                                        fc = 6 + 0*sx, uncorrelatedNoiseAmp = 0) 
result = clean.clean(stream, verbose = True, phi = 0.1, separateFreqs = 0, win_length_sec = win_length_sec, 
                              p_value=0.0001, freq_bin_width = freq_bin_width, show_plots = False,
                              sxList = s_list, syList = s_list)
            
plt.figure(6)                  
plt.subplot(2,2,1)
clean.plot_freq_slow_spec(result, 'xy', 'original')
plt.subplot(2,2,2)
clean.plot_freq_slow_spec(result, 'xy')
plt.subplot(2,2,3)
clean.polar_freq_slow_spec(result, 'fh')
plt.subplot(2,2,4)
clean.polar_freq_slow_spec(result, 'fa')

plt.tight_layout()
