#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import scipy
import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sqrt, sin, cos, exp
import obspy
import importlib
import mtspec

try:
    importlib.reload(clean)
    print('reloaded')
except:
    import clean

def approx_equal(x, y, p = 0.01):
    return np.abs(x-y)/x < p

#%% calc_fourier_window
st = clean.make_synth_stream(Nx = 2, Ny = 1, sx = 1, sy = 0, fl = [3], fh = [6], uncorrelatedNoiseAmp=0.5)

FT, freqs = clean.calc_fourier_window(st, raw = True)
plt.plot(np.abs(freqs), np.abs(FT[0,:])**2)

## Parseval's relation check for amplitude spectrum
power_time = np.var(st[0].data)
power_freq = np.sum(np.abs(FT[0,:])**2) * np.diff(freqs)[0]
print( (power_time, power_freq) )
assert approx_equal(power_time, power_freq), 'calc_fourier_window: Parseval\'s relation failed'

#%% calc_cross_spectrum
## Test different ways of estimating the cross spectrum. An on-diagonal component is shown here.
## The methods are scaled so that they satisfy Parseval's relation, given their different df. This
## also means they look about the same on a plot when the spectra are continuous (as opposed to spikes).
st = clean.make_synth_stream(1024, Nx = 4, Ny = 1, sx = 1, sy = 0, fl = [3], fh = [6], 
                                    amp=[1], uncorrelatedNoiseAmp=0)
## actual time lags should be 0.02 s (20 m, 1 s/km)
#st.plot()
plt.close(3)
fig = plt.figure(3)
splt = fig.subplots(2,1)
print(np.var(st[0].data))

## compare multiple single-taper cross-spectra
crossSpec, FT, freqs, dfN, dfD = clean.calc_cross_spectrum(st, nWelch = 1, freq_bin_width = 1, raw = True)
print(np.sum(FT[0,:]**2*np.diff(freqs)[0])/np.var(st[0].data))
print(np.sum(crossSpec[0,0,:]) * np.diff(freqs)[0] / np.var(st[0].data))
splt[0].plot(np.abs(freqs), np.abs(crossSpec[0,1,:]))
splt[1].plot(np.abs(freqs), np.angle(crossSpec[0,1,:])/(2*np.pi*freqs))
crossSpec, FT, freqs, dfN, dfD = clean.calc_cross_spectrum(st, nWelch = 4, freq_bin_width = 1)
splt[0].plot(np.abs(freqs), np.abs(crossSpec[0,1,:]))
splt[1].plot(np.abs(freqs), np.angle(crossSpec[0,1,:])/(2*np.pi*freqs))
crossSpec, FT, freqs, dfN, dfD = clean.calc_cross_spectrum(st, nWelch = 1, freq_bin_width = 4)
splt[0].plot(np.abs(freqs), np.abs(crossSpec[0,1,:]))
splt[1].plot(np.abs(freqs), np.angle(crossSpec[0,1,:])/(2*np.pi*freqs))
crossSpec, FT, freqs, dfN, dfD = clean.calc_cross_spectrum(st, nWelch = 2, freq_bin_width = 2)
splt[0].plot(np.abs(freqs), np.abs(crossSpec[0,1,:]))
splt[1].plot(np.abs(freqs), np.angle(crossSpec[0,1,:])/(2*np.pi*freqs))

## multitaper cross spectrum
#st.trim(st[0].stats.starttime + 1 + 0, st[0].stats.starttime + 1+2.555)
crossSpec, FT, freqs, dfN, dfD = clean.calc_cross_spectrum(st, taper = 'multitaper', taper_param=4)
#print(np.sum(FT[0,:]**2*np.diff(freqs)[0])/np.var(st[0].data))
print(np.sum(crossSpec[0,0,:]) * np.diff(freqs)[0] / np.var(st[0].data))
splt[0].plot(np.abs(freqs), np.abs(crossSpec[0,0,:]))
splt[1].plot(np.abs(freqs), np.angle(crossSpec[0,1,:])/(2*np.pi*freqs))
#splt[1].plot([0,50], [0,2*np.pi*50*0.02], 'k--')
splt[1].plot([0,50], [0.02,0.02], 'k--')
splt[0].set_xlim([0,10])
splt[1].set_xlim([0,10])
splt[0].legend(['plain', 'welch', 'smoothing', 'mix', 'multitaper'])

#%% Power conservation: special case of correlated wavefield
## In a dataset consisting solely of correlated waves, the clean spectrum should have the same power
## as the input cross-spectrum.

## Simple test with defaults (1 wave, 2.56 sec, sx=sy=0, signal freq 1-4, no noise)
stream = clean.make_synth_stream() # x,y are built into stream
result = clean.clean(stream, verbose = True, phi=0.2, nWelch = 4)

## power considerations:
original_power = np.einsum('iij ->', result['originalCrossSpec']) # total power in uncleaned cross spectrum
clean_power = np.sum(result['cleanSpec'])
print('')
print('Clean power/original power ratio')
print(np.real(clean_power / original_power)) # this is equal to the remaining power ratio printed at the end of clean
assert approx_equal(clean_power, original_power, 0.01), 'Power conservation, correlated wavefield: clean power != total power'
assert clean.check_output_power(result), 'Power conservation, correlated wavefield: power not conserved'

#%% Power Conservation: general case
## The sum of clean power and remaining power (noise) should always equal the original power. This 
## test includes both correlated signal and noise, so the clean spectrum will not be equal in power
## to the input, but the clean + remaining power will be.
nWelch = 8
freq_bin_width = 1
stream = clean.make_synth_stream(Nt = nWelch * freq_bin_width * 64, sx = [1, -2], sy = [2, 0], 
                                        Nx = 3, Ny = 1, fc = [6, 6], uncorrelatedNoiseAmp = 2) 
result = clean.clean(stream, verbose = True, phi = 0.2, separateFreqs = 0, nWelch = nWelch, 
                              p_value=0.9, freq_bin_width = freq_bin_width)
print('Power conservation ratio:')
assert clean.check_output_power(result), 'Power conservation, general case'
#%% p-value test
## When given pure uncorrelated noise and only one slowness to search, the probability
## of detecting something (a false positive, since this is pure noise should be equal to the 
## p-value. In real searches, the probability of false positives is somewhat higher because many
## slownesses are tested; however, it's not clear how this actual probability of false 
## positives could be calculated. That's basically what we see here.
N = 1000
p_value = 0.05
detections = np.zeros(N)
for i in range(N):
    stream = clean.make_synth_stream(1024, sx = [0], sy = [0], amp = [0], uncorrelatedNoiseAmp=10) # x,y are built into stream
    result = clean.clean(stream, verbose = False, phi=0.1, sxList = [0], syList = [0], 
                                  separateFreqs = 0, p_value = p_value, nWelch = 1, freq_max = 50)
    detections[i] = np.sum(result['cleanSpec']) > 0

## check that actual false positives are within 50% of expected false positives
observed_detections = np.sum(detections)/N # actual false positives
error_bar = 3*np.sqrt(p_value * (1-p_value) / N) # normal approx to binomial (3-sigma)
assert np.abs(p_value - observed_detections) < error_bar, 'p_value test: Unexpected # false positives'

print((p_value, observed_detections))

#0.65, 256: 0.112, 0.084, 0.09
#0.61, 1024: 0.024, 0.024


#0.7, 64: 0.012, 0.008
#0.6, 256: 0.086, 0.098, 0.098 .102
#0.58, 1024: 0.072, 0.096, 0.068, 0.092
#%%
confidence = 0.58
crossSpec, FT, freqList, dfN, dfD = clean.calc_cross_spectrum(stream, raw = None, nWelch = 1, freq_bin_width = 1)
stopF = scipy.stats.f.ppf(0.92, dfN*511, dfD*511) # more df means lower critical F

print(stopF) 

## in the test above, using only ONE slowness candidate, we know that a stopF of 1.68 is being used as a threshold, and it is getting
## fewer hits than the confidence level suggests (should be 0.3, actually is 0.01). So, stopF is 
## too high because the df is underestimated. If they're underestimated by the same ratio (likely), it 
## looks like a factor of 28, for a 64-sample series.

## 256 samples: stopF is currently 1.16, resulting in hit rate of .1, and as predicted, the df ratio 
## that results in that f for confidence 0.9 is approximately 127 (the number of freq bins added together).

## 1024 samples: current stopF is 1.086, resulting in hit rate of ~0.08 instead of 0.42. Scaling by 
## 511 (the number of freq bins added together) approximates this value, meaning confidence 0.919.
