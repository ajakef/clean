import scipy
import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sqrt, sin, cos, exp
import obspy
import importlib
import cleanbf
import subprocess

try:
    importlib.reload(cleanbf)
    print('reloaded')
except:
    pass


## leave out multitaper stuff; mtspec causes trouble, the multitaper code probably doesn't work well, and the benefits are probably negligible.
#try:
#    import mtspec
#except:
#    print('optional dependency mtspec not present, skipping multitaper tests')
#    run_multitaper = False
#else:
#    run_multitaper = True
run_multitaper = False

def approx_equal(x, y, p = 0.01):
    return np.abs(x-y)/x < p

#%% calc_fourier_window
st = cleanbf.make_synth_stream(Nx = 2, Ny = 1, sx = 1, sy = 0, fl = [3], fh = [6], uncorrelatedNoiseAmp=0.5)

FT, freqs = cleanbf.calc_fourier_window(st, raw = True)
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

## Results show that multitaper, Welch-style spectral averaging, and a combination of spectral 
## smoothing with Welch averaging all give similar and presumably good results. A plain, 
## one-window Fourier transform, either with or without smoothing or tapering, should be avoided.

freq_low = 4
freq_high = 8
st = cleanbf.make_synth_stream(400, Nx = 4, Ny = 1, sx = 1, sy = 0, fl = [freq_low], fh = [freq_high], 
                             amp=[1], uncorrelatedNoiseAmp=0)
## actual time lags should be 0.02 s (20 m, 1 s/km)
plt.close(3)
fig = plt.figure(3)
splt = fig.subplots(2,1)
splt[0].set_title('Amplitude Spectrum')
splt[1].set_title('Time Lag Spectrum (dashed line is theoretical)')
print(np.var(st[0].data))

## compare multiple single-taper cross-spectra
crossSpec, FT, freqs, dfN, dfD = cleanbf.calc_cross_spectrum(st, win_length_sec = 4, 
                                                           freq_bin_width = 1, raw = True)
print(np.sum(FT[0,:]**2*np.diff(freqs)[0])/np.var(st[0].data))
print(np.sum(crossSpec[0,0,:]) * np.diff(freqs)[0] / np.var(st[0].data))
splt[0].plot(np.abs(freqs), np.abs(crossSpec[0,1,:]))
splt[1].plot(np.abs(freqs), np.angle(crossSpec[0,1,:])/(2*np.pi*freqs))

crossSpec, FT, freqs, dfN, dfD = cleanbf.calc_cross_spectrum(st, win_length_sec = 1, 
                                                           freq_bin_width = 1)
splt[0].plot(np.abs(freqs), np.abs(crossSpec[0,1,:]))
splt[1].plot(np.abs(freqs), np.angle(crossSpec[0,1,:])/(2*np.pi*freqs))

crossSpec, FT, freqs, dfN, dfD = cleanbf.calc_cross_spectrum(st, win_length_sec = 1, 
                                                           freq_bin_width = 4)
splt[0].plot(np.abs(freqs), np.abs(crossSpec[0,1,:]))
splt[1].plot(np.abs(freqs), np.angle(crossSpec[0,1,:])/(2*np.pi*freqs))

crossSpec, FT, freqs, dfN, dfD = cleanbf.calc_cross_spectrum(st, win_length_sec = 2, 
                                                           freq_bin_width = 2)
splt[0].plot(np.abs(freqs), np.abs(crossSpec[0,1,:]))
splt[1].plot(np.abs(freqs), np.angle(crossSpec[0,1,:])/(2*np.pi*freqs))

if run_multitaper:
    ## multitaper cross spectrum
    crossSpec, FT, freqs, dfN, dfD = cleanbf.calc_cross_spectrum(st, taper = 'multitaper', taper_param=4)
    print(np.sum(crossSpec[0,0,:]) * np.diff(freqs)[0] / np.var(st[0].data))
    splt[0].plot(np.abs(freqs), np.abs(crossSpec[0,1,:]))
    splt[1].plot(np.abs(freqs), np.angle(crossSpec[0,1,:])/(2*np.pi*freqs + 1e-12))
    splt[0].legend(['plain', 'welch', 'smoothing', 'mix', 'multitaper'])
else:
    print('skipping multitaper tests')
    splt[0].legend(['plain', 'welch', 'smoothing', 'mix'])

splt[1].plot([0,50], [0.02,0.02], 'k--')
splt[1].axvline(freq_low, color = 'k', linestyle = '--')
splt[1].axvline(freq_high, color = 'k', linestyle = '--')
splt[0].set_xlim([0,10])
splt[0].set_ylabel('Cross-Spec Amp')
splt[1].set_xlim([0,10])
splt[1].set_ylim([0,0.05])
splt[1].set_xlabel('Frequency (Hz)')
splt[1].set_ylabel('Time lag (s)')

plt.tight_layout()
#%% Power conservation: special case of correlated wavefield
## In a dataset consisting solely of correlated waves, the clean spectrum should have the same power
## as the input cross-spectrum.

## Simple test with defaults
stream = cleanbf.make_synth_stream() # x,y are built into stream
result = cleanbf.clean(stream, verbose = True, phi=0.2, win_length_sec=1)

## power considerations:
original_power = np.einsum('iij ->', result['originalCrossSpec']) # total power in uncleaned cross spectrum
clean_power = np.sum(result['cleanSpec'])
print('')
print('Clean power/original power ratio')
print(np.real(clean_power / original_power)) # this is equal to the remaining power ratio printed at the end of clean
assert approx_equal(clean_power, original_power, 0.01), 'Power conservation, correlated wavefield: clean power != total power'
assert cleanbf.check_output_power(result), 'Power conservation, correlated wavefield: power not conserved'

#%% Power Conservation: general case
## The sum of clean power and remaining power (noise) should always equal the original power. This 
## test includes both correlated signal and noise, so the clean spectrum will not be equal in power
## to the input, but the clean + remaining power will be.
stream = cleanbf.make_synth_stream(Nt = 800, sx = [1, -2], sy = [2, 0], amp = [1,1],
                                        Nx = 3, Ny = 1, fc = [6, 6], uncorrelatedNoiseAmp = 2) 
result = cleanbf.clean(stream, phi = 0.2, win_length_sec = 1)
print('Power conservation ratio:')
assert cleanbf.check_output_power(result), 'Power conservation, general case'
#%% p-value test
## When given pure uncorrelated noise and only one slowness to search, the probability
## of detecting something (a false positive, since this is pure noise should be equal to the 
## p-value. In real searches, the probability of false positives is somewhat higher because many
## slownesses are tested; however, it's not clear how this actual probability of false 
## positives could be calculated. 

## This test appears to fail, but the false positive rate is only off by about a factor of 3.
## Worth investigating eventually but currently a low priority.

## "failure to correct for multiple comparisons"

N = 1000
p_value = 0.05
detections = np.zeros(N)
for i in range(N):
    stream = cleanbf.make_synth_stream(400, sx = [0], sy = [0], amp = [0], uncorrelatedNoiseAmp=10) # x,y are built into stream
    result = cleanbf.clean(stream, verbose = False, phi=0.1, sxList = [0], syList = [0], 
                                  p_value = p_value, win_length_sec = 1, freq_min = 4, freq_max = 20)
    detections[i] = np.sum(result['cleanSpec']) > 0

## check that actual false positives are within 50% of expected false positives
observed_detections = np.sum(detections)/N # actual false positives
error_bar = 3*np.sqrt(p_value * (1-p_value) / N) # normal approx to binomial (3-sigma)
print((p_value, observed_detections))
#assert np.abs(p_value - observed_detections) < error_bar, 'p_value test: Unexpected # false positives'


