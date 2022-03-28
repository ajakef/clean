from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import math
import warnings

import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import pandas as pd # needed for reading coords df as inv
import obspy
from obspy.signal.headers import clibsignal
from obspy.signal.invsim import cosine_taper
from obspy.signal.util import next_pow_2, util_geo_km
from obspy.signal.array_analysis import *
try: 
    import mtspec
except:
    print("Library mtspec is not available, meaning that multitaper spectra are not an option")

eps = 1e-12

def make_steering_vectors(geometry, stream, freqs, xSlowness, ySlowness = 0):
    if isinstance(geometry, list):
        geometry = np.array(geometry).transpose()
    ## define the frequency axis
    fs = stream[0].stats.sampling_rate
    deltaf = freqs[1] - freqs[0]
    nlow = 1
    nf = len(freqs)
    nhigh = freqs.max() / deltaf
    
    ## code adapted from get_timeshift
    mx = np.outer(geometry[:, 0], xSlowness)
    my = np.outer(geometry[:, 1], ySlowness)
    timeShiftTable = np.require(mx[:, :, np.newaxis].repeat(len(ySlowness), axis=2) +
        my[:, np.newaxis, :].repeat(len(xSlowness), axis=1), dtype=np.float32)
    ## code adapted from array_processing
    steering_vectors = np.empty((nf, len(xSlowness), len(ySlowness), len(stream)), dtype=np.complex128)
    clibsignal.calcSteer(len(stream), len(xSlowness), len(ySlowness), nf, nlow, deltaf, timeShiftTable, steering_vectors)
    return steering_vectors

def get_stream_coordinates1(st):
    try: # try converting lat and lon from stream coordinates, if available
        geometry_array = obspy.signal.array_analysis.get_geometry(stream)[:,:2]
    except: 
        # if stream coordinates are not lat and lon, just take the x and y directly
        # obspy get_geometry will re-center the array (bad) so don't use it for xy coords
        #geometry_array = obspy.signal.array_analysis.get_geometry(stream, 'xy')[:,:2]
        x = np.array([tr.stats.coordinates['x'] for tr in st])
        y = np.array([tr.stats.coordinates['y'] for tr in st])
        geometry_array = np.vstack([x,y]).transpose()
    return geometry_array

def make_steering_vectors_local(stream, geometry_src, freqs, c = 0.340):
    # plane wave steering vector dimensions are 
    # nf, nsx, nsy, nsta
    # exp(1j * 2 * pi * f * dt)
    #geometry_array = np.array([[d for d in tr.stats.coordinates.values()] for tr in stream])
    geometry_array = get_stream_coordinates(stream)
    if isinstance(geometry_array, list):
        geometry_array = np.array(geometry_array).transpose()
    fs = stream[0].stats.sampling_rate
    deltaf = freqs[1] - freqs[0]
    #freqs = freqs[1:]
    nf = len(freqs)
    nsta = geometry_array.shape[0]
    nsrc = geometry_src.shape[0]
    ## runtime difference of 10-100x when vectorized: result is shape (nf, nsrc, 1, nsta) including placeholder for compatibility
    ff, src_xx, pp, array_xx = np.meshgrid(freqs, geometry_src[:,0], [0], geometry_array[:,0], indexing = 'ij')
    ff, src_yy, pp, array_yy = np.meshgrid(freqs, geometry_src[:,1], [0], geometry_array[:,1], indexing = 'ij')
    dd = np.sqrt((array_xx-src_xx)**2 + (array_yy-src_yy)**2)
    #steering_vectors = np.exp(-2j * np.pi * ff * dd/c) / dd
    steering_vectors = np.exp(-2j * np.pi * ff * dd/c) / dd
    #steering_vectors = np.empty((nf, nsrc, 1, nsta), dtype=np.complex128) 
    #for i in range(nf):
    #    for j in range(nsrc):
    #        for k in range(1): # placeholder
    #            for l in range(nsta):
    #                distance = np.sqrt((geometry_array[l,0] - geometry_src[j,0])**2 + (geometry_array[l,1] - geometry_src[j,1])**2)
    #                steering_vectors[i,j,k,l] = np.exp(-2j * np.pi * freqs[i] * distance / c) / distance
    return steering_vectors

def calc_fourier_window(stream, offset=0, taper = None, taper_param = 4, raw=False, prewhiten = False):
    spoint = np.zeros(len(stream))
    nsamp = len(stream[0])
    nfft = next_pow_2(nsamp)
    nlow = 1
    nhigh = nfft // 2 - 1
    nf = nhigh - nlow + 1  # include upper and lower frequency
    freqList = np.arange(nlow, nlow+nf)/(nfft * stream[0].stats.delta)
    if (taper is None) or (taper.lower() == 'tukey'):
        taper = cosine_taper(nsamp, p=0.22)
    ft = np.empty((len(stream), nf), dtype=np.complex128)
    for i, tr in enumerate(stream):
        dat = tr.data[int(spoint[i] + offset):int(spoint[i] + offset + nsamp)]
        if (type(taper) is str) and (taper.lower() == 'multitaper'):
            ft[i, :] = mtspec(dat, stream[0].stats.delta, time_bandwidth = taper_param, nfft=nfft)[nlow:nlow + nf]
        else:
            if not raw:
                dat = scipy.signal.detrend(dat) * taper
            ft[i, :] = np.fft.rfft(dat, nfft)[nlow:nlow + nf]
    ft *= np.sqrt(2*stream[0].stats.delta / nsamp) # one-sided spectrum with physical amplitude units
    if prewhiten:
        eps = np.min(np.abs(ft)[ft != 0])
        ft /= np.abs(ft) + eps # apply a tiny waterlevel to avoid divide-by-zero
    return ft, freqList

def calc_cross_spectrum(stream, offset=0, taper=None, taper_param = 4, raw=False, win_length_sec = 1, 
                        freq_bin_width = 1, prewhiten = False, plot = False, criterion_function = None):
    """
    Calculates the cross-spectrum of traces in an obspy stream.
    
    Parameters
    ----------
    stream: obspy stream containing array data (and optionally coordinates in AttribDicts)
    taper: used to taper time windows before calculating spectrum. Can be either default 'Tukey' or 
         'multitaper', or a numpy.array containing the user-defined taper.
    taper_param: unused for Tukey; refers to the time-bandwidth product for multitaper.
    raw: if True, do not apply tapers or detrending when calculating spectrum and cross-spectrum
    freq_bin_width: number of adjacent samples in spectrum contained in each frequency bin
    prewhiten: if True, prewhiten spectra before calculating cross-spectrum
    plot: if True, sum the cross-spectrum's modulus over frequency and plot as image
    criterion_function: function of (stream) that returns a square boolean np.array,
         determining whether a time window is included in the cross-spectrum calculation.
    
    Returns
    -------
    cross_spec_binned: binned cross-spectrum (indices trace 1, trace 2, freq)
    FT: list of individual trace spectra
    freqs_binned: list of frequencies corresponding to frequency indices
    dfn: numerator degrees of freedom 
    dfd: denominator degrees of freedom
    """
    if criterion_function is None:
        def criterion_function(st): return np.ones([len(st), len(st)])
    cross_spec = 0
    FT = 0 
    if (taper is not None) and (taper.lower() == 'multitaper'):
        num_windows = 1
        #n_freq = int(2**np.ceil(np.log2(len(stream[0].data)))/2)
        n_freq = int(len(stream[0].data)/2)
        cross_spec = np.zeros([len(stream), len(stream), n_freq], dtype = 'complex')
        for i in range(len(stream)):
            for j in range(i):
                mt_output = mtspec.mt_coherence(stream[0].stats.delta, stream[i].data, 
                                                stream[j].data, tbp = taper_param, kspec = 5, 
                                                nf = n_freq, p = 0.95, cohe = True, phase = True, 
                                                speci = True, specj = True, freq = True)
                df = np.diff(mt_output['freq'])[0]
                ## Hack to enforce Parseval's relation. mtspec doesn't do it.
                scale_i = np.var(stream[i])/(np.sum(mt_output['speci']) * df)
                scale_j = np.var(stream[j])/(np.sum(mt_output['specj']) * df)
                cross_spec[i,i,:] = mt_output['speci'] * scale_i
                cross_spec[j,j,:] = mt_output['specj'] * scale_j
                cross_spec[i,j,:] = (mt_output['speci'] * mt_output['specj'] * scale_i * scale_j * 
                                    mt_output['cohe'])**0.5 * np.exp(1j * np.pi/180 * mt_output['phase'])
                cross_spec[j,i,:] = cross_spec[i,j].conj()
        freqs = mt_output['freq']                
    else: # use default Tukey taper
        win_length_samp = int(np.round(win_length_sec / stream[0].stats.delta))
        data_length_samp = len(stream[0].data)
        overlap = 0.5 # for possible future use. calculating degrees of freedom assumes non-overlapping windows.
        num_windows = 1 + int(np.ceil((data_length_samp - win_length_samp) / (win_length_samp * (1 - overlap))))
        if num_windows < 1:
            raise Exception('No analysis windows for calculating cross spectrum...empty stream?')
        criterion = np.zeros([len(stream), len(stream)])
        criterion_count = np.zeros([len(stream), len(stream)])
        
        for i in range(num_windows):
            win_start = int(np.round(i * (data_length_samp - win_length_samp) / (num_windows-1 + eps)))
            tmpStream = stream.copy()
            for tr in tmpStream:
                tr.data = tr.data[win_start:(win_start + win_length_samp)]
            criterion = criterion_function(tmpStream)
            criterion_count += criterion
            tmpFT, freqs = calc_fourier_window(tmpStream, offset, taper, taper_param, raw, prewhiten)
            cross_spec += np.einsum('ik,jk,ij->ijk', tmpFT, tmpFT.conj(), criterion) # identical to _r from obspy, but reordered indices. i: station, j: station, k: freq
            FT += np.abs(tmpFT) / num_windows
        cross_spec = np.einsum('ijk,ij->ijk', cross_spec, 1/criterion_count)
    ## bin smoothing
    n_bins = len(freqs) // freq_bin_width
    cross_spec_binned = np.zeros((len(stream), len(stream), n_bins), dtype = np.complex128)
    freqs_binned = np.zeros(n_bins)
    for i in range(n_bins):
        cross_spec_binned[:,:,i] += np.mean(cross_spec[:,:,(i*freq_bin_width):((i+1)*freq_bin_width)], axis = 2)
        freqs_binned[i] = np.mean(freqs[(i*freq_bin_width):((i+1)*freq_bin_width)])
    
    ## calculate degrees of freedom
    dfn = 2 * (freq_bin_width-1 + num_windows) # so that bin width 1, having no extra freqs, adds no extra df
    dfd = (len(stream)-1) * dfn
    if plot:
        image(np.einsum('ijk->ij', np.abs(cross_spec_binned)))
    return cross_spec_binned, FT, freqs_binned, dfn, dfd



####################################################
####################################################
####################################################
####################################################
####################################################
def calc_weights(steering_vectors):
    # denominator: RSS over stations for each freq and source. 
    weights_denom = np.sqrt(np.einsum('ijkl,ijkl->ijk', (steering_vectors**1).conj(), steering_vectors**1))
    weights = steering_vectors.copy()#.conj() # freq, sx, sy, station
    for i in range(weights.shape[3]):
        weights[:,:,:,i] /= weights_denom    
    ## now, sum of squares over stations is 1 for every freq-sx-sy combo
    return weights
 
def calc_freq_slowness_spec(cross_spec, steering_vectors = None, weights = None, freq_to_use = None, mask = None):
    if weights is None:
        weights = calc_weights(steering_vectors)
    if freq_to_use is None:
        freq_to_use = np.ones(cross_spec.shape[2])
    if mask is None:
        mask = np.ones(cross_spec.shape[:2])
    return np.abs(np.einsum('k,klmi,ijk,klmj,ij->klm', freq_to_use, weights.conj(), cross_spec, weights, mask)) # n_freq x n_slow x n_slow; absolute power over f, sx,sy

def clean_step(cleanSpec, cross_spec, wB, phi, stopF, separate_freqs, verbose, sxList, syList, show_plots, cross_spec_mask):
    originalCrossSpec = cross_spec.copy()
    dropped_power = 0 * cross_spec
    nsta = cross_spec.shape[0]
    done = False
    positive_freqs = np.ones(cross_spec.shape[2]) # these get set to zero when a frequency's power starts to get negative
    count = 0
    while not done:
        count += 1
        ## calculate frequency-slowness spectrum
        #P = np.abs(np.einsum('k,klmi,ijk,klmj,ij->klm', positive_freqs, wB.conj(), cross_spec, wB, cross_spec_mask)) # n_freq x n_slow x n_slow; absolute power over f, sx,sy
        P = calc_freq_slowness_spec(cross_spec, weights = wB, freq_to_use = positive_freqs, mask = cross_spec_mask)
        if separate_freqs == 1:  ## use this code to pick best slowness AND best frequency
            f_inds,imax,jmax= np.where(P == P.max()); f_inds = f_inds[:1]; imax=imax[0]; jmax=jmax[0]; #kmax=kmax[0]
            f_inds = f_inds == np.arange(len(positive_freqs)) ## convert f_inds to bool
            Pmax = P[f_inds,imax,jmax]
            Ptotal = (np.einsum('iik->k', np.abs(cross_spec)) * positive_freqs)[f_inds]
        elif separate_freqs == 2: ## try to be smart about which frequencies to work with
            PP = np.abs(np.einsum('ijk->jk', P))
            Pmax = PP.max()
            imax,jmax = np.where(PP == Pmax); imax=imax[0]; jmax=jmax[0]
            ## pick freqs that, at imax & jmax, are 1) near their max power and 2) significant
            #breakpoint()
            f_inds = (P[:,imax,jmax] > np.einsum('iik->k', np.abs(cross_spec)) / 
                               (1 + (nsta-1)/stopF)) & (P[:,imax,jmax] > (0.99*np.amax(P, (1,2))))
            Ptotal = np.einsum('iik->', np.abs(cross_spec[:,:,f_inds]))
            Pmax = np.sum(P[f_inds, imax, jmax])
        else: ## pick best slowness over all frequencies
            PP = np.abs(np.einsum('ijk->jk', P))
            Pmax = PP.max()
            imax,jmax = np.where(PP == Pmax); imax=imax[0]; jmax=jmax[0]
            #f_inds = cross_spec[0,0,:]*0 > -1 # throwaway to mean all f indices
            f_inds = (positive_freqs == 1)
            Ptotal = np.einsum('iik->', np.abs(cross_spec[:,:,f_inds]))
            Pmax = np.sum(P[f_inds, imax, jmax])

        ## Test whether the optimum slowness is coherent enough. If yes, adjust outputs accordingly. If not, break.
        stopCondition = Ptotal/(1 + (nsta-1)/stopF)
        remainingPower = np.abs(np.einsum('iij->', cross_spec) / np.einsum('iij->', originalCrossSpec))
        if (Pmax > stopCondition) and (remainingPower > 0.001):
            done = False
            compToRemove = phi * np.einsum('i,ij,ik->jki',P[:,imax,jmax], wB[:,imax,jmax,:],
                                     wB[:,imax,jmax,:].conj())
            for i in np.where(f_inds)[0]:
                #if any([any(newCrossSpec[i,i,:] < 0) for i in range(nsta)]):
                #if any(np.diag(cross_spec[:,:,i] - compToRemove[:,:,i]) < 0):
                diags = np.diag(cross_spec[:,:,i] - compToRemove[:,:,i])
                if np.mean(diags) < np.std(diags):# allow some negativity, but not much
                    print('Dropping freq index %d due to negative power' %i) 
                    positive_freqs[i] = 0 
                    dropped_power[:,:,i] = cross_spec[:,:,i]
                    
            # this section causes problems with separate_freqs = 1...comment for now                        
            #if np.sum(compToRemove[:,:,f_inds & (positive_freqs == 1)]) == 0:
            #    breakpoint()
            if remainingPower > (1+1e-9):
                breakpoint()
            cleanSpec[f_inds, imax, jmax] += phi * P[f_inds, imax, jmax] * positive_freqs[f_inds]
            if show_plots:
                plot_slowness_spectrum(cross_spec, compToRemove, cleanSpec, originalCrossSpec, wB, sxList, syList, count)
                
            cross_spec[:,:,f_inds] -= np.einsum('ij,ijk,k->ijk', cross_spec_mask, compToRemove, positive_freqs)[:,:,f_inds]
            #cross_spec = DropNegatives(cross_spec) ## shouldn't run anymore but sends an alert if it does
            if verbose: print('sx %.3f, sy %.3f, p/stop %.3f, remaining %.3f' % (sxList[imax], syList[jmax], Pmax/stopCondition, remainingPower))
        else:
            done = True

    return {'cleanSpec':cleanSpec, 'cross_spec':cross_spec, 'remainingSpec':P, 'droppedCrossSpec':dropped_power}

def plot_slowness_spectrum(cross_spec, compToRemove, cleanSpec, originalCrossSpec, wB, sxList, syList, count):
    """
    Used to plot snapshot slowness spectra showing clean power and remaining power at successive 
    clean iterations, using the "show_plots" option in clean().
    """
    fig, ax = plt.subplots(3,1,squeeze=False)
    fig.set_tight_layout(True)
    ref_power = np.abs(np.einsum('klmi,ijk,klmj->lm', wB.conj(), originalCrossSpec, wB)).max()
    P = np.abs(np.einsum('klmi,ijk,klmj->lm', wB.conj(), cross_spec, wB)) # n_slow x n_slow; absolute power over sx,sy
    im = image(P, sxList, syList, zmin = 0, zmax = ref_power, ax = ax[0,0])
    ax[0,0].set_xlabel('x slowness (s/km)')
    ax[0,0].set_ylabel('y slowness (s/km)')
    ax[0,0].set_title('Slowness Spectrum')
    cb = fig.colorbar(im, ax=ax[0,0])
    cb.ax.set_ylabel('Power ($Pa^2$)')    
    P = np.abs(np.einsum('klmi,ijk,klmj->lm', wB.conj(), compToRemove, wB)) # n_slow x n_slow; absolute power over sx,sy
    im = image(P, sxList, syList, zmin = 0, zmax = ref_power, ax = ax[1,0])
    ax[1,0].set_xlabel('x slowness (s/km)')
    ax[1,0].set_ylabel('y slowness (s/km)')
    ax[1,0].set_title('Component to Remove')
    cb = fig.colorbar(im, ax=ax[1,0])
    cb.ax.set_ylabel('Power ($Pa^2$)')
    P = np.abs(np.einsum('ijk->jk', cleanSpec)) # f, sx, sy; sum over f
    im = image(P, sxList, syList, zmin = 0, zmax = ref_power, ax = ax[2,0])
    ax[2,0].set_xlabel('x slowness (s/km)')
    ax[2,0].set_ylabel('y slowness (s/km)')
    ax[2,0].set_title('New Clean Spectrum')
    cb = fig.colorbar(im, ax=ax[2,0])
    cb.ax.set_ylabel('Power ($Pa^2$)')
    fig.set_figheight(10)
    fig.set_figwidth(4)
    fig.tight_layout()
    fig.savefig('/home/jake/Conferences/AGU2020/poster/plot_steps/plot_step_%.0f.png' % count)
    plt.close(fig)
    if count > 3: raise Exception()
#####################################################
## Function to step through a stream and clean all time windows
def clean_loop(stream, loop_step = 1, loop_width=2, x = None, y = None, sxList = None, syList = None, phi = 0.1, p_value = 0.01, 
          win_length_sec = 1, freq_bin_width = 1, freq_min = 0, freq_max = 15, separate_freqs = 0,
          rawFT = False, taper = 'Tukey', taper_param = 4, prewhiten = False,
          show_plots = False, verbose = True):
    t1 = loop_start = stream[0].stats.starttime
    loop_end = stream[0].stats.endtime
    nt = int(1 + (loop_end - loop_start - loop_width) // loop_step)
    z = np.zeros(nt)

    #clean_output = {'t':z, 'sx':z, 'sy':z, 'f':z, 'power':z, 'cleanRatio' : z}
    dirty_output = {'t':z, 'sx':z, 'sy':z, 'power':z}
    
    #while (t1 + loop_width) <= loop_end:
    for i in range(nt):
        st = stream.slice(t1, t1 + loop_width)
        halftime = t1 + loop_width/2 - loop_start
        print('%f of %f' % (halftime, loop_end - loop_start))
        result = clean(st, verbose = False, phi = phi, separate_freqs = separate_freqs, win_length_sec = win_length_sec,
                                  freq_bin_width = freq_bin_width, freq_min = freq_min, freq_max = freq_max, 
                                  sxList = sxList, syList = syList, prewhiten = False)
        if t1 == loop_start: # allocate the output variables
            freq = result['freq']
            output = {'t':np.zeros(nt), 'sx':sxList, 'sy':syList, 'f':freq, 
                      'original_power':np.zeros(nt), 'original_semblance':np.zeros(nt),
                      'original_sx':np.zeros(nt), 'original_sy':np.zeros(nt), 
                      'clean':np.zeros((nt, len(freq), len(sxList), len(syList))), 
                      'original':np.zeros((nt, len(sxList), len(syList))),
                      'all_data':[]}
        ## populate the output variables
        output['t'][i] = halftime
        output['all_data'] = result
        output['clean'][i,:,:,:] = result['cleanSpec']
        output['original'][i,:,:] = np.einsum('ijk->jk',result['originalSpec'])
        j_sx, k_sy = np.where(output['original'][i,:,:] == output['original'][i,:,:].max())
        output['original_power'][i] = output['original'][i,j_sx,k_sy]
        output['original_semblance'][i] = calc_semblance(result).max()
        output['original_sx'][i] = sxList[j_sx]
        output['original_sy'][i] = sxList[k_sy]
        t1 += loop_step
        
 
    output['original_sh'] = np.sqrt(output['original_sx']**2+output['original_sy']**2)
    output['original_az'] = np.arctan2(output['original_sx'], output['original_sy']) * 180/np.pi
    output['clean_polar'], output['az'], output['sh'] = _polar_transform(output['clean'], sxList, syList)

    return output    
#####################################################
## Function to iteratively remove wavefield components from obspy stream
def clean(stream, x = None, y = None, sxList = None, syList = None, phi = 0.1, p_value = 0.01, 
          win_length_sec = 1, freq_bin_width = 1, freq_min = 0, freq_max = 15, separate_freqs = 0,
          rawFT = False, taper = 'Tukey', taper_param = 4, prewhiten = False, 
          show_plots = False, verbose = True, steering_vectors = None, cross_spec = None, freqList = None,
          cross_spec_mask = None, criterion_function = None):
    """
    Iteratively deconvolves array response from cross-spectrum, stopping when the F statistic drops
    below a threshold and resulting in a sparse 3-D array (frequency, x-slowness, y-slowness).

    Parameters
    ----------
    stream: obspy stream containing array data (and optionally coordinates in AttribDicts)
    x, y: coordinates of array elements, if not contained in stream (km, for consistency with obspy)
    sxList, syList: lists of x slownesses and y slownesses (sec/km, for consistency with obspy)
    phi: in each step, the proportion of energy at the optimal slowness removed from the 
         cross-spectrum. Low phi means more precision but longer runtime.
    p_value: the probability of falsely detecting correlated signal at a given slowness in random 
         noise; used to determine the threshold F statistic for stopping. Note that because many
         slownesses are tested, not just one, the actual probability of false positives is higher.
    
    nWelch: number of non-overlapping windows used to calculate cross-spectrum.
    freq_bin_width: number of adjacent samples in spectrum contained in each frequency bin
    freq_min: lowest frequency used in calculations (Hz). This is not a filter's corner frequency.

    freq_max: highest frequency used in calculations (Hz). This is not a filter's corner frequency.

    separate_freqs: method used to determine peak of frequency-slowness spectrum in each iteration.
         Only '0' is currently recommended.
         --0 (default): sum power over all frequencies
         --1: treat each frequency-slowness combination separately
         --2: try to intelligently consider only frequencies that seem coherent at the same slowness
    rawFT: if True, do not apply tapers or detrending when calculating spectrum and cross-spectrum
    taper: used to taper time windows before calculating spectrum. Can be either default 'Tukey' or 
         'multitaper', or a numpy.array containing the user-defined taper.

    taper_param: unused for Tukey; refers to the time-bandwidth product for multitaper.
    prewhiten: if True, prewhiten spectra before calculating cross-spectrum
    show_plots: if True, show a slowness spectrum plot each iteration
    verbose: if True, give verbose output
    steering_vectors: if not None, has dimensions [f,nx,ny,nsta]
    cross_spec: if the cross spectrum has already been calculated, it can be provided to save time.
    cross_spec_mask: matrix of zeros and ones indicating which station pairs should be considered. 
    By default, all station pairs are used; however, the user might want to only consider station 
    pairs within some maximum distance.

    Returns
    -------
    Dict including clean frequency-slowness spectrum, original frequency-slowness spectrum and 
    cross-spectrum, frequency-slowness spectrum and cross-spectrum of what's left after cleaning,
    and frequency/slowness axes for the outputs. cross-spectra indices are [chan_i, chan_j, freq],
    and freq-slowness spectra have indices [freq, x_slow, y_slow].

    See the demo files for examples demonstrating use.
    """
    staLoc, sxList, syList = _process_inputs(stream, x, y, sxList, syList)
    nsta = staLoc.shape[0]    

    if (cross_spec is None) or (freqList is None):
        ## calculate the cross-spectrum
        if verbose: print('Calculating cross-spectrum')
        cross_spec, FT, freqList, dfN, dfD = calc_cross_spectrum(stream, raw = rawFT, win_length_sec = win_length_sec,
                                                                freq_bin_width = freq_bin_width, prewhiten = prewhiten,
                                                                taper = taper, taper_param = taper_param, criterion_function = criterion_function)
    ## drop freqs outside the user-defined range
    w = np.where((freqList <= freq_max) & (freqList >= freq_min))[0]
    cross_spec = cross_spec[:,:,w]
    freqList = freqList[w]

    if cross_spec_mask is None:
        cross_spec_mask = np.ones((nsta, nsta))
    
    ####### In calc_fourier_window or calc_cross_spectrum, need to consider the number of freqs as Z when defining dfN and dfD
    ## calculate the stopping F value
    if separate_freqs == 0:
        dfN *= len(freqList) 
        dfD *= len(freqList)
    stopF = scipy.stats.f.ppf(1-p_value, dfN, dfD)

    ## calculate the steering vectors and weights
    if verbose: print('Calculating steering vectors and weights')
    if steering_vectors is None:
        steering_vectors = make_steering_vectors(staLoc, stream, freqList, sxList, syList)
        #steering_vectors = steering_vectors.conj() ## fix this eventually
    #steering_vectors = 1/steering_vectors

    #wB_denom = np.sqrt(np.einsum('ijkl,ijkl->ijk', steering_vectors.conj(), steering_vectors))
    #wB = steering_vectors.copy().conj() # freq, sx, sy, station
    #for i in range(nsta):
    #    wB[:,:,:,i] /= wB_denom
    wB = calc_weights(steering_vectors)
    ## save the spectra before cleaning
    originalCrossSpec = cross_spec.copy()
    originalSpec = np.abs(np.einsum('klmi,ijk,klmj->klm', wB.conj(), cross_spec, wB)) # n_freq x n_slow x n_slow; absolute power over f, sx,sy
    ## Run the cleaning loop
    if verbose:
        print('Beginning cleaning loop')
        print('[sxList[imax], syList[jmax], Pmax/stopCondition]')
    cleanSpec = np.zeros(wB.shape[:3])
    clean_output = clean_step(cleanSpec, cross_spec, wB, phi, stopF, separate_freqs, verbose, sxList, syList, show_plots, cross_spec_mask)
    cleanSpec, cross_spec, remainingSpec, droppedCrossSpec = [clean_output[i] for i in ['cleanSpec', 'cross_spec', 'remainingSpec', 'droppedCrossSpec']]

    ## Print the remaining power and original power, and return
    print('Remaining power fraction: %.3f' % np.abs(np.einsum('iij->', cross_spec) / np.einsum('iij->', originalCrossSpec)))
    return {'cleanSpec':cleanSpec, 'originalSpec':originalSpec, 'remainingSpec':remainingSpec, 'remainingCrossSpec':cross_spec, 
            'originalCrossSpec':originalCrossSpec, 'droppedCrossSpec':droppedCrossSpec, 'sx':sxList, 'sy':syList, 'freq':freqList}

#####################
def _process_inputs(stream, x = None, y = None, sxList = None, syList = None):
    """
    Validate and apply defaults to clean inputs.
    """
    if len(stream) == 0:
        raise Exception('stream has no traces')
    if any([len(stream[0].data) != len(tr.data) for tr in stream]):
        raise Exception('traces are not all the same length')
    if len(stream[0].data) == 0:
        raise Exception('traces in stream are empty')
    
    ## work out the station geometry
    if (x is None) or (y is None):
        try:
            staLoc = get_geometry(stream) # https://docs.obspy.org/packages/autogen/obspy.signal.array_analysis.get_geometry.html
        except:
            staLoc = get_geometry(stream, 'xy')
    else:
        staLoc = np.array([x,y])
    
    ## work out the slownesses to test
    defaultSlownessList = np.arange(-0.004, 0.004, 0.0001) * 1000 # sec/km, for obspy consistency
    if sxList is None: 
        sxList = defaultSlownessList
    else: 
        sxList = np.array(sxList)
    if syList is None:  
        syList = defaultSlownessList
    else: 
        syList = np.array(syList)

    return (staLoc, sxList, syList)
###########################################
def make_synth_stream(Nt = 400, dt = 0.01, x = None, y = None, dx = None, dy = None, Nx = None, 
                    Ny = None, sx = None, sy = None, amp = None, fl = None, fh = None, fc = None, 
                    uncorrelatedNoiseAmp = 0, prewhiten = True):
    r"""
    Make synthetic array data including any number of waves with defined slowness, amplitude, and bandwidth. Each wave consists of filtered white noise.
    
    :type Nt: int
    :param Nt: output trace length (samples)
    :type dt: float
    :param dt: sample interval (seconds)
    :type x: numpy.array
    :param x: x coordinates of sensors in array (km)
    :type y: numpy.array
    :param y: y coordinates of sensors in array (km)
    :type dx: float
    :param dx: spacing between sensors in x-direction if x is not provided (km)
    :type dy: float
    :param dy: spacing between sensors in y-direction if y is not provided (km)
    :type Nx: int
    :param Nx: number of sensors in x direction if x is not provided (km)
    :type Ny: int
    :param Ny: number of sensors in y direction if y is not provided (km)
    :type sx: numpy.array
    :param sx: wave slownesses in x direction (s/km)
    :type sy: numpy.array
    :param sy: wave slownesses in y direction (s/km)
    :type amp: numpy.array
    :param amp: wave amplitudes
    :type fl: numpy.array
    :param fl: low corner frequencies of waves (Hz)
    :type fh: numpy.array
    :param fh: high corner frequencies of waves (Hz). If fh is too close to fl, instability can occur
    :type fc: numpy.array
    :param fc: if fl and fh are not provided, the logarithmic "center" of each wave's 2-octave passband (Hz)
    :type uncorrelatedNoiseAmp:
    :param uncorrelatedNoiseAmp: amplitude of uncorrelated noise added to each trace
    :return: :class:`obspy.stream`
    """
    x, y = _make_default_coords(x, y, dx, dy, Nx, Ny)
    sx, sy, amp, fl, fh = _make_default_waves(sx, sy, amp, fl, fh, fc)
    Nwaves = sx.size
    t = np.arange(Nt) * dt
    overSample=10; Ntt = 2*overSample*Nt; dtt = dt/overSample
    tt = (np.arange(Ntt)-Ntt/3)*dtt # provides padding and higher time res for interp (e.g., 0-1 s becomes -0.5 - 1.5 s)
    stf_list = []
    for i in range(Nwaves):
        [b,a]=scipy.signal.butter(4, [2*fl[i]*dtt,2*fh[i]*dtt], 'bandpass')
        signal = scipy.signal.lfilter(b, a, _make_noise(Ntt, prewhiten))
        signal *= amp[i]/np.std(signal)
        stf_list.append(scipy.interpolate.interp1d(tt, signal, fill_value='extrapolate', kind='cubic'))
    stream = obspy.Stream()
    for i in range(len(x)):
        data = _make_noise(Nt, prewhiten)
        data *= uncorrelatedNoiseAmp / np.std(data)
        for j in range(Nwaves):
            data += stf_list[j](t - x[i] * sx[j] - y[i] * sy[j])
        tr = obspy.Trace(data)
        loc = obspy.core.AttribDict()
        loc['x'] = x[i]
        loc['y'] = y[i]
        loc['elevation'] = 0
        tr.stats = obspy.core.Stats({**tr.stats, 'coordinates': loc})
        tr.stats.delta = dt
        tr.stats.location = str(i)
        stream += tr
    return stream

def make_synth_stream_local(Nt = 400, dt = 0.01, x = None, y = None, dx = None, dy = None, Nx = None, 
                    Ny = None, src_xyz = None, amp = None, fl = None, fh = None, fc = None, 
                    uncorrelatedNoiseAmp = 0, prewhiten = True, c = 0.340):
    x, y = _make_default_coords(x, y, dx, dy, Nx, Ny)
    #sx, sy, amp, fl, fh = _make_default_waves(sx, sy, amp, fl, fh, fc)
    nsrc = src_xyz.shape[0]
    nsta = len(x)
    sta_xyz = np.zeros([len(x), 3])
    sta_xyz[:,0] = x; sta_xyz[:,1] = y
    t = np.arange(Nt) * dt
    overSample=10; Ntt = 2*overSample*Nt; dtt = dt/overSample
    tt = (np.arange(Ntt)-Ntt/3)*dtt # provides padding and higher time res for interp (e.g., 0-1 s becomes -0.5 - 1.5 s)
    stf_list = []
    for i in range(nsrc):
        [b,a]=scipy.signal.butter(4, [2*fl[i]*dtt,2*fh[i]*dtt], 'bandpass')
        signal = scipy.signal.lfilter(b, a, _make_noise(Ntt, prewhiten))
        signal *= amp[i]/np.std(signal)
        stf_list.append(scipy.interpolate.interp1d(tt, signal, fill_value='extrapolate', kind='cubic'))
    stream = obspy.Stream()
    for i in range(nsta):
        data = _make_noise(Nt, prewhiten)
        data *= uncorrelatedNoiseAmp / np.std(data)
        for j in range(nsrc):
            distance = np.sqrt(np.sum((src_xyz[j,:] - sta_xyz[i,:])**2)) + 1e-6 # epsilon to prevent zero division
            data += stf_list[j](t - distance/c) / distance
        tr = obspy.Trace(data)
        loc = obspy.core.AttribDict()
        loc['x'] = x[i]
        loc['y'] = y[i]
        loc['elevation'] = 0
        tr.stats = obspy.core.Stats({**tr.stats, 'coordinates': loc})
        tr.stats.delta = dt
        tr.stats.location = str(i)
        stream += tr
    return stream

def _make_noise(N, prewhiten = True):
    x = np.random.normal(0,1,N)
    if prewhiten:
        s = np.fft.fft(x)
        x = np.real(np.fft.ifft(s/(1e-50 + np.abs(s)))) # water level to prevent div-by-0
    return x

def _make_default_coords(x=None, y=None, dx=None, dy=None, Nx=None, Ny=None):
    defaultDiff = 20/1000
    if (x is not None) and (y is not None):
        assert len(x) == len(y)
    elif (x is not None):
        x = np.array(x)
        y = 0 * x
    elif (y is not None):
        y = np.array(y)
        x = 0 * y
    else: # both x and y are None; create x and y as a grid
        if dx is None:
            dx = defaultDiff
        if dy is None:
            dy = defaultDiff
        if (Nx is not None) and (Nx > 1) and (Ny is not None) and (Ny > 1):
             xtmp = (np.arange(Nx) - (Nx-1)/2) * dx
             ytmp = (np.arange(Ny) - (Ny-1)/2) * dy
             x, y = np.meshgrid(xtmp, ytmp) # needs to be reshaped; see last line
        elif (Nx is not None) and (Nx > 1):
             x = (np.arange(Nx) - (Nx-1)/2) * dx
             y = 0 * x
        elif (Ny is not None) and (Ny > 1):
             y = (np.arange(Ny) - (Ny-1)/2) * dy
             x = 0 * y
        else:
            Nx = 3
            x = (np.arange(Nx) - (Nx-1)/2) * dx
            y = 0 * x
    return np.array(x).reshape(x.size), np.array(y).reshape(y.size)

def _assign_default_none(var, default):
    if var is None or np.array(var).size == 0:
        return np.array(default)
    else:
        return np.array(var)

def _make_default_waves(sx = None, sy = None, amp = None, fl = None, fh = None, fc = None):
    assert ((fl is None) and (fh is None)) or \
            ((fl is not None) and (fh is not None) and (len(np.array(fl)) == len(np.array(fh)))), \
            'Check fl and fh'
    
    sx = _assign_default_none(sx, default = 0)
    sy = _assign_default_none(sy, default = 0)
    amp = _assign_default_none(amp, default = 1)
    fc = _assign_default_none(fc, default = 3)
    fl = _assign_default_none(fl, default = fc / 2)
    fh = _assign_default_none(fh, default = fc * 2)

    ## make all the outputs the same length
    maxLength = max([var.size for var in (sx, sy, amp, fc, fl, fh)])
    assert (sx.size == maxLength) or (sx.size == 1)
    if sx.size == 1:
        sx = sx + np.zeros(maxLength)
    assert (sy.size == maxLength) or (sy.size == 1)
    if sy.size == 1:
        sy = sy + np.zeros(maxLength)
    assert (amp.size == maxLength) or (amp.size == 1)
    if amp.size == 1:
        amp = amp + np.zeros(maxLength)
    assert (fl.size == maxLength) or (fl.size == 1)
    if fl.size == 1:
        fl = fl + np.zeros(maxLength)
    assert (fh.size == maxLength) or (fh.size == 1)
    if fh.size == 1:
        fh = fh + np.zeros(maxLength)
    return sx, sy, amp, fl, fh

def _circle(r):
    az = np.arange(361) * np.pi/180
    plt.plot(r * np.cos(az), r * np.sin(az), 'k--', linewidth = 0.5)

def _comp_to_label(comp, backazimuth):
    if comp == 'f':
        return 'Frequency (Hz)'
    elif comp == 'x' and backazimuth:
        return 'x Back Slowness (s/km)'        
    elif comp == 'x' and not backazimuth:
        return 'x Forward Slowness (s/km)'
    elif comp == 'y' and backazimuth:
        return 'y Back Slowness'
    elif comp == 'y' and not backazimuth:
        return 'y Forward Slowness (s/km)'
    elif comp == 'a' and backazimuth:
        return 'Backazimuth (degrees)'
    elif comp == 'a' and not backazimuth:
        return 'Azimuth (degrees)'
    elif comp == 'h':
        return 'Horizontal Slowness (s/km)'
    else:
        return None
    
def calc_semblance(clean_output, type = 'original'):
    total_power = np.real(np.einsum('iik->', clean_output[type.lower() + 'CrossSpec']))
    return np.einsum('ijk->jk', clean_output[type.lower() + 'Spec'] / total_power)

def plot_freq_slow_spec(clean_output, plot_comp = 'fx', type = 'clean', semblance = True,
               fRange = [0, np.Inf], sxRange = [-4, 4], 
               syRange = [-4, 4], imageAdj = None, backazimuth = True):
    """
    Plot data from the 3-D frequency-slowness spectrum as a 2-D image.
    
    Parameters
    ----------
    clean_output: dict returned by clean()
    plot_comp: a string including one or two of 'f' (frequency), 'x' (x slowness), and 'y' (y 
         slowness). When two components are provided, they are the x and y axis, and power is 
         plotted as a color; when only one is provided, it is the x axis and power is the y axis.
    type: one of 'clean', 'original', or 'remaining'
    fRange: lower and upper limits for frequency axis
    sxRange: lower and upper limits for x slowness axis
    syRange: lower and upper limits for y slowness axis
    imageAdj: optional function to change scaling of power
    backazimuth: if True, flip the sx and sy axes to show the wave's direction of origin
    
    """
    if type.lower() == 'clean':
        spec = clean_output['cleanSpec']
    elif type.lower() == 'original' or type.lower() == 'remaining':
        if semblance:
            total_power = np.real(np.einsum('iik->', clean_output[type.lower() + 'CrossSpec']))
            spec = clean_output[type.lower() + 'Spec'] / total_power
        else:
            spec = clean_output[type.lower() + 'Spec']
    else:
        raise "Invalid 'type' in plot_freq_slow_spec"
    sx = clean_output['sx']
    sy = clean_output['sy']
    f = clean_output['freq']
    if backazimuth:
        spec = np.flip(spec, [1,2]) # flip it over dimensions x and y, but not f
        sx = -np.flip(sx)
        sy = -np.flip(sy)
    wf = (f >= fRange[0]) & (f <= fRange[1])
    wsx = (sx >= sxRange[0]) & (sx <= sxRange[1])
    wsy = (sy >= syRange[0]) & (sy <= syRange[1])
    mat = np.einsum('fxy->' + plot_comp, spec[np.ix_(wf, wsx, wsy)]) # np.ix_ does multidimensional broadcasting
    if imageAdj is not None:
        mat = imageAdj(mat)
    indVars = {'f':f[wf], 'x': sx[wsx], 'y': sy[wsy]}
    if len(plot_comp) == 1:
        plt.plot(indVars[plot_comp], mat)
    if len(plot_comp) == 2:

        iv0 = indVars[plot_comp[0]]
        iv1 = indVars[plot_comp[1]]
        image(mat, iv0, iv1, zmin = 0)
        if 'f' in plot_comp:
            #aspect = 'auto'
            pass
        else:
            plt.axis('square')
        plt.xlabel(_comp_to_label(plot_comp[0], backazimuth))
        plt.ylabel(_comp_to_label(plot_comp[1], backazimuth))
    if (plot_comp == 'xy') or (plot_comp == 'yx'):
        _circle(1000/325)
        _circle(1000/350)
        _circle(1) # seismic approximation
        if(type == 'original' or type == 'remaining'):
            #plt.clim([0,1])
            cbar = plt.colorbar()
            cbar.set_label('Semblance')
    plt.title(type)
    #print('Remember plt.show()') # necessary in terminal

def polar_freq_slow_spec(clean_output, plot_comp = 'fh', type = 'clean', 
                         fRange = [-np.Inf, np.Inf], azRange = [-np.Inf, np.Inf], 
               shRange = [0, 4], imageAdj = None, backazimuth = True):
    """
    Plot data from the 3-D frequency-slowness spectrum as a 2-D image.
    
    Parameters
    ----------
    clean_output: dict returned by clean()
    plot_comp: a string including one or two of 'f' (frequency), 'h' (horizontal slowness), and 'a' 
         (azimuth). When two components are provided, they are the x and y axis, and power is 
         plotted as a color; when only one is provided, it is the x axis and power is the y axis.
    type: one of 'clean', 'original', or 'remaining'
    fRange: lower and upper limits for frequency axis
    sxRange: lower and upper limits for x slowness axis
    syRange: lower and upper limits for y slowness axis
    imageAdj: optional function to change scaling of power
    backazimuth: if True, flip the sx and sy axes to show the wave's direction of origin
    
    """
    if type.lower() == 'clean':
        spec = clean_output['cleanSpec']
    elif type.lower() == 'original':
        spec = clean_output['originalSpec']
    elif type.lower() == 'remaining':
        spec = clean_output['remainingSpec']
    else:
        raise "Invalid 'type' in plot_freq_slow_spec"
    sx = clean_output['sx']
    sy = clean_output['sy']
    f = clean_output['freq']
    if backazimuth:
        spec = np.flip(spec, [1,2]) # flip it over dimensions x and y, but not f
        sx = -np.flip(sx)
        sy = -np.flip(sy)
    spec, az, sh = _polar_transform(spec, sx, sy)
    wf = (f >= fRange[0]) & (f <= fRange[1])
    waz = (az >= azRange[0]) & (az <= azRange[1])
    wsh = (sh >= shRange[0]) & (sh <= shRange[1])
    mat = np.einsum('fah->' + plot_comp, spec[np.ix_(wf, waz, wsh)]) # np.ix_ does multidimensional broadcasting
    if imageAdj is not None:
        mat = imageAdj(mat)
    indVars = {'f':f[wf], 'a': az[waz], 'h': sh[wsh]}
    if len(plot_comp) == 1:
        plt.plot(indVars[plot_comp], mat)
        plt.xlabel(_comp_to_label(plot_comp[0], backazimuth))
        plt.ylabel('Power')
    if len(plot_comp) == 2:
        iv0 = indVars[plot_comp[0]]
        iv1 = indVars[plot_comp[1]]
        image(mat, iv0, iv1, aspect = 'auto', zmin = 0)
        plt.xlabel(_comp_to_label(plot_comp[0], backazimuth))
        plt.ylabel(_comp_to_label(plot_comp[1], backazimuth))
    plt.title(type)
    #print('Remember plt.show()') # necessary in terminal
def _az_dist(a, b):
    """
    Find angular distance between two azimuths. Result is always between -180 and 180.
    """
    return ((a - b + 180) % 360) - 180

def _polar_transform(spec, sx_list, sy_list, az_start = 0, backazimuth = False):
    az_list = np.arange(36) * 10
    az_list = ((az_list - az_start) % 360) + az_start

    r_list = np.concatenate([sx_list[sx_list >= 0], [sx_list.max() + eps]])
    if len(spec.shape) == 3:
        polar_spec = np.zeros([spec.shape[0], len(az_list), len(r_list)])
    elif len(spec.shape) == 4:
        polar_spec = np.zeros([spec.shape[0], spec.shape[1], len(az_list), len(r_list)])
    for i, sx in enumerate(sx_list):
        for j, sy in enumerate(sy_list):
            if backazimuth:
                az = np.arctan2(-sx, -sy) * 180/np.pi
            else:
                az = np.arctan2(-sx, -sy) * 180/np.pi
            r = np.sqrt(sx**2 + sy**2)
            m = np.argmin(np.abs(_az_dist(az, az_list)))
            n = np.argmin(np.abs(r_list - r))
            if len(spec.shape) == 3:
                polar_spec[:,m,n] += spec[:,i,j]
            elif len(spec.shape) == 4:
                polar_spec[:,:,m,n] += spec[:,:,i,j]
    # drop the last r (unphysical excessive slowness)
    r_list = r_list[:-1]
    if len(spec.shape) == 3:
        polar_spec = polar_spec[:,:,:-1]
    elif len(spec.shape)==4:
        polar_spec = polar_spec[:,:,:,:-1]
       
    return (polar_spec, az_list, r_list)


def check_output_power(clean_output):
    """
    Verify that power is conserved by the cleaning operation. Original power from the cross-spectrum
    diagonal should be equal to the sum of the output clean power and remaining un-clean power.
    
    Parameters
    ----------
    clean_output: dict, output of clean()

    Result: ratio of total output to input power; should be very close to 1
    """
    original_power = np.einsum('iij ->', clean_output['originalCrossSpec'])
    remaining_power = np.einsum('iij ->', clean_output['remainingCrossSpec'])
    clean_power = np.sum(clean_output['cleanSpec'])
    
    ratio = np.real((clean_power + remaining_power)/original_power)
    print(ratio)
    return np.abs(ratio - 1) < 1e-3


 
def add_inv_coords(st, inv):
    """
    For each trace tr in stream st, adds a 'coordinates' AttribDict to tr.stats using location info
    from an inventory. See application in obspy.signal.array_analysis.array_processing.

    Parameters
    ----------
    st: stream that locations should be added to
    inv: obspy.Inventory or pandas.DataFrame containing locations for all traces in st

    Returns: None, changes st in place
    """
    if type(inv) is obspy.Inventory: 
        for tr in st:
            loc = obspy.core.AttribDict(inv.get_coordinates(tr.get_id()))
            tr.stats = obspy.core.Stats({**tr.stats, 'coordinates': loc})
    elif type(inv) is pd.DataFrame:
        for tr in st:
            id = tr.get_id()
            w = np.where(inv.SN == tr.stats.station)[0][0]
            loc = obspy.core.AttribDict(
                {'latitude':inv.loc[w,'lat'],
                 'longitude':inv.loc[w,'lon'],
                 'elevation':0,
                 'local_depth':0}
                )
            tr.stats = obspy.core.Stats({**tr.stats, 'coordinates': loc})
    
def image(Z, x = None, y = None, aspect = 'equal', zmin = None, zmax = None, ax = plt, crosshairs=True):
    # Z rows are x, columns are y
    if x is None:
        x = np.arange(Z.shape[0])
    if y is None:
        y = np.arange(Z.shape[1])
    #im = ax.imshow(Z.transpose(), extent = [x[0], x[-1], y[0], y[-1]], aspect = aspect, 
    #           origin = 'lower', vmin = zmin, vmax = zmax, cmap = 'YlOrRd')
    im = ax.pcolormesh(x, y, Z.T, vmin = zmin, vmax = zmax, cmap='YlOrRd', shading = 'auto')
    if crosshairs:
        ax.hlines(0, x[0], x[-1], 'k', linewidth=0.5)
        ax.vlines(0, y[0], y[-1], 'k', linewidth=0.5)
    return im

def get_stream_coordinates(x, y = None):
    if type(x) is obspy.Stream:
        try:
            staLoc = get_geometry(x) # https://docs.obspy.org/packages/autogen/obspy.signal.array_analysis.get_geometry.html
        except:
            #staLoc = get_geometry(x, 'xy') # re-centers array (bad)
            xx = np.array([tr.stats.coordinates['x'] for tr in x])
            yy = np.array([tr.stats.coordinates['y'] for tr in x])
            staLoc = np.array([xx,yy]).transpose()
    else:
        staLoc = np.array([x,y]).transpose()
    return staLoc

def calc_station_pair_distance(x, y = None):
    staLoc = get_stream_coordinates(x, y)
    nsta = staLoc.shape[0]
    distance = np.zeros([nsta, nsta])
    for i in range(nsta):
        for j in range(nsta):
            distance[i,j] = np.sqrt((staLoc[i,0] - staLoc[j,0])**2 + (staLoc[i,1] - staLoc[j,1])**2)

    return distance


def plot_distances(st, limit, ha = 'center', va = 'bottom'):
    coords = get_stream_coordinates(st)
    distance = calc_station_pair_distance(st)
    plt.plot(coords[:,0], coords[:,1], 'ko')
    nsta = coords.shape[0]
    for i in range(nsta):
        for j in range(i):
            if distance[i,j] < limit:
                plt.plot([coords[i,0], coords[j,0]], [coords[i,1], coords[j,1]], 'r-')
    for i in range(nsta):
        plt.text(coords[i,0], coords[i,1], st[i].stats.station + '.' + st[i].stats.location, va = va, ha = ha)

    plt.axis('square')

def make_station_mask(stream, exclude = []):
    nsta = len(stream)
    mask = np.ones([nsta, nsta])
    for i, tr in enumerate(stream):
        if tr.stats.station in exclude:
            mask[i,:] = 0
            mask[:,i] = 0
    return mask
