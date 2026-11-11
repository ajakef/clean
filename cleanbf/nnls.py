import numpy as np
from scipy.optimize import nnls

from cleanbf.core import (
    _process_inputs,
    calc_cross_spectrum,
    make_steering_vectors,
)


def nnls_beamform(stream, x=None, y=None, sxList=None, syList=None,
                  win_length_sec=1, freq_bin_width=1, freq_min=0,
                  freq_max=15, rawFT=False, taper='Tukey', taper_param=4,
                  prewhiten=False, smoothing=0, verbose=True):
    """Estimate nonnegative plane-wave power from the array cross-spectrum."""
    if smoothing < 0:
        raise ValueError('smoothing must be nonnegative')

    _, sxList, syList = _process_inputs(stream, x, y, sxList, syList)
    if verbose:
        print('Calculating cross-spectrum')
    cross_spec, _, freqList, _, _ = calc_cross_spectrum(
        stream,
        raw=rawFT,
        win_length_sec=win_length_sec,
        freq_bin_width=freq_bin_width,
        taper=taper,
        taper_param=taper_param,
        prewhiten=prewhiten,
    )

    selected = (freqList >= freq_min) & (freqList <= freq_max)
    cross_spec = cross_spec[:, :, selected]
    freqList = freqList[selected]
    steering = make_steering_vectors(stream, freqList, sxList, syList)
    steering /= np.sqrt(np.sum(np.abs(steering) ** 2, axis=-1, keepdims=True))

    nfreq, nsx, nsy, nsta = steering.shape
    cleanSpec = np.zeros((nfreq, nsx, nsy))
    remainingCrossSpec = cross_spec.copy()
    originalSpec = np.empty((nfreq, nsx, nsy))
    remainingSpec = np.empty_like(cleanSpec)
    n_candidates = 1

    for k in range(nfreq):
        covariance = cross_spec[:, :, k]
        vectors = steering[k].reshape(-1, nsta)
        beam_power = np.einsum(
            'ki,ij,kj->k', vectors.conj(), covariance, vectors
        ).real
        originalSpec[k] = np.maximum(beam_power, 0).reshape(nsx, nsy)
        remainingSpec[k] = originalSpec[k]

        total_power = np.trace(covariance).real
        if total_power <= 0 or beam_power.max() < 0.5 * total_power:
            continue

        candidates = np.argsort(beam_power)[-n_candidates:]
        vectors = vectors[candidates]
        atoms = np.einsum('ki,kj->kij', vectors, vectors.conj())
        design = atoms.reshape(n_candidates, -1).T
        target = covariance.reshape(-1)
        design = np.concatenate((design.real, design.imag), axis=0)
        target = np.concatenate((target.real, target.imag))
        if smoothing:
            design = np.vstack((design, np.sqrt(smoothing) * np.eye(n_candidates)))
            target = np.concatenate((target, np.zeros(n_candidates)))

        power, _ = nnls(design, target)
        cleanSpec[k].reshape(-1)[candidates] = power
        remainingCrossSpec[:, :, k] -= np.einsum(
            'k,kij->ij', power, atoms
        )
        remainingSpec[k] = np.maximum(
            np.einsum(
                'ki,ij,kj->k',
                steering[k].reshape(-1, nsta).conj(),
                remainingCrossSpec[:, :, k],
                steering[k].reshape(-1, nsta),
            ).real,
            0,
        ).reshape(nsx, nsy)

    return {
        'cleanSpec': cleanSpec,
        'originalSpec': originalSpec,
        'remainingSpec': remainingSpec,
        'remainingCrossSpec': remainingCrossSpec,
        'originalCrossSpec': cross_spec,
        'sx': sxList,
        'sy': syList,
        'freq': freqList,
    }
