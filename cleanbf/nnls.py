import numpy as np
import scipy.optimize
import scipy.linalg

EXACT_MAX = 6500 # largest number of slowness cells solved exactly (needs ~32*n^2 bytes); larger grids use a much slower iterative solver

from cleanbf.core import calc_cross_spectrum, make_steering_vectors, _process_inputs

def nnls_beamform(stream, x = None, y = None, sxList = None, syList = None, win_length_sec = 1,
                  freq_bin_width = 1, freq_min = 0, freq_max = 15, rawFT = False, taper = 'Tukey',
                  taper_param = 4, prewhiten = False, verbose = True, steering_vectors = None,
                  cross_spec = None, freqList = None, n_avg = None, cross_spec_mask = None,
                  criterion_function = None, threshold = 4, smoothing = 0.01, joint_freqs = True):
    """
    Fits the cross-spectrum as a sum of incoherent plane waves (one per slowness grid cell) with
    non-negative powers (non-negative least squares). Unlike clean(), this does not assume a small
    number of point sources, so it is better suited to distributed sources (e.g., line sources).

    Model: C(f) = sum_j p_j(f) v_j(f) v_j(f)^H, fit only to the off-diagonal (station-pair) elements
    of C, so that uncorrelated noise at each station does not affect the fit. Station noise is
    estimated afterward as whatever power the waves do not explain on the diagonal.

    Parameters
    ----------
    Same as clean() for stream, x, y, sxList, syList, win_length_sec, freq_bin_width, freq_min,
    freq_max, rawFT, taper, taper_param, prewhiten, verbose, steering_vectors, cross_spec, freqList,
    cross_spec_mask, and criterion_function.
    n_avg: number of independent averages in the cross-spectrum (windows + frequency bins - 1).
         Calculated automatically unless cross_spec is provided, in which case it is required if
         threshold > 0.
    threshold: detection threshold, in standard deviations of each slowness's fit statistic under
         pure uncorrelated noise. A slowness only gets power if the data support it by more than
         this. 0 disables the threshold (and then noise is fit as many weak waves).
    smoothing: ridge (Tikhonov) penalty on wave powers, relative to the typical size of a wave's
         contribution. Positive values spread power among slownesses that fit the data about equally
         well, which suits extended sources; 0 favors a sparse spectrum. In synthetic tests, 0.01 gave
         good line-source shapes while keeping point sources compact; 1 blurs point sources.
    joint_freqs: if True, first find which slownesses have significant power using all frequencies
         together (a common slowness map), then fit each frequency's power using only those
         slownesses. This uses the fact that aliases move with frequency while true slownesses
         don't, like clean() with separate_freqs = 0. If False, fit each frequency independently.

    Returns
    -------
    Dict with the same keys as clean(), so the same plotting functions work. 'cleanSpec' holds the
    fitted power of each slowness (in the same units as clean(), i.e., its contribution to the trace
    of the cross-spectrum). 'remainingCrossSpec' is the cross-spectrum minus the fitted waves.
    Additional keys: 'noiseSpec' (diagonal power not explained by waves, indices [freq, station];
    can be slightly negative), 'misfit' (relative off-diagonal misfit at each frequency), and
    'support' (boolean [sx, sy] map of slownesses allowed in the per-frequency fits).

    Memory use scales as (number of slowness cells)^2: about 35 MB for a 32x32 grid and 1.3 GB for
    80x80.
    """
    staLoc, sxList, syList = _process_inputs(stream, x, y, sxList, syList)
    nsta = staLoc.shape[0]

    if (cross_spec is None) or (freqList is None):
        ## calculate the cross-spectrum
        if verbose: print('Calculating cross-spectrum')
        cross_spec, FT, freqList, dfN, dfD = calc_cross_spectrum(stream, raw = rawFT, win_length_sec = win_length_sec,
                                                                freq_bin_width = freq_bin_width, prewhiten = prewhiten,
                                                                taper = taper, taper_param = taper_param, criterion_function = criterion_function)
        n_avg = dfN / 2
    elif (n_avg is None) and (threshold > 0):
        raise ValueError('n_avg must be provided along with cross_spec when threshold > 0')
    ## drop freqs outside the user-defined range
    w = np.where((freqList <= freq_max) & (freqList >= freq_min))[0]
    cross_spec = cross_spec[:,:,w]
    freqList = freqList[w]

    if cross_spec_mask is None:
        cross_spec_mask = np.ones((nsta, nsta))
    if steering_vectors is None:
        steering_vectors = make_steering_vectors(stream, freqList, sxList, syList)
    nf, nsx, nsy = steering_vectors.shape[:3]
    nsrc = nsx * nsy
    V = steering_vectors.reshape(nf, nsrc, nsta)
    scale = np.mean(np.einsum('iik->k', cross_spec).real) / nsta # keeps the numbers near 1
    C = cross_spec / scale

    ## standard deviation of each source's fit statistic (A^T b) under pure noise, for the threshold
    sd = np.zeros((nf, nsrc))
    if threshold > 0:
        for k in range(nf):
            sd[k] = _null_sd(C[:,:,k], V[k], cross_spec_mask, n_avg)

    if joint_freqs:
        ## Stage 1: common slowness map over all frequencies, with threshold -> support.
        ## Sum the normal equations (G = sum A_f^T A_f, h = sum A_f^T b_f) so memory doesn't grow with nf.
        if verbose: print('Finding significant slownesses using all frequencies')
        G, h = 0, 0
        for k in range(nf):
            A, b = _design_matrix(C[:,:,k], V[k], cross_spec_mask)
            G = G + A.T @ A
            h = h + A.T @ b
        c = 2 * threshold * np.sqrt(np.sum(sd**2, axis = 0))
        p_joint = _solve_qp(G, h, c, smoothing * np.mean(np.diag(G)))
        support = p_joint > 0
    else:
        support = np.ones(nsrc, dtype = bool)

    ## Stage 2: fit each frequency's wave powers (only within the support if joint_freqs)
    P = np.zeros((nf, nsrc))
    misfit = np.zeros(nf)
    S = np.where(support)[0]
    for k in range(nf):
        if verbose: print('Fitting frequency %d of %d (%.2f Hz)' % (k+1, nf, freqList[k]))
        A, b = _design_matrix(C[:,:,k], V[k][S], cross_spec_mask)
        if len(S) > 0:
            Gk = A.T @ A
            c = 0 if joint_freqs else 2 * threshold * sd[k] # with joint_freqs, significance was already established in stage 1
            P[k, S] = _solve_qp(Gk, A.T @ b, c, smoothing * np.mean(np.diag(Gk)))
            misfit[k] = np.linalg.norm(A @ P[k, S] - b) / np.linalg.norm(b) if np.any(b) else 0
        else:
            misfit[k] = 1

    P *= scale
    modelCrossSpec = np.einsum('kj,kji,kjl->ilk', P, V, V.conj())
    cleanSpec = (P * np.einsum('kji,kji->kj', V, V.conj()).real).reshape(nf, nsx, nsy) # contribution to trace
    remainingCrossSpec = cross_spec - modelCrossSpec
    noiseSpec = np.einsum('iik->ki', remainingCrossSpec).real
    weights = steering_vectors / np.linalg.norm(steering_vectors, axis = 3, keepdims = True) # same as calc_weights
    originalSpec = np.abs(np.einsum('klmi,ijk,klmj->klm', weights.conj(), cross_spec, weights))
    remainingSpec = np.abs(np.einsum('klmi,ijk,klmj->klm', weights.conj(), remainingCrossSpec, weights))
    if verbose:
        total = np.einsum('iik->', cross_spec).real
        print('Fitted wave power fraction: %.3f; unexplained station power fraction: %.3f' % (cleanSpec.sum()/total, noiseSpec.sum()/total))
    return {'cleanSpec':cleanSpec, 'originalSpec':originalSpec, 'remainingSpec':remainingSpec,
            'remainingCrossSpec':remainingCrossSpec, 'originalCrossSpec':cross_spec, 'noiseSpec':noiseSpec,
            'misfit':misfit, 'support':support.reshape(nsx, nsy), 'sx':sxList, 'sy':syList, 'freq':freqList}

def _design_matrix(C, v, mask):
    """
    Real-valued least-squares system for one frequency, using only off-diagonal elements of the
    [nsta, nsta] cross-spectrum C. v is [nsrc, nsta] steering vectors. Each complex pair appears twice
    in C (i,l and l,i), so it is weighted by sqrt(2) to make the misfit equal to the Frobenius norm of
    the off-diagonal residual.
    """
    iu, ju = np.triu_indices(C.shape[0], 1)
    w = np.sqrt(2) * mask[iu, ju]
    pair = v[:, iu] * v[:, ju].conj() # nsrc x npair
    A = np.vstack([(pair.real * w).T, (pair.imag * w).T])
    b = np.concatenate([C[iu, ju].real * w, C[iu, ju].imag * w])
    return A, b

def _null_sd(C, v, mask, n_avg):
    """
    Standard deviation of A^T b for each source column if the data were pure uncorrelated noise with
    station powers equal to diag(C). Each off-diagonal sample cross-spectrum element then has
    E|C_il|^2 = s_i s_l / n_avg.
    """
    iu, ju = np.triu_indices(C.shape[0], 1)
    s = np.diag(C).real
    var_pair = 2 * mask[iu, ju]**4 * s[iu] * s[ju] / n_avg
    return np.sqrt(np.abs(v[:, iu])**2 * np.abs(v[:, ju])**2 @ var_pair)

def _solve_qp(G, h, c, lam):
    """
    Minimizes p^T (G + lam I) p - 2 h^T p + c^T p subject to p >= 0. This is non-negative least
    squares in normal-equation form, with a ridge penalty lam and linear (L1, since p >= 0) penalty c.
    """
    n = len(h)
    lam = max(lam, 1e-6 * np.mean(np.diag(G))) # keeps the Cholesky factorization stable
    if n <= EXACT_MAX: # exact: rewrite as min ||L^T p - t||^2 with G + lam I = L L^T, and use NNLS
        L = scipy.linalg.cholesky(G + lam * np.eye(n), lower = True)
        t = scipy.linalg.solve_triangular(L, h - c/2, lower = True)
        p, _ = scipy.optimize.nnls(L.T, t, maxiter = 50 * n)
        return p
    ## iterative (FISTA, accelerated projected gradient); avoids a second n x n matrix
    Lip = _max_eig(G) + lam
    p = np.zeros(n); y = p.copy(); t = 1
    for i in range(20000):
        p_new = np.maximum(y - ((G @ y + lam * y) - h + c/2) / Lip, 0)
        t_new = (1 + np.sqrt(1 + 4*t**2)) / 2
        y = p_new + ((t - 1) / t_new) * (p_new - p)
        if np.linalg.norm(p_new - p) <= 1e-6 * max(np.linalg.norm(p_new), 1e-30):
            return p_new
        p, t = p_new, t_new
    return p

def _max_eig(G, n_iter = 100):
    x = np.random.default_rng(0).standard_normal(G.shape[0])
    for i in range(n_iter):
        x = G @ x; x /= np.linalg.norm(x)
    return x @ G @ x
