import numpy as np
from scipy import linalg
import mne
from mne.datasets import sample
from mne.viz import plot_sparse_source_estimates
import matplotlib.pyplot as plt
data_path = sample.data_path()
fwd_fname = data_path + '/MEG/sample/sample_audvis-meg-eeg-oct-6-fwd.fif'
ave_fname = data_path + '/MEG/sample/sample_audvis-ave.fif'
cov_fname = data_path + '/MEG/sample/sample_audvis-shrunk-cov.fif'
subjects_dir = data_path + '/subjects'
condition = 'Left Auditory'

# Read noise covariance matrix
noise_cov = mne.read_cov(cov_fname)
# Handling average file
evoked = mne.read_evokeds(ave_fname, condition=condition, baseline=(None, 0))
evoked.crop(tmin=0.00, tmax=None)

evoked = evoked.pick_types(eeg=True,meg=False)
# Handling forward solution
forward = mne.read_forward_solution(fwd_fname)


# %%

def apply_solver(solver, evoked, forward, noise_cov, loose=0.2, depth=0.8):
    """Call a custom solver on evoked data.

    This function does all the necessary computation:

    - to select the channels in the forward given the available ones in
      the data
    - to take into account the noise covariance and do the spatial whitening
    - to apply loose orientation constraint as MNE solvers
    - to apply a weigthing of the columns of the forward operator as in the
      weighted Minimum Norm formulation in order to limit the problem
      of depth bias.

    Parameters
    ----------
    solver : callable
        The solver takes 3 parameters: data M, gain matrix G, number of
        dipoles orientations per location (1 or 3). A solver shall return
        2 variables: X which contains the time series of the active dipoles
        and an active set which is a boolean mask to specify what dipoles are
        present in X.
    evoked : instance of mne.Evoked
        The evoked data
    forward : instance of Forward
        The forward solution.
    noise_cov : instance of Covariance
        The noise covariance.
    loose : float in [0, 1] | 'auto'
        Value that weights the source variances of the dipole components
        that are parallel (tangential) to the cortical surface. If loose
        is 0 then the solution is computed with fixed orientation.
        If loose is 1, it corresponds to free orientations.
        The default value ('auto') is set to 0.2 for surface-oriented source
        space and set to 1.0 for volumic or discrete source space.
    depth : None | float in [0, 1]
        Depth weighting coefficients. If None, no depth weighting is performed.

    Returns
    -------
    stc : instance of SourceEstimate
        The source estimates.
    """
    # Import the necessary private functions
    from mne.inverse_sparse.mxne_inverse import \
        (_prepare_gain, is_fixed_orient,
         _reapply_source_weighting, _make_sparse_stc)

    all_ch_names = evoked.ch_names

    # Handle depth weighting and whitening (here is no weights)
    forward, gain, gain_info, whitener, source_weighting, mask = _prepare_gain(
        forward, evoked.info, noise_cov, pca=False, depth=depth,
        loose=loose, weights=None, weights_min=None, rank=None)

    # Select channels of interest
    sel = [all_ch_names.index(name) for name in gain_info['ch_names']]
    M = evoked.data[sel]

    # Whiten data
    M = np.dot(whitener, M)
    n_orient = 1 if is_fixed_orient(forward) else 3
    X, active_set,K = solver(M, gain, n_orient)
    Xf = X
    X = _reapply_source_weighting(X, source_weighting, active_set)

    stc = _make_sparse_stc(X, active_set, forward, tmin=evoked.times[0],
                           tstep=1. / evoked.info['sfreq'])

    return stc, whitener, gain ,K,M, X, active_set, forward,Xf


def solver(M, G, n_orient):
    """Run L2 penalized regression and keep 10 strongest locations.

    Parameters
    ----------
    M : array, shape (n_channels, n_times)
        The whitened data.
    G : array, shape (n_channels, n_dipoles)
        The gain matrix a.k.a. the forward operator. The number of locations
        is n_dipoles / n_orient. n_orient will be 1 for a fixed orientation
        constraint or 3 when using a free orientation model.
    n_orient : int
        Can be 1 or 3 depending if one works with fixed or free orientations.
        If n_orient is 3, then ``G[:, 2::3]`` corresponds to the dipoles that
        are normal to the cortex.

    Returns
    -------
    X : array, (n_active_dipoles, n_times)
        The time series of the dipoles in the active set.
    active_set : array (n_dipoles)
        Array of bool. Entry j is True if dipole j is in the active set.
        We have ``X_full[active_set] == X`` where X_full is the full X matrix
        such that ``M = G X_full``.
    """
    inner = np.dot(G, G.T)
    trace = np.trace(inner)
    K = linalg.solve(inner + trace * 4e-6 * np.eye(G.shape[0]), G).T
    K /= np.linalg.norm(K, axis=1)[:, None]
    X = np.dot(K, M)

    indices = np.argsort(np.sum(X ** 2, axis=1))[:]
    active_set = np.zeros(G.shape[1], dtype=bool)
    for idx in indices:
        idx -= idx % n_orient
        active_set[idx:idx + n_orient] = True
    #X = X[active_set]
    return X, active_set, K


# %%
    # loose, depth = 0.2, 0.8  # corresponds to loose orientation
loose, depth = 0., 0  # corresponds to free orientation
stc,whitener, gain,K,whitened_data, X,active_set,forward, Xf = apply_solver(solver, evoked, forward, noise_cov, loose, depth)

# %%
time_series = np.zeros(evoked.data[0,:].shape)
for i in range(time_series.shape[0]):
    2+2
fig, axs = plt.subplots(2)
axs[0].plot(time_series)
axs[1].plot(np.dot(whitener,evoked.data)[0,:])


# %%
from scipy.optimize import nnls 
x,rnorm = nnls(G,np.dot(whitener,evoked.data)[:,0])



# %%
time_series = np.zeros(evoked.data[0,:].shape)
for i in range(time_series.shape[0]):
    x,rnorm = nnls(G,np.dot(whitener,evoked.data)[:,i])
    time_series[i] = np.dot(G,x)[0]

# %%
fig, axs = plt.subplots(3)
axs[0].plot(time_series)
axs[1].plot(np.dot(whitener,evoked.data)[0,:])
axs[2].plot(M[0,:])
# %%
fig, axs = plt.subplots(3)
axs[0].plot(( G @ Xf )[0,:])
axs[1].plot((G @ K @ M[:,:])[0,:])
axs[2].plot(M[0,:])