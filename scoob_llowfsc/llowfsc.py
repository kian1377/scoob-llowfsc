from .math_module import xp, xcipy, ensure_np_array
import scoob_llowfsc.utils as utils
from scoob_llowfsc.imshows import imshow1, imshow2, imshow3

import numpy as np
import astropy.units as u
import copy
from IPython.display import display, clear_output
import time

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

def inject_wfe(wfe_time_series, wfe_modes, freq, wfe_channel, offset=75e-6):
    Nsamps = wfe_time_series.shape[1]
    tsleep = 1/freq - offset
    try:
        print('Injecting WFE ...')
        i = 0
        while i<Nsamps+1:
            if i==Nsamps: i = 0
            wfe = np.sum( wfe_time_series[:, i, None, None] * wfe_modes, axis=0)
            wfe_channel.write(1e6 * wfe)
            time.sleep(tsleep)
            i += 1
    except KeyboardInterrupt:
        print('Stopped injecting WFE.')
        wfe_channel.write(np.zeros(wfe_channel.shape))

def calibrate_without_fsm(I, control_mask, dm_modes, amps=5e-9, plot=False):
    # time.sleep(2)
    Nmask = int(control_mask.sum())
    Nmodes = dm_modes.shape[0]
    if np.isscalar(amps):
        amps = [amps] * Nmodes

    if isinstance(control_mask, np.ndarray):
        responses = np.zeros((Nmodes, Nmask))
    else:
        responses = xp.zeros((Nmodes, Nmask))
    
    start = time.time()
    for i in range(Nmodes):
        amp = amps[i]
        mode = dm_modes[i]

        I.add_dm(amp*mode)
        im_pos = I.snap_locam()
        I.add_dm(-2*amp*mode)
        im_neg = I.snap_locam()
        I.add_dm(amp*mode)

        diff = im_pos - im_neg
        responses[i] = copy.copy(diff)[control_mask]/(2 * amp)
        
        if plot:
            imshow3(amp*mode, im_pos, diff, f'Mode {i+1}', 'Absolute Image', 'Difference', cmap1='viridis')
        
        print(f"\tCalibrated mode {i+1:d}/{dm_modes.shape[0]:d} in {time.time()-start:.3f}s", end='')
        print("\r", end="")

    response_matrix = responses.T

    return response_matrix

def update_locam_delta(response_matrix, modal_matrix, control_mask, dh_channel, locam_delta_channel,):
    del_ref_im = np.zeros(locam_delta_channel.shape)
    del_ref_im[control_mask] = response_matrix.dot(modal_matrix.dot(1e-6*dh_channel.grab_latest().ravel())/1024)
    locam_delta_channel.write(del_ref_im)
    return

def single_iteration(
    I,
    camlo_channel,
    camlo_ref_channel,
    camlo_delta_channel,  
    gain_channel, 
    control_matrix, 
    modal_matrix,
    control_mask, 
    leakage=0.0, 
    plot=False,
    clear=False,
    ):

    image = camlo_channel.grab_latest()
    del_im = image - (camlo_ref_channel.grab_latest() + camlo_delta_channel.grab_latest())

    # compute the DM command with the image based on the time delayed wavefront
    modal_coeff = -control_matrix.dot( del_im[control_mask] )
    modal_coeff *= gain_channel.grab_latest()[0]
    del_dm_command = modal_matrix.T.dot(modal_coeff).reshape(I.Nact,I.Nact)

    total_command = (1-leakage)*I.get_dm() + del_dm_command
    I.set_dm(total_command)

    if plot:
        dm_command = I.get_dm()
        pv_stroke = xp.max(dm_command) - xp.min(dm_command)
        rms_stroke = xp.sqrt(xp.mean(xp.square(dm_command[I.dm_mask])))
        imshow3(
            del_im, del_dm_command, dm_command, 
            'Measured Difference Image', 
            'Computed DM Correction',
            f'PV Stroke = {1e9*pv_stroke:.1f}nm\nRMS Stroke = {1e9*rms_stroke:.1f}nm', 
            cmap1='magma', cmap2='viridis', cmap3='viridis',
        )
        if clear: clear_output(wait=True)





