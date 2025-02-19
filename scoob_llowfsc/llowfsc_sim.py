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

def calibrate_without_fsm(
        M, 
        dm_modes, 
        control_mask, 
        channel=2,
        amps=3e-9, 
        plot=False,
    ):
    
    Nmask = int(control_mask.sum())
    Nmodes = dm_modes.shape[0]
    if np.isscalar(amps):
        amps = [amps] * Nmodes

    responses = xp.zeros((Nmodes, Nmask))
    response_cube = xp.zeros((Nmodes, M.ncamlo, M.ncamlo))
    
    start = time.time()
    for i in range(Nmodes):
        amp = amps[i]
        mode = dm_modes[i]

        M.add_dm(amp*mode, channel=channel)
        im_pos = M.snap_camlo()
        M.add_dm(-2*amp*mode, channel=channel)
        im_neg = M.snap_camlo()
        M.add_dm(amp*mode, channel=channel)

        diff = im_pos - im_neg
        response_cube[i] = copy.copy(diff) / (2 * amp)
        responses[i] = copy.copy(diff)[control_mask] / (2 * amp)
        
        if plot: imshow3(amp*mode, im_pos, diff, f'Mode {i+1}', 'Absolute Image', 'Difference', cmap1='viridis')
        
        print(f"\tCalibrated mode {i+1:d}/{dm_modes.shape[0]:d} in {time.time()-start:.3f}s", end='')
        print("\r", end="")

    response_matrix = responses.T

    return response_matrix, response_cube


def run(
        M, 
        static_amp, static_opd, 
        ref_im, 
        control_mask, 
        control_matrix, 
        wfe_time_series, 
        wfe_modes, 
        dm_modes,
        channel=2,
        gain=1/2,  
        leakage=0.0,
        plot=False, 
        plot_all=False,
        sleep=None, 
    ):
    print(f'Starting LLOWFSC control-loop simulation')

    Nitr = wfe_time_series.shape[1]
    camlo_ims = xp.zeros((Nitr, M.ncamlo, M.ncamlo))
    diff_ims = xp.zeros((Nitr, M.ncamlo, M.ncamlo))
    camsci_ims = xp.zeros((Nitr, M.ncamsci, M.ncamsci))
    lo_commands = xp.zeros((Nitr, M.Nact, M.Nact))
    injected_wfes = xp.zeros((Nitr, wfe_time_series[:, 0].shape[0]))
    
    # Apply the very first OPD in the time series
    new_opd = xp.sum( wfe_time_series[:, 0, None, None] * wfe_modes, axis=0)
    M.PREFPM_AMP = static_amp
    M.PREFPM_OPD = static_opd + new_opd

    for i in range(1, Nitr):
        camlo_im = M.snap_camlo()
        camsci_im = M.snap_camsci()
        del_im = camlo_im - ref_im

        # compute the DM command with the image based on the time delayed wavefront
        modal_coeff = - gain * control_matrix.dot(del_im[control_mask])
        del_dm_command = xp.sum( modal_coeff[:, None, None] * dm_modes, axis=0)
        total_lo_dm = (1 - leakage) * M.get_dm(channel) + del_dm_command
        M.set_dm(total_lo_dm, channel)

        # Apply the very first OPD in the time series
        new_opd = xp.sum( wfe_time_series[:, i, None, None] * wfe_modes, axis=0)
        M.PREFPM_OPD = static_opd + new_opd

        camlo_ims[i] = copy.copy(camlo_im)
        diff_ims[i] = copy.copy(del_im)
        camsci_ims[i] = copy.copy(camsci_im)
        lo_commands[i] = copy.copy(total_lo_dm)
        injected_wfes[i] = copy.copy(wfe_time_series[:, i])

        if sleep is not None: time.sleep(sleep)
        if plot or plot_all:
            imshow3(
                camlo_im, control_mask*del_im, camsci_im, 
                'LLOWFSC Image', 'Difference Image',
                cmap1='magma', cmap2='magma',
                lognorm3=True, vmin3=1e-9, 
            )
            rms_wfe = xp.sqrt(xp.mean(xp.square( new_opd[M.BAP_MASK] )))
            vmax_pup = 2*rms_wfe
            pupil_cmap = 'viridis'
            imshow3(
                new_opd, del_dm_command, total_lo_dm, 
                f'Current WFE: {rms_wfe:.2e}\nIteration = {i:d}s', 
                'LLOWFSC DM Command', 'Total DM Command',
                vmin1=-vmax_pup, vmax1=vmax_pup, 
                cmap1=pupil_cmap, cmap2=pupil_cmap, cmap3=pupil_cmap,
            )
            
            if not plot_all: clear_output(wait=True)

    sim_dict = {
        'llowfsc_ims':camlo_ims,
        'diff_ims':diff_ims, 
        'injected_wfes':injected_wfes,
        'llowfsc_ref':ref_im,
        'wfe_modes':wfe_modes, 
        'coro_ims':camsci_ims,
        'lo_commands':lo_commands,
    }
    
    return sim_dict



