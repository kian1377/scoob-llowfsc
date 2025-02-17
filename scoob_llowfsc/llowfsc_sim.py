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
        amps=3e-9, 
        NFRAMES=10, 
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

        M.add_dm(amp*mode)
        im_pos = M.snap_camlo()
        M.add_dm(-2*amp*mode)
        im_neg = M.snap_camlo()
        M.add_dm(amp*mode)

        diff = im_pos - im_neg
        response_cube[i] = copy.copy(diff) / (2 * amp)
        responses[i] = copy.copy(diff)[control_mask] / (2 * amp)
        
        if plot: imshow3(amp*mode, im_pos, diff, f'Mode {i+1}', 'Absolute Image', 'Difference', cmap1='viridis')
        
        print(f"\tCalibrated mode {i+1:d}/{dm_modes.shape[0]:d} in {time.time()-start:.3f}s", end='')
        print("\r", end="")

    response_matrix = responses.T

    return response_matrix, response_cube


def run_sim(
        M, 
        static_wfe, 
        ref_im, 
        control_mask, 
        control_matrix, 
        time_series, 
        wfe_modes, 
        dm_modes,
        gain=1/2,  
        leakage=0.0,
        dh_command=xp.zeros((34,34)),
        old_lo_command=0.0, 
        plot=False, 
        plot_all=False,
        sleep=None, 
    ):
    print(f'Starting LLOWFSC control-loop simulation')

    Nitr = time_series.shape[1]
    llowfsc_ims = xp.zeros((Nitr, M.nlocam, M.nlocam))
    diff_ims = xp.zeros((Nitr, M.nlocam, M.nlocam))
    coro_ims = xp.zeros((Nitr, M.npsf, M.npsf))
    lo_commands = xp.zeros((Nitr, M.Nact, M.Nact))
    injected_wfes = xp.zeros((Nitr, time_series[1:, 0].shape[0]))
    
    for i in range(Nitr):
        if sleep is not None: time.sleep(sleep)
        if i==0:
            del_dm_command = xp.zeros_like(M.DM.command)
        else:
            # compute the DM command with the image based on the time delayed wavefront
            modal_coeff = - gain * control_matrix.dot(del_im[control_mask])
            del_dm_command = xp.sum( modal_coeff[:, None, None] * dm_modes, axis=0)
        total_lo_dm = (1 - leakage) * old_lo_command + del_dm_command
        M.set_dm(dh_command + total_lo_dm)

        # apply the new wavefront to simulate a time delay 
        lo_wfe = xp.sum( time_series[1:, i, None, None] * wfe_modes, axis=0)
        M.setattr('WFE', static_wfe * xp.exp(1j * 2*np.pi/M.wavelength_c.to_value(u.m) * lo_wfe) )
        camlo_im = M.snap_camlo()
        coro_im = M.snap()
        del_im = camlo_im - ref_im

        llowfsc_ims[i] = copy.copy(camlo_im)
        diff_ims[i] = copy.copy(del_im)
        coro_ims[i] = copy.copy(coro_im)
        lo_commands[i] = copy.copy(total_lo_dm)
        injected_wfes[i] = copy.copy(time_series[1:, i])

        old_lo_command = copy.copy(total_lo_dm)

        if plot or plot_all:
            imshow3(camlo_im, control_mask*del_im, coro_im, 
                    'LLOWFSC Image', 'Difference Image',
                    cmap1='magma', cmap2='magma',
                    lognorm3=True, vmin3=1e-9, )
            rms_wfe = xp.sqrt(xp.mean(xp.square( lo_wfe[M.APMASK] )))
            vmax_pup = 2*rms_wfe
            pupil_cmap = 'viridis'
            imshow3(lo_wfe, del_dm_command, M.get_dm(), 
                    f'Current WFE: {rms_wfe:.2e}\nTime = {time_series[0][i]:.3f}s', 
                    'LLOWFSC DM Command', 'Total DM Command',
                    vmin1=-vmax_pup, vmax1=vmax_pup, 
                    cmap1=pupil_cmap, cmap2=pupil_cmap, cmap3=pupil_cmap,
                    )
            
            if not plot_all: clear_output(wait=True)

    sim_dict = {
        'llowfsc_ims':llowfsc_ims,
        'diff_ims':diff_ims, 
        'injected_wfes':injected_wfes,
        'llowfsc_ref':ref_im,
        'wfe_modes':wfe_modes, 
        'coro_ims':coro_ims,
        'lo_commands':lo_commands,
    }
    
    return sim_dict



