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
        ref_offset,
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
        camsci_vmin=1e-9,
        sleep=None, 
    ):
    print(f'Starting LLOWFSC control-loop simulation')

    Nitr = wfe_time_series.shape[1] - 1
    camlo_ims = xp.zeros((Nitr, M.ncamlo, M.ncamlo))
    diff_ims = xp.zeros((Nitr, M.ncamlo, M.ncamlo))
    camsci_ims = xp.zeros((Nitr, M.ncamsci, M.ncamsci))
    lo_commands = xp.zeros((Nitr, M.Nact, M.Nact))
    del_commands = xp.zeros((Nitr, M.Nact, M.Nact))
    injected_wfes = xp.zeros((Nitr, wfe_time_series[:, 0].shape[0]))
    
    # Apply the very first OPD in the time series
    new_opd = xp.sum( wfe_time_series[:, 0, None, None] * wfe_modes, axis=0)
    M.PREFPM_AMP = static_amp
    M.PREFPM_OPD = static_opd + new_opd

    start = time.time()
    for i in range(Nitr):
        # compute CAMLO image and the DM command with the new OPD applied
        camlo_im = M.snap_camlo()
        del_im = camlo_im - (ref_im + ref_offset)
        modal_coeff = - gain * control_matrix.dot(del_im[control_mask])
        del_dm_command = xp.sum( modal_coeff[:, None, None] * dm_modes, axis=0)
        total_lo_dm = (1 - leakage) * M.get_dm(channel) + del_dm_command
        M.set_dm(total_lo_dm, channel)

        # Apply the very first OPD in the time series
        new_opd = xp.sum( wfe_time_series[:, i+1, None, None] * wfe_modes, axis=0)
        M.PREFPM_OPD = static_opd + new_opd

        camsci_im = M.snap_camsci() # CAMSCI image is computed after applying the updated the OPD to simulate lag

        camlo_ims[i] = copy.copy(camlo_im)
        diff_ims[i] = copy.copy(del_im)
        camsci_ims[i] = copy.copy(camsci_im)
        del_commands[i] = copy.copy(del_dm_command)
        lo_commands[i] = copy.copy(total_lo_dm)
        injected_wfes[i] = copy.copy(wfe_time_series[:, i])

        if sleep is not None: time.sleep(sleep)
        if plot or plot_all:
            imshow3(
                camlo_im, control_mask*del_im, camsci_im, 
                'LLOWFSC Image', 'Difference Image',
                cmap1='magma', cmap2='magma',
                lognorm3=True, vmin3=camsci_vmin, 
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
        else:
            print(f"\tIteration {i+1:d}/{Nitr:d} completed in {time.time()-start:.3f}s", end='')
            print("\r", end="")
    print('\nLLOWFSC simulation complete.')

    sim_dict = {
        'camlo_ims':camlo_ims,
        'diff_ims':diff_ims, 
        'injected_wfes':injected_wfes,
        'camlo_ref_im':ref_im,
        'camlo_ref_offset':ref_offset,
        'wfe_modes':wfe_modes, 
        'camsci_ims':camsci_ims,
        'del_commands': del_commands,
        'lo_commands':lo_commands,
        'llowfsc_mask':control_mask,
    }
    
    return sim_dict


def calibrate_with_fsm(
        M, 
        dm_modes, 
        control_mask, 
        channel=2,
        amp=3e-9, # DM calibration amplitude in 
        plot=False,
    ):
    
    Nmask = int(control_mask.sum())
    Nmodes = dm_modes.shape[0] + 2

    responses = xp.zeros((Nmodes, Nmask))
    response_cube = xp.zeros((Nmodes, M.ncamlo, M.ncamlo))
    
    start = time.time()

    for i in range(Nmodes):
        if i==0:
            mode = M.PTT_MODES[1]
            amp_as = fsm_rms_to_as(amp, M.fsm_beam_diam)
            M.add_fsm(np.array([0, amp_as, 0]))
            im_pos = M.snap_camlo()
            M.add_fsm(np.array([0, -2*amp_as, 0]))
            im_neg = M.snap_camlo()
            M.add_fsm(np.array([0, amp_as, 0]))
        elif i==1:
            mode = M.PTT_MODES[2]
            amp_as = fsm_rms_to_as(amp, M.fsm_beam_diam)
            M.add_fsm(np.array([0, 0, amp_as]))
            im_pos = M.snap_camlo()
            M.add_fsm(np.array([0, 0, -2*amp_as]))
            im_neg = M.snap_camlo()
            M.add_fsm(np.array([0, 0, amp_as]))
        else:
            mode = dm_modes[i - 2]
            M.add_dm(amp*mode, channel=channel)
            im_pos = M.snap_camlo()
            M.add_dm(-2*amp*mode, channel=channel)
            im_neg = M.snap_camlo()
            M.add_dm(amp*mode, channel=channel)

        diff = im_pos - im_neg
        response_cube[i] = copy.copy(diff) / (2 * amp)
        responses[i] = copy.copy(diff)[control_mask] / (2 * amp)
        
        if plot: imshow3(amp*mode, im_pos, diff, f'Mode {i+1}', 'Absolute Image', 'Difference', cmap1='viridis')
        
        print(f"\tCalibrated mode {i+1:d}/{Nmodes:d} in {time.time()-start:.3f}s", end='')
        print("\r", end="")

    response_matrix = responses.T

    return response_matrix, response_cube

def run_with_fsm(
        M, 
        static_amp, static_opd, 
        ref_im, 
        ref_offset,
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
        camsci_vmax=1e-4,
        camsci_vmin=1e-9,
        sleep=None, 
    ):
    print(f'Starting LLOWFSC control-loop simulation')

    Nitr = wfe_time_series.shape[1] - 1
    camlo_ims = xp.zeros((Nitr, M.ncamlo, M.ncamlo))
    diff_ims = xp.zeros((Nitr, M.ncamlo, M.ncamlo))
    camsci_ims = xp.zeros((Nitr, M.ncamsci, M.ncamsci))
    fsm_commands = np.zeros((Nitr, 3))
    del_fsm_commands = np.zeros((Nitr, 3))
    lo_dm_commands = xp.zeros((Nitr, M.Nact, M.Nact))
    del_dm_commands = xp.zeros((Nitr, M.Nact, M.Nact))
    injected_wfes = xp.zeros((Nitr, wfe_time_series[:, 0].shape[0]))
    
    # Apply the very first OPD in the time series
    new_opd = xp.sum( wfe_time_series[:, 0, None, None] * wfe_modes, axis=0)
    M.PREFPM_AMP = static_amp
    M.PREFPM_OPD = static_opd + new_opd

    start = time.time()
    for i in range(Nitr):
        # compute CAMLO image and the DM command with the new OPD applied
        camlo_im = M.snap_camlo()
        del_im = camlo_im - (ref_im + ref_offset)
        modal_coeff = - gain * control_matrix.dot(del_im[control_mask])

        del_fsm_rms = ensure_np_array(modal_coeff[:2])
        del_fsm_as = fsm_rms_to_as(del_fsm_rms, M.fsm_beam_diam)
        del_fsm_command = np.array([0, del_fsm_as[0], del_fsm_as[1]])
        total_fsm_command = (1 - leakage) * M.get_fsm() + del_fsm_command
        M.set_fsm(total_fsm_command)

        del_dm_command = xp.sum( modal_coeff[2:, None, None] * dm_modes, axis=0)
        total_lo_dm = (1 - leakage) * M.get_dm(channel) + del_dm_command
        M.set_dm(total_lo_dm, channel)

        # Apply the very first OPD in the time series
        new_opd = xp.sum( wfe_time_series[:, i+1, None, None] * wfe_modes, axis=0)
        M.PREFPM_OPD = static_opd + new_opd

        camsci_im = M.snap_camsci() # CAMSCI image is computed after applying the updated the OPD to simulate lag

        camlo_ims[i] = copy.copy(camlo_im)
        diff_ims[i] = copy.copy(del_im)
        camsci_ims[i] = copy.copy(camsci_im)
        del_fsm_commands[i] = copy.copy(del_fsm_command)
        fsm_commands[i] = copy.copy(total_fsm_command)
        del_dm_commands[i] = copy.copy(del_dm_command)
        lo_dm_commands[i] = copy.copy(total_lo_dm)
        injected_wfes[i] = copy.copy(wfe_time_series[:, i])

        if sleep is not None: time.sleep(sleep)
        if plot or plot_all:
            imshow3(
                camlo_im, control_mask*del_im, camsci_im, 
                'LLOWFSC Image', 'Difference Image',
                cmap1='magma', cmap2='magma',
                lognorm3=True, vmin3=camsci_vmin,  vmax3=camsci_vmax,
            )
            rms_wfe = xp.sqrt(xp.mean(xp.square( new_opd[M.BAP_MASK] )))
            vmax_pup = 2*rms_wfe
            pupil_cmap = 'viridis'
            imshow3(
                new_opd, M.FSM_OPD, total_lo_dm, 
                f'Current WFE: {rms_wfe:.2e}\nIteration = {i:d}s', 
                'Total FSM Command', 
                'Total DM Command',
                vmin1=-vmax_pup, vmax1=vmax_pup, 
                cmap1=pupil_cmap, cmap2=pupil_cmap, cmap3=pupil_cmap,
            )
            
            if not plot_all: clear_output(wait=True)
        else:
            print(f"\tIteration {i+1:d}/{Nitr:d} completed in {time.time()-start:.3f}s", end='')
            print("\r", end="")
    print('\nLLOWFSC simulation complete.')

    sim_dict = {
        'camlo_ims':camlo_ims,
        'diff_ims':diff_ims, 
        'injected_wfes':injected_wfes,
        'camlo_ref_im':ref_im,
        'camlo_ref_offset':ref_offset,
        'wfe_modes':wfe_modes, 
        'camsci_ims':camsci_ims,
        'del_fsm_commands': del_fsm_commands,
        'fsm_commands':fsm_commands,
        'fsm_modes':M.PTT_MODES,
        'del_dm_commands': del_dm_commands,
        'lo_dm_commands':lo_dm_commands,
        'llowfsc_mask':control_mask,
    }
    
    return sim_dict

def fsm_rms_to_as(vals_rms, pupil_diam):
    vals_pv = vals_rms * 4
    vals_as = np.arctan(vals_pv / pupil_diam) * 206264.806 # radians * arcsec/radian
    return vals_as

def fsm_as_to_rms(vals_as, pupil_diam):
    vals_pv = np.tan(vals_as / 206264.806) * pupil_diam
    vals_rms = vals_pv / 4
    return vals_rms

def make_shear_chops(camlo_ref, control_mask, shear_pix=1/2, order=3, central_diff=False, plot=False):
    ncamlo = camlo_ref.shape[0]
    shear_chops = xp.zeros((2, ncamlo, ncamlo))
    if central_diff:
        shear_chops_x1 = ( xcipy.ndimage.shift(camlo_ref, (0,shear_pix), order=order))
        shear_chops_x2 = ( xcipy.ndimage.shift(camlo_ref, (0,-shear_pix), order=order)) 
        shear_chops[0] = ( shear_chops_x1 - shear_chops_x2 ) / (2*shear_pix)

        shear_chops_y1 = ( xcipy.ndimage.shift(camlo_ref, (shear_pix,0), order=order) )
        shear_chops_y2 = ( xcipy.ndimage.shift(camlo_ref, (-shear_pix,0), order=order))
        shear_chops[1] = ( shear_chops_y1 - shear_chops_y2 ) / (2*shear_pix)

    else:
        shear_chops_x = ( xcipy.ndimage.shift(copy.copy(camlo_ref), (0,shear_pix), order=order))
        shear_chops_y = ( xcipy.ndimage.shift(copy.copy(camlo_ref), (shear_pix,0), order=order))

        shear_chops[0] = ( shear_chops_x - camlo_ref ) / shear_pix
        shear_chops[1] = ( shear_chops_y - camlo_ref ) / shear_pix
    shear_chops[:] *= control_mask
    shear_responses = shear_chops[:, control_mask].T
    if plot: imshow2(shear_chops[0], shear_chops[1])
    return shear_responses, shear_chops

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LogNorm

def plot_responses(
        dm_modes, 
        response_cube, 
        figsize=(25,5),
        dpi=125,
        hspace=0.0,
        wspace=-0.05,
        title=None,
        title_fs=14,
    ):
    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = GridSpec(2, 10, figure=fig)
    fig.suptitle(title, fontsize=title_fs)

    for i in range(10):
        mode = ensure_np_array(dm_modes[i])
        response = ensure_np_array(response_cube[i])

        ax = fig.add_subplot(gs[0, i])
        ax.imshow(mode, cmap='viridis')
        ax.set_xticks([])
        ax.set_yticks([])

        ax = fig.add_subplot(gs[1, i])
        ax.imshow(response, cmap='magma',)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.subplots_adjust(hspace=hspace, wspace=wspace)


