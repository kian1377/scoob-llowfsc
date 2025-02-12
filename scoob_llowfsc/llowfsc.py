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

import threading
class Process(threading.Timer):  
    def run(self):
        while not self.finished.wait(self.interval):  
            self.function(*self.args, **self.kwargs)
# process = Repeat(0.1, print, ['Repeating']) 
# process.start()
# time.sleep(5)
# process.cancel()

def calibrate_without_fsm(
        camlo_stream, 
        dm_lo_stream, 
        dm_modes, 
        control_mask, 
        amps=3e-9, 
        NFRAMES=10, 
        dm_delay=0.001, 
        plot=False,
    ):
    Nmask = int(control_mask.sum())
    Nmodes = dm_modes.shape[0]

    if np.isscalar(amps): amps = [amps] * Nmodes

    responses = np.zeros((Nmodes, Nmask))
    response_cube = np.zeros((Nmodes, camlo_stream.shape[0], camlo_stream.shape[1]))
    
    start = time.time()
    for i in range(Nmodes):
        amp = amps[i]
        mode = dm_modes[i]

        dm_lo_stream.write(amp*mode*1e6)
        time.sleep(dm_delay)
        im_pos = np.mean( camlo_stream.grab_many(NFRAMES), axis=0 )
        dm_lo_stream.write(-2*amp*mode*1e6)
        time.sleep(dm_delay)
        im_neg = np.mean( camlo_stream.grab_many(NFRAMES), axis=0 )
        dm_lo_stream.write(amp*mode*1e6)

        # I.add_dm(amp*mode)
        # im_pos = I.snap_locam()
        # I.add_dm(-2*amp*mode)
        # im_neg = I.snap_locam()
        # I.add_dm(amp*mode)

        diff = im_pos - im_neg
        response_cube[i] = copy.copy(diff) / (2 * amp)
        responses[i] = copy.copy(diff)[control_mask] / (2 * amp)
        
        if plot:
            imshow3(amp*mode, im_pos, diff, f'Mode {i+1}', 'Absolute Image', 'Difference', cmap1='viridis')
        
        print(f"\tCalibrated mode {i+1:d}/{dm_modes.shape[0]:d} in {time.time()-start:.3f}s", end='')
        print("\r", end="")

    response_matrix = responses.T

    return response_matrix, response_cube

def update_ref_delta(
        response_matrix, 
        modal_matrix, 
        control_mask, 
        dm_dh_stream, 
        camlo_delta_stream,
    ):
    del_ref_im = np.zeros(camlo_delta_stream.shape)
    del_ref_im[control_mask] = response_matrix.dot(modal_matrix.dot(1e-6*dm_dh_stream.grab_latest().ravel())/1024)
    camlo_delta_stream.write(del_ref_im)
    return

# import skimage
# from skimage.registration import phase_cross_correlation

# def detect_ref_shear(current_ref, camlo_stream):
#     return new_ref

def create_control_mask(
        dims, 
        irad, 
        orad,  
        even=True,
    ):
    X = np.linspace(-dims[1]/2, dims[1]/2-1, dims[1]) + 1/2 if even else np.linspace(-dims[1]/2, dims[1]/2-1, dims[1])
    Y = np.linspace(-dims[0]/2, dims[0]/2-1, dims[0]) + 1/2 if even else np.linspace(-dims[0]/2, dims[0]/2-1, dims[0])
    x,y = np.meshgrid(X,Y)
    r = np.hypot(x, y)
    mask = (r > irad) * (r < orad)
    return mask

def single_iteration(
        dm_lo_stream,
        camlo_stream,
        camlo_ref_stream,
        camlo_delta_stream,  
        gains_stream,
        leak_stream, 
        control_matrix, 
        modal_matrix,
        control_mask, 
        plot=False,
        clear=False,
        Nact=34,
    ):

    image = camlo_stream.grab_latest()
    del_im = image - (camlo_ref_stream.grab_latest() + camlo_delta_stream.grab_latest())

    # compute the DM command with the image based on the time delayed wavefront
    modal_coeff = -control_matrix.dot( del_im[control_mask] )
    modal_coeff *= gains_stream.grab_latest()[0]
    del_dm_command = modal_matrix.T.dot(modal_coeff).reshape(Nact,Nact)

    total_command = (1-leak_stream.grab_latest()[0,0])*dm_lo_stream.grab_latest()/1e6 + del_dm_command
    dm_lo_stream.write(total_command * 1e6)

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

def inject_wfe(wfe_time_series, wfe_modes, wfe_stream, interval=75e-6):
    Nsamps = wfe_time_series.shape[1]
    try:
        print('Injecting WFE ...')
        i = 0
        while i<Nsamps+1:
            if i==Nsamps: i = 0
            wfe = np.sum( wfe_time_series[:, i, None, None] * wfe_modes, axis=0)
            wfe_stream.write(1e6 * wfe)
            time.sleep(interval)
            i += 1
    except KeyboardInterrupt:
        print('Stopped injecting WFE.')
        wfe_stream.write(np.zeros(wfe_stream.shape))



