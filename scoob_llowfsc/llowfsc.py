from .math_module import xp, xcipy, ensure_np_array
import scoob_llowfsc.utils as utils
from scoob_llowfsc.imshows import imshow

import numpy as np
import astropy.units as u
import copy
from IPython.display import display, clear_output
import time

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize, CenteredNorm
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

def create_control_mask(
        dims, 
        irad, 
        orad,  
        x_shift=0,
        y_shift=0,
        even=True,
    ):
    X = np.linspace(-dims[1]/2, dims[1]/2-1, dims[1]) + 1/2 if even else np.linspace(-dims[1]/2, dims[1]/2-1, dims[1])
    Y = np.linspace(-dims[0]/2, dims[0]/2-1, dims[0]) + 1/2 if even else np.linspace(-dims[0]/2, dims[0]/2-1, dims[0])
    X = X + x_shift
    Y = Y + y_shift
    x,y = np.meshgrid(X,Y)
    r = np.hypot(x, y)
    mask = (r > irad) * (r < orad)
    return mask

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
        dm_lo_stream.write(-amp*mode*1e6)
        time.sleep(dm_delay)
        im_neg = np.mean( camlo_stream.grab_many(NFRAMES), axis=0 )

        diff = im_pos - im_neg
        response_cube[i] = copy.copy(diff) / (2 * amp)
        responses[i] = copy.copy(diff)[control_mask] / (2 * amp)
        
        if plot:
            imshow(
                [amp*mode, diff], 
                titles=[f'Mode {i+1}', 'Difference'],
                cmaps=['viridis', 'magma'],
                # wspace=0.5,
                # figsize=(20,5),
            )
        
        print(f"\tCalibrated mode {i+1:d}/{dm_modes.shape[0]:d} in {time.time()-start:.3f}s", end='')
        print("\r", end="")
    
    dm_lo_stream.write(np.zeros_like(mode))

    response_matrix = responses.T

    return response_matrix, response_cube

def make_shear_chops(ref_locam_im, control_mask, shear_pix=1, order=3, central_diff=False, plot=False):
    nlocam = ref_locam_im.shape[0]
    shear_chops = xp.zeros((2, nlocam, nlocam))
    if central_diff:
        shear_chops_x1 = ( xcipy.ndimage.shift(ref_locam_im, (0,shear_pix), order=order) - ref_locam_im )
        shear_chops_x2 = ( xcipy.ndimage.shift(ref_locam_im, (0,-shear_pix), order=order) - ref_locam_im ) 
        shear_chops[0] = ( shear_chops_x1 - shear_chops_x2 ) / (2*shear_pix)

        shear_chops_y1 = ( xcipy.ndimage.shift(ref_locam_im, (shear_pix,0), order=order) - ref_locam_im )
        shear_chops_y2 = ( xcipy.ndimage.shift(ref_locam_im, (-shear_pix,0), order=order) - ref_locam_im )
        shear_chops[1] = ( shear_chops_y1 - shear_chops_y2 ) / (2*shear_pix)

    else:
        shear_chops[0] = ( xcipy.ndimage.shift(ref_locam_im, (0,shear_pix), order=order) - ref_locam_im ) / shear_pix
        shear_chops[1] = ( xcipy.ndimage.shift(ref_locam_im, (shear_pix,0), order=order) - ref_locam_im ) / shear_pix
    shear_chops[:] *= control_mask
    if plot: imshow2(shear_chops[0], shear_chops[1])
    return shear_chops

def update_ref_offset(
        response_matrix, 
        modal_matrix, 
        control_mask, 
        dm_dh_stream, 
        camlo_ref_offset_stream,
    ):
    del_ref_im = np.zeros(camlo_ref_offset_stream.shape)
    del_ref_im[control_mask] = response_matrix.dot(modal_matrix.dot(1e-6*dm_dh_stream.grab_latest().ravel())/1024)
    camlo_ref_offset_stream.write(del_ref_im)
    return

# import skimage
# from skimage.registration import phase_cross_correlation

# def detect_ref_shear(current_ref, camlo_stream):
#     return new_ref

def single_iteration(
        dm_lo_stream,
        camlo_stream,
        camlo_ref_stream,
        camlo_ref_offset_stream,  
        gains_stream,
        leak_stream, 
        control_matrix, 
        modal_matrix,
        control_mask, 
        plot=False,
        clear=False,
        Nact=34,
    ):

    image = camlo_stream.grab_latest() * control_mask
    del_im = image - (camlo_ref_stream.grab_latest() + camlo_ref_offset_stream.grab_latest())

    # compute the DM command with the image based on the time delayed wavefront
    modal_coeff = -control_matrix.dot( del_im[control_mask] )
    modal_coeff *= gains_stream.grab_latest()[0]
    del_dm_command = modal_matrix.T.dot(modal_coeff).reshape(Nact,Nact)

    total_command = (1-leak_stream.grab_latest()[0,0])*dm_lo_stream.grab_latest()/1e6 + del_dm_command
    dm_lo_stream.write(total_command * 1e6)

    if plot:
        imshow(
            [del_im, del_dm_command, total_command], 
            titles=['Measured Difference Image', 'Computed DM Correction', 'Total DM Command'], 
            cmaps=['magma', 'viridis', 'viridis'],
        )
        if clear: clear_output(wait=True)

def inject_wfe(
        wfe_time_series, 
        wfe_modes, 
        wfe_stream, 
        interval, 
        interval_offset=0.0,
    ):
    Nsamps = wfe_time_series.shape[1]
    delay = interval - interval_offset
    
    try:
        print('Injecting WFE ...')
        i = 0
        while i<Nsamps:
            wfe = np.sum( wfe_time_series[:, i, None, None] * wfe_modes, axis=0)
            wfe_stream.write(1e6 * wfe)
            # print(time)
            time.sleep(delay)
            i += 1
        print('Stopped injecting WFE.')
        wfe_stream.write(np.zeros(wfe_stream.shape))
    except KeyboardInterrupt:
        print('Stopped injecting WFE.')
        wfe_stream.write(np.zeros(wfe_stream.shape))

def inject_wfe_cube(
        wfe_cube,
        wfe_stream, 
        interval, 
        interval_offset=0.0,
    ):
    Nsamps = wfe_cube.shape[0]
    delay = interval - interval_offset
    
    try:
        print('Injecting WFE ...')
        i = 0
        while i<Nsamps:
            wfe_stream.write(wfe_cube[i])
            # print(time)
            time.sleep(delay)
            i += 1
            if i==Nsamps: i = 0
        print('Stopped injecting WFE.')
        wfe_stream.write(np.zeros(wfe_stream.shape))
    except KeyboardInterrupt:
        print('Stopped injecting WFE.')
        wfe_stream.write(np.zeros(wfe_stream.shape))

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
        fname=None,
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
    if fname is not None: fig.savefig(fname, format='pdf', bbox_inches="tight")
