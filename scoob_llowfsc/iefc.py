from .math_module import xp, xcipy, ensure_np_array
import scoob_llowfsc.utils as utils
from scoob_llowfsc.imshows import imshow1, imshow2, imshow3

import numpy as np
import astropy.units as u
import time
import copy
from IPython.display import display, clear_output
import matplotlib.pyplot as plt

# def take_measurement(system_interface, probe_cube, probe_amplitude, return_all=False, pca_modes=None):
def measure_probes(I, probe_cube, probe_amplitude, pca_modes=None, plot=False):
    N_probes = len(probe_cube)
    
    diff_ims = []
    ims = []
    for i in range(N_probes):
        probe = probe_cube[i]

        I.add_dm(probe_amplitude * probe) # add positive probe
        im_pos = I.snap_camsci()
        I.add_dm(-probe_amplitude * probe) # remove positive probe
        I.add_dm(-probe_amplitude * probe) # add negative probe
        im_neg = I.snap_camsci()
        I.add_dm(probe_amplitude*probe) # remove negative probe

        diff_ims.append((im_pos - im_neg) / (2 * probe_amplitude))

    diff_ims = xp.array(diff_ims)

    if plot:
        for i, diff_im in enumerate(diff_ims):
            imshow2(probe_cube[i], diff_im.reshape(I.npsf, I.npsf), 
                    f'Probe Command {i+1}', 'Difference Image', pxscl2=I.psf_pixelscale_lamDc,
                    cmap1='viridis')
    
    return diff_ims
    
def calibrate(
        I, 
        control_mask, 
        probe_amplitude, probe_modes, 
        calibration_amplitude, calibration_modes, 
        scale_factors=None, 
        plot_responses=False, 
    ):
    print('Calibrating iEFC...')

    Nprobes = probe_modes.shape[0]
    Nmodes = calibration_modes.shape[0]
    Nmask = control_mask.sum()
    print(Nmask)
    scale_factors = [1]*Nmodes if scale_factors is None else scale_factors

    calib_amps = []
    response_matrix = xp.zeros((int(Nprobes*Nmask), Nmodes))
    response_cube = xp.zeros((Nmodes, Nprobes, I.ncamsci, I.ncamsci))

    # Loop through all modes that you want to control
    start = time.time()
    for i, calib_mode in enumerate(calibration_modes):
        calib_amp = calibration_amplitude * scale_factors[i]
        calib_amps.append(calib_amp)

        I.add_dm(calib_amp * calib_mode) # add positive calibration mode
        pos_diff_ims = measure_probes(I, probe_modes, probe_amplitude)
        I.add_dm(-calib_amp * calib_mode) # remove positive calibration mode

        I.add_dm(-calib_amp * calib_mode) # add negative calibration mode
        neg_diff_ims = measure_probes(I, probe_modes, probe_amplitude)
        I.add_dm(-calib_amp * calib_mode) # remove positive calibration mode
        
        response = (pos_diff_ims - neg_diff_ims) / (2*calib_amp)

        # print(response.shape)
        # print(response[:, control_mask].ravel().shape)
        response_matrix[:,i] = copy.copy(response[:, control_mask].ravel())
        response_cube[i] = copy.copy(response)

        print(f"\tCalibrated mode {i+1:d}/{calibration_modes.shape[0]:d} in {time.time()-start:.3f}s", end='')
        print("\r", end="")

    # response_matrix = xp.array(response_matrix).T # this is the response matrix to be inverted
    # response_cube = xp.array(response_cube)

    print('\nCalibration complete.')
    
    if plot_responses:
        dm_response_map = xp.sqrt(xp.mean(xp.square(response_matrix.dot(calibration_modes.reshape(Nmodes, -1))), axis=0))
        dm_response_map = dm_response_map.reshape(I.Nact,I.Nact) / xp.max(dm_response_map)

        fp_response_map = xp.sqrt( xp.mean( xp.abs(response_cube), axis=(0,1))).reshape(I.ncamsci, I.ncamsci)
        imshow3(
            dm_response_map, fp_response_map, fp_response_map*control_mask,
            'DM RMS Actuator Responses', 'Focal Plane RMS Responses', 
            lognorm1=True, vmin1=1e-2,
            pxscl2=I.camsci_pxscl_lamDc, pxscl3=I.camsci_pxscl_lamDc,
        )
           
    return response_matrix, xp.array(response_cube)
    
def run(I, 
        data,
        control_matrix,
        probe_modes, probe_amplitude, 
        calibration_modes,
        control_mask,
        num_iterations=3,
        loop_gain=0.5, 
        leakage=0.0,
        plot_current=True,
        plot_all=False,
        plot_probes=False,
    ):
    
    print('Running iEFC...')
    start = time.time()
    starting_itr = len(data['images'])

    Nmodes = calibration_modes.shape[0]
    modal_matrix = calibration_modes.reshape(Nmodes, -1)

    total_command = copy.copy(data['commands'][-1]) if len(data['commands'])>0 else xp.zeros((I.Nact,I.Nact))

    for i in range(num_iterations):
        print(f"\tClosed-loop iteration {i+starting_itr} / {num_iterations+starting_itr-1}")
        I.subtract_dark = False
        diff_ims = measure_probes(I, probe_modes, probe_amplitude, plot=plot_probes)
        measurement_vector = diff_ims[:, control_mask].ravel()

        modal_coeff = -control_matrix.dot(measurement_vector)
        # print(modal_matrix.shape, modal_coeff.shape)

        del_command = modal_matrix.T.dot(modal_coeff).reshape(I.Nact,I.Nact)
        total_command = (1.0-leakage)*total_command + loop_gain*del_command
        I.set_dm(total_command)

        I.subtract_dark = True
        image_ni = I.snap()
        mean_ni = xp.mean(image_ni[control_mask])

        data['images'].append(copy.copy(image_ni))
        data['contrast'].append(copy.copy(mean_ni))
        data['commands'].append(copy.copy(total_command))
        data['del_commands'].append(copy.copy(del_command))
    
        if plot_current: 
            if not plot_all: clear_output(wait=True)
            imshow3(
                del_command, total_command, image_ni, 
                f'Iteration {starting_itr + i:d}: $\delta$DM', 
                'Total DM Command', 
                f'Image\nMean NI = {mean_ni:.3e}',
                cmap1='viridis', cmap2='viridis', 
                pxscl3=I.psf_pixelscale_lamDc, lognorm3=True, vmin3=1e-9,
            )
    
    print('Closed loop for given control matrix completed in {:.3f}s.'.format(time.time()-start))
    return data

def compute_hadamard_scale_factors(had_modes, scale_exp=1/6, scale_thresh=4, iwa=2.5, owa=13, oversamp=4, plot=False):
    Nact = had_modes.shape[1]

    ft_modes = []
    for i in range(had_modes.shape[0]):
        had_mode = had_modes[i]
        ft_modes.append(xp.fft.fftshift(xp.fft.fft2(xp.fft.ifftshift(utils.pad_or_crop(had_mode, Nact*oversamp)))))
    mode_freqs = xp.abs(xp.array(ft_modes))

    mode_freq_mask_pxscl = 1/oversamp
    x = (xp.linspace(-Nact*oversamp//2, Nact*oversamp//2-1, Nact*oversamp) + 1/2)*mode_freq_mask_pxscl
    x,y = xp.meshgrid(x,x)
    r = xp.sqrt(x**2+y**2)
    mode_freq_mask = (r>iwa)*(r<owa)
    if plot: imshow1(mode_freq_mask, pxscl=1/oversamp)

    sum_vals = []
    max_vals = []
    for i in range(had_modes.shape[0]):
        sum_vals.append(xp.sum(mode_freqs[i, mode_freq_mask]))
        max_vals.append(xp.max(mode_freqs[i, mode_freq_mask]**2))

    biggest_sum = xp.max(xp.array(sum_vals))
    biggest_max = xp.max(xp.array(max_vals))

    scale_factors = []
    for i in range(had_modes.shape[0]):
        scale_factors.append((biggest_max/max_vals[i])**scale_exp)
        # scale_factors.append((biggest_sum/sum_vals[i])**(1/2))
    scale_factors = ensure_np_array(xp.array(scale_factors))

    scale_factors[scale_factors>scale_thresh] = scale_thresh
    if plot: 
        plt.plot(scale_factors)
        plt.show()

    return scale_factors


