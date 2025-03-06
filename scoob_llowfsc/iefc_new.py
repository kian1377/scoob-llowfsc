from .math_module import xp, xcipy, ensure_np_array
from scoob_llowfsc import utils
from scoob_llowfsc import scoob_interface as scoobi
from scoob_llowfsc.imshows import imshow1, imshow2, imshow3

import numpy as np
import astropy.units as u
import time
import copy
from IPython.display import display, clear_output
import matplotlib.pyplot as plt

# def take_measurement(system_interface, probe_cube, probe_amplitude, return_all=False, pca_modes=None):
def take_measurement(
        camsci_stream, 
        dm_stream, 
        im_params,
        ref_psf_params,
        probe_amplitude, 
        probe_modes,
        NCAMSCI=10,
        plot=False):
    
    Ncamsci = camsci_stream.shape[0]
    Nprobes = probe_modes.shape[0]

    current_command = dm_stream.grab_latest()*1e-6
    
    diff_ims = []
    ims = []
    for i in range(Nprobes):
        probe = ensure_np_array(probe_amplitude * probe_modes[i])

        dm_stream.write( (current_command + probe) * 1e6 )
        im_pos = np.mean(camsci_stream.grab_many(NCAMSCI), axis=0)
        im_pos_ni = scoobi.normalize_camsci_image(im_pos, im_params, ref_psf_params)

        dm_stream.write( (current_command - probe) * 1e6 )
        im_neg = np.mean(camsci_stream.grab_many(NCAMSCI), axis=0)
        im_neg_ni = scoobi.normalize_camsci_image(im_neg, im_params, ref_psf_params)

        diff_ims.append((im_pos_ni - im_neg_ni) / (2*probe_amplitude))

    diff_ims = xp.array(diff_ims)
    dm_stream.write( current_command * 1e6 )

    if plot:
        for i, diff_im in enumerate(diff_ims):
            imshow2(
                probe_modes[i], diff_im.reshape(Ncamsci, Ncamsci), 
                f'Probe Command {i+1}', 'Difference Image',
                cmap1='viridis')
    
    return diff_ims
    
def calibrate(
        camsci_stream, 
        dm_stream, 
        control_mask, 
        probe_amplitude, 
        probe_modes, 
        calibration_amplitude, 
        calibration_modes,
        NCAMSCI=10,
        scale_factors=None, 
        plot_responses=False, 
    ):
    print('Calibrating iEFC...')

    Nact = probe_modes.shape[1]
    Nprobes = probe_modes.shape[0]
    Nmodes = calibration_modes.shape[0]
    Ncamsci = camsci_stream.shape[0]

    current_command = dm_stream.grab_latest()*1e-6

    response_matrix = []
    calib_amps = []
    response_cube = []
    
    # Loop through all modes that you want to control
    start = time.time()
    for ci, calibration_mode in enumerate(calibration_modes):
        response = 0
        for s in [-1, 1]: # We need a + and - probe to estimate the jacobian
            dm_mode = calibration_mode.reshape(Nact, Nact)
            amp = calibration_amplitude * scale_factors[ci] if scale_factors is not None else calibration_amplitude
            calib_mode = ensure_np_array(amp * dm_mode)

            dm_stream.write( (current_command + s * calib_mode) * 1e6)
            # Compute reponse with difference images of probes
            diff_ims = take_measurement(
                camsci_stream, 
                dm_stream, 
                probe_modes, 
                probe_amplitude, 
                NCAMSCI=NCAMSCI,
            )
            calib_amps.append(amp)
            response += s * diff_ims.reshape(Nprobes, Ncamsci**2) / (2 * amp)
            
            dm_stream.write( (current_command - s * calib_mode) * 1e6) # Remove the mode from the DMs
        
        print(f"\tCalibrated mode {ci+1:d}/{calibration_modes.shape[0]:d} in {time.time()-start:.3f}s", end='')
        print("\r", end="")
        
        if probe_modes.shape[0]==2:
            response_matrix.append( xp.concatenate([response[0, control_mask.ravel()],
                                                    response[1, control_mask.ravel()]]) )
        elif probe_modes.shape[0]==3: # if 3 probes are being used
            response_matrix.append( xp.concatenate([response[0, control_mask.ravel()], 
                                                    response[1, control_mask.ravel()],
                                                    response[2, control_mask.ravel()]]) )
        
        response_cube.append(response)
    print('\nCalibration complete.')

    response_matrix = xp.array(response_matrix).T # this is the response matrix to be inverted
    response_cube = xp.array(response_cube)
    
    if plot_responses:
        dm_response_map = xp.sqrt(xp.mean(xp.square(response_matrix.dot(calibration_modes.reshape(Nmodes, -1))), axis=0))
        dm_response_map = dm_response_map.reshape(Nact,Nact) / xp.max(dm_response_map)

        fp_response_map = xp.sqrt( xp.mean( xp.abs(response_cube), axis=(0,1))).reshape(Ncamsci, Ncamsci)
        fp_response_map = fp_response_map / xp.max(fp_response_map)
        imshow2(
            dm_response_map, fp_response_map, 
            'DM RMS Actuator Responses', 
            lognorm1=True, vmin1=1e-2,
        )
            
    return response_matrix, response_cube
    
def run(camsci_stream,
        dm_stream,
        im_params,
        ref_psf_params, 
        data_dict,
        control_matrix,
        probe_amplitude, 
        probe_modes, 
        calibration_modes,
        control_mask,
        dark_frame, 
        channel=3,
        num_iterations=3,
        gain=0.75, 
        leakage=0.0,
        plot_current=True,
        plot_all=False,
        vmin=1e-9,
        NCAMSCI=10, 
       ):
    
    print('Running iEFC...')
    start = time.time()
    starting_itr = len(data_dict['images'])

    Nact = probe_modes.shape[1]
    Nmodes = calibration_modes.shape[0]
    modal_matrix = calibration_modes.reshape(Nmodes, -1).T

    total_command = copy.copy(data_dict['commands'][-1]) if len(data_dict['commands'])>0 else xp.zeros((Nact,Nact))

    for i in range(num_iterations):
        print(f"\tClosed-loop iteration {i+starting_itr} / {num_iterations+starting_itr-1}")
        diff_ims = take_measurement(
                camsci_stream, 
                dm_stream, 
                probe_modes, 
                probe_amplitude, 
                NCAMSCI=NCAMSCI,
            )
        measurement_vector = diff_ims[:, control_mask].ravel()

        modal_coeff = -control_matrix.dot(measurement_vector)
        del_command = gain * modal_matrix.dot(modal_coeff).reshape(Nact,Nact)
        total_command = (1.0-leakage) * total_command + del_command
        
        dm_stream.write( total_command * 1e6 )

        new_image = np.mean(camsci_stream.grab_many(NCAMSCI), axis=0)
        image_ni = scoobi.normalize_camsci_image(new_image, im_params, ref_psf_params)
        mean_ni = xp.mean(image_ni[control_mask])

        data['raw_images']
        data['images'].append(copy.copy(image_ni))
        data['contrasts'].append(copy.copy(mean_ni))
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
                pxscl3=sysi.camsci_pxscl_lamDc, lognorm3=True, vmin3=vmin,
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


