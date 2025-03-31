from .math_module import xp, xcipy, ensure_np_array
from scoob_llowfsc import utils
from scoob_llowfsc import scoob_interface as scoobi

import numpy as np
import astropy.units as u
import time
import copy
from IPython.display import display, clear_output

import matplotlib.pyplot as plt
plt.rcParams['image.origin'] = 'lower'
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm, Normalize, CenteredNorm

# def take_measurement(system_interface, probe_cube, probe_amplitude, return_all=False, pca_modes=None):
def take_measurement(
        camsci_stream, 
        dm_stream, 
        im_params,
        ref_psf_params,
        probe_modes,
        probe_amplitude, 
        NFRAMES=10,
        delay=0.01,
        plot=False
    ):
    
    Ncamsci = camsci_stream.shape[0]
    Nprobes = probe_modes.shape[0]

    current_command = dm_stream.grab_latest() * 1e-6
    
    diff_ims = []
    ims = []
    for i in range(Nprobes):
        probe = ensure_np_array(probe_amplitude * probe_modes[i])

        dm_stream.write( (current_command + probe) * 1e6 )
        time.sleep(delay)
        im_pos = scoobi.snap(camsci_stream, NFRAMES, 0, im_params, ref_psf_params)

        dm_stream.write( (current_command - probe) * 1e6 )
        time.sleep(delay)
        im_neg = scoobi.snap(camsci_stream, NFRAMES, 0, im_params, ref_psf_params)

        diff_im = (im_pos - im_neg) / (2 * probe_amplitude)
        diff_ims.append( diff_im )

    diff_ims = np.array(diff_ims)
    dm_stream.write( current_command * 1e6 )

    if plot:
        for i, diff_im in enumerate(diff_ims):
            utils.imshow(
                [probe_modes[i], diff_im], 
                titles=[f'Probe Command {i+1}', 'Difference Image'],
                cmaps=['viridis', 'magma'], 
            )
    
    return diff_ims
    
def calibrate(
        camsci_stream, 
        dm_stream, 
        control_mask, 
        probe_amplitude, 
        probe_modes, 
        calibration_amplitude, 
        calibration_modes,
        im_params,
        ref_psf_params, 
        NFRAMES=10,
        delay=0.01,
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
    for i, calibration_mode in enumerate(calibration_modes):
        response = 0
        for s in [-1, 1]: # We need a + and - probe to estimate the jacobian
            dm_mode = calibration_mode.reshape(Nact, Nact)
            amp = calibration_amplitude * scale_factors[i] if scale_factors is not None else calibration_amplitude
            calib_mode = ensure_np_array(amp * dm_mode)

            dm_stream.write( (current_command + s * calib_mode) * 1e6)
            time.sleep(delay)
            # Compute reponse with difference images of probes
            diff_ims = take_measurement(
                camsci_stream, 
                dm_stream, 
                im_params,
                ref_psf_params,
                probe_modes,
                probe_amplitude, 
                NFRAMES=NFRAMES,
                delay=delay,
            )
            calib_amps.append(amp)
            response += s * diff_ims.reshape(Nprobes, Ncamsci**2) / (2 * amp)
            
            # dm_stream.write( (current_command - s * calib_mode) * 1e6) # Remove the mode from the DMs
        
        print(f"\tCalibrated mode {i+1:d}/{calibration_modes.shape[0]:d} in {time.time()-start:.3f}s", end='')
        print("\r", end="")
        
        dm_stream.write( current_command * 1e6 )
        if probe_modes.shape[0]==2:
            response_matrix.append( np.concatenate([response[0, control_mask.ravel()],
                                                    response[1, control_mask.ravel()]]) )
        elif probe_modes.shape[0]==3: # if 3 probes are being used
            response_matrix.append( np.concatenate([response[0, control_mask.ravel()], 
                                                    response[1, control_mask.ravel()],
                                                    response[2, control_mask.ravel()]]) )
        
        response_cube.append(response)
    print('\nCalibration complete.')

    response_matrix = np.array(response_matrix).T # this is the response matrix to be inverted
    response_cube = np.array(response_cube)
    
    if plot_responses:
        dm_response_map = np.sqrt(np.mean(np.square(response_matrix.dot(calibration_modes.reshape(Nmodes, -1))), axis=0))
        dm_response_map = dm_response_map.reshape(Nact,Nact) / np.max(dm_response_map)

        fp_response_map = np.sqrt( np.mean( np.abs(response_cube), axis=(0,1))).reshape(Ncamsci, Ncamsci)
        fp_response_map = fp_response_map / np.max(fp_response_map)
        utils.imshow(
            [dm_response_map, fp_response_map], 
            titles=['DM Response Map', 'Focal Plane Response Map'],
            norms=[LogNorm(1e-2), None]
        )
            
    return response_matrix, response_cube
    
def run(iefc_data,
        camsci_stream,
        dm_stream,
        control_matrix,
        probe_amplitude, 
        probe_modes, 
        modal_matrix,
        control_mask,
        im_params,
        ref_psf_params, 
        dark_frame, 
        NFRAMES=10, 
        delay=0.01,
        num_iterations=3,
        gain=0.75, 
        leakage=0.0,
        plot_current=True,
        plot_all=False,
        vmin=1e-9,
    ):
    
    start = time.time()
    starting_itr = len(iefc_data['images'])

    Nact = probe_modes.shape[1]
    Nmodes = modal_matrix.shape[1]

    total_command = copy.copy(iefc_data['commands'][-1]) if len(iefc_data['commands'])>0 else np.zeros((Nact,Nact))

    for i in range(num_iterations):
        print(f"Running iteration {i+starting_itr} / {num_iterations+starting_itr-1}")
        diff_ims = take_measurement(
            camsci_stream, 
            dm_stream, 
            im_params,
            ref_psf_params,
            probe_modes,
            probe_amplitude, 
            NFRAMES=NFRAMES,
            delay=delay,
        )
        measurement_vector = diff_ims[:, control_mask].ravel()

        modal_coeff = -control_matrix.dot(measurement_vector)
        del_command = gain * modal_matrix.dot(modal_coeff).reshape(Nact, Nact)
        total_command = (1.0 - leakage) * total_command + del_command
        
        dm_stream.write( total_command * 1e6 )
        time.sleep(delay)

        print(f"Measuring dark hole state ...")
        image_ni = scoobi.snap(camsci_stream, NFRAMES, dark_frame, im_params, ref_psf_params)
        contrast = np.mean(image_ni[control_mask])

        iefc_data['images'].append(copy.copy(image_ni))
        iefc_data['contrasts'].append(copy.copy(contrast))
        iefc_data['commands'].append(copy.copy(total_command))
        iefc_data['del_commands'].append(copy.copy(del_command))
    
        if plot_current: 
            if not plot_all: clear_output(wait=True)
            utils.imshow(
                [del_command, total_command, image_ni], 
                titles=[f'Iteration {starting_itr + i:d}: $\delta$DM', 
                        'Total DM Command', 
                        f'Normalized Image\nMean Contrast = {contrast:.3e}'],
                cmaps=['viridis', 'viridis', 'magma'],
                pxscls=[None, None, None],
                norms=[CenteredNorm(), None, LogNorm(vmin=vmin)],
            )

    print(f'Completed {num_iterations:d} iterations in {time.time()-start:.3f}s.')
    return iefc_data

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
    if plot: utils.imshow([mode_freq_mask], pxscls=[1/oversamp])

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

def plot_data_with_ref(
        data, 
        im1vmin=1e-9, im1vmax=1e-4,
        im2vmin=1e-9, im2vmax=1e-4, 
        vmin=1e-9, vmax=1e-4, 
        xticks=None,
        exp_name='',
        fname=None,
    ):
    ims = ensure_np_array( xp.array(data['images']) ) 
    control_mask = ensure_np_array( data['control_mask'] )
    # print(type(control_mask))
    Nitr = ims.shape[0]
    npsf = ims.shape[1]
    psf_pixelscale_lamD = data['pixelscale']

    mean_nis = np.mean(ims[:,control_mask], axis=1)
    ibest = np.argmin(mean_nis)
    ref_im = ensure_np_array(data['images'][0])
    best_im = ensure_np_array(data['images'][ibest])

    fig,ax = plt.subplots(nrows=1, ncols=3, figsize=(15,10), dpi=125, gridspec_kw={'width_ratios': [1, 1, 1], })
    ext = psf_pixelscale_lamD*npsf/2
    extent = [-ext, ext, -ext, ext]

    w = 0.225
    im1 = ax[0].imshow(ref_im, norm=LogNorm(vmax=im1vmax, vmin=im1vmin), cmap='magma', extent=extent)
    ax[0].set_title(f'Initial Image:\nMean Contrast = {mean_nis[0]:.2e}', fontsize=14)
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes("right", size="4%", pad=0.075)
    cbar = fig.colorbar(im1, cax=cax)
    cbar.ax.set_ylabel('NI', rotation=0, labelpad=7)
    ax[0].set_position([0, 0, w, w]) # [left, bottom, width, height]

    im2 = ax[1].imshow( best_im, norm=LogNorm(vmax=im2vmax, vmin=im2vmin), cmap='magma', extent=extent)
    ax[1].set_title('Best Iteration' + exp_name + f':\nMean Contrast = {mean_nis[ibest]:.2e}', fontsize=14)
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes("right", size="4%", pad=0.075)
    cbar = fig.colorbar(im2, cax=cax,)
    cbar.ax.set_ylabel('NI', rotation=0, labelpad=7)
    ax[1].set_position([0.23, 0, w, w])

    ax[0].set_ylabel('Y [$\lambda/D$]', fontsize=12, labelpad=-5)
    ax[0].set_xlabel('X [$\lambda/D$]', fontsize=12, labelpad=5)
    ax[1].set_xlabel('X [$\lambda/D$]', fontsize=12, labelpad=5)

    ax[2].set_title('Mean Contrast per Iteration' + exp_name, fontsize=14)
    ax[2].semilogy(mean_nis, label='3.6% Bandpass')
    ax[2].grid()
    ax[2].set_xlabel('Iteration Number', fontsize=12, )
    ax[2].set_ylabel('Mean Contrast', fontsize=14, labelpad=1)
    ax[2].set_ylim([vmin, vmax])
    xticks = np.arange(0,Nitr,2) if xticks is None else xticks
    ax[2].set_xticks(xticks)
    ax[2].set_position([0.525, 0, 0.3, w])

    if fname is not None: fig.savefig(fname, format='pdf', bbox_inches="tight")

