from .math_module import xp, xcipy, ensure_np_array
from scoob_llowfsc import utils

import numpy as np
import astropy.units as u
import time
import copy
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, CenteredNorm, Normalize

# def take_measurement(system_interface, probe_cube, probe_amplitude, return_all=False, pca_modes=None):
def take_measurement(I, probe_cube, probe_amplitude, channel=3, pca_modes=None, plot=False):
    N_probes = len(probe_cube)
    
    diff_ims = []
    ims = []
    for i in range(N_probes):
        probe = probe_cube[i]

        I.add_dm(probe_amplitude * probe, channel=channel) # add positive probe
        im_pos = I.snap_camsci()
        I.add_dm(-probe_amplitude * probe, channel=channel) # remove positive probe
        I.add_dm(-probe_amplitude * probe, channel=channel) # add negative probe
        im_neg = I.snap_camsci()
        I.add_dm(probe_amplitude*probe, channel=channel) # remove negative probe

        diff_ims.append((im_pos - im_neg) / (2*probe_amplitude))

    diff_ims = xp.array(diff_ims)

    if plot:
        for i, diff_im in enumerate(diff_ims):
            utils.imshow(
                [probe_cube[i], diff_im], 
                titles=[f'Probe Command {i+1}', 'Difference Image'], 
                pxscls=[None, I.camsci_pxscl_lamDc],
                cmaps=['viridis', 'magma'],
            )
    
    return diff_ims
    
def calibrate(
        I, 
        control_mask, 
        probe_amplitude, 
        probe_modes, 
        calibration_amplitude, 
        calibration_modes, 
        channel=3,
        scale_factors=None, 
        plot_responses=False, 
    ):
    print('Calibrating iEFC...')
    
    Nprobes = probe_modes.shape[0]
    Nmodes = calibration_modes.shape[0]

    response_matrix = []
    calib_amps = []
    response_cube = []
    
    # Loop through all modes that you want to control
    start = time.time()
    for ci, calibration_mode in enumerate(calibration_modes):
        response = 0
        for s in [-1, 1]: # We need a + and - probe to estimate the jacobian
            dm_mode = calibration_mode.reshape(I.Nact, I.Nact)
            calib_amp = calibration_amplitude * scale_factors[ci] if scale_factors is not None else calibration_amplitude

            # Add the mode to the DMs
            I.add_dm(s * calib_amp * dm_mode, channel=channel)
            
            # Compute reponse with difference images of probes
            diff_ims = take_measurement(I, probe_modes, probe_amplitude, channel=channel)
            calib_amps.append(calib_amp)
            response += s * diff_ims.reshape(Nprobes, I.ncamsci**2) / (2 * calib_amp)
            
            # Remove the mode form the DMs
            I.add_dm(-s * calib_amp * dm_mode, channel=channel) # remove the mode
        
        print(f"\tCalibrated mode {ci+1:d}/{calibration_modes.shape[0]:d} in {time.time()-start:.3f}s", end='')
        print("\r", end="")
        
        if probe_modes.shape[0]==2:
            response_matrix.append( 
                xp.concatenate([response[0, control_mask.ravel()],
                                response[1, control_mask.ravel()]]) 
            )
        elif probe_modes.shape[0]==3: # if 3 probes are being used
            response_matrix.append( 
                xp.concatenate([response[0, control_mask.ravel()], 
                                response[1, control_mask.ravel()],
                                response[2, control_mask.ravel()]]) 
                )
        
        response_cube.append(response)
    print('\nCalibration complete.')

    response_matrix = xp.array(response_matrix).T # this is the response matrix to be inverted
    response_cube = xp.array(response_cube)
    
    if plot_responses:
        dm_response_map = xp.sqrt(xp.mean(xp.square(response_matrix.dot(calibration_modes.reshape(Nmodes, -1))), axis=0))
        dm_response_map = dm_response_map.reshape(I.Nact,I.Nact) / xp.max(dm_response_map)

        fp_response_map = xp.sqrt( xp.mean( xp.abs(response_cube), axis=(0,1))).reshape(I.ncamsci, I.ncamsci)
        fp_response_map = fp_response_map / xp.max(fp_response_map)
        utils.imshow(
            [dm_response_map, fp_response_map],
            titles=['DM RMS Actuator Responses', 'Focal Plane Response'], 
            norms=[LogNorm(1e-2), None],
            pxscls=[None, I.camsci_pxscl_lamDc], 
            cmaps=['plasma', 'magma'],
        )
            
    return response_matrix, response_cube
    
def run(I, 
        data,
        control_matrix,
        probe_modes, probe_amplitude, 
        calibration_modes,
        control_mask,
        channel=3,
        num_iterations=3,
        gain=0.75, 
        leakage=0.0,
        plot_current=True,
        plot_all=False,
        vmin=1e-9,
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
        diff_ims = take_measurement(I, probe_modes, probe_amplitude, channel=channel, plot=plot_probes,)
        measurement_vector = diff_ims[:, control_mask].ravel()

        modal_coeff = -control_matrix.dot(measurement_vector)

        del_command = gain * modal_matrix.T.dot(modal_coeff).reshape(I.Nact,I.Nact)
        total_command = (1.0 - leakage)*total_command + del_command
        I.set_dm(total_command, channel=channel)

        image_ni = I.snap_camsci()
        mean_ni = xp.mean(image_ni[control_mask])

        data['images'].append(copy.copy(image_ni))
        data['contrasts'].append(copy.copy(mean_ni))
        data['commands'].append(copy.copy(total_command))
        data['del_commands'].append(copy.copy(del_command))
    
        if plot_current: 
            if not plot_all: clear_output(wait=True)
            utils.imshow(
                [del_command, total_command, image_ni], 
                titles=[f'Iteration {starting_itr + i:d}: $\delta$DM', 
                        'Total DM Command', 
                        f'Normalized Image\nMean Contrast = {mean_ni:.3e}'],
                cmaps=['viridis', 'viridis', 'magma'],
                pxscls=[None, None, I.camsci_pxscl_lamDc],
                norms=[CenteredNorm(), None, LogNorm(vmin=vmin)],
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

import matplotlib.pyplot as plt
plt.rcParams['image.origin'] = 'lower'
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm, Normalize, CenteredNorm
from IPython.display import display, clear_output

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
