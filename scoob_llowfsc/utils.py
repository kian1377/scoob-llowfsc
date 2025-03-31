from .math_module import xp, xcipy, ensure_np_array

import numpy as np
import scipy
import astropy.units as u
from astropy.io import fits
import poppy
import pickle

import skimage

import matplotlib.pyplot as plt
plt.rcParams['image.origin'] = 'lower'
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm, Normalize, CenteredNorm
from matplotlib.patches import Circle, Rectangle
from IPython.display import display, clear_output

def mean(array, mask=None):
    MEAN = xp.mean(array) if mask is None else xp.mean(array[mask])
    return MEAN

def rms(array, mask=None):
    RMS = xp.sqrt( xp.mean( xp.square(array))) if mask is None else xp.sqrt( xp.mean( xp.square(array[mask])))
    return RMS

def make_grid(npix, pixelscale=1, half_shift=False):
    if half_shift:
        y,x = (xp.indices((npix, npix)) - npix//2 + 1/2)*pixelscale
    else:
        y,x = (xp.indices((npix, npix)) - npix//2)*pixelscale
    return x,y

def pad_or_crop( arr_in, npix ):
    n_arr_in = arr_in.shape[0]
    if n_arr_in == npix:
        return arr_in
    elif npix < n_arr_in:
        x1 = n_arr_in // 2 - npix // 2
        x2 = x1 + npix
        arr_out = arr_in[x1:x2,x1:x2].copy()
    else:
        arr_out = xp.zeros((npix,npix), dtype=arr_in.dtype)
        x1 = npix // 2 - n_arr_in // 2
        x2 = x1 + n_arr_in
        arr_out[x1:x2,x1:x2] = arr_in
    return arr_out

def imshow(
        arrs,
        titles=[], 
        xlabels=[],
        ylabels=[],
        title_fzs=[],
        label_fzs=[],
        pxscls=[],
        npix=[],
        cmaps=[],
        norms=[],
        cbar_labels=[],
        grids=[],
        xticks=[],
        yticks=[], 
        all_patches=[],
        figsize=None,
        dpi=125,
        Nrows=1,
        Ncols=None, 
        wspace=None, 
        hspace=None, 
        return_fig=False,
    ):

    Nax = len(arrs)
    titles.extend([None] * (Nax - len(titles)))
    xlabels.extend([None] * (Nax - len(xlabels)))
    ylabels.extend([None] * (Nax - len(ylabels)))
    title_fzs.extend([None] * (Nax - len(title_fzs)))
    label_fzs.extend([None] * (Nax - len(label_fzs)))
    cmaps.extend(['magma'] * (Nax - len(cmaps)))
    norms.extend([None] * (Nax - len(norms)))
    cbar_labels.extend([None] * (Nax - len(cbar_labels)))
    grids.extend([None] * (Nax - len(grids)))
    xticks.extend([None] * (Nax - len(xticks)))
    yticks.extend([None] * (Nax - len(yticks)))
    pxscls.extend([None] * (Nax - len(pxscls)))
    npix.extend([None] * (Nax - len(npix)))
    all_patches.extend([None] * (Nax - len(all_patches)))

    if figsize is None:
        if Nax==1:
            figsize = (4,4)
        elif Nax==2:
            figsize = (10,4)
        elif Nax==3:
            figsize = (16,4)
        else:
            figsize = (10,10)
    
    if Nrows==1 and Ncols is None:
        Ncols = Nax
    fig, axs = plt.subplots(nrows=Nrows, ncols=Ncols, figsize=figsize, dpi=dpi)
    print(np.ndim(axs))

    row_ind = 0
    col_ind = 0
    for i in range(Nax):
        arr = arrs[i]
        title = titles[i]
        xlabel = xlabels[i]
        ylabel = ylabels[i]
        title_fz = title_fzs[i]
        label_fz = label_fzs[i]
        cmap = cmaps[i]
        norm = norms[i]
        cbar_label = cbar_labels[i]
        xtick = xticks[i]
        ytick = yticks[i]
        pxscl = pxscls[i]
        grid = grids[i]
        patches = all_patches[i]
        narr = npix[i]

        if narr is not None: 
            arr = pad_or_crop(arr, narr)

        Nwidth = arr.shape[1]
        Nheight = arr.shape[0]
        extent = None if pxscl is None else [-Nwidth/2*pxscl, Nwidth/2*pxscl, -Nheight/2*pxscl, Nheight/2*pxscl]

        if np.ndim(axs)==0:
            ax = axs
        elif np.ndim(axs)==1:
            ax = axs[i]
        elif np.ndim(axs)==2:
            row_ind = i//Ncols
            col_ind = i%Ncols
            ax = axs[row_ind, col_ind]

        im = ax.imshow(ensure_np_array(arr), cmap=cmap, norm=norm, extent=extent)
        ax.set_title(title, fontsize=title_fz)
        ax.set_xlabel(xlabel, fontsize=label_fz)
        ax.set_ylabel(ylabel, fontsize=label_fz)
        if xtick is not None: ax.set_xticks(xtick)
        if ytick is not None: ax.set_yticks(ytick)
        if grid is not None: ax.grid()
        if patches is not None: 
            for patch in patches:
                ax.add_patch(patch)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4%", pad=0.075)
        cbar = fig.colorbar(im, cax=cax)
        cbar.ax.set_ylabel(cbar_label, rotation=0, labelpad=7)
    
    plt.subplots_adjust(wspace=wspace, hspace=hspace)
    plt.close()
    
    if return_fig:
        return fig, axs
    else:
        display(fig)

def save_fits(fpath, data, header=None, ow=True, quiet=False):
    data = ensure_np_array(data)
    if header is not None:
        keys = list(header.keys())
        hdr = fits.Header()
        for i in range(len(header)):
            hdr[keys[i]] = header[keys[i]]
    else: 
        hdr = None
    hdu = fits.PrimaryHDU(data=data, header=hdr)
    hdu.writeto(str(fpath), overwrite=ow) 
    if not quiet: print('Saved data to: ', str(fpath))

def load_fits(fpath, header=False):
    data = xp.array(fits.getdata(fpath))
    if header:
        hdr = fits.getheader(fpath)
        return data, hdr
    else:
        return data

def save_pickle(fpath, data, quiet=False):
    out = open(str(fpath), 'wb')
    pickle.dump(data, out)
    out.close()
    if not quiet: print('Saved data to: ', str(fpath))

def load_pickle(fpath):
    infile = open(str(fpath),'rb')
    pkl_data = pickle.load(infile)
    infile.close()
    return pkl_data  

def shift_arr(
        arr,
        x_shift=0, 
        y_shift=0,
        order=0,
    ):
    shifted = xcipy.ndimage.rotate(xp.array(arr), (y_shift, x_shift), order=order)
    return shifted

def rotate_arr(
        arr, 
        rotation, 
        reshape=False, 
        order=3,
    ):
    if arr.dtype == complex:
        arr_r = xcipy.ndimage.rotate(xp.real(arr), angle=rotation, reshape=reshape, order=order)
        arr_i = xcipy.ndimage.rotate(xp.imag(arr), angle=rotation, reshape=reshape, order=order)
        rotated_arr = arr_r + 1j*arr_i
    else:
        rotated_arr = xcipy.ndimage.rotate(arr, angle=rotation, reshape=reshape, order=order)
    return rotated_arr

def interp_arr(
        arr, 
        pixelscale, 
        new_pixelscale, 
        order=1,
    ):
    Nold = arr.shape[0]
    old_xmax = pixelscale * (Nold/2)
    Nnew = 2*int(np.round(old_xmax/new_pixelscale))
    new_xmax = new_pixelscale * (Nnew/2)

    x,y = xp.ogrid[-old_xmax:old_xmax - pixelscale:Nold*1j,
                   -old_xmax:old_xmax - pixelscale:Nold*1j]

    newx,newy = xp.mgrid[-new_xmax:new_xmax-new_pixelscale:Nnew*1j,
                         -new_xmax:new_xmax-new_pixelscale:Nnew*1j]
    
    x0 = x[0,0]
    y0 = y[0,0]
    dx = x[1,0] - x0
    dy = y[0,1] - y0

    ivals = (newx - x0)/dx
    jvals = (newy - y0)/dy

    coords = xp.array([ivals, jvals])

    interped_arr = xcipy.ndimage.map_coordinates(arr, coords, order=order)
    return interped_arr

def create_zernike_modes(pupil_mask, nmodes=15, remove_modes=0):
    if remove_modes>0:
        nmodes += remove_modes
    zernikes = poppy.zernike.arbitrary_basis(pupil_mask, nterms=nmodes, outside=0)[remove_modes:]

    return zernikes

def lstsq(modes, data):
    """Least-Squares fit of modes to data.

    Parameters
    ----------
    modes : iterable
        modes to fit; sequence of ndarray of shape (m, n)
    data : numpy.ndarray
        data to fit, of shape (m, n)
        place NaN values in data for points to ignore

    Returns
    -------
    numpy.ndarray
        fit coefficients

    """
    mask = xp.isfinite(data)
    data = data[mask]
    modes = xp.asarray(modes)
    modes = modes.reshape((modes.shape[0], -1))  # flatten second dim
    modes = modes[:, mask.ravel()].T  # transpose moves modes to columns, as needed for least squares fit
    c, *_ = xp.linalg.lstsq(modes, data, rcond=None)
    return c

def TikhonovInverse(A, rcond=1e-15):
    U, s, Vt = xp.linalg.svd(A, full_matrices=False)
    s_inv = s/(s**2 + (rcond * s.max())**2)
    return (Vt.T * s_inv).dot(U.T)

def beta_reg(S, beta=-1, return_np=False):
    # S is the sensitivity matrix also known as the Jacobian
    if return_np: S = xp.array(S)
    sts = xp.matmul(S.T, S)
    rho = xp.diag(sts)
    alpha2 = rho.max()

    control_matrix = xp.matmul( xp.linalg.inv( sts + alpha2*10.0**(beta) * xp.eye(sts.shape[0]) ), S.T)
    if return_np:
        control_matrix = ensure_np_array(control_matrix)
    return control_matrix

def create_circ_mask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = (int(w//2), int(h//2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])
        
    Y, X = xp.ogrid[:h, :w]
    dist_from_center = xp.sqrt((X - center[0] + 1/2)**2 + (Y - center[1] + 1/2)**2)

    mask = dist_from_center <= radius
    return mask

def create_annular_mask(
        N, 
        pixelscale, 
        irad, 
        orad,  
        edge=None,
        x_shift=0,
        y_shift=0,
        rotation=0,
    ):
    x = (xp.linspace(-N/2, N/2-1, N) + 1/2) * pixelscale
    x,y = xp.meshgrid(x,x)
    r = xp.hypot(x, y)
    mask = (r > irad) * (r < orad)
    if edge is not None: mask *= (x > edge)
    
    mask = xcipy.ndimage.rotate(mask, rotation, reshape=False, order=0)
    mask = xcipy.ndimage.shift(mask, (y_shift, x_shift), order=0)
    
    return mask

def create_annular_focal_plane_mask(
        npsf, 
        psf_pixelscale, 
        irad, 
        orad,  
        edge=None,
        rotation=0,
    ):
    x = (xp.linspace(-npsf/2, npsf/2-1, npsf) + 1/2) * psf_pixelscale
    x,y = xp.meshgrid(x,x)
    r = xp.hypot(x, y)
    mask = (r > irad) * (r < orad)
    if edge is not None: mask *= (x > edge)
    
    mask = xcipy.ndimage.rotate(mask, rotation, reshape=False, order=0)
        
    return mask

# def create_hadamard_modes(dm_mask): 
#     Nacts = dm_mask.sum().astype(int)
#     Nact = dm_mask.shape[0]
#     np2 = 2**int(xp.ceil(xp.log2(Nacts)))
#     hmodes = xp.array(scipy.linalg.hadamard(np2))
    
#     had_modes = []

#     inds = xp.where(dm_mask.flatten().astype(int))
#     for hmode in hmodes:
#         hmode = hmode[:Nacts]
#         mode = xp.zeros((dm_mask.shape[0]**2))
#         mode[inds] = hmode
#         had_modes.append(mode)
#     had_modes = xp.array(had_modes).reshape(np2, Nact, Nact)
    
#     return had_modes
    
# def create_fourier_modes(
#         dm_mask, npsf, psf_pixelscale_lamD, iwa, owa, 
#         rotation=0, 
#         fourier_sampling=0.75,
#         which='both', 
#         return_fs=False,
#     ):
#     Nact = dm_mask.shape[0]
#     nfg = int(xp.round(npsf * psf_pixelscale_lamD/fourier_sampling))
#     if nfg%2==1: nfg += 1
#     yf, xf = (xp.indices((nfg, nfg)) - nfg//2 + 1/2) * fourier_sampling
#     # fourier_cm = create_annular_focal_plane_mask(nfg, fourier_sampling, iwa-fourier_sampling, owa+fourier_sampling, edge=iwa-fourier_sampling, rotation=rotation)
#     fourier_cm = create_annular_mask(
#         nfg, 
#         fourier_sampling, 
#         iwa-fourier_sampling, 
#         owa+fourier_sampling, 
#         edge=iwa-fourier_sampling, 
#         rotation=rotation,
#     )
#     ypp, xpp = (xp.indices((Nact, Nact)) - Nact//2 + 1/2)

#     sampled_fs = xp.array([xf[fourier_cm], yf[fourier_cm]]).T

#     fourier_modes = []
#     for i in range(len(sampled_fs)):
#         fx = sampled_fs[i,0]
#         fy = sampled_fs[i,1]
#         if which=='both' or which=='cos':
#             fourier_modes.append( dm_mask * xp.cos(2 * np.pi * (fx*xpp + fy*ypp)/Nact) )
#         if which=='both' or which=='sin':
#             fourier_modes.append( dm_mask * xp.sin(2 * np.pi * (fx*xpp + fy*ypp)/Nact) )
    
#     if return_fs:
#         return xp.array(fourier_modes), sampled_fs
#     else:
#         return xp.array(fourier_modes)

# def create_fourier_probes(
#         dm_mask, npsf, psf_pixelscale_lamD, iwa, owa, 
#         rotation=0, 
#         fourier_sampling=0.75, 
#         shifts=None, nprobes=2,
#         use_weighting=False, 
#     ): 
#     Nact = dm_mask.shape[0]

#     cos_modes, fs = create_fourier_modes(
#         dm_mask, npsf, psf_pixelscale_lamD, iwa, owa, rotation,
#         fourier_sampling=fourier_sampling, 
#         return_fs=True,
#         which='cos',
#     )

#     sin_modes = create_fourier_modes(
#         dm_mask, npsf, psf_pixelscale_lamD, iwa, owa, rotation,
#         fourier_sampling=fourier_sampling, 
#         which='sin',
#     )

#     nfs = fs.shape[0]

#     probes = xp.zeros((nprobes, Nact, Nact))
#     if use_weighting:
#         fmax = xp.max(np.sqrt(fs[:,0]**2 + fs[:,1]**2))
#         sum_cos = 0
#         sum_sin = 0
#         for i in range(nfs):
#             f = np.sqrt(fs[i][0]**2 + fs[i][1]**2)
#             weight = f/fmax
#             sum_cos += weight*cos_modes[i]
#             sum_sin += weight*sin_modes[i]
#         sum_cos = sum_cos
#         sum_sin = sum_sin
#     else:
#         sum_cos = cos_modes.sum(axis=0)
#         sum_sin = sin_modes.sum(axis=0)
    
#     # nprobes=2 will give one probe that is purely the sum of cos and another that is the sum of sin
#     cos_weights = np.linspace(1,0,nprobes)
#     sin_weights = np.linspace(0,1,nprobes)
    
#     shifts = [(0,0)]*nprobes if shifts is None else shifts

#     for i in range(nprobes):
#         probe = cos_weights[i]*sum_cos + sin_weights[i]*sum_sin
#         probe = xcipy.ndimage.shift(probe, (shifts[i][1], shifts[i][0]))
#         probes[i] = probe/xp.max(probe)

#     return probes

# def make_f(h=10, w=6, shift=(0,0), Nact=34):
#     f_command = xp.zeros((Nact, Nact))

#     top_row = Nact//2 + h//2 + shift[1]
#     mid_row = Nact//2 + shift[1]
#     row0 = Nact//2 - h//2 + shift[1]

#     col0 = Nact//2 - w//2 + shift[0] + 1
#     right_col = Nact//2 + w//2 + shift[0] + 1

#     rows = xp.arange(row0, top_row)
#     cols = xp.arange(col0, right_col)

#     f_command[rows, col0] = 1
#     f_command[top_row,cols] = 1
#     f_command[mid_row,cols] = 1
#     return f_command

# def make_ring(rad=15, Nact=34, thresh=1/2):
#     y,x = (xp.indices((Nact, Nact)) - Nact//2 + 1/2)
#     r = xp.sqrt(x**2 + y**2)
#     ring = (rad-thresh<r) * (r < rad+thresh)
#     ring = ring.astype(float)
#     return ring

# def make_fourier_command(x_cpa=10, y_cpa=10, Nact=34):
#     # cpa = cycles per aperture
#     # max cpa must be Nact/2
#     if x_cpa>Nact/2 or y_cpa>Nact/2:
#         raise ValueError('The cycles per aperture is too high for the specified number of actuators.')
#     y,x = xp.indices((Nact, Nact)) - Nact//2
#     fourier_command = xp.cos(2*np.pi*(x_cpa*x + y_cpa*y)/Nact)
#     return fourier_command

# def make_cross_command(xc=[0], yc=[0], Nact=34):
#     y,x = (xp.indices((Nact, Nact)) - Nact//2 + 1/2)
#     cross = xp.zeros((Nact,Nact))
#     for i in range(len(xc)):
#         cross[(xc[i]-0.5<=x) & (x<xc[i]+0.5)] = 1
#         cross[(yc[i]-0.5<=y) & (y<yc[i]+0.5)] = 1
#     # cross
#     return cross

def get_radial_dist(shape, scaleyx=(1.0, 1.0), cenyx=None):
    '''
    Compute the radial separation of each pixel
    from the center of a 2D array, and optionally 
    scale in x and y.
    '''
    indices = np.indices(shape)
    if cenyx is None:
        cenyx = ( (shape[0] - 1) / 2., (shape[1] - 1)  / 2.)
    radial = np.sqrt( (scaleyx[0]*(indices[0] - cenyx[0]))**2 + (scaleyx[1]*(indices[1] - cenyx[1]))**2 )
    return radial

def get_radial_contrast(im, mask, nbins=50, cenyx=None):
    im = ensure_np_array(im)
    mask = ensure_np_array(mask)
    radial = get_radial_dist(im.shape, cenyx=cenyx)
    bins = np.linspace(0, radial.max(), num=nbins, endpoint=True)
    digrad = np.digitize(radial, bins)
    profile = np.asarray([np.mean(im[ (digrad == i) & mask]) for i in np.unique(digrad)])
    return bins, profile
    
def plot_radial_contrast(im, mask, pixelscale, nbins=30, cenyx=None, xlims=None, ylims=None):
    bins, contrast = get_radial_contrast(im, mask, nbins=nbins, cenyx=cenyx)
    r = bins * pixelscale

    fig,ax = plt.subplots(nrows=1, ncols=1, dpi=125, figsize=(6,4))
    ax.semilogy(r,contrast)
    ax.set_xlabel('radial position [$\lambda/D$]')
    ax.set_ylabel('Contrast')
    ax.grid()
    if xlims is not None: ax.set_xlim(xlims[0], xlims[1])
    if ylims is not None: ax.set_ylim(ylims[0], ylims[1])
    plt.close()
    display(fig)

def measure_waffle_center_and_angle(
        waffle_im, 
        psf_pixelscale_lamD, 
        im_thresh=1e-4, 
        r_thresh_min=12,
        r_thresh_max=18, 
        verbose=True, 
        plot=True,
    ):
    npsf = waffle_im.shape[0]
    y,x = (xp.indices((npsf, npsf)) - npsf//2)*psf_pixelscale_lamD
    r = xp.sqrt(x**2 + y**2)
    waffle_mask = (waffle_im > im_thresh) * (r>r_thresh_min) * (r<r_thresh_max)

    centroids = []
    for i in [0,1]:
        for j in [0,1]:
            arr = waffle_im[j*npsf//2:(j+1)*npsf//2, i*npsf//2:(i+1)*npsf//2]
            mask = waffle_mask[j*npsf//2:(j+1)*npsf//2, i*npsf//2:(i+1)*npsf//2]
            cent = np.flip(skimage.measure.centroid(ensure_np_array(mask*arr)))
            cent[0] += i*npsf//2
            cent[1] += j*npsf//2
            centroids.append(cent)
            # print(cent)
            # imshow3(mask, arr, mask*arr, lognorm2=True,
            #         patches1=[Circle(cent, 1, fill=True, color='cyan')])
    centroids.append(centroids[0])
    centroids = np.array(centroids)
    centroids[[2,3]] = centroids[[3,2]]
    if verbose: print('Centroids:\n', centroids)

    if plot: 
        patches = []
        for i in range(4): patches.append(Circle(centroids[i], 1, fill=False, color='black'))
        imshow(
            [waffle_mask, waffle_im, waffle_mask*waffle_im], 
            norms=[LogNorm(np.max(waffle_im)/1e4)],
            all_patches=[patches],
        )

    mean_angle = 0.0
    for i in range(4):
        angle = np.arctan2(centroids[i+1][1] - centroids[i][1], centroids[i+1][0] - centroids[i][0]) * 180/np.pi
        if angle<0:
            angle += 360
        if 0<angle<90:
            angle = 90-angle
        elif 90<angle<180:
            angle = 180-angle
        elif 180<angle<270:
            angle = 270-angle
        elif 270<angle<360:
            angle = 360-angle
        mean_angle += angle/4
    if verbose: print('Angle: ', mean_angle)

    m1 = (centroids[0][1] - centroids[2][1])/(centroids[0][0] - centroids[2][0])
    m2 = (centroids[1][1] - centroids[3][1])/(centroids[1][0] - centroids[3][0])
    # print(m1,m2)
    b1 = -m1*centroids[0][0] + centroids[0][1]
    b2 =  -m2*centroids[1][0] + centroids[1][1]
    # print(b1,b2)

    # m1*x + b1 = m2*x + b2
    # (m1-m2) * x = b2 - b1
    xc = (b2 - b1) / (m1 - m2)
    yc = m1*xc + b1
    print('Measured center in X: ', xc)
    print('Measured center in Y: ', yc)

    xshift = np.round(npsf/2 - xc)
    yshift = np.round(npsf/2 - yc)
    print('Required shift in X: ', xshift)
    print('Required shift in Y: ', yshift)

    return xshift, yshift, mean_angle




