from .math_module import xp, xcipy, ensure_np_array
from scoob_llowfsc import utils

import numpy as np
import scipy
import astropy.units as u

import matplotlib.pyplot as plt
plt.rcParams['image.origin'] = 'lower'
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm, Normalize, CenteredNorm
from IPython.display import display, clear_output

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
    cmaps.extend([None] * (Nax - len(cmaps)))
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
            figsize = (15,7)
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
            arr = utils.pad_or_crop(arr, narr)

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

def imshow1(
        arr, 
        title=None, 
        xlabel=None,
        npix=None,
        lognorm=False, 
        vmin=None, 
        vmax=None,
        cmap='magma',
        pxscl=None,
        axlims=None,
        patches=None,
        grid=False, 
        figsize=(4,4), 
        dpi=125, 
        display_fig=True, 
        return_fig=False,
    ):
    fig,ax = plt.subplots(nrows=1, ncols=1, figsize=figsize, dpi=dpi)
    
    if npix is not None:
        arr = utils.pad_or_crop(arr, npix)

    arr = ensure_np_array(arr)
    
    if pxscl is not None:
        if isinstance(pxscl, u.Quantity):
            pxscl = pxscl.value
        vext = pxscl * arr.shape[0]/2
        hext = pxscl * arr.shape[1]/2
        extent = [-vext,vext,-hext,hext]
    else:
        extent=None
    
    norm = LogNorm(vmin=vmin,vmax=vmax) if lognorm else Normalize(vmin=vmin,vmax=vmax)
    
    im = ax.imshow(arr, cmap=cmap, norm=norm, extent=extent)
    if axlims is not None:
        ax.set_xlim(axlims[:2])
        ax.set_ylim(axlims[2:])
    ax.tick_params(axis='x', labelsize=9, rotation=30)
    ax.tick_params(axis='y', labelsize=9, rotation=30)
    ax.set_xlabel(xlabel)
    if patches: 
        for patch in patches:
            ax.add_patch(patch)
    ax.set_title(title)
    if grid: ax.grid()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.075)
    fig.colorbar(im, cax=cax)
    plt.close()
    
    if display_fig: 
        display(fig)
    if return_fig: 
        return fig,ax
    
def imshow2(
        arr1, arr2, 
        title1=None, title2=None,
        xlabel=None, xlabel1=None, xlabel2=None,
        npix=None, npix1=None, npix2=None,
        pxscl=None, pxscl1=None, pxscl2=None,
        axlims=None, axlims1=None, axlims2=None,
        grid=False, grid1=False, grid2=False,
        cmap1='magma', cmap2='magma',
        lognorm=False, lognorm1=False, lognorm2=False,
        vmin1=None, vmax1=None, vmin2=None, vmax2=None,
        patches1=None, patches2=None,
        display_fig=True, 
        return_fig=False, 
        figsize=(10,4), dpi=125, wspace=0.2,
    ):
    fig,ax = plt.subplots(nrows=1, ncols=2, figsize=figsize, dpi=dpi)
    
    npix1, npix2 = (npix, npix) if npix is not None else (npix1, npix2)
    if npix1 is not None: arr1 = utils.pad_or_crop(arr1, npix1)
    if npix2 is not None: arr2 = utils.pad_or_crop(arr2, npix2)
    
    arr1 = ensure_np_array(arr1)
    arr2 = ensure_np_array(arr2)

    pxscl1, pxscl2 = (pxscl, pxscl) if pxscl is not None else (pxscl1, pxscl2)
    if pxscl1 is not None:
        if isinstance(pxscl1, u.Quantity):
            vext = pxscl1.value * arr1.shape[0]/2
            hext = pxscl1.value * arr1.shape[1]/2
            extent1 = [-vext,vext,-hext,hext]
        else:
            vext = pxscl1 * arr1.shape[0]/2
            hext = pxscl1 * arr1.shape[1]/2
            extent1 = [-vext,vext,-hext,hext]
    else:
        extent1=None
        
    if pxscl2 is not None:
        if isinstance(pxscl2, u.Quantity):
            vext = pxscl2.value * arr2.shape[0]/2
            hext = pxscl2.value * arr2.shape[1]/2
            extent2 = [-pxscl2.value *arr2.shape[0]/2,vext,-hext,hext]
        else:
            vext = pxscl2 * arr2.shape[0]/2
            hext = pxscl2 * arr2.shape[1]/2
            extent2 = [-vext,vext,-hext,hext]
    else:
        extent2=None
    
    axlims1, axlims2 = (axlims, axlims) if axlims is not None else (axlims1, axlims2) # overide axlims
    xlabel1, xlabel2 = (xlabel, xlabel) if xlabel is not None else (xlabel1, xlabel2)
    
    norm1 = LogNorm(vmin=vmin1,vmax=vmax1) if lognorm1 or lognorm else Normalize(vmin=vmin1,vmax=vmax1)
    norm2 = LogNorm(vmin=vmin2,vmax=vmax2) if lognorm2 or lognorm else Normalize(vmin=vmin2,vmax=vmax2)
    
    # first plot
    im = ax[0].imshow(arr1, cmap=cmap1, norm=norm1, extent=extent1)
    if axlims1 is not None:
        ax[0].set_xlim(axlims1[:2])
        ax[0].set_ylim(axlims1[2:])
    if grid or grid1: ax[0].grid()
    ax[0].tick_params(axis='x', labelsize=9, rotation=30)
    ax[0].tick_params(axis='y', labelsize=9, rotation=30)
    ax[0].set_xlabel(xlabel1)
    if patches1: 
        for patch1 in patches1:
            ax[0].add_patch(patch1)
    ax[0].set_title(title1)
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes("right", size="4%", pad=0.075)
    fig.colorbar(im, cax=cax)
    
    # second plot
    im = ax[1].imshow(arr2, cmap=cmap2, norm=norm2, extent=extent2)
    if axlims2 is not None:
        ax[1].set_xlim(axlims2[:2])
        ax[1].set_ylim(axlims2[2:])
    if grid or grid2: ax[1].grid()
    ax[1].tick_params(axis='x', labelsize=9, rotation=30)
    ax[1].tick_params(axis='y', labelsize=9, rotation=30)
    ax[1].set_xlabel(xlabel2)
    if patches2: 
        for patch2 in patches2:
            ax[1].add_patch(patch2)
    ax[1].set_title(title2)
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes("right", size="4%", pad=0.075)
    fig.colorbar(im, cax=cax)
        
    plt.subplots_adjust(wspace=wspace)
    
    plt.close()
    
    if display_fig: display(fig)
    if return_fig: return fig,ax

def imshow3(
        arr1, arr2, arr3,
        title1=None, title2=None, title3=None, titlesize=12,
        npix=None, npix1=None, npix2=None, npix3=None,
        pxscl=None, pxscl1=None, pxscl2=None, pxscl3=None, 
        axlims=None, axlims1=None, axlims2=None, axlims3=None,
        xlabel=None, xlabel1=None, xlabel2=None, xlabel3=None,
        cmap1='magma', cmap2='magma', cmap3='magma',
        lognorm=False, lognorm1=False, lognorm2=False, lognorm3=False,
        vmin1=None, vmax1=None, vmin2=None, vmax2=None, vmin3=None, vmax3=None, 
        patches1=None, patches2=None, patches3=None,
        grid=False, grid1=False, grid2=False, grid3=False,
        display_fig=True, 
        return_fig=False,
        figsize=(14,7), dpi=125, wspace=0.3
    ):
    fig,ax = plt.subplots(nrows=1, ncols=3, figsize=figsize, dpi=dpi)
    
    npix1, npix2, npix3 = (npix, npix, npix) if npix is not None else (npix1, npix2, npix3)
    if npix1 is not None: arr1 = utils.pad_or_crop(arr1, npix1)
    if npix2 is not None: arr2 = utils.pad_or_crop(arr2, npix2)
    if npix3 is not None: arr3 = utils.pad_or_crop(arr3, npix3)

    arr1 = ensure_np_array(arr1)
    arr2 = ensure_np_array(arr2)
    arr3 = ensure_np_array(arr3)
    
    pxscl1, pxscl2, pxscl3 = (pxscl, pxscl, pxscl) if pxscl is not None else (pxscl1, pxscl2, pxscl3)
    if pxscl1 is not None:
        if isinstance(pxscl1, u.Quantity):
            vext = pxscl1.value * arr1.shape[0]/2
            hext = pxscl1.value * arr1.shape[1]/2
            extent1 = [-vext,vext,-hext,hext]
        else:
            vext = pxscl1 * arr1.shape[0]/2
            hext = pxscl1 * arr1.shape[1]/2
            extent1 = [-vext,vext,-hext,hext]
    else:
        extent1=None
        
    if pxscl2 is not None:
        if isinstance(pxscl2, u.Quantity):
            vext = pxscl2.value * arr2.shape[0]/2
            hext = pxscl2.value * arr2.shape[1]/2
            extent2 = [-vext,vext,-hext,hext]
        else:
            vext = pxscl2 * arr2.shape[0]/2
            hext = pxscl2 * arr2.shape[1]/2
            extent2 = [-vext,vext,-hext,hext]
    else:
        extent2=None
        
    if pxscl3 is not None:
        if isinstance(pxscl3, u.Quantity):
            vext = pxscl3.value * arr3.shape[0]/2
            hext = pxscl3.value * arr3.shape[1]/2
            extent3 = [-vext,vext,-hext,hext]
        else:
            vext = pxscl3 * arr3.shape[0]/2
            hext = pxscl3 * arr3.shape[1]/2
            extent3 = [-vext,vext,-hext,hext]
    else:
        extent3 = None
    
    axlims1, axlims2, axlims3 = (axlims, axlims, axlims) if axlims is not None else (axlims1, axlims2, axlims3) # overide axlims
    xlabel1, xlabel2, xlabel3 = (xlabel, xlabel, xlabel) if xlabel is not None else (xlabel1, xlabel2, xlabel3)
    
    norm1 = LogNorm(vmin=vmin1,vmax=vmax1) if lognorm1 or lognorm else Normalize(vmin=vmin1,vmax=vmax1)
    norm2 = LogNorm(vmin=vmin2,vmax=vmax2) if lognorm2 or lognorm else Normalize(vmin=vmin2,vmax=vmax2)
    norm3 = LogNorm(vmin=vmin3,vmax=vmax3) if lognorm3 or lognorm else Normalize(vmin=vmin3,vmax=vmax3)
    
    # first plot
    im = ax[0].imshow(arr1, cmap=cmap1, norm=norm1, extent=extent1)
    if axlims1 is not None:
        ax[0].set_xlim(axlims1[:2])
        ax[0].set_ylim(axlims1[2:])
    if grid or grid1: ax[0].grid()
    ax[0].tick_params(axis='x', labelsize=9, rotation=30)
    ax[0].tick_params(axis='y', labelsize=9, rotation=30)
    ax[0].set_xlabel(xlabel1)
    if patches1: 
        for patch1 in patches1:
            ax[0].add_patch(patch1)
    ax[0].set_title(title1)
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes("right", size="4%", pad=0.075)
    fig.colorbar(im, cax=cax)
    
    # second plot
    im = ax[1].imshow(arr2, cmap=cmap2, norm=norm2, extent=extent2)
    if axlims2 is not None:
        ax[1].set_xlim(axlims2[:2])
        ax[1].set_ylim(axlims2[2:])
    if grid or grid2: ax[1].grid()
    ax[1].tick_params(axis='x', labelsize=9, rotation=30)
    ax[1].tick_params(axis='y', labelsize=9, rotation=30)
    ax[1].set_xlabel(xlabel2)
    if patches2: 
        for patch2 in patches2:
            ax[1].add_patch(patch2)
    ax[1].set_title(title2)
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes("right", size="4%", pad=0.075)
    fig.colorbar(im, cax=cax)
    
    # third plot
    im = ax[2].imshow(arr3, cmap=cmap3, norm=norm3, extent=extent3)
    if axlims3 is not None:
        ax[2].set_xlim(axlims3[:2])
        ax[2].set_ylim(axlims3[2:])
    if grid or grid3: ax[2].grid()
    ax[2].tick_params(axis='x', labelsize=9, rotation=30)
    ax[2].tick_params(axis='y', labelsize=9, rotation=30)
    ax[2].set_xlabel(xlabel3)
    if patches3: 
        for patch3 in patches3:
            ax[2].add_patch(patch3)
    ax[2].set_title(title3)
    divider = make_axes_locatable(ax[2])
    cax = divider.append_axes("right", size="4%", pad=0.075)
    fig.colorbar(im, cax=cax)
        
    plt.subplots_adjust(wspace=wspace)
    
    plt.close()
    
    if display_fig: display(fig)
    if return_fig: return fig,ax

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

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LogNorm

def plot_howfsc(
        data, 
        dm_flat, 
        im1vmin=1e-9, im1vmax=1e-4,
        im2vmin=1e-9, im2vmax=1e-4, 
        vmin=1e-9, vmax=1e-4, 
        xticks=None,
        title_fz = 16,
        plot_position=[0.6, 0.3, 0.4, 0.4],
        exp_name='',
        hspace=None, wspace=None, 
        figsize=(20,15), 
        dpi=125,
        fname=None,
    ):
    flat_command = ensure_np_array(dm_flat) * 1e9
    ims = ensure_np_array( xp.array(data['images']) ) 
    control_mask = ensure_np_array( data['control_mask'] )

    # print(type(control_mask))
    Nitr = ims.shape[0]
    npsf = ims.shape[1]
    psf_pixelscale_lamD = data['pixelscale']
    ext = psf_pixelscale_lamD*npsf/2
    extent = [-ext, ext, -ext, ext]

    mean_nis = np.mean(ims[:,control_mask], axis=1)
    ibest = np.argmin(mean_nis)
    ref_im = ensure_np_array(data['images'][0])
    best_im = ensure_np_array(data['images'][ibest])
    best_command = ensure_np_array(data['commands'][ibest-1]) * 1e9

    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = GridSpec(2, 3, figure=fig)

    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(flat_command, cmap='viridis')
    ax.set_title('DM Flat Command', fontsize=title_fz)
    # ax.set_xticks([])
    # ax.set_yticks([])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.075)
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.set_ylabel('nm', rotation=0, labelpad=7)

    ax = fig.add_subplot(gs[0, 1])
    im = ax.imshow(best_command, cmap='viridis',)
    ax.set_title('Best HOWFSC Command', fontsize=title_fz)
    # ax.set_xticks([])
    # ax.set_yticks([])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.075)
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.set_ylabel('nm', rotation=0, labelpad=7)

    ax = fig.add_subplot(gs[1, 0])
    im = ax.imshow(ref_im, norm=LogNorm(vmax=im1vmax, vmin=im1vmin), cmap='magma', extent=extent)
    ax.set_title(f'Initial Image:\nMean Contrast = {mean_nis[0]:.2e}', fontsize=title_fz)
    ax.set_ylabel('Y [$\lambda/D$]', fontsize=12, labelpad=-5)
    ax.set_xlabel('X [$\lambda/D$]', fontsize=12, labelpad=5)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.075)
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.set_ylabel('NI', rotation=0, labelpad=7)

    ax = fig.add_subplot(gs[1, 1])
    im = ax.imshow(best_im, norm=LogNorm(vmax=im2vmax, vmin=im2vmin), cmap='magma', extent=extent)
    ax.set_title('Best Iteration' + exp_name + f':\nMean Contrast = {mean_nis[ibest]:.2e}', fontsize=title_fz)
    # ax.set_ylabel('Y [$\lambda/D$]', fontsize=12, labelpad=-5)
    ax.set_xlabel('X [$\lambda/D$]', fontsize=12, labelpad=5)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.075)
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.set_ylabel('NI', rotation=0, labelpad=7)
    
    plt.subplots_adjust(wspace=wspace, hspace=hspace)

    ax = fig.add_subplot(gs[:, 2])
    ax.set_title('Mean Contrast per Iteration' + exp_name, fontsize=14)
    ax.semilogy(mean_nis, label='3.6% Bandpass')
    ax.grid()
    ax.set_xlabel('Iteration Number', fontsize=12, )
    ax.set_ylabel('Mean Contrast', fontsize=14, labelpad=1)
    ax.set_ylim([vmin, vmax])
    xticks = np.arange(0,Nitr,2) if xticks is None else xticks
    ax.set_xticks(xticks)
    ax.set_position(plot_position)    

    if fname is not None: fig.savefig(fname, format='pdf', bbox_inches="tight")
    
    