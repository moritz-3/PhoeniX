# -*- coding: utf-8 -*-
"""
Visualization tools to display processed or analyzed data in a nice and coherent manner
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def show_incoherent_img(inc_img, mask=None, v_limits='auto'):
    if mask is None:
        if v_limits == 'auto':
            v_limits = (np.percentile(inc_img, 10), np.percentile(inc_img, 90)) # automatically set the color bar limits within [10%-90%] of the data range
        plt.figure()        
        img = plt.imshow(inc_img, vmin=v_limits[0], vmax=v_limits[1], origin='lower')
        plt.colorbar(img)
        plt.xlabel(r'Pixel$_x$')
        plt.ylabel(r'Pixel$_y$')
        plt.tight_layout()
        plt.show()
    else:
        inc_img_cor = np.multiply(inc_img, mask.array)
        if v_limits == 'auto':
            v_limits = (np.percentile(inc_img_cor, 10), np.percentile(inc_img_cor, 90)) # automatically set the color bar limits within [10%-90%] of the data range
        fig, ax = plt.subplots(1,3, figsize=(12,4))
        img_original = ax[0].imshow(inc_img, vmin=v_limits[0], vmax=v_limits[1], origin='lower')
        fig.colorbar(img_original, ax=ax[0])
        ax[0].set_title('Incoherent Img.')
        ax[0].set_xlabel(r'Pixel$_x$')
        ax[0].set_ylabel(r'Pixel$_y$')
        # mask
        ax[1].imshow(mask.array, origin='lower')
        ax[1].set_title('Mask')
        ax[1].set_xlabel(r'Pixel$_x$')
        ax[1].set_ylabel(r'Pixel$_y$')
        # corrected image
        img_cor = ax[2].imshow(inc_img_cor, vmin=v_limits[0], vmax=v_limits[1], origin='lower')
        fig.colorbar(img_cor, ax=ax[2])
        ax[2].set_title('Corrected Img.')
        ax[2].set_xlabel(r'Pixel$_x$')
        ax[2].set_ylabel(r'Pixel$_y$')
        plt.tight_layout()
        plt.show()
        
def show_SAXS_curve(SAXS_Q, SAXS_IofQ, SAXS_errorIofQ=None, y_limits='auto'):
    if y_limits == 'auto':
        y_limits = (0.9*np.nanmin(SAXS_IofQ),1.1*np.nanmax(SAXS_IofQ))
    plt.figure()
    if SAXS_errorIofQ is None:
        plt.plot(SAXS_Q*1e-9, SAXS_IofQ)
    else:
        plt.errorbar(SAXS_Q*1e-9, SAXS_IofQ, yerr=SAXS_errorIofQ)
    plt.xlabel(r'Q (nm$^{-1}$)')
    plt.ylabel('Counts (arb. u.)')
    plt.yscale('log')
    plt.ylim(y_limits)
    plt.show()

def show_evolution_plots(common_x, all_y, norm_reference_idx=0):
    colorbar_limits = (np.percentile(all_y, 10), np.percentile(all_y, 90))
    
    # waterfall plot
    waterfall_dim_y, waterfall_dim_x = all_y.shape
    fig, ax = plt.subplots(1,3, figsize=(14,4))
    pcm = ax[0].pcolormesh(common_x, np.arange(waterfall_dim_y), all_y, shading='auto', cmap='viridis', norm=LogNorm(vmin=colorbar_limits[0], vmax=colorbar_limits[1]))
    plt.colorbar(pcm, label='Intensity (arb. u.)')
    ax[0].set_ylabel('#Steps')
    ax[0].set_title('Absolute Intensity')

    # waterfall plot as difference to reference
    relative_data = (all_y - all_y[norm_reference_idx,:]) / all_y[norm_reference_idx,:]
    relative_data_limits = (np.percentile(relative_data, 10), np.percentile(relative_data, 90))
    pcm = ax[1].pcolormesh(common_x, np.arange(waterfall_dim_y), relative_data, shading='auto', cmap='viridis', vmin=relative_data_limits[0], vmax=relative_data_limits[1])#, norm=LogNorm(vmin=relative_data_limits[0], vmax=relative_data_limits[1]))
    plt.colorbar(pcm, label='Relative Intensity (arb. u.)')
    ax[1].set_ylabel('#Steps')
    ax[1].set_title('Relative Intensity')
    
    # change in data all plotted together
    cmap = plt.get_cmap('viridis')
    colors = cmap(np.linspace(0, 1, waterfall_dim_y))
    for idx_y in range(waterfall_dim_y):
        if idx_y == 0:
            ax[2].semilogy(common_x, all_y[idx_y,:], color=colors[idx_y], label='Start')
        elif idx_y == waterfall_dim_y - 1:
            ax[2].semilogy(common_x, all_y[idx_y,:], color=colors[idx_y], label='End')
        else:
            ax[2].semilogy(common_x, all_y[idx_y,:], color=colors[idx_y], label='')
    plt.legend()
    plt.tight_layout() 
    plt.show()

def get_auto_plot_limits_ttcf(ttcf, percentil_min=25, percentil_max=75):
    tmp = np.copy(ttcf)
    np.fill_diagonal(tmp, np.nan)
    v_limits = (np.nanpercentile(tmp, percentil_min), np.nanpercentile(tmp, percentil_max))
    return v_limits

def show_single_ttcf(ttcf, v_limits='auto'):
    if v_limits == 'auto':
        v_limits = get_auto_plot_limits_ttcf(ttcf)
    plt.figure()
    img = plt.imshow(ttcf, vmin=v_limits[0], vmax=v_limits[1], origin='lower')
    plt.colorbar(img)
    plt.xlabel('#frames')
    plt.ylabel('#frames')
    plt.tight_layout()
    plt.show()

def show_all_ttcf(all_ttcf, v_limits='auto'):
    number_ttcfs, _, _ = all_ttcf.shape
    plot_columns = 4
    plot_rows = int(np.ceil(number_ttcfs / plot_columns))
    subfigure_size = 3
    fig, axes = plt.subplots(plot_rows, plot_columns, figsize=(plot_columns * subfigure_size, plot_rows*subfigure_size))
    axes = axes.flatten()
    if v_limits == 'auto':
        auto_determine_limits = True
    for i in range(number_ttcfs):
        if auto_determine_limits:
            v_limits = get_auto_plot_limits_ttcf(all_ttcf[i,:,:], percentil_min=40, percentil_max=60)
        img = axes[i].imshow(all_ttcf[i,:,:], vmin=v_limits[0], vmax=v_limits[1], origin='lower')
        plt.colorbar(img)
        axes[i].set_xlabel('#frames')
        axes[i].set_ylabel('#frames')
        axes[i].set_title(f'TTCF #{i}')
    plt.tight_layout()
    plt.show()
        
def show_single_g2(times_s, g2, g2error=None, y_limits='auto'):
    plt.figure()
    if g2error is None:
        plt.plot(times_s, g2)
    else:
        plt.errorbar(times_s, g2, yerr=g2error)
    plt.xscale('log')
    if y_limits != 'auto':
        plt.ylim(y_limits)
    plt.xlabel('Time (s)')
    plt.ylabel('g2')
    plt.show()

def show_all_g2s(times_s, g2s, g2errors=None, y_limits='auto'):
    number_g2s = g2s.shape[0]
    cmap = plt.get_cmap('viridis')
    colors = cmap(np.linspace(0, 1, number_g2s))
    plt.figure()
    for i in range(number_g2s):
        if g2errors is None:
            plt.plot(times_s, g2s[i,:], color=colors[i], label=f'g2 #{i}')
        else:
            plt.errorbar(times_s, g2s[i,:], yerr=g2errors[i,:], color=colors[i], label=f'g2 #{i}')
    plt.xscale('log')
    if y_limits != 'auto':
        plt.ylim(y_limits)
    plt.xlabel('Time (s)')
    plt.ylabel('g2')
    plt.legend(loc="upper left",bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    plt.show()

def show_all_g2s_separately(times_s, g2s, g2errors=None, y_limits='auto'):
    number_g2s = g2s.shape[0]
    plot_columns = 4
    plot_rows = int(np.ceil(number_g2s / plot_columns))
    subfigure_size = 3
    fig, axes = plt.subplots(plot_rows, plot_columns, figsize=(plot_columns * subfigure_size, plot_rows*subfigure_size))
    axes = axes.flatten()
    for i in range(number_g2s):
        if g2errors is None:
            axes[i].plot(times_s, g2s[i,:])
        else:
            axes[i].errorbar(times_s, g2s[i,:], yerr=g2errors[i,:])
        if y_limits != 'auto':
            axes[i].ylim(y_limits)
        axes[i].set_xlabel('Time (s)')
        axes[i].set_ylabel('g2')
        axes[i].set_xscale('log')
        axes[i].set_title(f'g2 #{i}')
    plt.tight_layout()
    plt.show()
    
""" Include show q_rings function that can be called with QRings class
# def show_q_rings(self, SAXS_Q, SAXS_IofQ, y_limits=None):
#     fig, ax = plt.subplots()
#     ax.plot(SAXS_Q*1e-9, SAXS_IofQ)
#     if y_limits is None:
#         y_limits = (0.9*np.nanmin(SAXS_IofQ), 1.1*np.nanmax(SAXS_IofQ))
#     ax.set_yscale('log')
#     ax.set_ylim(y_limits)
#     q_centers = self.info['q_centers']
#     q_widths = self.info['q_widths']
#     q_mins = q_centers - q_widths/2
#     num_rings = len(q_widths)
#     # define colors for ymap
#     cmap = plt.get_cmap('jet')
#     colors = cmap(np.linspace(0, 1, num_rings))
#     for i in range(num_rings):
#         rectangle = patches.Rectangle((q_mins[i]*1e-9, y_limits[0]), q_widths[i]*1e-9, y_limits[1] - y_limits[0], alpha=0.4, edgecolor='black', facecolor=colors[i], linewidth=1)
#         ax.add_patch(rectangle)
#     plt.show()
"""