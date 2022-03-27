#!/usr/local/bin/python

import os
import glob
import numpy as np
import pandas as pd
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, ScalarFormatter

plt.rcParams['mathtext.fontset'] = 'stix'

# Plotting
def make_all(
    # basic
    save=True,
    show=True,
    label_size=28.,
    tick_size=22.,
    LW=1.25,
    major=15.,
    minor=10.,
    direction='inout',
    # ensemble plot
    path_to_dists='data/distributions/dteff_100_K_dlogg_0.1_dex/*',
    double=True,
    which='teff',
    res_teff=100.,
    res_logg=0.1,
    # hr diagram
    path_to_sample='data/sara_revised.csv', 
    continuous=True,
    discrete=True,
    lower_x=6650.,
    upper_x=3650.,
    lower_y=5.15,
    upper_y=3.62,
    ):

    ensemble_plot(path_to_dists=path_to_dists, double=double, which=which,)
    rotation_hists(path_to_sample=path_to_sample,)
    rotation_hrdiagram(path_to_sample=path_to_sample,)
    
    

def get_dists(path_to_dists):
    xs, ys, t, l = [], [], [], []
    if glob.glob(path_to_dists):
        for file in glob.glob(path_to_dists):
            values = '.'.join(file.split('/')[-1].split('.')[:-1]).split('_')
            t.append((float(values[1]) + float(values[2]))/2.)
            l.append((float(values[4]) + float(values[5]))/2.)
            with open(file, "r") as f:
                lines = [line for line in f.readlines() if not line.startswith("#")]
            xs.append([float(line.strip().split()[0]) for line in lines])
            ys.append([float(line.strip().split()[1]) for line in lines])
    else:
        print('ERROR: incorrect path to distributions.\n      Please try again.')
    return xs, ys, t, l


def multiline(xs, ys, c, ax=None, **kwargs,):
    from matplotlib.collections import LineCollection
    # find axes
    ax = plt.gca() if ax is None else ax
    # create LineCollection
    segments = [np.column_stack([x, y]) for x, y in zip(xs, ys)]
    lc = LineCollection(segments, **kwargs)
    # set coloring of line segments
    #    Note: I get an error if I pass c as a list here... not sure why.
    lc.set_array(np.asarray(c))
    # add lines to axes and rescale 
    #    Note: adding a collection doesn't autoscalee xlim/ylim
    ax.add_collection(lc)
    ax.autoscale()
    return lc


def ensemble_plot(double=True, which='teff', path_to_dists=None, save=True, show=True, label_size=28., 
                  tick_size=22., LW=1.25, major=15., minor=10., direction='inout',):
    d = {'teff':{'axis_label':r'$\rm T_{eff} \,\, [K]$', 'file_name':'ensemble_teff.png',
                 'tick_label':[r'$3500$', r'$4000$', r'$4500$', r'$5000$', r'$5500$', r'$6000$'],}, 
         'logg':{'axis_label':r'$\mathrm{log}\,g \,\, [\mathrm{dex}]$', 'file_name':'ensemble_logg.png',
                 'tick_label':[r'$3.8$', r'$4.0$', r'$4.2$', r'$4.4$', r'$4.6$', r'$4.8$', r'$5.0$'],},
        }
    if path_to_dists is None:
        path_to_dists = 'data/distributions/dteff_%d_K_dlogg_%f_dex/*'%(int(res_teff),res_logg)
    xs, ys, t, l = get_dists(path_to_dists)
    if double:
        fig = plt.figure(figsize=(10,12))
        ax = plt.subplot(2,1,1)
    else:
        fig = plt.figure(figsize=(10,6))
        ax = plt.subplot(1,1,1)
    if which == 'teff':
        c = t[:]
        other = 'logg'
    else:
        c = l[:]
        other = 'teff'
    lc = multiline(xs, ys, c, cmap='cividis', lw=2)
    ax.set_ylabel(r'$\rm P_{rot} \,\, [days]$', fontsize=28)
    ax.tick_params(axis='both', which='minor', length=minor, width=LW, direction=direction)
    ax.tick_params(axis='both', which='major', length=major, width=LW, direction=direction)
    ax.tick_params(labelsize=tick_size)
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    if not double:
        ax.set_xlabel(r'$F\,\mathrm{(P_{rot}\, \vert \,T_{eff}, log}g\mathrm{)}$', fontsize=label_size)
        ax.set_xticklabels([r'$0.0$', r'$0.2$', r'$0.4$', r'$0.6$', r'$0.8$', r'$1.0$'])
    else:
        ax.set_xticklabels([])
    ax.xaxis.set_minor_locator(MultipleLocator(0.05))
    ax.set_xlim([0.,1.])
    ax.set_yticks([10., 20., 30., 40., 50., 60.])
    ax.set_yticklabels([r'$10$', r'$20$', r'$30$', r'$40$', r'$50$', r'$60$'])
    ax.yaxis.set_minor_locator(MultipleLocator(5))
    ax.set_ylim([0.,68.])
    cbar = fig.colorbar(lc, pad=0.01)
    cbar.set_label(d[which]['axis_label'], size=label_size)
    cbar.ax.set_yticklabels(d[which]['tick_label'])
    cbar.ax.tick_params(labelsize=tick_size)
    cbar.ax.tick_params(which='major', length=major, width=LW, direction=direction)

    if double:
        ax = plt.subplot(2,1,2)
        if other == 'teff':
            c = t[:]
        else:
            c = l[:]
        lc = multiline(xs, ys, c, cmap='cividis', lw=2)
        ax.set_ylabel(r'$\rm P_{rot} \,\, [days]$', fontsize=28)
        ax.set_xlabel(r'$F\,\mathrm{(P_{rot}\, \vert \,T_{eff}, log}g\mathrm{)}$', fontsize=label_size)
        ax.tick_params(axis='both', which='minor', length=minor, width=LW, direction=direction)
        ax.tick_params(axis='both', which='major', length=major, width=LW, direction=direction)
        ax.tick_params(labelsize=tick_size)
        ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_xticklabels([r'$0.0$', r'$0.2$', r'$0.4$', r'$0.6$', r'$0.8$', r'$1.0$'])
        ax.xaxis.set_minor_locator(MultipleLocator(0.05))
        ax.set_xlim([0.,1.])
        ax.set_yticks([10., 20., 30., 40., 50., 60.])
        ax.set_yticklabels([r'$10$', r'$20$', r'$30$', r'$40$', r'$50$', r'$60$'])
        ax.yaxis.set_minor_locator(MultipleLocator(5))
        ax.set_ylim([0.,68.])
        cbar = fig.colorbar(lc, pad=0.01)
        cbar.set_label(d[other]['axis_label'], size=label_size)
        cbar.ax.set_yticklabels(d[other]['tick_label'])
        cbar.ax.tick_params(labelsize=tick_size)
        cbar.ax.tick_params(which='major', length=major, width=LW, direction=direction)
        f_name = 'ensemble_double.png'
    else:
        f_name = d[which]['file_name']

    plt.tight_layout()
    if save:
        plt.savefig('plots/%s'%f_name, dpi=250)
    if show:
        plt.show()
    plt.close()


def rotation_hists(path_to_sample='data/sara_revised.csv', log=True, save=True, show=True, label_size=28., 
                   tick_size=22., LW=1.25, major=15., minor=10., direction='inout', n_bins=40,):
    df = pd.read_csv(path_to_sample)
    # divide by spectral type
    # Early F stars
    EF = df.query("spectral_type == 'F0V' or spectral_type == 'F1V' or spectral_type == 'F2V' or spectral_type == 'F3V' or spectral_type == 'F4V'")
    # Late F stars
    LF = df.query("spectral_type == 'F5V' or spectral_type == 'F6V' or spectral_type == 'F7V' or spectral_type == 'F8V' or spectral_type == 'F9V'")
    # Early G stars
    EG = df.query("spectral_type == 'G0V' or spectral_type == 'G1V' or spectral_type == 'G2V' or spectral_type == 'G3V' or spectral_type == 'G4V'")
    # Late G stars
    LG = df.query("spectral_type == 'G5V' or spectral_type == 'G6V' or spectral_type == 'G7V' or spectral_type == 'G8V' or spectral_type == 'G9V'")
    # All K stars
    K = df.query("spectral_type == 'K0V' or spectral_type == 'K1V' or spectral_type == 'K2V' or spectral_type == 'K3V' or spectral_type == 'K4V' or spectral_type == 'K5V' or spectral_type == 'K6V' or spectral_type == 'K7V' or spectral_type == 'K8V' or spectral_type == 'K9V'")

    # Histogram of periods versus spectral type
    if log:
        bins = np.logspace(-1,2,n_bins)
    else:
        bins = np.linspace(min(df.period.values),max(df.period.values),n_bins)
    plt.figure(figsize=(10,7))
    ax = plt.subplot(1,1,1)
    ax.hist(EF.period.values, bins=bins, facecolor='orange', zorder=3, edgecolor='k', label=r'$\rm Early\,\, F$')
    ax.hist(LF.period.values, bins=bins, facecolor='0.5', zorder=0, edgecolor='k', label=r'$\rm Late \,\,F$')
    ax.hist(EG.period.values, bins=bins, facecolor='dodgerblue', zorder=1, edgecolor='k', label=r'$\rm Early\,\, G$')
    ax.hist(LG.period.values, bins=bins, facecolor='green', zorder=2, edgecolor='k', label=r'$\rm Late \,\,G$')
    ax.hist(K.period.values, bins=bins, facecolor='magenta', zorder=4, edgecolor='k', label=r'$\rm All \,\,K$')
    ax.set_xlabel(r'$\rm P_{rot} \,\, [days]$', fontsize=label_size)
    ax.set_xscale('log')
    locations = list(np.arange(10,75,10))
    labels = [r'$%d$'%loc for loc in locations]
    if log:
        ax.set_xticks([0.1, 1.0, 10., 100.,])
        ax.set_xticklabels([r'$0.1$', r'$1$', r'$10$', r'$100$'])
    else:
        ax.set_xticklabels(labels)
    ax.set_yticks(locations)
    ax.set_yticklabels(labels)
    ax.tick_params(axis='both', which='minor', length=minor, width=LW, direction=direction)
    ax.tick_params(axis='both', which='major', length=major, width=LW, direction=direction)
    ax.tick_params(labelsize=tick_size)
    ax.legend(fontsize=tick_size, handletextpad=0.5, labelspacing=0.25, borderpad=0.25, loc='upper left')

    plt.tight_layout()
    if save:
        plt.savefig('plots/sptype_rotation.png', dpi=250)
    if show:
        plt.show()
    plt.close()


def rotation_hrdiagram(path_to_sample='data/sara_revised.csv', save=True, show=True, label_size=28., 
                       tick_size=22., LW=1.25, major=15., minor=10., direction='inout', lower_x=6650.,
                       upper_x=3650., lower_y=5.15, upper_y=3.62, continuous=True, discrete=True,):

    df = pd.read_csv(path_to_sample)
    s = np.argsort(df.period.values)
    # continuous colorbar
    if continuous:
        plt.figure(figsize = (10,8))
        ax = plt.subplot(1, 1, 1)
        plt.scatter(df.teff.values[s][::-1], df.logg.values[s][::-1], c=df.period.values[s][::-1], s=75., lw=0.75, edgecolor='k', cmap='cividis_r', norm=colors.LogNorm(vmin=0.25, vmax=35.0))
        ax.set_ylabel(r'$\mathrm{log} g \,\, \mathrm{[dex]}$', fontsize=label_size)
        ax.set_xlabel(r'$\rm T_{eff} \,\,[K]$', fontsize=label_size)
        ax.tick_params(axis='both', which='minor', length=minor, width=LW, direction=direction)
        ax.tick_params(axis='both', which='major', length=major, width=LW, direction=direction)
        ax.tick_params(labelsize=tick_size)
        ax.set_xticks([4000., 4500., 5000., 5500., 6000., 6500.])
        ax.set_xticklabels([r'$4000$', r'$4500$', r'$5000$', r'$5500$', r'$6000$', r'$6500$'])
        ax.set_yticks([3.8, 4.0, 4.2, 4.4, 4.6, 4.8, 5.0])
        ax.set_yticklabels([r'$3.8$', r'$4.0$', r'$4.2$', r'$4.4$', r'$4.6$', r'$4.8$', r'$5.0$'])
        ax.xaxis.set_minor_locator(MultipleLocator(100))
        ax.yaxis.set_minor_locator(MultipleLocator(0.1))
        ax.set_xlim(lower_x, upper_x)
        ax.set_ylim(lower_y,upper_y)
        ticks, labels = [0.3, 1., 3., 10., 30.], [r'$0.3$', r'$1$', r'$3$', r'$10$', r'$30$']
        cbar = plt.colorbar(ticks=ticks, pad=0.01)
        cbar.set_label(label=r'$\rm P_{rot} \,\, [days]$', size=label_size)
        cbar.ax.set_yticklabels(labels)
        cbar.ax.tick_params(labelsize=tick_size)
        cbar.ax.tick_params(which='major', length=major, width=LW, direction=direction)

        plt.tight_layout()
        if save:
            plt.savefig('plots/hrdiagram_rotation_1', dpi=250)
        if show:
            plt.show()
        plt.close()

    # discrete colorbar
    if discrete:
        bounds = np.array([0, 5, 10, 15, 20, 25, 30])
        norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
  
        plt.figure(figsize = (10,8))
        ax = plt.subplot(1, 1, 1)
        plt.scatter(df.teff.values[s][::-1], df.logg.values[s][::-1], c=df.period.values[s][::-1], s=75., lw=0.75, edgecolor='k', norm=norm, cmap='cividis_r')
        ax.set_ylabel(r'$\mathrm{log} g \,\, \mathrm{[dex]}$', fontsize=label_size)
        ax.set_xlabel(r'$\rm T_{eff} \,\,[K]$', fontsize=label_size)
        ax.tick_params(axis='both', which='minor', length=minor, width=LW, direction=direction)
        ax.tick_params(axis='both', which='major', length=major, width=LW, direction=direction)
        ax.tick_params(labelsize=tick_size)
        ax.set_xticks([4000., 4500., 5000., 5500., 6000., 6500.])
        ax.set_xticklabels([r'$4000$', r'$4500$', r'$5000$', r'$5500$', r'$6000$', r'$6500$'])
        ax.set_yticks([3.8, 4.0, 4.2, 4.4, 4.6, 4.8, 5.0])
        ax.set_yticklabels([r'$3.8$', r'$4.0$', r'$4.2$', r'$4.4$', r'$4.6$', r'$4.8$', r'$5.0$'])
        ax.xaxis.set_minor_locator(MultipleLocator(100))
        ax.yaxis.set_minor_locator(MultipleLocator(0.1))
        ax.set_xlim(lower_x, upper_x)
        ax.set_ylim(lower_y,upper_y)
        cbar = plt.colorbar(ticks = [0, 5., 10., 15., 20., 25., 30], pad=0.01)
        cbar.set_label(label = r'$\rm P_{rot} \,\, [days]$', size=label_size)
        cbar.ax.set_yticklabels([r'$0$', r'$5$', r'$10$', r'$15$', r'$20$', r'$25$', r'$30$'])
        cbar.ax.tick_params(labelsize=tick_size)
        cbar.ax.tick_params(which='major', length=major, width=LW, direction=direction)

        plt.tight_layout()
        if save:
            plt.savefig('plots/hrdiagram_rotation_2.png', dpi=250)
        if show:
            plt.show()
        plt.close()

