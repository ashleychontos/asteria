#!/usr/local/bin/python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, ScalarFormatter

plt.rcParams['mathtext.fontset'] = 'stix'

# Plotting
save = True
show = True
lower_x = 6650.
upper_x = 3650.
lower_y = 5.15
upper_y = 3.62

# ALL F/G/K STARS
kepler = pd.read_csv('../../Info/kepler.csv')
k2 = pd.read_csv('../../Info/k2.csv')
print(len(kepler), len(k2))

sara = pd.read_csv('../../Info/sara_revised.csv')
print(len(sara))

type = results.spectral_type.values.tolist()
teff = results.teff.values.tolist()
logg = results.logg.values.tolist()
per = results.period.values.tolist()
lw = results.linewidth.values.tolist()
lw = [((freq*10**-6)**-1)/60./60./24. for freq in lw]

# BY SPECTRAL TYPE

# Early F stars
EF = results.query("type == 'F0V' or type == 'F1V' or type == 'F2V' or type == 'F3V' or type == 'F4V'")
type_ef = EF.type.values.tolist()
teff_ef = EF.teff.values.tolist()
logg_ef = EF.logg.values.tolist()
per_ef = EF.period.values.tolist()
per_err_ef = EF.period_err.values.tolist()
frac_err_ef = [y/x for x,y in zip(per_ef, per_err_ef)]
lw_ef = EF.linewidth.values.tolist()
lw_ef = [((freq*10**-6)**-1)/60./60./24. for freq in lw_ef]

# Late F stars
LF = results.query("type == 'F5V' or type == 'F6V' or type == 'F7V' or type == 'F8V' or type == 'F9V'")
type_lf = LF.type.values.tolist()
teff_lf = LF.teff.values.tolist()
logg_lf = LF.logg.values.tolist()
per_lf = LF.period.values.tolist()
per_err_lf = LF.period_err.values.tolist()
frac_err_lf = [y/x for x,y in zip(per_lf, per_err_lf)]
lw_lf = LF.linewidth.values.tolist()
lw_lf = [((freq*10**-6)**-1)/60./60./24. for freq in lw_lf]

# Early G stars
EG = results.query("type == 'G0V' or type == 'G1V' or type == 'G2V' or type == 'G3V' or type == 'G4V'")
type_eg = EG.type.values.tolist()
teff_eg = EG.teff.values.tolist()
logg_eg = EG.logg.values.tolist()
per_eg = EG.period.values.tolist()
per_err_eg = EG.period_err.values.tolist()
frac_err_eg = [y/x for x,y in zip(per_eg, per_err_eg)]
lw_eg = EG.linewidth.values.tolist()
lw_eg = [((freq*10**-6)**-1)/60./60./24. for freq in lw_eg]

# Late G stars
LG = results.query("type == 'G5V' or type == 'G6V' or type == 'G7V' or type == 'G8V' or type == 'G9V'")
type_lg = LG.type.values.tolist()
teff_lg = LG.teff.values.tolist()
logg_lg = LG.logg.values.tolist()
per_lg = LG.period.values.tolist()
per_err_lg = LG.period_err.values.tolist()
frac_err_lg = [y/x for x,y in zip(per_lg, per_err_lg)]
lw_lg = LG.linewidth.values.tolist()
lw_lg = [((freq*10**-6)**-1)/60./60./24. for freq in lw_lg]

# All K stars
K = results.query("type == 'K0V' or type == 'K1V' or type == 'K2V' or type == 'K3V' or type == 'K4V' or type == 'K5V' or type == 'K6V' or type == 'K7V' or type == 'K8V' or type == 'K9V'")
type_k = K.type.values.tolist()
teff_k = K.teff.values.tolist()
logg_k = K.logg.values.tolist()
per_k = K.period.values.tolist()
per_err_k = K.period_err.values.tolist()
frac_err_k = [y/x for x,y in zip(per_k, per_err_k)]
lw_k = K.linewidth.values.tolist()
lw_k = [((freq*10**-6)**-1)/60./60./24. for freq in lw_k]


# Histogram of periods versus spectral type
bins = np.arange(1,31,1)

plt.figure(figsize = (8,10))
ax1 = plt.subplot(2,1,1)
ax1.hist(per_ef, bins = bins, facecolor = 'orange', zorder = 3, edgecolor = 'k', label = r'$\rm Early\,\, F$')
ax1.hist(per_lf, bins = bins, facecolor = '0.5', zorder = 0, edgecolor = 'k', label = r'$\rm Late \,\,F$')
ax1.hist(per_eg, bins = bins, facecolor = 'dodgerblue', zorder = 1, edgecolor = 'k', label = r'$\rm Early\,\, G$')
ax1.hist(per_lg, bins = bins, facecolor = 'green', zorder = 2, edgecolor = 'k', label = r'$\rm Late \,\,G$')
ax1.hist(per_k, bins = bins, facecolor = 'magenta', zorder = 4, edgecolor = 'k', label = r'$\rm All \,\,K$')
ax1.set_xlim(1.1, 27.5)
ax1.set_xlabel(r'$\rm P_{rot} \,\, (days)$', fontsize = 22)
ax1.yaxis.set_minor_locator(MultipleLocator(10))
ax1.yaxis.set_major_locator(MultipleLocator(50))
ax1.xaxis.set_minor_locator(MultipleLocator(1))
ax1.xaxis.set_major_locator(MultipleLocator(5))
ax1.tick_params(axis = 'both', which = 'minor', length = 5, width = 1, direction = 'inout')
ax1.tick_params(axis = 'both', which = 'major', length = 8, width = 2, direction = 'inout')
ax1.tick_params(labelsize = 22)
ax1.set_xticks([5., 10., 15., 20., 25.])
ax1.set_xticklabels([r'$5$', r'$10$', r'$15$', r'$20$', r'$25$'])
ax1.set_yticks([50., 100., 150., 200.])
ax1.set_yticklabels([r'$50$', r'$100$', r'$150$', r'$200$'])
ax1.legend(fontsize = 22, handletextpad = 0.5, labelspacing = 0.25, borderpad = 0.25, loc = 'upper right')

bins = np.arange(0.008, 0.037, 0.001)

ax2 = plt.subplot(2,1,2)
ax2.hist(frac_err_ef, bins = bins, facecolor = 'orange', zorder = 3, edgecolor = 'k')
ax2.hist(frac_err_lf, bins = bins, facecolor = '0.5', zorder = 0, edgecolor = 'k')
ax2.hist(frac_err_eg, bins = bins, facecolor = 'dodgerblue', zorder = 1, edgecolor = 'k')
ax2.hist(frac_err_lg, bins = bins, facecolor = 'green', zorder = 2, edgecolor = 'k')
ax2.hist(frac_err_k, bins = bins, facecolor = 'magenta', zorder = 4, edgecolor = 'k')
ax2.set_xlabel(r'$\rm \sigma_{P_{rot}}/P_{rot}$', fontsize = 22)
ax2.yaxis.set_minor_locator(MultipleLocator(5))
ax2.yaxis.set_major_locator(MultipleLocator(20))
ax2.xaxis.set_minor_locator(MultipleLocator(0.001))
ax2.xaxis.set_major_locator(MultipleLocator(0.005))
ax2.tick_params(axis = 'both', which = 'minor', length = 5, width = 1, direction = 'inout')
ax2.tick_params(axis = 'both', which = 'major', length = 8, width = 2, direction = 'inout')
ax2.tick_params(labelsize = 22)
ax2.set_xticks([0.01, 0.015, 0.02, 0.025, 0.03, 0.035])
ax2.set_xticklabels([r'$0.010$', r'$0.015$', r'$0.020$', r'$0.025$', r'$0.030$', r'$0.035$'])
ax2.set_yticks([20, 40, 60, 80, 100, 120])
ax2.set_yticklabels([r'$20$', r'$40$', r'$60$', r'$80$', r'$100$', r'$120$'])

plt.tight_layout()
plt.savefig('../Plots/Sara/Spectral_type_rotation.png', dpi = 150)
plt.show()
plt.close()


# Histogram of mode lifetime versus spectral type
bins = np.arange(0.0, 5.5, 0.25)

plt.figure(figsize = (10,6))
ax1 = plt.subplot(1,1,1)
ax1.hist(lw_ef, bins = bins, facecolor = 'orange', zorder = 3, edgecolor = 'k', label = r'$\rm Early\,\, F$')
ax1.hist(lw_lf, bins = bins, facecolor = '0.5', zorder = 0, edgecolor = 'k', label = r'$\rm Late \,\,F$')
ax1.hist(lw_eg, bins = bins, facecolor = 'dodgerblue', zorder = 1, edgecolor = 'k', label = r'$\rm Early\,\, G$')
ax1.hist(lw_lg, bins = bins, facecolor = 'green', zorder = 2, edgecolor = 'k', label = r'$\rm Late \,\,G$')
ax1.hist(lw_k, bins = bins, facecolor = 'magenta', zorder = 4, edgecolor = 'k', label = r'$\rm All \,\,K$')
ax1.set_xlim(0.25, 5.25)
ax1.set_xlabel(r'$\rm Mode \,\, lifetime \,\, [days]$', fontsize = 22)
ax1.yaxis.set_minor_locator(MultipleLocator(10))
ax1.yaxis.set_major_locator(MultipleLocator(40))
ax1.xaxis.set_major_locator(MultipleLocator(1))
ax1.xaxis.set_minor_locator(MultipleLocator(0.25))
ax1.tick_params(axis = 'both', which = 'minor', length = 5, width = 1, direction = 'inout')
ax1.tick_params(axis = 'both', which = 'major', length = 8, width = 2, direction = 'inout')
ax1.tick_params(labelsize = 22)
ax1.set_xticks([1., 2., 3., 4., 5.])
ax1.set_xticklabels([r'$1$', r'$2$', r'$3$', r'$4$', r'$5$'])
ax1.set_yticks([40., 80., 120.])
ax1.set_yticklabels([r'$40$', r'$80$', r'$120$'])
ax1.legend(fontsize = 22, handletextpad = 0.5, labelspacing = 0.25, borderpad = 0.25)

plt.tight_layout()
if save:
    plt.savefig('../Plots/Sara/Spectral_type_linewidths.png', dpi = 150)
if show:
    plt.show()
plt.close()


plt.figure(figsize = (10,8))

ax = plt.subplot(1, 1, 1)
plt.scatter(teff, logg, c = per, s = 75., lw = 0.75, edgecolor = 'k', cmap = 'viridis_r')
ax.set_ylabel(r'$\rm log(g) \,\,[dex]$', fontsize = 26)
ax.set_xlabel(r'$\rm T_{eff} \,\,[K]$', fontsize = 26)
ax.yaxis.set_minor_locator(MultipleLocator(0.05))
ax.yaxis.set_major_locator(MultipleLocator(0.2))
ax.xaxis.set_minor_locator(MultipleLocator(100))
ax.xaxis.set_major_locator(MultipleLocator(500))
ax.tick_params(axis = 'both', which = 'minor', length = 5, width = 1, direction = 'inout')
ax.tick_params(axis = 'both', which = 'major', length = 8, width = 2, direction = 'inout')
ax.tick_params(labelsize = 22)
ax.ticklabel_format(useOffset = False)
ax.set_xticks([4000., 4500., 5000., 5500., 6000., 6500.])
ax.set_xticklabels([r'$4000$', r'$4500$', r'$5000$', r'$5500$', r'$6000$', r'$6500$'])
ax.set_yticks([3.8, 4.0, 4.2, 4.4, 4.6, 4.8, 5.0])
ax.set_yticklabels([r'$3.8$', r'$4.0$', r'$4.2$', r'$4.4$', r'$4.6$', r'$4.8$', r'$5.0$'])
ax.set_xlim(lower_x, upper_x)
ax.set_ylim(lower_y,upper_y)
cbar = plt.colorbar(ticks = [5, 10, 15, 20, 25, 30, 35], pad = 0.01)
cbar.set_label(label = r'$\rm P_{rot} \,\, [days]$', size = 26)
cbar.ax.set_yticklabels([r'$5$', r'$10$', r'$15$', r'$20$', r'$25$', r'$30$', r'$35$'])
cbar.ax.tick_params(labelsize = 22)
cbar.ax.tick_params(which = 'major', length = 5, width = 2, direction = 'inout')

plt.tight_layout()
plt.savefig('../Plots/Sara/HRdiagram_rotation_1.png', dpi = 150)
plt.close()

bounds = np.array([0, 5, 10, 15, 20, 25, 30])
norm = colors.BoundaryNorm(boundaries = bounds, ncolors = 256)
  
plt.figure(figsize = (10,8))

ax = plt.subplot(1, 1, 1)
plt.scatter(teff, logg, c = per, s = 75., lw = 0.75, edgecolor = 'k', norm = norm, cmap = 'seismic_r')
ax.set_ylabel(r'$\rm log(g) \,\,[dex]$', fontsize = 26)
ax.set_xlabel(r'$\rm T_{eff} \,\,[K]$', fontsize = 26)
ax.yaxis.set_minor_locator(MultipleLocator(0.05))
ax.yaxis.set_major_locator(MultipleLocator(0.2))
ax.xaxis.set_minor_locator(MultipleLocator(100))
ax.xaxis.set_major_locator(MultipleLocator(500))
ax.tick_params(axis = 'both', which = 'minor', length = 5, width = 1, direction = 'inout')
ax.tick_params(axis = 'both', which = 'major', length = 8, width = 2, direction = 'inout')
ax.tick_params(labelsize = 22)
ax.ticklabel_format(useOffset = False)
ax.set_xticks([4000., 4500., 5000., 5500., 6000., 6500.])
ax.set_xticklabels([r'$4000$', r'$4500$', r'$5000$', r'$5500$', r'$6000$', r'$6500$'])
ax.set_yticks([3.8, 4.0, 4.2, 4.4, 4.6, 4.8, 5.0])
ax.set_yticklabels([r'$3.8$', r'$4.0$', r'$4.2$', r'$4.4$', r'$4.6$', r'$4.8$', r'$5.0$'])
ax.set_xlim(lower_x, upper_x)
ax.set_ylim(lower_y,upper_y)
cbar = plt.colorbar(ticks = [0, 5., 10., 15., 20., 25., 30], pad = 0.01)
cbar.set_label(label = r'$\rm P_{rot} \,\, [days]$', size = 26)
cbar.ax.set_yticklabels([r'$0$', r'$5$', r'$10$', r'$15$', r'$20$', r'$25$', r'$30$'])
cbar.ax.tick_params(labelsize = 22)
cbar.ax.tick_params(which = 'major', length = 5, width = 2, direction = 'inout')

plt.tight_layout()
plt.savefig('../Plots/Sara/HRdiagram_rotation_2.png', dpi = 150)
plt.close()

plt.figure(figsize = (10,8))

ax = plt.subplot(1, 1, 1)
plt.scatter(teff, logg, c = lw, vmin = min(lw), vmax = max(lw), s = 75., lw = 0.75, edgecolor = 'k', cmap = plt.cm.get_cmap('viridis_r'))
ax.set_ylabel(r'$\rm log(g) \,\,[dex]$', fontsize = 26)
ax.set_xlabel(r'$\rm T_{eff} \,\,[K]$', fontsize = 26)
ax.yaxis.set_minor_locator(MultipleLocator(0.05))
ax.yaxis.set_major_locator(MultipleLocator(0.2))
ax.xaxis.set_minor_locator(MultipleLocator(100))
ax.xaxis.set_major_locator(MultipleLocator(500))
ax.tick_params(axis = 'both', which = 'minor', length = 5, width = 1, direction = 'inout')
ax.tick_params(axis = 'both', which = 'major', length = 8, width = 2, direction = 'inout')
ax.tick_params(labelsize = 22)
ax.ticklabel_format(useOffset = False)
ax.set_xticks([4000., 4500., 5000., 5500., 6000., 6500.])
ax.set_xticklabels([r'$4000$', r'$4500$', r'$5000$', r'$5500$', r'$6000$', r'$6500$'])
ax.set_yticks([3.8, 4.0, 4.2, 4.4, 4.6, 4.8, 5.0])
ax.set_yticklabels([r'$3.8$', r'$4.0$', r'$4.2$', r'$4.4$', r'$4.6$', r'$4.8$', r'$5.0$'])
ax.set_xlim(lower_x, upper_x)
ax.set_ylim(lower_y,upper_y)
cbar = plt.colorbar(ticks = [1., 2., 3., 4., 5.], pad = 0.01)
cbar.set_label(label = r'$\rm Mode \,\, lifetime \,\, [days]$', size = 26)
cbar.ax.set_yticklabels([r'$1.0$', r'$2.0$', r'$3.0$', r'$4.0$', r'$5.0$'])
cbar.ax.tick_params(labelsize = 22)
cbar.ax.tick_params(which = 'major', length = 5, width = 2, direction = 'inout')

plt.tight_layout()
plt.savefig('../Plots/Sara/HRdiagram_linewidth.png', dpi = 150)
plt.close()
