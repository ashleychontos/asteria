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
label_size = 28.
tick_size = 22.


df = pd.read_csv('../../Info/sara_revised.csv')

# BY SPECTRAL TYPE

# Early F stars
EF = df.query("type == 'F0V' or type == 'F1V' or type == 'F2V' or type == 'F3V' or type == 'F4V'")

# Late F stars
LF = df.query("type == 'F5V' or type == 'F6V' or type == 'F7V' or type == 'F8V' or type == 'F9V'")

# Early G stars
EG = df.query("type == 'G0V' or type == 'G1V' or type == 'G2V' or type == 'G3V' or type == 'G4V'")

# Late G stars
LG = df.query("type == 'G5V' or type == 'G6V' or type == 'G7V' or type == 'G8V' or type == 'G9V'")

# All K stars
K = df.query("type == 'K0V' or type == 'K1V' or type == 'K2V' or type == 'K3V' or type == 'K4V' or type == 'K5V' or type == 'K6V' or type == 'K7V' or type == 'K8V' or type == 'K9V'")

# Histogram of periods versus spectral type
bins = np.logspace(-1,100,40)

plt.figure(figsize=(10,7))
ax1 = plt.subplot(1,1,1)
ax1.hist(EF.period.values, bins=bins, facecolor='orange', zorder=3, edgecolor='k', label=r'$\rm Early\,\, F$')
ax1.hist(LF.period.values, bins=bins, facecolor='0.5', zorder=0, edgecolor='k', label=r'$\rm Late \,\,F$')
ax1.hist(EG.period.values, bins=bins, facecolor='dodgerblue', zorder=1, edgecolor='k', label=r'$\rm Early\,\, G$')
ax1.hist(LG.period.values, bins=bins, facecolor='green', zorder=2, edgecolor='k', label=r'$\rm Late \,\,G$')
ax1.hist(K.period.values, bins=bins, facecolor='magenta', zorder=4, edgecolor='k', label=r'$\rm All \,\,K$')
ax1.set_xlim(1.1, 27.5)
ax1.set_xlabel(r'$\rm P_{rot} \,\, (days)$', fontsize=label_size)
ax1.yaxis.set_minor_locator(MultipleLocator(10))
ax1.yaxis.set_major_locator(MultipleLocator(50))
ax1.xaxis.set_minor_locator(MultipleLocator(1))
ax1.xaxis.set_major_locator(MultipleLocator(5))
ax1.tick_params(axis = 'both', which = 'minor', length = 5, width = 1, direction = 'inout')
ax1.tick_params(axis = 'both', which = 'major', length = 8, width = 2, direction = 'inout')
ax1.tick_params(labelsize=tick_size)
ax1.set_xticks([5., 10., 15., 20., 25.])
ax1.set_xticklabels([r'$5$', r'$10$', r'$15$', r'$20$', r'$25$'])
ax1.set_yticks([50., 100., 150., 200.])
ax1.set_yticklabels([r'$50$', r'$100$', r'$150$', r'$200$'])
ax1.legend(fontsize=tick_size, handletextpad = 0.5, labelspacing = 0.25, borderpad = 0.25, loc = 'upper right')

plt.tight_layout()
plt.savefig('../Plots/Sara/Spectral_type_rotation.png', dpi=250)
plt.show()
plt.close()



plt.figure(figsize = (10,8))

ax = plt.subplot(1, 1, 1)
plt.scatter(df.teff.values, df.logg.values, c=per, s=75., lw=0.75, edgecolor='k', cmap='viridis_r')
ax.set_ylabel(r'$\rm log(g) \,\,[dex]$', fontsize=label_size)
ax.set_xlabel(r'$\rm T_{eff} \,\,[K]$', fontsize=label_size)
ax.yaxis.set_minor_locator(MultipleLocator(0.05))
ax.yaxis.set_major_locator(MultipleLocator(0.2))
ax.xaxis.set_minor_locator(MultipleLocator(100))
ax.xaxis.set_major_locator(MultipleLocator(500))
ax.tick_params(axis='both', which='minor', length=10, width=1.25, direction='inout')
ax.tick_params(axis='both', which='major', length=15, width=1.25, direction='inout')
ax.tick_params(labelsize=tick_size)
ax.ticklabel_format(useOffset=False)
ax.set_xticks([4000., 4500., 5000., 5500., 6000., 6500.])
ax.set_xticklabels([r'$4000$', r'$4500$', r'$5000$', r'$5500$', r'$6000$', r'$6500$'])
ax.set_yticks([3.8, 4.0, 4.2, 4.4, 4.6, 4.8, 5.0])
ax.set_yticklabels([r'$3.8$', r'$4.0$', r'$4.2$', r'$4.4$', r'$4.6$', r'$4.8$', r'$5.0$'])
ax.set_xlim(lower_x, upper_x)
ax.set_ylim(lower_y,upper_y)
cbar = plt.colorbar(ticks = [5, 10, 15, 20, 25, 30, 35], pad = 0.01)
cbar.set_label(label = r'$\rm P_{rot} \,\, [days]$', size=label_size)
cbar.ax.set_yticklabels([r'$5$', r'$10$', r'$15$', r'$20$', r'$25$', r'$30$', r'$35$'])
cbar.ax.tick_params(labelsize=tick_size)
cbar.ax.tick_params(which='major', length=15, width=1.25, direction='inout')

plt.tight_layout()
plt.savefig('hrdiagram_rotation_1', dpi=250)
plt.show()
plt.close()

bounds = np.array([0, 5, 10, 15, 20, 25, 30])
norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
  
plt.figure(figsize = (10,8))

ax = plt.subplot(1, 1, 1)
plt.scatter(df.teff.values, df.logg.values, c=per, s=75., lw=0.75, edgecolor='k', norm=norm, cmap='seismic_r')
ax.set_ylabel(r'$\rm log(g) \,\,[dex]$', fontsize=label_size)
ax.set_xlabel(r'$\rm T_{eff} \,\,[K]$', fontsize=label_size)
ax.yaxis.set_minor_locator(MultipleLocator(0.05))
ax.yaxis.set_major_locator(MultipleLocator(0.2))
ax.xaxis.set_minor_locator(MultipleLocator(100))
ax.xaxis.set_major_locator(MultipleLocator(500))
ax.tick_params(axis='both', which='minor', length=10, width=1.25, direction='inout')
ax.tick_params(axis='both', which='major', length=15, width=1.25, direction='inout')
ax.tick_params(labelsize=tick_size)
ax.ticklabel_format(useOffset=False)
ax.set_xticks([4000., 4500., 5000., 5500., 6000., 6500.])
ax.set_xticklabels([r'$4000$', r'$4500$', r'$5000$', r'$5500$', r'$6000$', r'$6500$'])
ax.set_yticks([3.8, 4.0, 4.2, 4.4, 4.6, 4.8, 5.0])
ax.set_yticklabels([r'$3.8$', r'$4.0$', r'$4.2$', r'$4.4$', r'$4.6$', r'$4.8$', r'$5.0$'])
ax.set_xlim(lower_x, upper_x)
ax.set_ylim(lower_y,upper_y)
cbar = plt.colorbar(ticks = [0, 5., 10., 15., 20., 25., 30], pad=0.01)
cbar.set_label(label = r'$\rm P_{rot} \,\, [days]$', size=label_size)
cbar.ax.set_yticklabels([r'$0$', r'$5$', r'$10$', r'$15$', r'$20$', r'$25$', r'$30$'])
cbar.ax.tick_params(labelsize=tick_size)
cbar.ax.tick_params(which='major', length=15, width=1.25, direction='inout')

plt.tight_layout()
plt.savefig('hrdiagram_rotation_2.png', dpi=250)
plt.show()
plt.close()
