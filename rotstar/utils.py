import os
import random
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats, interpolate


fig_path = '/Users/ashleychontos/Desktop/'


def multiline(xs, ys, c, ax=None, **kwargs):
    """
    Plot lines with different colorings

    Parameters
    ----------
    xs : iterable container of x coordinates
    ys : iterable container of y coordinates
    c : iterable container of numbers mapped to colormap
    ax (optional): Axes to plot on.
    kwargs (optional): passed to LineCollection

    Notes:
        len(xs) == len(ys) == len(c) is the number of line segments
        len(xs[i]) == len(ys[i]) is the number of points for each line (indexed by i)

    Returns
    -------
    lc : LineCollection instance.

    """
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

def fix_distribution(x, y):
    indices=[0]
    for i in range(1,len(y)):
        if y[i] != y[i-1]:
            indices.append(i)
    x, y = x[indices], y[indices]
    return x, y


def get_inverse(query, n_bins=100, log=False):
    kernel = stats.gaussian_kde(query.period.values)
    lower, upper = min(query.period.values), max(query.period.values)
    if log:
        x = np.logspace(np.log10(lower),np.log10(upper),n_bins)
    else:
        x = np.linspace(lower,upper,n_bins)
    y = np.cumsum(kernel(x))/np.sum(kernel(x))
    x, y = fix_distribution(x, y)
    try:
        spline = interpolate.CubicSpline(y, x)
    except ValueError:
        return None, None
    else:
        xnew = np.linspace(min(y),max(y),n_bins)
        return xnew, spline


def get_periods(path_to_sample='rotation.csv', min_sample=20, res_teff=100., res_logg=0.1,):
    period=[]
    # read in known rotation periods to draw samples from
    df = pd.read_csv(args.path_to_sample)
    if args.path_to_stars is not None:
        # read in targets of interest to estimate rotation periods for
        stars = pd.read_csv(args.path_to_stars)
        args.teff, args.logg = stars.teff.values, stars.logg.values
    # iterate through stars to estimate rotation periods
    for teff, logg in zip(args.teff, args.logg):
        # select stars near target in HR diagram
        query = df.query("teff >= %f and teff < %f and logg >= %f and logg < %f"%(teff-(args.res_teff/2.), teff+(args.res_teff/2.), logg-(args.res_logg/2.), logg+(args.res_logg/2.)))
        if len(query) < args.min_sample:
            if args.verbose:
                print('WARNING: not enough in the sample to create an accurate distribution.\nTry changing the resolution of the grid to include more stars!')
                print('Currently using teff +/- %.1f K and logg +/- %.2f dex'%(args.res_teff/2.,args.res_logg/2.))
            per = np.nan
        else:
            _, spline = get_inverse(args, query)
            # draw random number to map back to period distribution
            per = spline(random.random())+0.
        period.append(per)
    if args.path_to_stars is not None:
        # save new period estimates
        stars['period'] = np.array(period)
        stars.to_csv(args.path_to_stars, index=False)
    else:
        if args.returnn:
            return np.array(period)
        else:
            print(period)


# Main function to import when not using CLI
def get_period(teff, logg, path='rotation.csv', min_sample=20, res_teff=100., res_logg=0.1, log=False, n_bins=100, verbose=False):
    period=[]
    # read in known rotation periods and get limits
    df = pd.read_csv(path)
    for tt, ll in zip(teff, logg):
        per = np.nan
        # select stars near target in HR diagram
        query = df.query("teff >= %f and teff < %f and logg >= %f and logg < %f"%(tt-(res_teff/2.), tt+(res_teff/2.), ll-(res_logg/2.), ll+(res_logg/2.)))
        if len(query) < min_sample:
            if verbose:
                print('WARNING: not enough in the sample to create an accurate distribution.\nTry changing the resolution of the grid to include more stars!')
                print('Currently using teff +/- %.1f K and logg +/- %.2f dex'%(res_teff/2.,res_logg/2.))
        else:
            kernel = stats.gaussian_kde(query.period.values)
            lower, upper = min(query.period.values), max(query.period.values)
            if log:
                x = np.logspace(np.log10(lower),np.log10(upper),n_bins)
            else:
                x = np.linspace(lower,upper,n_bins)
            y = np.cumsum(kernel(x))/np.sum(kernel(x))
            x, y = fix_distribution(x, y)
            try:
                spline = interpolate.CubicSpline(y, x)
            except ValueError:
                continue
            else:
                # draw random number to map back to period distribution
                per = spline(random.random())+0.
        period.append(per)
    return np.array(period)


def save_file(x, y, path, formats=[">10.4f", ">10.2f"]):
    header = '#      CDF      prot\n#             (days)\n'
    with open(path, "w") as f:
        f.write(header)
        for xx, yy in zip(x, y):
            values = [xx, yy]
            text = '{:{}}'*len(values) + '\n'
            fmt = sum(zip(values, formats), ())
            f.write(text.format(*fmt))


def ensemble_plot_teff(path='rotation.csv', path_save='distributions', min_sample=20, 
                       res_teff=100., res_logg=0.1, log=False, save=True, show=True, verbose=True):
    import os
    from matplotlib import cm
    if not os.path.exists(os.path.join(os.path.abspath(os.getcwd()), path_save)):
        os.mkdir(os.path.join(os.path.join(os.path.abspath(os.getcwd()), path_save)))
    # read in known rotation periods and get limits
    df = pd.read_csv(path)
    # for effective temperature
    min_teff, max_teff = np.floor(df.teff.min()/res_teff)*res_teff, np.ceil(df.teff.max()/res_teff)*res_teff
    edges_teff = np.arange(min_teff, max_teff+res_teff, res_teff)
    # for surface gravity
    min_logg, max_logg = np.floor(df.logg.min()/res_logg)*res_logg, np.ceil(df.logg.max()/res_logg)*res_logg
    edges_logg = np.arange(min_logg, max_logg+res_logg, res_logg)
    # let's try a pretty plot
    tnorm = matplotlib.colors.Normalize(vmin=min_teff, vmax=max_teff)
    norm = matplotlib.colors.Normalize(vmin=min_logg, vmax=max_logg)
    xs, ys, c, al = [], [], [], []
    fig = plt.figure(figsize=(10,6))
    ax = plt.subplot(1,1,1)
    for i in range(len(edges_teff)-1):
        mid_teff = (edges_teff[i]+edges_teff[i+1])/2.
        for j in range(len(edges_logg)-1):
            mid_logg = (edges_logg[j]+edges_logg[j+1])/2.
            query = df.query("teff >= %f and teff < %f and logg >= %f and logg < %f"%(edges_teff[i], edges_teff[i+1], edges_logg[j], edges_logg[j+1]))
            if verbose:
                print("teff~[%d,%d]; logg~[%f,%f]; n=%d"%(int(edges_teff[i]), int(edges_teff[i+1]), edges_logg[j], edges_logg[j+1], len(query)))
            if len(query) >= min_sample:
                x, spline = get_inverse(query, log=log)
                if x is not None:
                    fname = os.path.join(path_save,'teff_%d_%d_logg_%.1f_%.1f.txt'%(edges_teff[i], edges_teff[i+1], edges_logg[j], edges_logg[j+1]))
                    save_file(x, spline(x), fname)
                    xs.append(x)
                    ys.append(spline(x))
                    c.append(mid_teff)
                    al.append(mid_logg)
    lc = multiline(xs, ys, c, cmap='cividis', lw=2)
    ax.set_ylabel(r'$\rm P_{rot} \,\, [days]$', fontsize=28)
    ax.set_xlabel(r'$F\,\mathrm{(P_{rot}\, \vert \,T_{eff}, log}g\mathrm{)}$', fontsize=28)
    ax.tick_params(axis='both', which='minor', length=10, width=1.25, direction='inout')
    ax.tick_params(axis='both', which='major', length=15, width=1.25, direction='inout')
    ax.tick_params(labelsize=22)
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xticklabels([r'$0.0$', r'$0.2$', r'$0.4$', r'$0.6$', r'$0.8$', r'$1.0$'])
    ax.xaxis.set_minor_locator(MultipleLocator(0.05))
    ax.set_xlim([0.,1.])
    if log:
        ax.set_yscale('log')
        ax.set_yticks([0.3, 1., 3., 10., 30.])
        ax.set_yticklabels([r'$0.3$', r'$1$', r'$3$', r'$10$', r'$30$'])
        ax.set_ylim([0.1,68.])
    else:
        ax.set_yticks([10., 20., 30., 40., 50., 60.])
        ax.set_yticklabels([r'$10$', r'$20$', r'$30$', r'$40$', r'$50$', r'$60$'])
        ax.yaxis.set_minor_locator(MultipleLocator(5))
        ax.set_ylim([0.,68.])
    cbar = fig.colorbar(lc, pad=0.01)
    cbar.set_label(r'$\rm T_{eff} \,\, [K]$', size=28)
    cbar.ax.set_yticklabels([r'$3500$', r'$4000$', r'$4500$', r'$5000$', r'$5500$', r'$6000$'])
    cbar.ax.tick_params(labelsize=22)
    cbar.ax.tick_params(which='major', length=15, width=1.25, direction='inout')

    plt.tight_layout()
    if save:
        if log:
            plt.savefig('%steff_cb_logscale.png'%fig_path, dpi=250)
        else:
            plt.savefig('%steff_cb.png'%fig_path, dpi=250)
    if show:
        plt.show()
    plt.close()



def ensemble_plot_logg(path='rotation.csv', path_save='distributions', min_sample=20, 
                       res_teff=100., res_logg=0.1, log=False, save=True, show=True, verbose=False):
    import os
    from matplotlib import cm
    if not os.path.exists(os.path.join(os.path.abspath(os.getcwd()), path_save)):
        os.mkdir(os.path.join(os.path.join(os.path.abspath(os.getcwd()), path_save)))
    # read in known rotation periods and get limits
    df = pd.read_csv(path)
    # for effective temperature
    min_teff, max_teff = np.floor(df.teff.min()/res_teff)*res_teff, np.ceil(df.teff.max()/res_teff)*res_teff
    edges_teff = np.arange(min_teff, max_teff+res_teff, res_teff)
    # for surface gravity
    min_logg, max_logg = np.floor(df.logg.min()/res_logg)*res_logg, np.ceil(df.logg.max()/res_logg)*res_logg
    edges_logg = np.arange(min_logg, max_logg+res_logg, res_logg)
    # let's try a pretty plot
    tnorm = matplotlib.colors.Normalize(vmin=min_teff, vmax=max_teff)
    norm = matplotlib.colors.Normalize(vmin=min_logg, vmax=max_logg)
    xs, ys, c, al = [], [], [], []
    fig = plt.figure(figsize=(10,6))
    ax = plt.subplot(1,1,1)
    for i in range(len(edges_teff)-1):
        mid_teff = (edges_teff[i]+edges_teff[i+1])/2.
        for j in range(len(edges_logg)-1):
            mid_logg = (edges_logg[j]+edges_logg[j+1])/2.
            query = df.query("teff >= %f and teff < %f and logg >= %f and logg < %f"%(edges_teff[i], edges_teff[i+1], edges_logg[j], edges_logg[j+1]))
            if verbose:
                print("teff~[%d,%d]; logg~[%f,%f]; n=%d"%(int(edges_teff[i]), int(edges_teff[i+1]), edges_logg[j], edges_logg[j+1], len(query)))
            if len(query) >= min_sample:
                x, spline = get_inverse(query, log=log)
                if x is not None:
                    fname = os.path.join(path_save,'teff_%d_%d_logg_%.1f_%.1f.txt'%(edges_teff[i], edges_teff[i+1], edges_logg[j], edges_logg[j+1]))
                    save_file(x, spline(x), fname)
                    xs.append(x)
                    ys.append(spline(x))
                    c.append(mid_teff)
                    al.append(mid_logg)
    lc = multiline(xs, ys, al, cmap='cividis', lw=2)
    ax.set_ylabel(r'$\rm P_{rot} \,\, [days]$', fontsize=28)
    ax.set_xlabel(r'$F\,\mathrm{(P_{rot}\, \vert \,T_{eff}, log}g\mathrm{)}$', fontsize=28)
    ax.tick_params(axis='both', which='minor', length=10, width=1.25, direction='inout')
    ax.tick_params(axis='both', which='major', length=15, width=1.25, direction='inout')
    ax.tick_params(labelsize=22)
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xticklabels([r'$0.0$', r'$0.2$', r'$0.4$', r'$0.6$', r'$0.8$', r'$1.0$'])
    ax.xaxis.set_minor_locator(MultipleLocator(0.05))
    ax.set_xlim([0.,1.])
    if log:
        ax.set_yscale('log')
        ax.set_yticks([0.3, 1., 3., 10., 30.])
        ax.set_yticklabels([r'$0.3$', r'$1$', r'$3$', r'$10$', r'$30$'])
        ax.set_ylim([0.1,68.])
    else:
        ax.set_yticks([10., 20., 30., 40., 50., 60.])
        ax.set_yticklabels([r'$10$', r'$20$', r'$30$', r'$40$', r'$50$', r'$60$'])
        ax.yaxis.set_minor_locator(MultipleLocator(5))
        ax.set_ylim([0.,68.])
    cbar = fig.colorbar(lc, pad=0.01)
    cbar.set_label(r'$\mathrm{log}\,g \,\, [\mathrm{dex}]$', size=28)
    cbar.ax.set_yticklabels([r'$3.8$', r'$4.0$', r'$4.2$', r'$4.4$', r'$4.6$', r'$4.8$', r'$5.0$'])
    cbar.ax.tick_params(labelsize=22)
    cbar.ax.tick_params(which='major', length=15, width=1.25, direction='inout')

    plt.tight_layout()
    if save:
        if log:
            plt.savefig('%slogg_cb_logscale.png'%fig_path, dpi=250)
        else:
            plt.savefig('%slogg_cb.png'%fig_path, dpi=250)
    if show:
        plt.show()
    plt.close()


def ensemble_plot_double(path='../../Info/rotation.csv', path_save='distributions', min_sample=20, 
                         res_teff=100., res_logg=0.1, log=False, save=True, show=True, verbose=False):
    import os
    from matplotlib import cm
    if not os.path.exists(os.path.join(os.path.abspath(os.getcwd()), path_save)):
        os.mkdir(os.path.join(os.path.join(os.path.abspath(os.getcwd()), path_save)))
    # read in known rotation periods and get limits
    df = pd.read_csv(path)
    # for effective temperature
    min_teff, max_teff = np.floor(df.teff.min()/res_teff)*res_teff, np.ceil(df.teff.max()/res_teff)*res_teff
    edges_teff = np.arange(min_teff, max_teff+res_teff, res_teff)
    # for surface gravity
    min_logg, max_logg = np.floor(df.logg.min()/res_logg)*res_logg, np.ceil(df.logg.max()/res_logg)*res_logg
    edges_logg = np.arange(min_logg, max_logg+res_logg, res_logg)
    # let's try a pretty plot
    tnorm = matplotlib.colors.Normalize(vmin=min_teff, vmax=max_teff)
    norm = matplotlib.colors.Normalize(vmin=min_logg, vmax=max_logg)
    xs, ys, c, al = [], [], [], []

    fig = plt.figure(figsize=(10,12))
    ax = plt.subplot(2,1,1)
    for i in range(len(edges_teff)-1):
        mid_teff = (edges_teff[i]+edges_teff[i+1])/2.
        for j in range(len(edges_logg)-1):
            mid_logg = (edges_logg[j]+edges_logg[j+1])/2.
            query = df.query("teff >= %f and teff < %f and logg >= %f and logg < %f"%(edges_teff[i], edges_teff[i+1], edges_logg[j], edges_logg[j+1]))
            if verbose:
                print("teff~[%d,%d]; logg~[%f,%f]; n=%d"%(int(edges_teff[i]), int(edges_teff[i+1]), edges_logg[j], edges_logg[j+1], len(query)))
            if len(query) >= min_sample:
                x, spline = get_inverse(query, log=log)
                if x is not None:
                    fname = os.path.join(path_save,'teff_%d_%d_logg_%.1f_%.1f.txt'%(edges_teff[i], edges_teff[i+1], edges_logg[j], edges_logg[j+1]))
                    save_file(x, spline(x), fname)
                    xs.append(x)
                    ys.append(spline(x))
                    c.append(mid_teff)
                    al.append(mid_logg)
    lc = multiline(xs, ys, c, cmap='cividis', lw=2)
    ax.set_ylabel(r'$\rm P_{rot} \,\, [days]$', fontsize=28)
#    ax.set_xlabel(r'$F\,\mathrm{(P_{rot}\, \vert \,T_{eff}, log}g\mathrm{)}$', fontsize=28)
    ax.tick_params(axis='both', which='minor', length=10, width=1.25, direction='inout')
    ax.tick_params(axis='both', which='major', length=15, width=1.25, direction='inout')
    ax.tick_params(labelsize=22)
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xticklabels([])
#    ax.set_xticklabels([r'$0.0$', r'$0.2$', r'$0.4$', r'$0.6$', r'$0.8$', r'$1.0$'])
    ax.xaxis.set_minor_locator(MultipleLocator(0.05))
    ax.set_xlim([0.,1.])
    if log:
        ax.set_yscale('log')
        ax.set_yticks([0.3, 1., 3., 10., 30.])
        ax.set_yticklabels([r'$0.3$', r'$1$', r'$3$', r'$10$', r'$30$'])
        ax.set_ylim([0.1,68.])
    else:
        ax.set_yticks([10., 20., 30., 40., 50., 60.])
        ax.set_yticklabels([r'$10$', r'$20$', r'$30$', r'$40$', r'$50$', r'$60$'])
        ax.yaxis.set_minor_locator(MultipleLocator(5))
        ax.set_ylim([0.,68.])
    cbar = fig.colorbar(lc, pad=0.01)
    cbar.set_label(r'$\rm T_{eff} \,\, [K]$', size=28)
    cbar.ax.set_yticklabels([r'$3500$', r'$4000$', r'$4500$', r'$5000$', r'$5500$', r'$6000$'])
    cbar.ax.tick_params(labelsize=22)
    cbar.ax.tick_params(which='major', length=15, width=1.25, direction='inout')

    ax = plt.subplot(2,1,2)
    lc = multiline(xs, ys, al, cmap='cividis', lw=2)
    ax.set_ylabel(r'$\rm P_{rot} \,\, [days]$', fontsize=28)
    ax.set_xlabel(r'$F\,\mathrm{(P_{rot}\, \vert \,T_{eff}, log}g\mathrm{)}$', fontsize=28)
    ax.tick_params(axis='both', which='minor', length=10, width=1.25, direction='inout')
    ax.tick_params(axis='both', which='major', length=15, width=1.25, direction='inout')
    ax.tick_params(labelsize=22)
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xticklabels([r'$0.0$', r'$0.2$', r'$0.4$', r'$0.6$', r'$0.8$', r'$1.0$'])
    ax.xaxis.set_minor_locator(MultipleLocator(0.05))
    ax.set_xlim([0.,1.])
    if log:
        ax.set_yscale('log')
        ax.set_yticks([0.3, 1., 3., 10., 30.])
        ax.set_yticklabels([r'$0.3$', r'$1$', r'$3$', r'$10$', r'$30$'])
        ax.set_ylim([0.1,68.])
    else:
        ax.set_yticks([10., 20., 30., 40., 50., 60.])
        ax.set_yticklabels([r'$10$', r'$20$', r'$30$', r'$40$', r'$50$', r'$60$'])
        ax.yaxis.set_minor_locator(MultipleLocator(5))
        ax.set_ylim([0.,68.])
    cbar = fig.colorbar(lc, pad=0.01)
    cbar.set_label(r'$\mathrm{log}\,g \,\, [\mathrm{dex}]$', size=28)
    cbar.ax.set_yticklabels([r'$3.8$', r'$4.0$', r'$4.2$', r'$4.4$', r'$4.6$', r'$4.8$', r'$5.0$'])
    cbar.ax.tick_params(labelsize=22)
    cbar.ax.tick_params(which='major', length=15, width=1.25, direction='inout')

    plt.tight_layout()
    if save:
        if log:
            plt.savefig('%sdouble_panel_logscale.png'%fig_path, dpi=250)
        else:
            plt.savefig('%sdouble_panel.png'%fig_path, dpi=250)
    if show:
        plt.show()
    plt.close()

