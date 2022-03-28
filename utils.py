import os
import random
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats, interpolate



def fix_distribution(x, y):
    indices=[0]
    for i in range(1,len(y)):
        if y[i] != y[i-1]:
            indices.append(i)
    x, y = x[indices], y[indices]
    return x, y


def get_inverse(query, n_bins=100,):
    kernel = stats.gaussian_kde(query.period.values)
    lower, upper = min(query.period.values), max(query.period.values)
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


def get_periods(path_to_sample='data/sara_revised.csv', path_to_known='data/kepler.csv', 
                teff=None, logg=None,  min_sample=20, res_teff=100., res_logg=0.1, 
                n_bins=100, verbose=False,):
    period=[]
    if teff is not None and logg is not None:
        assert len(teff) == len(logg), "# ERROR: input arrays must be the same length"
    else:
        if os.path.exists(path_to_sample):
            df = pd.read_csv(path_to_sample)
            teff, logg = np.copy(df.teff.values), np.copy(df.logg.values)
        else:
            print('# ERROR: incorrect path -- cannot find sample\n#      Please try again.')
            return
    # read in known rotation periods and get limits
    df = pd.read_csv(path_to_known)
    for tt, ll in zip(teff, logg):
        per = np.nan
        # select stars near target in HR diagram
        query = df.query("teff >= %f and teff < %f and logg >= %f and logg < %f"%(tt-(res_teff/2.), tt+(res_teff/2.), ll-(res_logg/2.), ll+(res_logg/2.)))
        if len(query) < min_sample:
            if verbose:
                print('WARNING: not enough in the sample to create an accurate distribution.\nTry changing the resolution of the grid to include more stars!')
                print('Currently using teff +/- %.1f K and logg +/- %.2f dex'%(res_teff/2.,res_logg/2.))
        else:
            _, spline = get_inverse(query)
            # draw random number to map back to period distribution
            per = spline(random.random())+0.
            if per <= 0.0:
                while per <= 0.0:
                    per = spline(random.random())+0.
        period.append(per)
    return np.array(period)


def make_distributions(path_to_dists=None, path_to_known='data/kepler.csv', min_sample=20, save=True,
                       res_teff=100., res_logg=0.1, show=True, verbose=False,):
    if path_to_dists is None:
        path_to_dists = 'data/distributions/dteff_%d_K_dlogg_%.1f_dex/'%(int(res_teff), res_logg)
    if not os.path.exists(path_to_dists):
        os.mkdir(path_to_dists)
    # read in known rotation periods and get limits
    df = pd.read_csv(path_to_known)
    # for effective temperature
    min_teff, max_teff = np.floor(df.teff.min()/res_teff)*res_teff, np.ceil(df.teff.max()/res_teff)*res_teff
    edges_teff = np.arange(min_teff, max_teff+res_teff, res_teff)
    # for surface gravity
    min_logg, max_logg = np.floor(df.logg.min()/res_logg)*res_logg, np.ceil(df.logg.max()/res_logg)*res_logg
    edges_logg = np.arange(min_logg, max_logg+res_logg, res_logg)
    for i in range(len(edges_teff)-1):
        for j in range(len(edges_logg)-1):
            query = df.query("teff >= %f and teff < %f and logg >= %f and logg < %f"%(edges_teff[i], edges_teff[i+1], edges_logg[j], edges_logg[j+1]))
            if verbose:
                print("teff~[%d,%d]; logg~[%f,%f]; n=%d"%(int(edges_teff[i]), int(edges_teff[i+1]), edges_logg[j], edges_logg[j+1], len(query)))
            if len(query) >= min_sample:
                x, spline = get_inverse(query)
                if x is not None:
                    f_name = '%steff_%d_%d_logg_%.1f_%.1f.txt'%(path_to_dists, edges_teff[i], edges_teff[i+1], edges_logg[j], edges_logg[j+1])
                    save_file(x, spline(x), f_name)


def save_file(x, y, path, formats=[">10.4f", ">10.2f"]):
    header = '#      CDF      prot\n#             (days)\n'
    with open(path, "w") as f:
        f.write(header)
        for xx, yy in zip(x, y):
            values = [xx, yy]
            text = '{:{}}'*len(values) + '\n'
            fmt = sum(zip(values, formats), ())
            f.write(text.format(*fmt))
