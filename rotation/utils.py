import os
import random
import numpy as np
import pandas as pd
from scipy import stats, interpolate


def main(args):
    check_inputs(args)
    get_periods(args)


def check_inputs(args):
    assert isinstance(args.path_to_sample,str) and os.path.exists(args.path_to_sample), "Cannot find the Kepler/K2 sample of rotation periods."
    if args.path_to_stars is None and args.teff is None and args.logg is None:
        assert False, "No sample file or arrays were provided. Please try again."
    if args.path_to_stars is not None:
        assert isinstance(args.path_to_stars,str) and os.path.exists(args.path_to_stars), "Cannot find the file to the star list."
    if args.teff is not None and args.logg is not None:
        assert len(args.teff) == len(args.logg), "The two arrays do not have matching lengths."
    if (args.teff is not None and args.logg is None) or (args.teff is None and args.logg is not None):
        assert False, "Only one array was provided but it needs both to run. Please try again."


def fix_distribution(x, y):
    indices=[0]
    for i in range(1,len(y)):
        if y[i] != y[i-1]:
            indices.append(i)
    x, y = x[indices], y[indices]
    return x, y


def get_inverse(args, query, n_bins=100, log=False):
    kernel = stats.gaussian_kde(query.period.values)
    lower, upper = min(query.period.values), max(query.period.values)
    if args.log:
        x = np.logspace(np.log10(lower),np.log10(upper),args.n_bins)
    else:
        x = np.linspace(lower,upper,args.n_bins)
    y = np.cumsum(kernel(x))/np.sum(kernel(x))
    x, y = fix_distribution(x, y)
    try:
        spline = interpolate.CubicSpline(y, x)
    except ValueError:
        return None, None
    else:
        xnew = np.linspace(min(y),max(y),args.n_bins)
        return xnew, spline


def get_periods(args, path_to_sample='../../Info/rotation.csv', min_sample=20, res_teff=100., res_logg=0.1, period=[]):
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
def get_period(teff, logg, period=[], path='../../Info/rotation.csv', min_sample=20, res_teff=100., res_logg=0.1, log=False, n_bins=100, verbose=True):
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
