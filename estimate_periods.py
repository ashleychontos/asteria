#!/usr/local/bin/python
import os
import random
import argparse
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats, interpolate



def main(args):
    check_inputs(args)
    get_periods(args)


def check_inputs(args):
    assert isinstance(args.path_to_known,str) and os.path.exists(args.path_to_known), "Cannot find the Kepler/K2 sample of rotation periods."
    if args.path_to_sample is None and args.teff is None and args.logg is None:
        assert False, "No sample file or arrays were provided. Please try again."
    if args.path_to_sample is not None:
        assert isinstance(args.path_to_sample,str) and os.path.exists(args.path_to_sample), "Cannot find the file to the star list."
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


def get_inverse(args, query, n_bins=100,):
    kernel = stats.gaussian_kde(query.period.values)
    lower, upper = min(query.period.values), max(query.period.values)
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


def get_periods(args, path_to_sample='data/sara_revised.csv', path_to_known='data/kepler.csv', 
                min_sample=20, res_teff=100., res_logg=0.1,):
    period=[]
    # read in known rotation periods to draw samples from
    if os.path.exists(args.path_to_known):
        df = pd.read_csv(args.path_to_known)
    else:
        print('# ERROR: incorrect path to known periods.\n#      Please try again.')
        return
    if args.path_to_sample is not None:
        # read in targets of interest to estimate rotation periods for
        stars = pd.read_csv(args.path_to_sample)
        args.teff, args.logg = stars.teff.values.tolist(), stars.logg.values.tolist()
    # iterate through stars to estimate rotation periods
    for teff, logg in zip(args.teff, args.logg):
        per = np.nan
        # select stars near target in HR diagram
        query = df.query("teff >= %f and teff < %f and logg >= %f and logg < %f"%(teff-(args.res_teff/2.), teff+(args.res_teff/2.), logg-(args.res_logg/2.), logg+(args.res_logg/2.)))
        if len(query) < args.min_sample:
            if args.verbose:
                print('WARNING: not enough in the sample to create an accurate distribution.\nTry changing the resolution of the grid to include more stars!')
                print('Currently using teff +/- %.1f K and logg +/- %.2f dex'%(args.res_teff/2.,args.res_logg/2.))
        else:
            _, spline = get_inverse(args, query)
            # draw random number to map back to period distribution
            per = spline(random.random())+0.
            if per <= 0.0:
                while per <= 0.0:
                    per = spline(random.random())+0.
        period.append(per)
    if args.path_to_sample is not None:
        # save new period estimates
        stars['period'] = np.array(period)
        stars.to_csv(args.path_to_sample, index=False)



def save_file(x, y, path, formats=[">10.4f", ">10.2f"]):
    header = '#      CDF      prot\n#             (days)\n'
    with open(path, "w") as f:
        f.write(header)
        for xx, yy in zip(x, y):
            values = [xx, yy]
            text = '{:{}}'*len(values) + '\n'
            fmt = sum(zip(values, formats), ())
            f.write(text.format(*fmt))


##########################################################################################
#                                                                                        #
#                                        INITIATE                                        #
#                                                                                        #
##########################################################################################


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This is a script to estimate rotation periods for a list of stars.')
    parser.add_argument('--sample', metavar='str', help='Path to targets of interest', type=str, default='data/sara_revised.csv', dest='path_to_sample')
    parser.add_argument('--dist', metavar='str', help='Path to distributions', type=str, default=None, dest='path_to_dists')
    parser.add_argument('--logg', metavar='float', help='Array of surface gravities', type=float, default=None, nargs='*', dest='logg')
    parser.add_argument('--min', '--minsample', metavar='int', help='Minimum number of stars to use to construct PDF', type=int, dest='min_sample', default=20)
    parser.add_argument('--nbins', metavar='int', help='Number of bins to use to construct PDF', type=int, dest='n_bins', default=100)
    parser.add_argument('--known', metavar='str', help='Path to Kepler/K2 samples', dest='path_to_known', type=str, default='data/kepler.csv')
    parser.add_argument('--dlogg', '--reslogg', metavar='float', help='Resolution grid in surface gravity', type=float, dest='res_logg', default=0.1)
    parser.add_argument('--dteff', '--resteff', metavar='float', help='Resolution grid in effective temperature', type=float, dest='res_teff', default=100.0)
    parser.add_argument('--teff', metavar='float', help='Array of effective temperatures', type=float, default=None, nargs='*', dest='teff')
    parser.add_argument('-v', '--verbose', help='Turn on verbose output', type=bool, default=False, dest='verbose')
    main(parser.parse_args())
