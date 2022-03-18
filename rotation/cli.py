import argparse

import rotation
from rotation import utils
from rotation import DATADIR


def main():

#####################################################################
# Initiate parser
#

    parser = argparse.ArgumentParser(
                                     description="prot: a quick package for estimating stellar rotation periods given their teff and logg", 
                                     prog='prot',
    )
    parser.add_argument('-version', '--version',
                        action='version',
                        version="%(prog)s {}".format(prot.__version__),
                        help="Print version number and exit.",
    )

#####################################################################
# Parent parser contains arguments and options common to all modes
#

    main_parser = argparse.ArgumentParser()

    main_parser.add_argument('--input', '--list', '--stars', 
                             metavar='str', 
                             help='Path to targets of interest', 
                             default=None, 
                             dest='path_to_stars',
                             type=str, 
    )
    main_parser.add_argument('-l', '--log', 
                             help='Use log bins to construct PDFs', 
                             default=False, 
                             dest='log', 
                             action='store_true',
    )
    main_parser.add_argument('--logg',
                             metavar='float', 
                             help='Array of surface gravities', 
                             default=None, 
                             nargs='*', 
                             dest='logg',
                             type=float, 
    )
    main_parser.add_argument('--min', '--minsample', 
                             metavar='int', 
                             help='Minimum number of stars to use to construct PDF', 
                             default=20,
                             dest='min_sample', 
                             type=int, 
    )
    main_parser.add_argument('--nbins', 
                             metavar='int', 
                             help='Number of bins to use to construct PDF',   
                             default=100,
                             dest='n_bins',
                             type=int,
    )
    main_parser.add_argument('--path', '--sample', 
                             metavar='str', 
                             help='Path to Kepler/K2 samples', 
                             default=os.path.join(DATADIR,'rotation.csv'),
                             dest='path_to_sample',
                             type=str, 
    )
    main_parser.add_argument('-r', '--return', 
                             help='Return array of estimated periods', 
                             default=False, 
                             dest='returnn', 
                             action='store_true',
    )
    main_parser.add_argument('--resl', '--reslogg', 
                             metavar='float', 
                             help='Resolution grid in surface gravity', 
                             default=0.1,
                             dest='res_logg',
                             type=float, 
    )
    main_parser.add_argument('--rest', '--resteff', 
                             metavar='float', 
                             help='Resolution grid in effective temperature',  
                             default=100.0,
                             dest='res_teff', 
                             type=float,
    )
    main_parser.add_argument('--teff',
                             metavar='float', 
                             help='Array of effective temperatures',  
                             default=None, 
                             nargs='*', 
                             dest='teff',
                             type=float,
    )
    main_parser.add_argument('-v', '--verbose', 
                             help='Verbose output', 
                             default=False, 
                             dest='verbose', 
                             action='store_true',
    )

    main_parser.set_defaults(func=utils.main)

    args = parser.parse_args()
    args.func(args)



if __name__ == '__main__':

    main()
