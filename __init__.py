import os

from .utils import *

__all__ = ['cli', 'utils']

__version__ = '0.0.1'

DATADIR = None

if 'PROT' in os.environ:
    DATADIR = os.environ['PROT']
else:
    DATADIR = os.path.join(os.path.expanduser('~'), '.prot')

if not os.path.isdir(DATADIR):
    # If DATADIR doesn't exist, make a new directory to accommodate data
    try:
        os.mkdir(DATADIR)
    except OSError:
        # If cannot make new directory, we assume it to be CWD
        DATADIR = '.'