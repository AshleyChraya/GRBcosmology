import argparse  # Commandline input
from collections import OrderedDict as odict

import getdist.plots as gdplt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cobaya.run import run
from getdist.mcsamples import MCSamplesFromCobaya
from scipy.integrate import quad
from scipy.interpolate import interp1d
from mpi4py import MPI
from cobaya.log import LoggedError