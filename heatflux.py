#!/usr/local/bin/python3

import argparse
import os
import cq_global
import scipy as sp
import matplotlib.pyplot as plt
from pdb import set_trace
from md_tools import HeatFlux
from cq_io import ConquestParser

parser = argparse.ArgumentParser(description='Compute heat flux autocorrelation \
        and thermal conductivity', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-d', '--dirs', nargs='+', default='.', dest='dirs',
                    action='store', help='Directories to compare')
parser.add_argument('--skip', action='store', dest='nskip', default=0,
                    type=int, help='Number of equilibration steps to skip')
parser.add_argument('--stride', action='store', dest='stride', default=1,
                    type=int, help='Only analyse every nth step of frames file')
parser.add_argument('--window', action='store', dest='window', default=1000.0,
                    type=float, help='Window for autocorrelation functions in fs')
parser.add_argument('--dump', action='store_true', dest='dump', 
                    help='Dump secondary data used to generate plots')

opts = parser.parse_args()

# Parser heat flux files, compute HFACF
first_dir = True
for d in opts.dirs:
  if first_dir:
    cq_parser = ConquestParser(path=d)
    dt = cq_parser.dt
    volume = cq_parser.structure.volume
    temp = float(cq_parser.get_flag('AtomMove.IonTemperature'))
    hf = HeatFlux(dt, opts.window)
    first_dir = False
  hf_path = os.path.join(d, cq_global.cq_heatflux_file)
  hf.parse_heatflux_file(hf_path)
  hf.update_G_2(opts.nskip)
hf.norm_HFACF()

# Plot the HFACF
hf.plot_HFACF()

fname = "hfacf.dat"

# Dump data
if opts.dump:
  hf.dump_HFACF()

# Compute thermal conductivity
kappa = hf.get_kappa(volume, temp)
print(kappa)
