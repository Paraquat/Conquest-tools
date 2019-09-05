#!/usr/local/bin/python3

import argparse
import sys
import re
import os.path
import scipy as sp
import matplotlib.pyplot as plt
from frame import Frame
from cq_io import ConquestParser, ConquestWriter
from md_tools import MSER
from pdb import set_trace

ha2ev = 27.211399
ha2k = 3.15737513e5

# Command line arguments
parser = argparse.ArgumentParser(description='Plot statistics for a Conquest MD \
        trajectory', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-c', '--compare', action='store_true', default=False,
                    dest='compare', help='Compare statistics of trajectories \
                    in directories specified by -d')
parser.add_argument('-d', '--dirs', nargs='+', default='.', dest='dirs',
                    action='store', help='Directories to compare')
parser.add_argument('--description', nargs='+', default='', dest='desc',
                    action='store', help='Description of graph for legend \
                    (only if using --compare)')
parser.add_argument('--skip', action='store', dest='nskip', default=0,
                    type=int, help='Number of equilibration steps to skip')
parser.add_argument('--stop', action='store', dest='nstop', default=-1, 
                    type=int, help='Number of last frame in analysis')
parser.add_argument('--equil', action='store', dest='nequil', default=0, 
                    type=int, help='Number of equilibration steps')
parser.add_argument('--landscape', action='store_true', dest='landscape', 
                    help='Generate plot with landscape orientation')
parser.add_argument('--mser', action='store', dest='mser_var', default=None,
                    type=str, help='Compute MSER for the given property')

opts = parser.parse_args()

if not opts.compare:
  # Parse the input structure and Conquest_input files
  cq_parser = ConquestParser()
  natoms = cq_parser.structure.natoms
  dt = float(cq_parser.get_flag('AtomMove.Timestep'))
  ensemble = cq_parser.get_flag('MD.Ensemble')
  plot_nskip = int(opts.nskip*dt)
  thermo_type = cq_parser.get_flag('MD.Thermostat')
  baro_type = cq_parser.get_flag('MD.Barostat')
  if opts.nequil == 0:
    opts.nequil = opts.nskip
  cq_parser.calc_mean(opts.nequil)

  # Plot the statistics
  if opts.landscape:
    fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, sharex=True, figsize=(11,7))
    plt.tight_layout(pad=6.5)
  else:
    fig1, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1, sharex=True, figsize=(7,10))

  time = cq_parser.get_stats("time", opts.nskip, opts.nstop)
  pe = cq_parser.get_stats("pe", opts.nskip, opts.nstop)
  ke = cq_parser.get_stats("ke", opts.nskip, opts.nstop)
  if ensemble[2] == 't':
    try:
      thermo = cq_parser.get_stats("thermostat", opts.nskip, opts.nstop)
    except KeyError:
      thermo = cq_parser.get_stats("nhc", opts.nskip, opts.nstop)
  hprime = cq_parser.get_stats("H\'", opts.nskip, opts.nstop)
  T = cq_parser.get_stats("T", opts.nskip, opts.nstop)
  P = cq_parser.get_stats("P", opts.nskip, opts.nstop)
  if ensemble[1] == 'p':
    baro = cq_parser.get_stats("barostat", opts.nskip, opts.nstop)
    pv = cq_parser.get_stats("pV", opts.nskip, opts.nstop)
    V = cq_parser.get_stats("V", opts.nskip, opts.nstop)

  ax1.plot(time, pe, 'r-', label='Potential energy')
  ax1a = ax1.twinx()
  ax1a.plot(time, ke, 'b-', label='Kinetic energy')
  if ensemble[2] == 't':
    if thermo_type == 'nhc':
      ax1a.plot(time, thermo, 'g-', label='Thermostat energy')
    if thermo_type == 'svr':
      ax1a.plot(time, thermo, 'g-', label='Thermostat energy')
  if ensemble[1] == 'p':
    if 'ssm' in baro_type or 'pr' in baro_type:
      ax1a.plot(time, baro, 'c-', label='Barostat energy')
    ax1a.plot(time, pv, 'm-', label='pV')
  ax2.plot(time, hprime)
  ax2.plot((plot_nskip,time[-1]), (cq_parser.get_mean("H\'"), cq_parser.get_mean("H\'")), '-',
        label=r'$\langle H\' \rangle$ = {0:>12.4f} $\pm$ {1:<12.4f}'.format(cq_parser.get_mean("H\'"), cq_parser.get_std("H\'")))
  ax3.plot(time, T)
  ax3.plot((plot_nskip, time[-1]), (cq_parser.get_mean("T"),cq_parser.get_mean("T")), '-',
        label=r'$\langle T \rangle$ = {0:>12.4f} $\pm$ {1:<12.4f}'.format(cq_parser.get_mean("T"), cq_parser.get_std("T")))
  ax4.plot(time, P, 'b-')
  ax4.plot((plot_nskip,time[-1]), (cq_parser.get_mean("P"),cq_parser.get_mean("P")), 'b--',
        label=r'$\langle P \rangle$ = {0:>12.4f} $\pm$ {1:<12.4f}'.format(cq_parser.get_mean("P"), cq_parser.get_std("P")))
  if ensemble[1] == 'p':
    ax4a = ax4.twinx()
    ax4a.plot(time, V, 'r-')
    ax4a.plot((plot_nskip,time[-1]), (cq_parser.get_mean("V"),cq_parser.get_mean("V")), 'r--',
              label=r'$\langle V \rangle$ = {0:>12.4f} $\pm$ {1:<12.4f}'.format(cq_parser.get_mean("V"), cq_parser.get_std("V")))
  ax1.set_ylabel("E (Ha)")
  ax2.set_ylabel("H$'$ (Ha)")
  ax3.set_ylabel("T (K)")
  ax4.set_ylabel("P (GPa)", color='b')
  if ensemble[1] == 'p':
    ax4a.set_ylabel("V ($a_0^3$)", color='r')
  ax4.set_xlabel("time (fs)")
  ax1.legend(loc="upper left")
  ax1a.legend(loc="lower right")
  ax2.legend()
  ax3.legend()
  ax4.legend(loc="upper left")
  if ensemble[1] == 'p':
    ax4a.legend(loc="lower right")
  plt.xlim((plot_nskip,time[-1]))
  fig1.subplots_adjust(hspace=0)
  fig1.savefig("stats.pdf", bbox_inches='tight')
else:
  # If we're comparing statistics in several directories, use a simplified plot
  fig1, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(7,7))
  ax1.grid(which='both', axis='both')
  ax2.grid(which='both', axis='both')
  ax3.grid(which='both', axis='both')
  ax1a = ax1.twinx()
  for ind, d in enumerate(opts.dirs):
    cq_parser = ConquestParser(path=d)
    natoms = cq_parser.structure.natoms
    dt = float(cq_parser.get_flag('AtomMove.Timestep'))
    ensemble = cq_parser.get_flag('MD.Ensemble')
    plot_nskip = int(opts.nskip*dt)
    thermo_type = cq_parser.get_flag('MD.Thermostat')
    baro_type = cq_parser.get_flag('MD.Barostat')
    if opts.nequil == 0:
      opts.nequil = opts.nskip

    time = cq_parser.get_stats("time", opts.nskip, opts.nstop)
    hprime = cq_parser.get_stats("H\'", opts.nskip, opts.nstop)
    T = cq_parser.get_stats("T", opts.nskip, opts.nstop)
    P = cq_parser.get_stats("P", opts.nskip, opts.nstop)
    if ensemble[1] == 'p':
      V = cq_parser.get_stats("V", opts.nskip, opts.nstop)

    ax1.plot(time, hprime, linewidth=0.5, label=opts.desc[ind])
    y1,y2 = ax1.get_ylim()
    ax1a.set_ylim(y1*ha2k,y2*ha2k)
    ax2.plot(time, T, linewidth=0.5, label=opts.desc[ind])
    ax3.plot(time, P, linewidth=0.5, label=opts.desc[ind])

  ax1.set_ylabel("H$'$ (Ha)")
  ax1a.set_ylabel("H$'$ (K)")
  ax2.set_ylabel("T (K)")
  ax3.set_ylabel("P (GPa)")
  ax3.set_xlabel("time (fs)")
  ax1.legend()
  plt.xlim((opts.nskip,time[-1]))
  fig1.subplots_adjust(hspace=0)
  fig1.savefig("stats.pdf", bbox_inches='tight')

# Plot MSER
if opts.mser_var:
  nsteps = cq_parser.stats_parser.nstep
  stat = cq_parser.get_stats(opts.mser_var, 0, nsteps)
  step = cq_parser.get_stats('step', 0, nsteps)
  traj = MSER(nsteps, opts.mser_var, stat)
  traj.get_mser()
  traj.plot_mser(step)
