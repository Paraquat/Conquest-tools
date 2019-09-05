#!/usr/local/bin/python3

import argparse
import sys
import re
import os.path
import scipy as sp
import matplotlib.pyplot as plt
from frame import Frame
from cq_io import ConquestParser, ConquestWriter
from trajectory import Trajectory
from pdb import set_trace

# Command line arguments
parser = argparse.ArgumentParser(description='Analyse a Conquest MD \
        trajectory', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-d', '--dirs', nargs='+', default='.', dest='dirs',
                    action='store', help='Directories to compare')
parser.add_argument('--skip', action='store', dest='nskip', default=0,
                    type=int, help='Number of equilibration steps to skip')
parser.add_argument('--stride', action='store', dest='stride', default=1,
                    type=int, help='Only analyse every nth step of frames file')
parser.add_argument('--snap', action='store', dest='snap', default=-1, 
                    type=int, help='Analyse Frame of a single snapshot')
parser.add_argument('--stop', action='store', dest='nstop', default=-1, 
                    type=int, help='Number of last frame in analysis')
parser.add_argument('--equil', action='store', dest='nequil', default=0, 
                    type=int, help='Number of equilibration steps')
parser.add_argument('--vacf', action='store_true', dest='vacf', 
                    help='Plot velocity autocorrelation function')
parser.add_argument('--msd', action='store_true', dest='msd', 
                    help='Plot mean squared deviation')
parser.add_argument('--rdf', action='store_true', dest='rdf', 
                    help='Plot radial distribution function')
parser.add_argument('--stress', action='store_true', dest='stress', 
                    help='Plot stress')
parser.add_argument('--nbins', action='store', dest='nbins', default=100,
                    help='Number of histogram bins')
parser.add_argument('--rdfwidth', action='store', dest='rdfwidth',
                    type=float, default=0.05, help='RDF histogram bin width (A)')
parser.add_argument('--rdfcut', action='store', dest='rdfcut', default=8.0,
                    type=float, help='Distance cutoff for RDF in Angstrom')
parser.add_argument('--window', action='store', dest='window', default=1000.0,
                    type=float, help='Window for autocorrelation functions in fs')
parser.add_argument('--fitstart', action='store', dest='fitstart', default=-1.0,
                    type=float, help='Start time for curve fit')
parser.add_argument('--dump', action='store_true', dest='dump', 
                    help='Dump secondary data used to generate plots')

opts = parser.parse_args()

flags = {}
if opts.rdf:
  flags['rdf'] = True
  flags['rdfcut'] = opts.rdfcut
  flags['rdfwidth'] = opts.rdfwidth
  print("Computing radial distribution function g(r)")
if opts.vacf:
  flags['vacf'] = True
  print("Computing velocity autocorrelation function")
if opts.msd:
  flags['msd'] = True
  print("Computing mean squared deviation")
if opts.stress:
  flags['stress'] = True
if opts.dump:
  flags['dump'] = True
flags['window'] = opts.window
flags['stride'] = opts.stride
flags['fitstart'] = opts.fitstart


first_dir = True
for d in opts.dirs:
  cq_parser=ConquestParser(path=d)
  flags['nspecies'] = cq_parser.nspecies
  flags['natoms'] = cq_parser.natoms
  flags['dt'] = cq_parser.dt
  flags['species'] = cq_parser.get_flag('species')
  flags['species_count'] = cq_parser.species_count
  if first_dir:
    full_traj = Trajectory(**flags)
  traj = Trajectory(**flags)
  fcount = -1
  # Decide which frames to process
  while True:
    fcount += 1
    if fcount < opts.nskip:
      continue
    if fcount%opts.stride != 0:
      continue
    if opts.snap != -1:
      if fcount != opts.snap:
        continue
      if fcount > opts.snap:
        break
    # Parse the frames file
    frame = cq_parser.parse_frame(fcount)
    if not frame:
      break

    traj.update_frame(frame)
    traj.process_frame()
    if opts.stress:
      traj.update_stress()

    if opts.snap != -1:
      break
    if (fcount >= opts.nstop) and (opts.nstop > 0):
      break

  traj.finalise()
  full_traj.cat(traj)
  first_dir = False
print()
print("Analysing {} frames...".format(fcount))

# full_traj.finalise()
full_traj.plot()
# Plot the stress
if opts.stress:
  plt.figure("Stress")
  if cq_params['MD.Ensemble'][1] == "p":
    fig2, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True)
  else:
    fig2, (ax1,) = plt.subplots(nrows=1, ncols=1)

  plt.xlabel("t (fs)")
  ax1.set_ylabel("Stress (GPa)")
  ax2.set_ylabel("Cell dimension ($a_0$)")
  plt.xlim((traj.time[0], traj.time[-1]))
  ax1.plot(traj.time, traj.stress[:,0,0], 'r-', label='xx', linewidth=1.0)
  ax1.plot(traj.time, traj.stress[:,1,1], 'g-', label='yy', linewidth=1.0)
  ax1.plot(traj.time, traj.stress[:,2,2], 'b-', label='zz', linewidth=1.0)
  ax1.plot((traj.time[0],traj.time[-1]),
            (traj.mean_stress[0,0], traj.mean_stress[0,0]), 'r-',
          label=r'$\langle S_{{xx}} \rangle$ = {0:<10.4f}'.format(traj.mean_stress[0,0]))
  ax1.plot((traj.time[0],traj.time[-1]),
            (traj.mean_stress[1,1], traj.mean_stress[1,1]), 'g-',
          label=r'$\langle S_{{yy}} \rangle$ = {0:<10.4f}'.format(traj.mean_stress[1,1]))
  ax1.plot((traj.time[0],traj.time[-1]),
            (traj.mean_stress[2,2], traj.mean_stress[2,2]), 'b-',
          label=r'$\langle S_{{zz}} \rangle$ = {0:<10.4f}'.format(traj.mean_stress[2,2]))

  if cq_params['MD.Ensemble'][1] == "p":
    ax2.plot(traj.time, traj.lat[:,0,0], 'r-', label='a', linewidth=1.0)
    ax2.plot(traj.time, traj.lat[:,1,1], 'g-', label='b', linewidth=1.0)
    ax2.plot(traj.time, traj.lat[:,2,2], 'b-', label='c', linewidth=1.0)
    ax2.plot((traj.time[0],traj.time[-1]),
              (traj.mean_lat[0,0], traj.mean_lat[0,0]), 'r-',
            label=r'$\langle a \rangle$ = {0:<10.4f}'.format(traj.mean_lat[0,0]))
    ax2.plot((time[0],time[-1]),
              (traj.mean_lat[1,1], traj.mean_lat[1,1]), 'g-',
            label=r'$\langle b \rangle$ = {0:<10.4f}'.format(traj.mean_lat[1,1]))
    ax2.plot((time[0],time[-1]),
              (traj.mean_lat[2,2], traj.mean_lat[2,2]), 'b-',
            label=r'$\langle c \rangle$ = {0:<10.4f}'.format(traj.mean_lat[2,2]))
    ax1.legend(bbox_to_anchor=(1.05,1), loc=2, borderaxespad=0.)
    ax2.legend(bbox_to_anchor=(1.05,1), loc=2, borderaxespad=0.)
    fig2.subplots_adjust(hspace=0)
    plt.setp([a.get_xticklabels() for a in fig1.axes[:-1]], visible=False)
    fig2.savefig("stress.pdf", bbox_inches='tight')
