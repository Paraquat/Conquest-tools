#!/usr/local/bin/python3

import argparse
import sys
import re
import os.path
import cq_global
import scipy as sp
import matplotlib.pyplot as plt
from cq_data import ConquestStructure
from cq_io import ConquestParser
from md_tools import Pairdist
from frame import Frame
from pdb import set_trace

# Command line arguments
parser = argparse.ArgumentParser(description='Analyse a CONQUEST-formatted structure',
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-i', '--infile', action='store', dest='infile',
                    default='coord_next.dat', help='Conquest format structure file')
parser.add_argument('--bonds', action='store_true', dest='bonds', 
                    help='Compute average and minimum bond lengths')
parser.add_argument('--density', action='store_true', dest='density', 
                    help='Compute density')
parser.add_argument('--nbins', action='store', dest='nbins', default=100,
                    help='Number of histogram bins')
parser.add_argument('-c', '--cutoff', nargs='+', action='store', default=None,
                    dest='cutoff', 
                    help='Bond length cutoff matrix (upper triangular part, in rows')
parser.add_argument('--printall', action='store_true', default=False,
                    dest='printall', help='Print all bond lengths')

opts = parser.parse_args()

cq_parser = ConquestParser()
structure = ConquestStructure(opts.infile)
f = structure.make_frame()

if cq_parser.get_flag('IO.FractionalAtomicCoords') == 'T':
  f.frac2cart()

if opts.bonds:
  cutoff = [float(bit) for bit in opts.cutoff]
  bondcut = sp.zeros((structure.nspecies,structure.nspecies))
  k = 0
  for i in range(structure.nspecies):
    for j in range(i, structure.nspecies):
      bondcut[i,j] = cutoff[k]
      bondcut[j,i] = cutoff[k]
      k+=1

  small = 0.1
  rdfwidth = 0.1
  rdfcut = min(structure.latvec[0,0], structure.latvec[1,1], structure.latvec[2,2])/2.0 + small

  pairdist = Pairdist(structure.natoms, structure.nspecies, rdfcut,
                      rdfwidth, cq_parser.get_flag('species'),
                      structure.species_count)
  pairdist.update_rdf(f)
  pairdist.get_bondlength(bondcut, f, opts.printall)

if opts.density:
  v = sp.dot(f.lat[:,0], sp.cross(f.lat[:,1],f.lat[:,2]))
  m = 0.0
  for s in cq_parser.get_flag('species').keys():
    m += structure.species_count[s]*cq_parser.get_flag('mass')[s]
  v = v * cq_global.bohr2ang**3 * cq_global.ang2cm**3
  m = m / cq_global.avogadro
  density = m/v
  print(f"Density: {density:>10.4f} g/cm^3")
