import re
import cq_global
import scipy as sp
from frame import Frame
from os import path

def strip_comments(line, separator):
  for s in separator:
    i = line.find(s)
    if i >= 0:
      line = line[:i]
  return line.strip()

class ConquestFlags:
  """A parser for the Conquest_input file. Generates a dictionary of
  input flags"""

  def __init__(self, inputfile):
    self.cq_input_file = inputfile
    self.flags = {}
    self.parse_input()
    self.parse_flags(self.cq_input_file)

  def get_flag(self, flag):
    # convert to lower case to avoid ambiguity
    return self.flags[flag.lower()]

  def set_flag(self, flag, value):
    # convert to lower case to avoid ambiguity
    self.flags[flag.lower()] = value

  def parse_flags(self, fname):
    with open(fname, 'r') as infile:
      for line in infile:
        stripped = strip_comments(line, "#%!")
        if stripped:
          bits = stripped.split()
          key = bits[0]
          if len(bits[1:]) == 1:
            value = bits[1]
          else:
            value = ' '.join(bits[1:])
          self.set_flag(key, value)

  def parse_input(self):
    # get the species labels
    specblock_re = re.compile(r'%block ChemicalSpeciesLabel\n(.*?)\n%endblock',
                              re.M | re.S | re.I)
    self.flags['species'] = {}
    self.flags['mass'] = {}
    with open(self.cq_input_file, 'r') as infile:
      m = re.search(specblock_re, infile.read())
      specinfo = m.group(1).splitlines()
      for line in specinfo:
        try:
          index, mass, spec  = line.split()
        except ValueError:
          index, mass, spec, ionfile = line.split()
        self.flags['species'][int(index)] = spec
        self.flags['mass'][int(index)] = float(mass)

class ConquestStructure:
  """A parser for Conquest-formatted structure files. Requires parsing
  of Conquest_input for species labels"""

  def __init__(self, inputfile):
    self.structure_file = inputfile
    self.parse_structure()

  def parse_structure(self):
    with open(self.structure_file, 'r') as infile:
      a = [float(bit) for bit in infile.readline().strip().split()]
      b = [float(bit) for bit in infile.readline().strip().split()]
      c = [float(bit) for bit in infile.readline().strip().split()]
      self.latvec = sp.array([a,b,c])
      natoms = int(infile.readline().strip())
      self.natoms = natoms
      self.coords = []
      self.species = []
      for i in range(natoms):
        x, y, z, spec, cx, cy, cz = infile.readline().strip().split()
        self.coords.append([float(x), float(y), float(z)])
        self.species.append(int(spec))
      self.coords = sp.array(self.coords)
      self.species = sp.array(self.species)
      scount = {}
      for i in range(natoms):
        if self.species[i] in scount.keys():
          scount[self.species[i]] += 1
        else:
          scount[self.species[i]] = 1
      self.species_count = scount
      self.nspecies = len(scount.keys())
      self.get_volume()

  def get_volume(self):
    """Get volume in A^3"""
    self.volume = self.latvec[0,0]*self.latvec[1,1]*self.latvec[2,2]

  def make_frame(self):
    f = Frame(self.natoms, 1)
    f.r = self.coords
    f.lat = self.latvec
    f.species = self.species
    return f


class MDStats:
  """A parser for the Conquest molecular dynamics statistics file"""

  def __init__(self, dt, inputfile):
    self.stat_file = inputfile
    self.nstep = 0
    self.dt = dt
    self.columns = {}
    self.avg = {}
    self.std = {}
    if path.exists(inputfile):
      self.parse_stats()

  def get_stats(self, key, nstart, nstop):
    return self.columns[key][nstart:nstop]

  def parse_stats(self):
    header = True
    with open(self.stat_file, 'r') as statfile:
      for line in statfile:
        if header:
          col_id = line.strip().split()
          for col in col_id:
            self.columns[col] = []
          header = False
        else:
          bits = line.strip().split()
          for i, bit in enumerate(bits):
            if i==0:
              info = int(bit)
            else:
              info = float(bit)
            self.columns[col_id[i]].append(info)
        self.nstep += 1
      for key in self.columns:
        self.columns[key] = sp.array(self.columns[key])
      self.columns['time'] = sp.array([float(s)*self.dt for s in self.columns['step']])

  def calc_mean(self, nstart, nstop):
    """Compute the mean of a quantity"""
    for key in self.columns:
      self.avg[key] = sp.mean(self.columns[key][nstart:nstop])

  def calc_std(self, nstart, nstop):
    """Compute the standard deviation of a quantity"""
    for key in self.columns:
      self.std[key] = sp.std(self.columns[key][nstart:nstop])

  def get_mean(self, key):
    """Get the mean of a thermodyanmic quantity (key)"""
    return self.avg[key]

  def get_std(self, key):
    """Get the standard deviation of thermodynamic quantity (key)"""
    return self.std[key]
