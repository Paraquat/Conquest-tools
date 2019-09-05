import os
import cq_global
from cq_data import ConquestFlags, ConquestStructure, MDStats
from frame import FrameParser
from pdb import set_trace

class ConquestParser:
  """A wrapper for parsing all Conquest files. Read inputs and outputs in a
  given directory"""

  def __init__(self, path='./'):
    self.path = path
    self.flags = ConquestFlags(os.path.join(path, cq_global.cq_input_file))
    if os.path.exists(os.path.join(path, cq_global.cq_input_log_file)):
      self.flags.parse_flags(os.path.join(path, cq_global.cq_input_log_file))
    structure_inp = self.get_flag('IO.Coordinates')
    self.structure = ConquestStructure(os.path.join(path, structure_inp))
    self.nspecies = self.structure.nspecies
    self.species = self.structure.species
    self.species_count = self.structure.species_count
    self.natoms = self.structure.natoms
    self.dt = float(self.get_flag('AtomMove.Timestep'))
    if os.path.exists(os.path.join(path, cq_global.cq_frames_file)):
      fpath = os.path.join(path, cq_global.cq_frames_file)
      self.frame_parser = FrameParser(fpath)
    elif os.path.exists(os.path.join(path, cq_global.cq_frames_old)):
      fpath = os.path.join(path, cq_global.cq_frames_old)
      self.frame_parser = FrameParser(fpath)
    else:
      print("No frames file found!")
    dt = float(self.get_flag('AtomMove.Timestep'))
    spath = None
    if os.path.exists(os.path.join(path, cq_global.cq_stats_file)):
      spath = os.path.join(path, cq_global.cq_stats_file)
      self.stats = MDStats(dt, spath)
    elif os.path.exists(os.path.join(path, cq_global.cq_stats_old)):
      spath = os.path.join(path, cq_global.cq_stats_old)
      self.stats = MDStats(dt, spath)
    else:
      print("No stats file found!")

  def get_flag(self, flag):
    return self.flags.get_flag(flag)

  def get_stats(self, key, nstart, nstop):
    return self.stats.get_stats(key, nstart, nstop)

  def calc_mean(self, nstart, nstop=-1):
    self.stats.calc_mean(nstart, nstop)
    self.stats.calc_std(nstart, nstop)

  def get_mean(self, key):
    return self.stats.get_mean(key)

  def get_std(self, key):
    return self.stats.get_std(key)

  def parse_frame(self, n):
    return next(self.frame_parser.get_frame(nframe=n))

class ConquestWriter:
  """Write Conquest-formatted files"""

  def __init__(self, structure, flags):
    self.structure = structure
    self.flags = flags
    self.units = 'atomic'
    self.frac = True
    self.lat_fmt = "{0:>20.12f}{1:>20.12f}{2:>20.12f}\n"
    self.pos_fmt = "{0:>24.16e}{1:>24.16e}{2:>24.16e}{3:>2d} T T T\n"

  def write_config(self, conf_file_name):
    with open(conf_file_name, 'w') as outfile:
      outfile.write(self.lat_fmt.format(self.structure.latvec[:,0]))
      outfile.write(self.lat_fmt.format(self.structure.latvec[:,1]))
      outfile.write(self.lat_fmt.format(self.structure.latvec[:,2]))
      outfile.write(f'{self.structure.natoms}\n')
      for i in range(self.structure.natoms):
        outfile.write(self.pos_fmt.format(structure.coords[:,i],
                                          structure.species[i]))

  def write_frame(self, f, conf_file_name):
    f.cart2frac()
    with open(conf_file_name, 'w') as outfile:
      outfile.write(self.lat_fmt.format(*f.lat[0,:]))
      outfile.write(self.lat_fmt.format(*f.lat[1,:]))
      outfile.write(self.lat_fmt.format(*f.lat[2,:]))
      outfile.write(f'{f.nat}\n')
      for i in range(f.nat):
        outfile.write(self.pos_fmt.format(*f.r[i,:], f.species[i]))
