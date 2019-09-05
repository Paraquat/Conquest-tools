import re
import sys
import scipy as sp
from scipy.linalg import inv
from os import path
from pdb import set_trace
import md_tools

# Regular expressions
cell_re = re.compile('cell_vectors(.*?)end cell_vectors', re.M | re.S)
stress_re = re.compile('stress_tensor(.*?)end stress_tensor', re.M | re.S)
position_re = re.compile('positions(.*?)end positions', re.M | re.S)
position_re = re.compile('positions(.*?)end positions', re.M | re.S)
velocity_re = re.compile('velocities(.*?)end velocities', re.M | re.S)
force_re = re.compile('forces(.*?)end forces', re.M | re.S)
frame_re = re.compile('frame\s+\d+')
endframe_re = re.compile('end frame')

class FrameIterator:
  """Iterate through a frames file"""

  def __init__(self, fname, start):
    self.fname = fname
    self.start = start
    self.parser = FrameParser(fname)

  def __iter__(self):
    self.nframes = 0
    return self

  def __next__(self):
    self.nframes += 1
    while True:
      f = self.parser.get_frame()
      if f.step >= self.start:
        break
    return f

  def reset(self):
    self.parser = FrameParser(self.fname)

class FrameParser:
  """Parse a Frame file"""

  def __init__(self, fname):
    if path.exists(fname):
      self.framef = open(fname, 'r')
    self.first_call = True

  def close(self):
    if self.framef:
      self.framef.close()
      self.framef = None

  def get_frame(self, nframe=None):
    """Extract a single frame from a file"""
    buf = ""
    save_frame = False
    while True:
      line = self.framef.readline()
      if not line:
        yield None
        break
      if re.match(frame_re, line):
        save_frame = True
        n = int(line.split()[1])
        if nframe:
          if n != nframe:
            save_frame = False
          else:
            sys.stdout.write("Processing frame {}\r".format(n))

      if re.match(endframe_re, line):
        if save_frame:
          break

      if save_frame:
        buf += line
    fr = self.parse_frame(n, buf)
    yield fr

  def parse_frame(self, n, buf):
    """Read frame data from a string buffer"""

    m = re.search(position_re, buf)
    lines = m.group(1).strip().splitlines()
    if self.first_call:
      self.natoms = len(lines)
      self.first_call = False
    fr = Frame(self.natoms, n)
    for i in range(self.natoms):
      bits = lines[i].strip().split()
      bits.pop(0)
      fr.species[i] = int(bits.pop(0))
      for j in range(3):
        fr.r[i,j] = float(bits[j])

    m = re.search(cell_re, buf)
    lines = m.group(1).strip().splitlines()
    for i in range(3):
      bits = lines[i].strip().split()
      for j in range(3):
        fr.lat[i,j] = float(bits[j])

    m = re.search(stress_re, buf)
    if m:
      lines = m.group(1).strip().splitlines()
      for i in range(3):
        bits = lines[i].strip().split()
        for j in range(3):
          fr.stress[i,j] = float(bits[j])

    m = re.search(velocity_re, buf)
    lines = m.group(1).strip().splitlines()
    for i in range(self.natoms):
      bits = lines[i].strip().split()
      bits.pop(0)
      bits.pop(0)
      for j in range(3):
        fr.v[i,j] = float(bits[j])

    m = re.search(force_re, buf)
    lines = m.group(1).strip().splitlines()
    for i in range(self.natoms):
      bits = lines[i].strip().split()
      bits.pop(0)
      bits.pop(0)
      for j in range(3):
        fr.f[i,j] = float(bits[j])

    return fr

class Frame:
  """Stores a frame from a MD trajectory"""

  def __init__(self, nat, step):
    self.step = step
    self.nat = nat
    self.species = sp.zeros(nat, dtype=int)
    self.r = sp.zeros((nat, 3), dtype='float')
    self.v = sp.zeros((nat, 3), dtype='float')
    self.f = sp.zeros((nat, 3), dtype='float')
    self.lat = sp.zeros((3, 3), dtype='float')
    self.stress = sp.zeros((3, 3), dtype='float')
    self.ke = 0.
    self.pe = 0.
    self.E = 0.
    self.T = 0.
    self.P = 0.
    self.vmax = 1.0
    self.cart = True

  def diff_mic(self, r1, r2, lat=None):
    """Minimum image convention relative vector (orthorhombic cell only)"""
    if lat is None:
      lat = self.lat
    diff = r1 - r2
    for k in range(3):
      diff[k] -= round(diff[k]/lat[k,k])*lat[k,k]
    return diff

  def cart2frac(self):
    """Convert Cartesian coordinates to fractional"""
    if self.cart:
      lat_inv = inv(self.lat)
      for i in range(self.nat):
        self.r[i,:] = sp.matmul(lat_inv, self.r[i,:])
      self.cart = False

  def frac2cart(self):
    """Convert fractional coordinates to Cartesian"""
    for i in range(self.nat):
      self.r[i,:] = sp.matmul(self.lat, self.r[i,:])
    self.cart = True
