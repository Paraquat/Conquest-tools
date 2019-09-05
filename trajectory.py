from pdb import set_trace
from md_tools import Pairdist, MSD, VACF
from frame import Frame
import scipy as sp

class Trajectory:
  """Accumulate dynamics (MSD, VACF, RDF) data from an MD trajectory by
  parsing frames"""

  def __init__(self, **flags):
    self.nframes = 0
    self.flags = {}
    self.time = []
    for key, value in flags.items():
      self.flags[key] = value
    self.natoms = self.flags['natoms']
    self.dt = self.flags['dt']
    if 'rdf' in self.flags:
      self.pairdist = Pairdist(self.natoms, self.flags['nspecies'],
                               self.flags['rdfcut'], self.flags['rdfwidth'],
                               self.flags['species'], self.flags['species_count'])
    if 'msd' in self.flags:
      self.msd = MSD(self.flags['window'], self.natoms, self.dt,
                     self.flags['stride'])
    if 'vacf' in self.flags:
      self.vacf = VACF(self.flags['window'], self.natoms, self.dt,
                       self.flags['stride'])
    if 'stress' in self.flags:
      self.stress = []
      self.lat = []
    if not self.flags['window']:
      sys.exit('No window given')

  def cat(self, other):
    """Concatenate two trajectories"""
    nf_self = float(self.nframes)
    nf_other = float(other.nframes)
    weight_self = nf_self/(nf_self + nf_other)
    weight_other = nf_other/(nf_self + nf_other)
    self.nframes += other.nframes
    self.time.extend(other.time)
    if 'rdf' in self.flags:
      self.pairdist.gr_total = self.pairdist.gr_total * weight_self + \
        other.pairdist.gr_total * weight_other
      self.pairdist.gr = self.pairdist.gr * weight_self + \
        other.pairdist.gr * weight_other
      self.pairdist.freq_total += other.pairdist.freq_total
      self.pairdist.freq += other.pairdist.freq
      self.pairdist.coord_total = self.pairdist.coord_total * weight_self + \
        other.pairdist.coord_total * weight_other
      if self.pairdist.nspec > 1:
        self.pairdist.coord = self.pairdist.coord * weight_self + \
          other.pairdist.coord * weight_other
    if 'vacf' in self.flags:
      self.vacf.vacf = self.vacf.vacf * weight_self + \
        other.vacf.vacf * weight_other
      self.vacf.nwindows += other.vacf.nwindows
    if 'msd' in self.flags:
      self.msd.msd = self.msd.msd * weight_self + \
        other.msd.msd * weight_other
      self.msd.nwindows += other.msd.nwindows

  def udpate_flags(self, **flags):
    for key, value in flags.items():
      self.flags[key] = value

  def update_frame(self, frame):
    """Load a new frame"""
    self.frame = frame
    self.time.append(float(frame.step)*self.dt)
    self.nframes += 1

  def finalise(self):
    """Normalise quantities"""
    self.time = sp.array(self.time)
    if 'rdf' in self.flags:
      self.pairdist.norm_rdf()
      self.pairdist.get_coordination()
    if 'vacf' in self.flags:
      # self.vacf.finalise()
      self.vacf.norm_vacf()
    if 'msd' in self.flags:
      # self.msd.finalise()
      self.msd.norm_msd()
    if 'stress' in self.flags:
      self.stress = sp.array(self.stress)
      self.lat = sp.array(self.lat)
      self.mean_stress = sp.zeros((3,3))
      self.mean_lat = sp.zeros((3,3))
      for i in range(3):
        for j in range(3):
          self.mean_stress[i,j] += sp.mean(self.stress[:,i,j])
          self.mean_lat[i,j] += sp.mean(self.lat[:,i,j])

  def plot(self):
    """Plot things"""
    if 'rdf' in self.flags:
      if 'dump' in self.flags:
        self.pairdist.dump_gr()
      self.pairdist.plot_gr()
    if 'vacf' in self.flags:
      if 'dump' in self.flags:
        self.vacf.dump_vacf()
      self.vacf.plot_vacf()
    if 'msd' in self.flags:
      if 'dump' in self.flags:
        self.msd.dump_msd()
      self.msd.plot_msd(self.flags['fitstart'])

  def process_frame(self):
    if 'rdf' in self.flags:
      self.update_rdf()
    if 'vacf' in self.flags:
      self.update_vacf()
    if 'msd' in self.flags:
      self.update_msd()
    if 'stress' in self.flags:
      self.update_stress()

  def update_stress(self):
    self.stress.append(self.frame.stress)
    self.lat.append(self.frame.lat)

  def update_rdf(self):
    self.pairdist.update_rdf(self.frame)

  def update_vacf(self):
    self.vacf.update_vacf(self.frame)

  def update_msd(self):
    self.msd.update_msd(self.frame)
