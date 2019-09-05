#!/usr/local/bin/python3

import cq_global
import scipy as sp
import matplotlib.pyplot as plt
from scipy.linalg import norm
from scipy.integrate import cumtrapz, trapz
from scipy.signal import correlate
from scipy.optimize import curve_fit, leastsq
from scipy import histogram
from math import ceil, pi
from pdb import set_trace


def autocorr(x, y=None):
  """Autocorrelation function"""
  if y.any():
    result = correlate(x, y, mode='full')
  else:
    result = correlate(x, x, mode='full')
  return result[result.size // 2:]

def diff_mic(pos1, pos2, cell):
  """Minimum image convention relative vector (orthorhombic cell only)"""
  diff = pos2 - pos1
  for i in range(3):
    diff[i] -= round(diff[i]/cell[i])*cell[i]
  return diff

def linear(x, a, b):
  return a*x + b

def linearfit(y, x):
  a, b = sp.polyfit(y, x, 1)
  return (a, b)


class Pairdist:
  """Object for computing pair distribution functions"""

  def __init__(self, nat, nspec, rcut, binwidth, species, species_count):
    self.nframes = 0
    self.nat = nat
    self.nspec = nspec
    self.rcut = rcut
    self.binwidth = binwidth
    self.nbins = ceil(rcut/binwidth)+1
    self.spec_count = species_count
    self.species = species
    self.bins = []
    for i in range(self.nbins):
      self.bins.append((float(i)*binwidth + binwidth/2.))
    self.bins = sp.array(self.bins)
    self.dt = sp.zeros((self.nat,self.nat), dtype='float')
    self.freq_total = sp.zeros(self.nbins, dtype='int')
    self.freq = sp.zeros((self.nbins,self.nspec,self.nspec), dtype='int')
    self.nfac_total = sp.zeros(self.nbins, dtype='float')
    self.nfac = sp.zeros((self.nbins,self.nspec,self.nspec), dtype='float')
    self.gr_total = sp.zeros(self.nbins, dtype='float')
    self.gr = sp.zeros((self.nbins,self.nspec,self.nspec), dtype='float')
    self.coord_total = sp.zeros(self.nbins, dtype='float')
    if self.nspec > 1:
      self.coord = sp.zeros((self.nbins,self.nspec,self.nspec), dtype='float')

  def update_rdf(self, frame):
    self.nframes += 1
    cell = sp.zeros(3)
    for i in range(3):
      cell[i] = frame.lat[i,i]

    self.volume = cell[0]*cell[1]*cell[2]*cq_global.bohr2ang**3
    self.rho = float(self.nat)/self.volume

    for i in range(self.nat):
      for j in range(i+1, self.nat):
        diff = frame.diff_mic(frame.r[i,:], frame.r[j,:])*cq_global.bohr2ang
        self.dt[i,j] = norm(diff)
        self.dt[j,i] = norm(diff)
        if self.dt[i,j] < self.rcut:
          ind = int(round((self.dt[i,j]+self.binwidth)/self.binwidth))-1
          self.freq_total[ind] += 2
          if self.nspec > 1:
            for ispec in range(self.nspec):
              for jspec in range(ispec, self.nspec):
                if (ispec == frame.species[i]-1 and jspec == frame.species[j]-1):
                  self.freq[ind, ispec, jspec] += 2

  def norm_rdf(self):
    """Normalise the RDF"""
    const1 = 4.0*pi*(self.binwidth**3)/3.0
    const2 = self.rho*self.nat*self.nframes
    for i in range(self.nbins):
      vshell = (float(i+1)**3 - float(i)**3)*const1
      self.nfac_total[i] = vshell*const2
      if self.nspec > 1:
        for ispec in range(self.nspec):
          for jspec in range(self.nspec):
            const3 = self.rho*self.spec_count[ispec+1]*self.spec_count[jspec+1]/self.nat
            self.nfac[i,ispec,jspec] = vshell*const3*self.nframes
    self.gr_total = self.freq_total.astype(float)/self.nfac_total
    if self.nspec > 1:
      self.gr = self.freq.astype(float)/self.nfac

  def get_coordination(self):
    """Compute coordination"""
    gxrsq = self.gr_total*self.bins**2
    self.coord_total[1:] = cumtrapz(gxrsq,self.bins)
    self.coord_total *= 4.*pi*self.rho
    if self.nspec > 1:
      for ispec in range(self.nspec):
        for jspec in range(ispec,self.nspec):
          gxrsq = self.gr[:,ispec,jspec]*self.bins**2
          self.coord[1:,ispec,jspec] = cumtrapz(gxrsq[:], self.bins)
          self.coord *= 4.*pi*self.rho # check this

  def plot_gr(self):
    """Plot the radial distribution function"""
    plt.figure("RDF")
    filename = "rdf.pdf"
    if (self.nspec > 1):
      fig3, (axl, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True)
    else:
      fig3, axl = plt.subplots()
    axl.minorticks_on()
    axl.grid(b=True, which='major', axis='x', color='gray', linestyle='-')
    axl.grid(b=True, which='minor', axis='x', color='gray', linestyle='--')
    axl.grid(b=True, which='major', axis='y', color='gray', linestyle='-')
    # axl.grid(b=True, which='minor', axis='y', color='gray', linestyle='--')
    axr = axl.twinx()
    axl.set_ylabel("g(r)", color='b')
    axr.set_ylabel("Coordination", color='r')
    axl.plot(self.bins, self.gr_total,'b-', label="total", linewidth=1.0)
    axr.plot(self.bins, self.coord_total, 'r-', label="total", linewidth=1.0)
    plt.xlim((0,self.rcut))
    axl.set_ylim(bottom=0)
    axr.set_ylim(bottom=axl.get_ylim()[0], top=axl.get_ylim()[1]*10.0)
    if self.nspec > 1:
      ax2.minorticks_on()
      ax2.grid(b=True, which='major', axis='x', color='gray', linestyle='-')
      ax2.grid(b=True, which='minor', axis='x', color='gray', linestyle='--')
      ax2.grid(b=True, which='major', axis='y', color='gray', linestyle='-')
      # ax2.grid(b=True, which='minor', axis='y', color='gray', linestyle='--')
      for ispec in range(self.nspec):
        for jspec in range(ispec,self.nspec):
          pair = "{}-{}".format(self.species[ispec+1], self.species[jspec+1])
          ax2.plot(self.bins, self.gr[:,ispec,jspec], label=pair, linewidth=1.0)
      ax2.set_ylim(bottom=0)
      ax2.set_xlabel("r (A)")
      ax2.set_ylabel("partial g(r)")
      ax2.legend(loc="upper right")
    else:
      axl.set_xlabel("r (A)")
    fig3.savefig(filename, bbox_inches='tight')

  def dump_gr(self):
    """Dump the radial distributino function data"""
    filename = "rdf.dat"
    header_bit = "{0:>16s}"
    rdf_bit = "{0:>16.6f}"
    header_fmt = "{0:>16s}{1:>16s}{2:>16s}"
    rdf_fmt = "{0:>16.6f}{1:>16.6f}{2:>16.6f}"

    header = header_fmt.format("r (A)", "total", "coordination")
    if self.nspec > 1:
      for ispec in range(self.nspec):
        for jspec in range(ispec,self.nspec):
          pair = "{}-{}".format(self.species[ispec+1], self.species[jspec+1])
          header += header_bit.format(pair)
    header += "\n"

    with open(filename, 'w') as outfile:
      outfile.write(header)
      for i in range(self.nbins):
        rdf_line = rdf_fmt.format(self.bins[i], self.gr_total[i],
                                  self.coord_total[i])
        if self.nspec > 1:
          for ispec in range(self.nspec):
            for jspec in range(ispec,self.nspec):
              rdf_line += rdf_bit.format(self.gr[i,ispec,jspec])
        rdf_line += "\n"
        outfile.write(rdf_line)


  def get_bondlength(self, bondcut, frame, printall):
    """Compute bond lengths given bond cutoff"""

    bond_tot = sp.zeros((self.nspec, self.nspec), dtype=float)
    bondsq_tot = sp.zeros((self.nspec, self.nspec), dtype=float)
    bond_avg = sp.zeros((self.nspec, self.nspec), dtype=float)
    bond_sd = sp.zeros((self.nspec, self.nspec), dtype=float)
    bond_min = sp.zeros((self.nspec, self.nspec), dtype=float)
    nbonds = sp.zeros((self.nspec, self.nspec), dtype=int)

    bond_min = sp.copy(bondcut)
    for i in range(self.nat):
      for j in range(i+1, self.nat):
        s1 = frame.species[i]-1
        s2 = frame.species[j]-1
        if self.dt[i,j] < bondcut[s1,s2]:
          bond_tot[s1,s2] += self.dt[i,j]
          bondsq_tot[s1,s2] += self.dt[i,j]**2
          nbonds[s1,s2] += 1
          if s1 != s2:
            nbonds[s2,s1] += 1
          if self.dt[i,j] < bond_min[s1,s2]:
            bond_min[s1,s2] = self.dt[i,j]

          if printall:
            pair = "{}--{}".format(self.species[s1+1], self.species[s2+1])
            print(f'{pair}: {i:>4d}--{j:<4d} {self.dt[i,j]:>8.4f}')

    print("Mean bond lengths:")
    for i in range(self.nspec):
      for j in range(i,self.nspec):
        if nbonds[i,j] > 0:
          bond_avg[i,j] = bond_tot[i,j]/float(nbonds[i,j])
          bond_sd[i,j] = sp.sqrt(bondsq_tot[i,j]/nbonds[i,j] - bond_avg[i,j]**2)
          pair = "{}-{}".format(self.species[i+1], self.species[j+1])
          print(f'{pair}: {bond_avg[i,j]:>8.4f} +/- {bond_sd[i,j]:>8.4f} ({nbonds[i,j]} bonds)')

    print("Minimum bond lengths:")
    for i in range(self.nspec):
      for j in range(i,self.nspec):
        if nbonds[i,j] > 0:
          pair = "{}-{}".format(self.species[i+1], self.species[j+1])
          print(f'{pair}: {bond_min[i,j]:>8.4f}')

class MSER:
  """Marginal Standard Error Rule heuristic 
  --- K P White, Simulation 69, 323 (1997)"""

  def __init__(self, nframes, varname, var_traj):
    self.n_j = nframes-1
    self.propname = varname
    self.traj = var_traj
    self.mser = sp.zeros(self.n_j, dtype='float')
    # stop before the end otherwise the MSER becomes very noisy
    self.mser_cut = 200

  def get_point(self, d_j):
    prefac = 1.0/(self.n_j-d_j)**2
    ybar_ij = sp.mean(self.traj[d_j:])
    variance = 0.0
    for i in range(d_j+1,self.n_j):
      variance += (self.traj[i] - ybar_ij)**2
    return prefac*variance

  def get_mser(self):
    for i in range(self.n_j):
      self.mser[i] = self.get_point(i)

  def mser_min(self):
    return sp.argmin(self.mser[:-self.mser_cut])

  def plot_mser(self, steps):
    plt.figure("{} MSER".format(self.propname))
    plt.xlabel("step")
    plt.ylabel("MSER ({})".format(self.propname))
    plt.plot(steps[:-200], self.mser[:-200], 'k-')
    mser_min = self.mser_min()
    lab = "Minimum at step {}".format(mser_min)
    plt.axvline(x=mser_min, label=lab)
    plt.legend(loc="upper left")
    plt.savefig("mser.pdf", bbox_inches='tight')

  def dump_mser(self, steps):
    mser_fmt = "{0:>8d}{1:>16.6f}\n"
    filename = "mser.dat"
    with open(filename, 'w') as outfile:
      for i in range(self.n_j):
        outfile.write(mser_fmt.format(steps[i], self.mser[i]))

class VACF:
  """Velocity autocorrelation function"""

  def __init__(self, window, nat, dt, stride):
    self.window = window
    self.stride = stride
    self.dt = dt
    self.wsteps = int(window/dt/stride)
    self.nframes = 0
    self.nat = nat
    self.vacf = sp.zeros(self.wsteps)
    self.vacf_window = sp.zeros(self.wsteps)
    self.nwindows = sp.zeros(self.wsteps)
    self.time = sp.linspace(0.0, window, num=self.wsteps, endpoint=False)

  # def get_vacf(self):
  #   """Compute the full vacf"""
  #   for n in range(self.wsteps-1): # iterate over starting points
  #     for lag in range(n,wsteps): # iterate over values of lag

  def set_v_0(self, frame):
    """Set the reference velocity"""
    self.v_0 = frame.v

  def update_vacf(self, frame):
    """Update VACF from a given frame"""
    if self.nframes == 0:
      self.set_v_0(frame)
    for i in range(self.nat):
      self.vacf_window[self.nframes] += sp.dot(self.v_0[i,:], frame.v[i,:])
    self.nwindows[self.nframes] += 1
    self.nframes += 1
    if self.nframes >= self.wsteps:
      self.nframes = 0
      self.vacf += self.vacf_window
      self.vacf_window.fill(0.)

  def finalise(self):
    self.vacf += self.vacf_window

  def norm_vacf(self):
    """Normalise the VACF"""
    self.vacf = self.vacf/self.nwindows[-1]/self.nat

  def plot_vacf(self):
    filename = "vacf.pdf"
    plt.figure("VACF")
    plt.xlabel("t (fs)")
    plt.ylabel("C(t)")
    plt.xlim((self.time[0],self.time[-1]))
    plt.plot(self.time, self.vacf)
    plt.plot((0,self.time[-1]), (0, 0), 'k-')
    plt.savefig(filename, bbox_inches='tight')

  def dump_vacf(self):
    """Dump the VACF data"""
    filename = "vacf.dat"
    vacf_fmt = "{0:>12.4f}{1:>16.6f}\n"
    with open(filename, 'w') as outfile:
      for i in range(self.nframes):
        outfile.write(vacf_fmt.format(self.time[i], self.vacf[i]))

class MSD:
  """Mean Squared Deviation"""

  def __init__(self, window, nat, dt, stride):
    self.window = window
    self.stride = stride
    self.dt = dt
    self.wsteps = int(window/dt/stride)
    self.nframes = 0
    self.nat = nat
    self.r_prev = sp.zeros((self.nat,3))
    self.displ = sp.zeros((self.nat,3))
    self.msd = sp.zeros(self.wsteps)
    self.msd_window = sp.zeros(self.wsteps)
    self.nwindows = sp.zeros(self.wsteps)
    self.time = sp.linspace(0.0, window, num=self.wsteps, endpoint=False)

  def update_msd(self, frame):
    """Update the mean squared deviation from a given frame. The displacement
    is computed incrementally (per time step) since it is trickier to define
    the displacement in cases when the cell boundary is crossed and the volume
    is not fixed."""
    if self.nframes == 0:
      self.r_prev = frame.r
    for i in range(self.nat):
      diff = frame.diff_mic(frame.r[i,:], self.r_prev[i,:], lat=frame.lat)
      self.displ[i,:] += diff*cq_global.bohr2ang
      self.msd_window[self.nframes] += sp.sum(self.displ[i,:]**2)
    self.nwindows[self.nframes] += 1
    self.nframes += 1
    self.r_prev = frame.r
    if self.nframes >= self.wsteps:
      # reset the window
      self.nframes = 0
      self.msd += self.msd_window
      self.msd_window.fill(0.)
      self.displ.fill(0.)

  def finalise(self):
    self.msd += self.msd_window

  def norm_msd(self):
    """Normalise the MSD"""
    self.msd = self.msd/self.nwindows[-1]/self.nat

  def get_msd_diffusion(self, t_0):
    m, c = linearfit(self.msd[t_0:], self.time[t_0:])
    return m, c

  def plot_msd(self, fit_start):
    """Plot the MSD"""
    if fit_start > 0.0:
      t_0 = int(fit_start/self.dt)
      m, c  = linearfit(self.time[t_0:], self.msd[t_0:])
      d = m/6.0 * cq_global.ang2cm**2 / cq_global.fs2s
      linfit = linear(self.time, m, c)
    filename = "msd.pdf"
    plt.figure(r"MSD")
    plt.xlabel("t (fs)")
    plt.ylabel("MSD ($\AA ^2$)")
    plt.xlim((self.time[0],self.time[-1]))
    plt.plot(self.time, self.msd, label=None)
    if fit_start > 0.0:
      plt.plot(self.time, linfit, label=r"D = {0:<12.4e} $cm^2/s$".format(d))
    plt.ylim(ymin=0)
    plt.legend(loc='lower right')
    plt.savefig(filename, bbox_inches='tight')

  def dump_msd(self):
    """Dumpt the MSD data"""
    filename = "msd.dat"
    msd_fmt = "{0:>12.4f}{1:>16.6f}\n"
    with open(filename, 'w') as outfile:
      for i in range(self.nframes):
        outfile.write(msd_fmt.format(self.time[i], self.msd[i]))

class HeatFlux:
  """Heat flux autocorrelation"""

  def __init__(self, dt, window):
    self.dt = dt
    self.window = window
    self.ntimesteps = int(self.window // self.dt)
    self.nruns = 0
    self.J = []
    self.t = sp.linspace(0, self.window-self.dt, self.ntimesteps)
    self.G = sp.zeros((3,3,self.ntimesteps))
    self.S = sp.zeros(self.ntimesteps)
    # heat flux in W/m^2 = [E]/[t][d]^2
    # convert from ha/fs a^3
    self.hf_conv = cq_global.ha2j / cq_global.bohr2m**2
    # HFACF plotted in  W / m K ps
    self.hfacf_conv = self.hf_conv/1000.0

  def parse_heatflux_file(self, fname):
    """Reset J, and read the heat flux file"""
    self.J = []
    self.nsteps = 0
    with open(fname, 'r') as infile:
      for line in infile:
        step, Jx, Jy, Jz = line.split()
        step = int(step)
        Jx = float(Jx)*self.hf_conv
        Jy = float(Jy)*self.hf_conv
        Jz = float(Jz)*self.hf_conv
        self.J.append([Jx, Jy, Jz])

    self.nsteps = len(self.J)
    self.J = sp.array(self.J)

  def update_G(self, nskip):
    """Update the autocorrelation function with the new J"""
    nstart = nskip
    nstop = self.nsteps
    nwindows = int(float(nstop - nstart) // float(self.ntimesteps))
    for i in range(3):
      for j in range(3):
        for k in range(nwindows):
          self.nruns += 1
          start = nstart + k*self.ntimesteps
          finish = nstart + k*self.ntimesteps + self.ntimesteps
          self.G[i,j,:] += autocorr(self.J[start:finish,i], self.J[start:finish,j])

  def update_G_2(self, nskip):
    """Update the autocorrelation function with the new J"""
    nstart = nskip
    nstop = self.nsteps
    nwindows = int(float(nstop - nstart) // float(self.ntimesteps))

    for i in range(3):
      for j in range(3):
        for k in range(nwindows):
          for lag in range(self.ntimesteps): # iterate through values of tau
            end = self.ntimesteps - lag
            corr = 0.0
            for m in range(end):
              corr += self.J[m,i]*self.J[m+lag,j]
            self.G[i,j,lag] = corr/float(end)

          self.nruns += 1

  def norm_HFACF(self):
    self.G = self.G / float(self.nruns)
    # self.S = self.S / float(self.nruns)
    # convert units to W/mKps
    self.G = self.G * self.hfacf_conv
    # self.S = self.S * self.hfacf_conv

  def dump_HFACF(self):
    prefix = "hfacf"
    for i in range(3):
      for j in range(3):
        direction = str(i+1) + str(j+1)
        fname = prefix + direction + ".dat"
        with open(fname, 'w') as outfile:
          for k in range(self.ntimesteps):
            outfile.write(f'{self.t[k] :<8.2f}{self.G[i,j,k] :<16.8f}\n')


  def plot_HFACF(self):
    plt.figure("HFACF")
    plt.xlabel("t (fs)")
    plt.ylabel("HFACF")
    plt.xlim((0, self.t[-1]))
    # plt.plot(self.t[:], self.S[:], 'b-', label='S', linewidth=0.5)
    avg = sp.mean(self.G[1,1,:])
    plt.plot(self.t[:], self.G[0,0,:], 'r-', label='G_{xx}', linewidth=0.5)
    # plt.plot((self.t[0], self.t[-1]), (avg, avg), 'k-')
    # plt.plot(self.t[:], self.G[1,1,:], 'g-', label='G_{yy}', linewidth=0.5)
    # plt.plot(self.t[:], self.G[2,2,:], 'b-', label='G_{zz}', linewidth=0.5)
    # plt.plot(self.t[:], self.G[0,1,:], 'k-', label='G_{xy}', linewidth=0.5)
    # plt.plot(self.t[:], self.G[0,2,:], 'k-', label='G_{xz}', linewidth=0.5)
    # plt.plot(self.t[:], self.G[1,2,:], 'k-', label='G_{yz}', linewidth=0.5)
    plt.plot((self.t[0], self.t[-1]), (0.0, 0.0), 'k-')
    plt.legend(loc='upper right')
    plt.savefig('hfacf.pdf', bbox_inches='tight')

  def get_kappa(self, volume, temp):
    volume = volume*(cq_global.bohr2m**3)
    kappa = sp.zeros((3,3))
    for i in range(3):
      for j in range(3):
        kappa[i,j] = trapz(self.G[i,j,:], dx=self.dt)
    kappa = kappa*(volume/cq_global.k_B/temp**2)
    return kappa
