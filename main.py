'''
perform calculation of glass transition temperature through MD simulation in LAMMPS

Usage:

python main.py /path/to/lmp.data [optional: /path/to/workdir]

RESTRICTIONS:
1) mpirun available
2) LAMMPS build in parallel
3) GAFF force field


PACKAGES:
numpy, pandas, scipy, pwlf, periodictable, psutil

'''

import sys, os, re, shutil, tempfile
import subprocess
import multiprocessing as mp
import numpy as np
import pandas as pd
from scipy import optimize
import pwlf
import matplotlib.pyplot as plt
from io import StringIO
import psutil
import periodictable

LAMMPS_EXE = 'lmp'
THREADS_PER_TASK = 32

_elements = [elem for elem in periodictable.elements]

_in_templ = '''
log             %(name)s.log

units           real
dimension       3
processors      * * *
boundary        p p p
atom_style      full
pair_style      lj/cut/coul/long 10
bond_style      harmonic
angle_style     harmonic
dihedral_style  fourier
improper_style  cvff 
special_bonds   amber  
pair_modify     tail yes mix arithmetic
kspace_style    pppm 1.0e-4
timestep        %(dt)s

restart         %(nrst)s ./%(name)s.*.restart
read_data       %(datafile)s
fix             1 all npt temp %(tstart)s %(tend)s 100 iso 1 1 1000
dump            wr all custom %(ndump)s %(name)s.lammpstrj id element x y z
dump_modify     wr element %(elemstr)s
thermo          100
thermo_style    custom step time etotal pe press temp vol density evdwl ecoul elong ebond eangle edihed eimp etail ecouple fmax fnorm

velocity        all create %(tstart)s 539487526

run             %(nstep)s
write_data      %(name)s.data
'''

def _get_field(fobj, field):
    fobj.seek(0)
    content = []
    # size = set()
    lines = iter(fobj.readlines())
    for line in lines:
        line = line.partition('#')[0]
        if field in line:
            _ = next(lines)
            while True:
                line = next(lines)
                line = line.partition('#')[0]
                words = line.split()
                if len(words)==0:
                    return content
                    # continue
                # size.add(len(words))
                # if len(size)>1:
                    # return content
                content.append(words)
    print(f'Warning: no field "{field}" in data file')

def elem_by_mass(mass: float):
    '''
    Guess chemical element from given atomic mass in Da. If difference is greater than 1 Da, 
    warning will be printed
    '''
    e = min(_elements, key = lambda x: abs(x.mass - mass))
    if abs(e.mass - mass) >= 1:
        print(f'Warning: mass difference for guessed element {e} was {abs(e.mass - mass)} Da')
    return e

def get_elements_str(datafile):
    '''
    convert `Masses` section in LAMMPS data file to string containg elements to 
    use in `dump_modify` command in input file
    
    '''
    elements = []
    with open(datafile,'r') as dat:
        masses = _get_field(dat,'Masses')
    if masses==None:
        print('unable to guess elements in data file, exit')
        sys.exit()
    for i in masses:
        m = float(i[1])
        el = str(elem_by_mass(m))
        elements.append(el)
    return ' '.join(elements)

def free_cpus():
    '''
    Returns number of logical CPUs (threads) which are available at the moment
    '''
    return int(psutil.cpu_count() * (1 - psutil.cpu_percent()*0.01))

def check_prereq():
    '''
    Checks for presence of mpirun and LAMMPS executables, raises  NotImplementedError 
    if anything is missing
    
    '''
    global LAMMPS_EXE
    lmp = next((loc for loc in (LAMMPS_EXE, 'lmp_mpi', 'lmp') if shutil.which(loc) !=None), None)
    mpirun = shutil.which('mpirun')
    possible_lmp = []
    if mpirun is None:
        raise NotImplementedError('mpirun not found in $PATH')
    else:
        print(f'mpirun located in $PATH: {mpirun}')
    if lmp is None:
        raise NotImplementedError('LAMMPS executable not found in $PATH')
    else:
        LAMMPS_EXE = lmp
        print(f'LAMMPS located in $PATH: {LAMMPS_EXE}')

def prepare_input(template : str, args : dict):
    '''
    Concatenates a template string with a dictionary and returns the formatted string. 
    If any arguments are missed, prints an error message and closes the interpreter
    '''
    try:
        istring = template % args
    except KeyError as ke:
        print(f'***ERROR***\n\ncould not write LAMMPS input file: {ke} argument missing')
        sys.exit(1)
    return istring

def run_lammps(wd : str, istr : str, files : list[str] = [], np = THREADS_PER_TASK):
    '''
    Description
    ---
    Runs a LAMMPS simulation in the specified directory, waits for it to complete 
    and returns tuple with the exit code, stdout, and stderr
    
    Arguments
    ---
    
    `infile` [str] : lines that will be written to the LAMMPS input file;
    
    `wd` [str] : path to the directory where LAMMPS will be started;
    
    `files` [list[str]] : list with paths to additional files required for input (data, dump etc.);
    
    `np` [int] : number of MPI threads LAMMPS should use (will be passed to mpirun)
    
    Restrictions
    ---
    It is expected that `mpirun` is installed and available;
    
    LAMMPS should be compiled in parallel mode (see https://docs.lammps.org/Build_basics.html)
    
    '''
    iname = os.path.join(wd, 'input.lammps')
    with open(iname, 'w') as fi:
        fi.write(istr)
    for f in files:
        shutil.copy(f, os.path.join(wd, os.path.basename(f)))
    cmd = ['mpirun', '-np', str(np), LAMMPS_EXE, '-i', iname]
    proc = subprocess.Popen(cmd, cwd=wd, shell=False, text=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    out, err = proc.communicate()
    return proc.returncode, out, err

def read_log(log : str):
    '''
    Description
    ---
    Read LAMMPS log file, returns pandas.core.frame.DataFrame with MD data
    
    Arguments
    ---
    `log` [str] : absolute path to log file OR file content as string
    
    Restricitons
    ---
    Log file should contain exactly 1 table
    
    
    '''
    if os.path.exists(log):
        with open(log, 'r') as fi:
            data = fi.read()
    else:
        data = log
    pat = re.compile(r'Per MPI rank .*\n([\S|\s]+)\nLoop time')
    res = re.search(pat, data)
    if res:
        table = res.group(1)
    else:
        raise ValueError('Could not parse table from file!')
    return pd.read_table(StringIO(table), sep = '\\s+')

def _reduce_data(xdata, ydata, xlo, xhi):
    if xlo > xhi:
        print(f'ERROR (reduce_data) no data between {xlo}-{xhi}')
    reducedx = []; reducedy = [];  
    for x, y in zip(xdata, ydata):
        if x >= xlo and x <= xhi:
            reducedx.append(x); reducedy.append(y);
    return reducedx, reducedy

def _fit_hyperbola(x, y, xlo, xhi, minimum_convergence=None, initial_guess=False, maxiter=10**4):
    '''
    Adapted from Lunar code: https://github.com/CMMRLab/LUNAR
    
    Fits 2D data to hyperbolic curve according to method proposed in paper: https://doi.org/10.1016/j.polymer.2016.01.074
    '''
      
    # define default outputs (in-case something goes wrong)
    xout = list(x); yout = len(y)*[0];
    params = [0, 0, 0, 0, 0]; center = [0, 0];
    slopes = [0, 0]; transition = [0, 0];
    
    # Convert to float64 and then lists to numpy arrays
    xx = np.array(x); yy = np.array(y);
    
    # Setup intial guess
    p0 = None
    if initial_guess:
        slopeguess = (yy[-1]-yy[0])/(xx[-1]-xx[0])
        p0 = (np.mean(xx),np.mean(yy), slopeguess, slopeguess, np.log((xx[-1]-xx[0])**2/100))

    # Define the hyperbola equation (eqn 1 in paper)
    def hyberbola(t, t0, v0, a, b, c):
        h0 = 0.5*(t-t0) + np.sqrt( ((t-t0)*(t-t0))/4 + np.exp(c) )
        v = v0 + a*(t-t0) + b*h0
        return v
    
    # Find best fit    t0          v0        a        b        c
    parm_bounds = ((np.min(xx), np.min(yy), -np.inf, -np.inf, -np.inf), # lower
                   (np.max(xx), np.max(yy),  np.inf,  np.inf,  np.inf)) # upper
    param, param_cov = optimize.curve_fit(hyberbola, xx, yy, p0=p0, method='trf', bounds=parm_bounds, maxfev=maxiter)
    t0, v0, a, b, c = param # extract out fitting coeffs
    
    # update defaults
    params = list(param)
    yout = list(hyberbola(xout, *param))
    center = [t0, v0]
    slopes = [a, (a+b)]
    
    # Find where function is transitioning (eqn 5 in paper)
    if minimum_convergence is not None:
        dtemp = (np.exp(c/2)*(2*minimum_convergence-1))/(np.sqrt(minimum_convergence*(1-minimum_convergence)))
        transition = [t0-dtemp, t0+dtemp]
    return xout, yout, params, center, slopes, transition

def fit_hyperbola(x, y, xlo, xhi, minimum_convergence=None, initial_guess=False, maxiter=10**4):
    # xlo = min(x); xhi = max(x) # use all available data
    reduced_x, reduced_y = _reduce_data(x, y, xlo, xhi)
    if reduced_x and reduced_y:        
        xout, yout, params, center, slopes, transition = _fit_hyperbola(x, y, xlo, xhi, minimum_convergence, initial_guess, maxiter)
    else:
        xout = [0, 1]; yout = [0, 1]; slopes = [0, 1]
        center = [0, 0]; params = [0, 0, 0, 0, 0];
        transition = [];
        raise ValueError(f'ERROR no (hyperbola) LAMMPS data in xrange {xlo} - {xhi}')
    return xout, yout, params, center, slopes, transition

def _piecewise_regression(x, y, xlo, xhi, n):
    '''
    Adapted from Lunar code: https://github.com/CMMRLab/LUNAR
    
    Segmented bilinear regression
    '''
    npx = np.array(x); npy = np.array(y);
    
    # Perform peicewise regression
    my_pwlf = pwlf.PiecewiseLinFit(npx, npy)
    xbreaks = my_pwlf.fit(n+1)
    ybreaks = my_pwlf.predict(np.array(xbreaks))
    xout = np.linspace(npx.min(), npx.max(), 100)
    yout = my_pwlf.predict(xout)
    
    # Compute the slopes
    def slope(x1, y1, x2, y2):
        m = (y2-y1)/(x2-x1)
        return m
    
    npoints = min([len(xbreaks), len(ybreaks)])-1
    slopes = {(i, i+1):slope(xbreaks[i], ybreaks[i], xbreaks[i+1], ybreaks[i+1]) for i in range(npoints)}
    return xout, yout, xbreaks, ybreaks, slopes

def piecewise_regression(x, y, xlo, xhi, n):
    # xlo = min(x); xhi = max(x) # use all available data
    reduced_x, reduced_y = _reduce_data(x, y, xlo, xhi)
    if reduced_x and reduced_y:
        xout, yout, xbreaks, ybreaks, slopes = _piecewise_regression(reduced_x, reduced_y, xlo, xhi, n)
    else:
        xout = [0, 1]; yout = [0, 1]; slopes = {(0,1):1}
        xbreaks = [0, 1]; ybreaks = [0, 1]
        raise ValueError(f'ERROR no (peicewise-regression) LAMMPS data in xrange {xlo} - {xhi}')
    return xout, yout, xbreaks, ybreaks, slopes

def compute_Tg(df : pd.DataFrame, xval : str = 'Temperature', yval : str = 'Volume', method : str = 'segmented', n = 4, title = None, s : float = 0.1, save = True, show = False):
    '''
    Description
    ---
    Computes temperature of glass transition (Tg) and coefficients of thermal expansion (CTE) below and under Tg, using 
    Pandas DataFrame as source of MD data. Returns tuple containg floating numbers (Tg, cte_below, cte_under)
    
    Arguments
    ---
    `df` [pandas.core.frame.DataFrame] : DataFrame containg MD data;
    `xval` [str] : name of column with temperature data;
    `yval` [str] : name of column with volumetric data;
    `method` [str] = "segmented" : which regression model to use ("hyperbolic", "segmented", "disconnected");
    `n` [int] = 4 : integer number indicating which fraction of data to use in disconnected model 
    (1/n will be used to fit lines);
    `title` [str] = None : title of plot;
    `s` [float] = 0.1 : size of data points on plot;
    `save` [bool] = True : save png image;
    `show` [bool] = False : show plot in interpreter
    '''
    _methods = ['hyperbolic', 'segmented', 'disconnected']
    col_names = df.columns.to_list()
    xlab = next((s for s in col_names if xval in s), None)
    ylab = next((s for s in col_names if yval in s), None)
    assert xlab != None and ylab !=None, f'Properties "{xval}", "{yval}" not found in dataframe: {col_names}'
    x = df[xlab].to_numpy()
    y = df[ylab].to_numpy()
    xstart = x[: x.size//1000].mean()
    xend = x[-x.size//1000 :].mean()
    ystart = y[: y.size//1000].mean()
    yend = y[-y.size//1000 :].mean()
    cooling = False
    if xstart > xend:
        xstart = x[-x.size//1000 :].mean()
        xend = x[: x.size//1000].mean()
        cooling = True
        ystart = y[-y.size//1000 :].mean()
        yend = y[: y.size//1000].mean()
    plt.scatter(x, y, s=s, marker='o', color='gray', label = 'Raw data')
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    if method == 'hyperbolic':
        try:
            # xout, yout, params, center, slopes, transition
            xh, yh, _parms, center, sl, _tr = fit_hyperbola(x, y, xstart, xend)
            # plt.axvline(x=center[0], color='r', linestyle='--', label=f'xfit = {center[0]:.2f}')
            plt.plot(xh, yh, color = 'blue', label = 'Hyperbolic fit')
            plt.plot([min(xh), center[0], max(xh)], [min(yh), center[1], max(yh)], color = 'blue', linestyle='--')
            plt.plot(*center, 'bo', label = f'Center of h-fit: ({center[0]:.1f}, {center[1]:.1f})')
            Tg = center[0]
            if cooling: # cooling, tstart > Tg > tend
                cte_under = (center[1] - yend)/(center[0] - xend)/yend
                cte_below = (-center[1] + ystart)/(-center[0] + xstart)/center[1]
            else: # heating, tstart < Tg < tend
                cte_below = (center[1] - ystart)/(center[0] - xstart)/ystart
                cte_under = (-center[1] + yend)/(-center[0] + xend)/center[1]
        except:
            raise ValueError('Hyperbolic fit failed')
    elif method == 'segmented':
        try:
            xh, yh, xbreaks, ybreaks, sl = piecewise_regression(x, y, xstart, xend, n=1)
            plt.plot(xh, yh, color = 'red', label = 'Bilinear fit')
            plt.plot(xbreaks[1], ybreaks[1], 'ro', label = f'Center of bl-fit: ({xbreaks[1]:.1f}, {ybreaks[1]:.1f})')
            Tg = xbreaks[1]
            if cooling: # cooling, tstart > Tg > tend
                cte_under = (ybreaks[1] - yend)/(xbreaks[1] - xend)/yend
                cte_below = (-ybreaks[1] + ystart)/(-xbreaks[1] + xstart)/ybreaks[1]
            else: # heating, tstart < Tg < tend
                cte_below = (ybreaks[1] - ystart)/(xbreaks[1] - xstart)/ystart
                cte_under = (-ybreaks[1] + yend)/(-xbreaks[1] + xend)/ybreaks[1]
        except:
            raise ValueError('Segmented bilinear regression failed')
    elif method == 'disconnected':
        try:
            xstart = x[: x.size//1000].mean()
            xend = x[-x.size//1000 :].mean()
            x0 = x[: x.size//n] # first 1/n
            x1 = x[-x.size//n :] # last (n-1)/n
            y0 = y[: y.size//n] # first 1/n
            y1 = y[-y.size//n :] # last (n-1)/n
            k0, b0 = np.polyfit(x0,y0, 1)
            k1, b1 = np.polyfit(x1,y1, 1)
            ix = -(b0 - b1) / (k0 - k1) # x intersection point
            iy = ix * k0 + b0 # y intersection point
            xfit = np.array([xstart, ix, xend]) # fitted x points
            yfit = np.array([xstart*k0+b0, iy, xend*k1+b1]) # fitted y points
            plt.plot(xfit, yfit, color = 'green', label = f'Linear fit, n = {n}')
            plt.plot(ix,iy, 'go', label = f'Center of l-fit: ({ix:.1f}, {iy:.1f})')
            Tg = ix
            if cooling: # cooling, tstart > Tg > tend
                cte_under = (iy - yend)/(ix[1] - xend)/yend
                cte_below = (-iy + ystart)/(-ix[1] + xstart)/iy[1]
            else: # heating, tstart < Tg < tend
                cte_below = (iy - ystart)/(ix[1] - xstart)/ystart
                cte_under = (-iy + yend)/(-ix[1] + xend)/iy[1]
        except:
            raise ValueError('Disconnected bilinear regression failed')
    else:
        raise ValueError(f'Unknown method requested: "{method}", available are: ' + ', '.join(_methods))
    plt.legend()
    if title != None:
        plt.title(title)
    if save:
        plt.savefig(f'{yval}_{xval}.png')
    if show:
        plt.show()
    return Tg, cte_below, cte_under

def volume_expansion(thermo_df, temp=None, press=1.0, mass=None, init=0, last=None):
    """
    Adapted from RadonPy code:
    https://github.com/RadonPy/RadonPy

    Calculate (isobaric volumetric) thermal expansion coefficient from thermodynamic data in a log file
    alpha_P = Cov(V, H) / (V*kB*T**2)

    Args:
        thermo_df: Pandas Data Frame of thermodynamic data

    Optional args:
        temp: Temperature (If None, temperature in thermodynamic data is used) (float, K)
        press: Pressure (float, atm)
        init: Initial step (int)
        last: Last step (int)

    Return:
        volume expansion (float, K**-1)
    """
    # Conversion factors
    atm2pa = 101325
    cal2j = 4.184

    # Physical constants
    kB = 1.3806504e-23 # J/K
    NA = 6.02214076e+23 # mol^-1
    
    if 'Volume' in thermo_df.columns:
        V = thermo_df['Volume'].to_numpy() * 1e-30 # Angstrom**3 -> m**3
    else:
        V = thermo_df['Lx'].to_numpy() * thermo_df['Ly'].to_numpy() * thermo_df['Lz'].to_numpy() * 1e-30 # Angstrom**3 -> m**3

    T = thermo_df['Temp'].to_numpy() # K

    U = thermo_df['TotEng'].to_numpy() * cal2j * 1000 / NA # kcal/mol -> J
    P = press * atm2pa # Pa = J / m**3
    H = U + P * V

    mV = V[init:last].mean()
    mT = T[init:last].mean() if temp is None else temp
    VH_cov = np.sum((V[init:last] - mV)*(H[init:last] - H[init:last].mean())) / len(V[init:last])

    alpha_P = VH_cov / (mV * kB * mT**2) # m**3 * J / (m**3 * J/K * K**2) = 1/K

    return alpha_P

def main(wd : str, datafile : str, template, settings : dict):
    '''
    Description
    ---
    Main routine to run LAMMPS simulation and compute glass transition temperature
    
    Arguments
    ---
    `wd` [str] : working directory in which LAMMPS job will be started;
    
    `datafile` [str] : absolute path to LAMMPS data file (.data, .lmps);
    
    `template` [str] : string with named fields which represents LAMMPS input file
    
    `settings` [dict] : dictionary with template field names as keys
    
    Restrictions
    ---
    `wd` directory should exist;
    
    system must be parameterized in the GAFF2 force field
    '''
    check_prereq()
    wd = os.path.abspath(wd)
    df_copy = os.path.join(wd, os.path.basename(datafile))
    shutil.copy(datafile, df_copy)
    settings[datafile] = df_copy
    istr = prepare_input(template, settings)
    avail = free_cpus()
    if THREADS_PER_TASK > avail:
        print(f'***ERROR***\n\nNumber of available threads is {avail} but required number is {THREADS_PER_TASK}. Reduce the number of requested threads or wait for more resources to become available')
        sys.exit(1)
    rc, out, err = run_lammps(wd, istr, np=THREADS_PER_TASK)
    if rc != 0:
        print(f'***ERROR***\n\nLAMMPS terminated with following error message:\n\n{err}\n')
        sys.exit(1)
    df = read_log(out)
    try:
        Tg, cte1, cte2 = compute_Tg(df, 'Temp', 'Volume', method = 'segmented', show=False, save = True)
    except:
        print(f'***ERROR***\n\nFailed to build regression model. This may be caused by too rapid \
            annealing, poorly relaxed initial structure or incompatible force field parameters \
                in data file. Try to relax your system before running simulation or decrease rate \
                    of temperature change')
        sys.exit(1)
    return Tg, cte1, cte2

if __name__ == '__main__':
    tstart = 100 # 100 K/ns
    tend = 1100
    datafile = sys.argv[1]
    try:
        wd = sys.argv[2]
        os.makedirs(wd, exist_ok=True)
    except IndexError:
        wd = None
    md_params = {
        'name' : os.path.basename(datafile).partition('.')[0],
        'dt' : 1, # 1 fs
        'nrst' : 1000000, # 1 ns
        'ndump' : 10000, # 10 ps
        'nstep' : 10000000, # 10 ns
        'datafile' : datafile,
        'tstart' : tstart,
        'tend' : tend, 
        'elemstr' : get_elements_str(datafile),
        'name' : 'test'
        }
    if wd == None:
        with tempfile.TemporaryDirectory() as wd:
            Tg, cte1, cte2 = main(
                wd=wd,
                datafile=datafile,
                settings=md_params,
                template=_in_templ
            )
    else:
        Tg, cte1, cte2 = main(
            wd=wd,
            datafile=datafile,
            settings=md_params,
            template=_in_templ
        )
    print(f'Polymer: {os.path.basename(datafile).partition(".")[0]}')
    print(f'Temperature range: {tstart} - {tend} K')
    print(f'Estimated temperature of glass transition (Tg) = {Tg:.1f} K')
    print(f'Coefficient of thermal expansion below Tg = {cte1:.2e} K^-1')
    print(f'Coefficient of thermal expansion under Tg = {cte2:.2e} K^-1')
    print('\nNormal termination')
        
        
    
    