import os, re, time
import multiprocessing as mp
import subprocess, csv
from io import StringIO
import psutil

GMX_EXE = '/home/md/gromacs-2025.1/build/bin/gmx_mpi'
KNOWN_TERMS = ['Bond', 'Angle', 'Fourier-Dih.', 'Ryckaert-Bell.', 'LJ-14', 'Coulomb-14', 'LJ-(SR)', 'Disper.-corr.', 'Coulomb-(SR)', 'Coul.-recip.', 'Potential', 'Kinetic-En.', 'Total-Energy', 'Conserved-En.', 'Temperature', 'Pres.-DC', 'Pressure', 'Box-X', 'Box-Y', 'Box-Z', 'Volume', 'Density', 'pV', 'Enthalpy', 'Vir-XX', 'Vir-XY', 'Vir-XZ', 'Vir-YX', 'Vir-YY', 'Vir-YZ', 'Vir-ZX', 'Vir-ZY', 'Vir-ZZ', 'Pres-XX', 'Pres-XY', 'Pres-XZ', 'Pres-YX', 'Pres-YY', 'Pres-YZ', 'Pres-ZX', 'Pres-ZY', 'Pres-ZZ', '#Surf*SurfTen', 'Box-Vel-XX', 'Box-Vel-YY', 'Box-Vel-ZZ', 'T-System']

def new_name(filename : str, ext : str):
    '''
    creates filename by template with different extension.
    
    Example:
    
    data = '/path/to/benzene.lmps'
    new_name(data, 'gro')
    > '/path/to/benzene.gro'
    '''
    if ext.startswith('.'):
        ext = ext[1:]
    old = os.path.basename(filename)
    new = '.'.join([old.split('.')[0], ext])
    return filename.replace(old, new)

def free_cpus():
    '''
    `None` --> `int`
    
    Returns number of logical CPUs (threads) which are available at the moment.
    '''
    return int(psutil.cpu_count() * (1 - psutil.cpu_percent()*0.01))

def _get_gpu_usage():
    '''
    Returns a dictionary of the form {n : int : m : float} where `n` is numeric GPU ID and `m` is 
    its current utilization in %
    '''
    try:
        out = {}
        process = subprocess.Popen(['nvidia-smi', '--query-gpu=index,utilization.gpu', '--format=csv,noheader,nounits'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        gpu_usage = stdout.decode('utf-8').strip()
        reader = csv.reader(StringIO(gpu_usage))
        for row in reader:
            gpu_id, usage = row
            out[int(gpu_id)] = float(usage)
        return out
    except FileNotFoundError:
        raise ValueError("nvidia-smi not found. Is NVIDIA driver installed?")
    except Exception as e:
        raise e

def free_gpu(max_usage = 75):
    '''
    Selects the least loaded GPU and returns its numeric ID [`int`] if its utilization 
    is below `max_usage` (75% by default). Otherwise returns `None`
    '''
    current_gpu_load = _get_gpu_usage()
    gpu_id : int = min(current_gpu_load, key = current_gpu_load.get)
    if current_gpu_load[gpu_id] <= max_usage:
        return  gpu_id

def monitor_gpu_load(log = 'gpu_usage.csv', timeout = 600, gpu=30, screen = False):
    '''
    Monitor GPU load over time. Outputs information to .csv format (or screen) if `screen` = True.
    '''
    t0 = time.time()
    gpu_ids = sorted(_get_gpu_usage())
    with open(log, 'w') as fo:
        fo.write(','.join(['time (s)'] + [str(i) for i in gpu_ids] + ['free']) + '\n')
        if screen:
            print('    '.join(['time (s)'] + [str(i) for i in gpu_ids] + ['free']))
        while time.time() - t0 < timeout:
            # fo.write('\n')
            t = f'{(time.time() - t0):.1f}'
            fg = str(free_gpu(max_usage=gpu))
            cu = _get_gpu_usage()
            usage = [f'{cu[i]:.1f}' for i in gpu_ids]
            fo.write(','.join([t, *usage, fg]) + '\n')
            if screen:
                print('    '.join([t, *usage, fg]))
            time.sleep(1)





def _gmx_proc(cmd : list, wd = None, message = None):
    '''
    Calls gromacs subprogram and, optionally, make communication
    '''
    if wd == None:
        wd = os.getcwd()
    proc = subprocess.Popen(cmd, text=True, stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.PIPE, cwd=wd)
    if message:
        stdout, stderr = proc.communicate(message)
        returncode = proc.returncode
    else:
        stdout, stderr = proc.communicate()
        returncode = proc.returncode
    return (returncode, stdout, stderr)

def grompp(wd, mdp, gro, top, tpr=None):
    '''
    Run `gmx grompp` with given mdp, coordinate and topology files. If exit code is 0, 
    returns path to tpr file, othervise returns None.
    '''
    if not os.path.isdir(wd):
        try:
            os.makedirs(wd)
        except:
            raise ValueError(f'Could not create dir: {wd}')
    if tpr == None:
        tpr = new_name(top, 'tpr')
    rc , stdout, stderr = _gmx_proc([GMX_EXE, 'grompp', '-f', mdp, '-c', gro, '-p', top, '-o', os.path.join(wd, tpr)], wd = wd)
    if rc == 0:
        return os.path.join(wd, tpr)
    print('WARNING: gmx grompp failed: ', stderr)

def mdrun(wd : str, gpu_id, np, bk = False):
    '''
    Run `gmx mdrun` command in given directory in which .tpr file is expected. 
    `gpu_id` corresponds to numeric index of GPU device displayed by `nvidia-smi` and 
    `np` corresponds to number of OMP threads (-ntomp flag). If `bk` is True, program will be 
    started in background (`bk` script is required and should be added to PATH).
    '''
    tpr = next((f for f in os.listdir(wd) if f.endswith('tpr')), None)
    if tpr  == None:
        raise ValueError(f'tpr file not present in directory: {wd}')
    cpt = new_name(tpr, 'cpt')
    if bk:
        rc = subprocess.Popen(['bk', GMX_EXE, 'mdrun', '-ntomp', str(np), '--gpu_id', str(gpu_id), '-s', os.path.join(wd,tpr), '-cpi', os.path.join(wd,cpt)], cwd=wd)
        # print(rc.communicate())
    else:
        rc , stdout, stderr = _gmx_proc([GMX_EXE, 'mdrun', '-ntomp', str(np), '--gpu_id', str(gpu_id), '-s', os.path.join(wd,tpr), '-cpi', os.path.join(wd,cpt)], wd = wd)
        if rc != 0:
            print(f'WARNING: mdrun failed in directory: {wd}')

def energy(wd : str, sim_property : str):
    '''
    Run "gmx energy" and save xvg file with required property
    
    For example, `energy('/path/to/gmx/calc', 'density')` is equal to: `cd /path/to/gmx/calc; gmx_mpi energy -f ener.edr -o density.xvg`:
    
    Raises ValueError if property could not be calculated ('Density' or 'Pressure' for NVT simulation)
    
    Returns absolute path to xvg file
    '''
    sim_property = sim_property.strip()
    edr = next((f for f in os.listdir(wd) if f.endswith('.edr')), None)
    if edr == None:
        raise ValueError(f'EDR file not present in dir: {wd}')
    edr = os.path.join(wd, edr)
    xvg = os.path.join(wd,f"{re.sub('[!@#$()-.*]', '', sim_property)}.xvg")
    if os.path.exists(xvg):
        os.remove(xvg)
    assert sim_property in KNOWN_TERMS, f'available properties are: ' + ', '.join([p for p in KNOWN_TERMS])
    assert os.path.exists(edr), f'no ener.edr in folder: {wd}'
    returncode, stdout, stderr = _gmx_proc([GMX_EXE, 'energy', '-f', edr, '-o', xvg], wd=wd, message=f'{sim_property}\n\n')
    if returncode !=0:
        raise ValueError(f"`gmx energy` terminated with error: {stderr}")
    return xvg

def free_volume(wd : str):
    '''
    Gromacs command example: 
    `gmx_mpi freevolume -f traj_comp.xtc -s polyvinylalcohol.tpr  -o fv.xvg`
    '''
    files = os.listdir(wd)
    xtc = next((f for f in files if f.endswith('.xtc')), None)
    if xtc == None:
        raise ValueError(f"XTC file not present in directory: {wd}")
    tpr = next((f for f in files if f.endswith('.tpr')), None)
    if tpr == None:
        raise ValueError(f"TPR file not present in directory: {wd}")
    xtc = os.path.join(wd, xtc)
    tpr = os.path.join(wd, tpr)
    xvg = os.path.join(wd, 'freevolume.xvg')
    if os.path.exists(xvg):
        os.remove(xvg)
    returncode, stdout, stderr = _gmx_proc(
        [GMX_EXE, 'freevolume', '-f', xtc, '-s', tpr, '-o', xvg], wd=wd)
    if returncode != 0:
        raise ValueError(f'Error in free volume computation: {stderr}')
    return xvg

def trjconv(wd : str, f : str, s : str, o : str, **kwargs):
    '''https://manual.gromacs.org/current/onlinehelp/gmx-trjconv.html'''
    # gmx_mpi trjconv -f traj_comp.xtc -s polymer.tpr -dt 1000 -sep -o traj.gro
    assert os.path.basename(f) != os.path.basename(o), 'output file should have different name'
    try:
        base, ext = os.path.basename(o).split('.')
    except ValueError:
        raise ValueError('ouput name should contain exactly 1 dot')
    files0 = set([f for f in os.listdir(wd) if f.endswith(ext)]) - \
        set([f for f in os.listdir(wd) if base in f])
    dt = kwargs.get('dt', None)
    sep = kwargs.get('sep', False)
    pbc = kwargs.get('pbc', None)
    cmd = [GMX_EXE, 'trjconv', '-f', f, '-s', s, '-o', o]
    if sep == True:
        cmd+=['-sep']
    if pbc !=None:
        assert pbc in ['mol', 'res', 'atom', 'nojump', 'cluster', 'whole'], 'unknown parm for pbc'
        cmd += ['-pbc', pbc]
    if dt !=None:
        dt = int(dt)
        cmd += ['-dt', str(dt)]
    returncode, stdout, stderr = _gmx_proc(cmd, wd, message='0\n\n')
    if returncode == 0:
        files1 = set([f for f in os.listdir(wd) if f.endswith(ext)])
        output = files1 - files0
        if sep:
            if len(output) == 1:
                return output.pop()
            elif len(output) > 1:
                return list(output)
            else:
                raise ValueError(f'Something went wrong in `trjconv` subroutine: expected new .{ext} file(s) in {wd}')
        else:
            return os.path.basename(o)
    else:
        print('WARNING: error in `trjconv` subroutine: ', stderr)
        return


