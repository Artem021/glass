import os, subprocess, shutil
import time
from tools.mdcalc import grompp, mdrun, free_cpus, free_gpu


# copy files on 22

def copy_wd(wd : str, files : list[str]):
    '''
    make a reduced copy of working dir with required files to copy via `scp`.
    
    `files` may contain relative path to file: npt/confout.gro
    '''
    tmpdir = os.path.join(wd, 'tmp')
    if os.path.exists(tmpdir):
        shutil.rmtree(tmpdir, ignore_errors = True)
    # files = ['npt/confout.gro', 'npt/polymer.tpr', 'input.top']
    for d in os.listdir(wd):
        for f in files:
            src = os.path.join(wd, d, f)
            dest = os.path.join(tmpdir, d, os.path.dirname(f))
            if not os.path.exists(dest):
                os.makedirs(dest)
            if os.path.exists(src):
                shutil.copy(src, dest)
            else:
                print(f'{d} was skipped, file {src} not found')
                shutil.rmtree(dest, ignore_errors=True)
    return tmpdir










# STEP X: generate 10ns trajectories to make 10 samples
print('='*50)
print('Generation of samples')
print('='*50 + '\n\n')
generation_dir = '/home/md/md/Tg/all_data/c_rescale'
calc_root = '/home/md/md/Tg/all_data/c_rescale/samples'
exclude = []
mdp_dir = '/home/md/md/Tg/gromacs_test/mdp/c-rescale'

try: 
    new_calc_dirs = []
    for d in os.listdir(generation_dir):
        if d not in exclude:
            pdb = os.path.join(generation_dir, d, 'npt', 'confout.gro')
            top = os.path.join(generation_dir, d, 'npt', 'polymer.tpr')
            calcdir = os.path.join(calc_root, d)
            if os.path.exists(pdb):
                os.makedirs(calcdir)
                shutil.copy(pdb, calcdir)
                shutil.copy(top, calcdir)
                new_calc_dirs.append(calcdir)
            else:
                print(f'WARNING: pdb file not found in {d}, skipping it')
    print(f'{len(new_calc_dirs)} candidates were found')
    if len(new_calc_dirs) == 0:
        print('Nothing to do, exit')
        exit()
except Exception as e:
    print('Failed to copy initial structures from `generation folder`: ', e)
    exit()

# STEP X+1: run NPT relaxation at 500 K for 10 ns
try:
    npt_tasks = []
    for d in new_calc_dirs:
        npt = os.path.join(d, 'npt')
        os.makedirs(npt)
        confout = os.path.join(d, 'confout.gro')
        top = os.path.join(d, 'polymer.tpr')
        if not os.path.exists(confout):
            print(f'WARNING: grompp could not be started because NVT job failed: {os.path.join(d, "nvt")}, skip')
            continue
        res = grompp(npt, os.path.join(mdp_dir, 'npt.mdp'), confout, top, 'polymer.tpr')
        if res != None:
            npt_tasks.append(res)
        else:
            print(f'WARNING: grompp failed at {npt}, skip')
    if len(npt_tasks) == 0:
        print('grompp failed everywhere fo some reason, exit')
        exit()
    res = wait(npt_tasks)
    if not res:
        print('waiting for NPT failed, exit')
        exit()
except Exception as e:
    print('Failed to run NPT relaxation: ', e)
    exit()

exit()


mdp_dir = '/home/md/md/Tg/gromacs_test/mdp/c-rescale'
calc_root = '/home/md/md/Tg/all_data/c_rescale'
generation_dir = '/home/md/md/Tg/generation/'
exclude = []
CPU_PER_TASK = 32
MIN_GPU_LOAD = 30
CALCULATED_POLYMERS = os.path.join(generation_dir, 'calculated_cr.txt')

below400 = [0, 1, 2, 3, 7, 8, 9, 12, 16, 17, 19, 21, 22, 23, 24, 28, 29, 30, 31, 34, 36, 37, 39, 40, 42, 43, 44, 46, 48, 50, 54, 56, 57, 59, 60, 61, 62, 63, 64, 66, 68, 69, 72, 73]
up400 = [4, 5, 6, 10, 11, 13, 14, 15, 18, 20, 25, 26, 27, 32, 33, 35, 38, 41, 45, 47, 49, 51, 52, 53, 55, 58, 65, 67, 70, 71, 74, 75]
exclude +=[str(i) for i in up400]
exclude += [str(i) for i in below400 if i>=40] # calc only first 16 'below 400 K' that were generated

try:
    with open(CALCULATED_POLYMERS, 'r') as log:
        exclude += log.read().strip().split('\n')
except:
    open(CALCULATED_POLYMERS, 'w').close()

exclude = list(set(exclude))


bk = '''
#!/bin/sh
#
# @(#)$Id: bk.sh,v 1.9 2008/06/25 16:43:25 jleffler Exp $"
#
# Run process in background
# Immune from logoffs -- output to file log

(
echo "Date: `date`"
echo "Command: $*"
nice nohup "$@"
echo "Completed: `date`"
echo
) >>${LOGFILE:=log} 2>&1 &
'''



def check_completed(wd):
    '''
    `bk` script required and should be added to PATH
    '''
    with open(os.path.join(wd, 'log'), 'r') as log:
        cont = log.read().strip().split('\n')
    return 'Completed' in cont[-1]
    

def wait(tprs : list[str], timeout = 5 * 24 * 60 * 60, cpu = CPU_PER_TASK, gpu = MIN_GPU_LOAD):
    running = []
    print(f'Started at {time.ctime()}')
    t0 = time.time()
    ok = False
    while True:
        if (len(tprs) == 0) and (len(running) == 0):
            print('All tasks completed')
            ok = True
            break
        if time.time() - t0 >= timeout:
            print('Maximum time reached. Uncompleted tasks: ')
            print(tprs)
            print('Running tasks: ')
            print(running)
            break
        for tpr in tprs.copy():
            cpu_av = free_cpus()
            gpu_av = free_gpu(max_usage=gpu)
            if (cpu_av >= cpu) & (gpu_av != None):
                wd = os.path.dirname(tpr)
                mdrun(wd, gpu_av, cpu, True)
                tprs.remove(tpr)
                print(f'New task started at {wd} ({cpu} procs, gpu {gpu_av})')
                running.append(wd)
                time.sleep(20)
            else:
                break
        for job in running.copy():
            if check_completed(job):
                running.remove(job)
        time.sleep(60)
    print(f'Ended at {time.ctime()}')
    return ok


# STEP 1: copy files from generation folder to calc folder
start = time.time()
print('='*50)
print('Calculation of glass transition temperature')
print('='*50 + '\n\n')
try: 
    new_calc_dirs = []
    for d in os.listdir(generation_dir):
        if d not in exclude:
            pdb = os.path.join(generation_dir, d, 'input.pdb')
            top = os.path.join(generation_dir, d, 'input.top')
            calcdir = os.path.join(calc_root, d)
            if os.path.exists(pdb):
                os.makedirs(calcdir)
                shutil.copy(pdb, calcdir)
                shutil.copy(top, calcdir)
                new_calc_dirs.append(calcdir)
            else:
                print(f'WARNING: pdb file not found in {d}, skipping it')
    print(f'{len(new_calc_dirs)} candidates were found')
    if len(new_calc_dirs) == 0:
        print('Nothing to do, exit')
        exit()
except Exception as e:
    print('Failed to copy initial structures from `generation folder`: ', e)
    exit()


# STEP 2: run NVT relaxation at 500 K for 10 ns
try:
    nvt_tasks = []
    for d in new_calc_dirs:
        nvt = os.path.join(d, 'nvt')
        os.makedirs(nvt)
        res = grompp(nvt, os.path.join(mdp_dir, 'nvt.mdp'), os.path.join(d, 'input.pdb'), os.path.join(d, 'input.top'), 'polymer.tpr')
        if res != None:
            nvt_tasks.append(res)
        else:
            print(f'WARNING: grompp failed at {nvt}, skip')
    if len(nvt_tasks) == 0:
        print('grompp failed everywhere fo some reason, exit')
        exit()
    res = wait(nvt_tasks)
    if not res:
        print('waiting for NVT failed, exit')
        exit()
except Exception as e:
    print('Failed to run NVT relaxation: ', e)
    exit()


# STEP 3: run NPT relaxation at 500 K for 10 ns
try:
    npt_tasks = []
    for d in new_calc_dirs:
        npt = os.path.join(d, 'npt')
        os.makedirs(npt)
        confout = os.path.join(d, 'nvt', 'confout.gro')
        if not os.path.exists(confout):
            print(f'WARNING: grompp could not be started because NVT job failed: {os.path.join(d, "nvt")}, skip')
            continue
        res = grompp(npt, os.path.join(mdp_dir, 'npt.mdp'), confout, os.path.join(d, 'input.top'), 'polymer.tpr')
        if res != None:
            npt_tasks.append(res)
        else:
            print(f'WARNING: grompp failed at {npt}, skip')
    if len(npt_tasks) == 0:
        print('grompp failed everywhere fo some reason, exit')
        exit()
    res = wait(npt_tasks)
    if not res:
        print('waiting for NPT failed, exit')
        exit()
except Exception as e:
    print('Failed to run NPT relaxation: ', e)
    exit()


# STEP 4: run cooling from 500 to 100 K for 100 ns (4K / ns)

try:
    tasks = []
    for d in new_calc_dirs:
        cool = os.path.join(d, 'cool')
        os.makedirs(cool)
        confout = os.path.join(d, 'npt', 'confout.gro')
        if not os.path.exists(confout):
            print(f'WARNING: grompp could not be started because NPT job failed: {os.path.join(d, "npt")}, skip')
            continue
        res = grompp(cool, os.path.join(mdp_dir, 'cooling.mdp'), confout, os.path.join(d, 'input.top'), 'polymer.tpr')
        if res != None:
            tasks.append(res)
        else:
            print(f'WARNING: grompp failed at {cool}, skip')
    if len(tasks) == 0:
        print('grompp failed everywhere fo some reason, exit')
        exit()
    res = wait(tasks)
    if not res:
        print('waiting for cooling failed, exit')
        exit()
except Exception as e:
    print('Failed to run cooling: ', e)
    exit()

success = []
for d in new_calc_dirs:
    confout = os.path.join(d, 'cool', 'confout.gro')
    if not os.path.exists(confout):
        print(f'WARNING: cooling failed: {os.path.join(d, "cool")}, skip')
        continue
    success.append(os.path.basename(d))
    
with open(CALCULATED_POLYMERS, 'a') as log:
    for p in success:
        log.write(p + '\n')

print(f'Polymers calculated: {len(success)}/{len(new_calc_dirs)}')

end = time.time()
dt = (end-start)/3600
print('='*50 + '\n')
print(f'Normal termination. Total time: {int(dt):.0f}h {60*(dt-int(dt)):.0f}m')

