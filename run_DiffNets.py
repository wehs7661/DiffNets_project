import os
import sys
import argparse
import numpy as np
import mdtraj as md
import time
import datetime
import psutil
from contextlib import contextmanager
from argparse import RawTextHelpFormatter
from diffnets.nnutils import split_inds
from memory_profiler import memory_usage

class ImproperlyConfigured(Exception):
    '''The given configuration is incomplete or otherwise not usable.'''
    pass

def initialize():
    parser = argparse.ArgumentParser(
        description="This code applies DiffNets to the simulation outputs of insulin glycoform to generate \n"
        "predictors of insulin properties. It automates the following steps: \n"
        "   (1) Create symbolic links for the simulation output files\n"
        "   (2) Prepare DiffNets input files (mostly the npy files)\n"
        "   (3) Perform data whitening transformation and train a DiffNet"
        "Note that config.yml should be present if you want to train a DiffNet.",
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument(
        "-m",
        "--model",
        required=True,
        nargs='+',
        choices=['4EYD', '4EY9', '4EY1', '3I3Z', '2MVC', 'all'],
        help="The wildtype model that all the variants of interest are based on."
    )
    parser.add_argument(
        "-p",
        "--property",
        required=True,
        choices=['1', '2'],
        help="The insulin property of interest: 1 for dimerization propensity and 2 for proteolytic stability."
    )
    parser.add_argument(
        "-v",
        "--variants",
        type=int,
        nargs='+',
        help="Serial numbers of the variants of interest. Note that by defulat, training a DiffNets for "
        "dimerization propensity requires consideration of GF 1 (WT), 9, 10, 13, while for proteolytic"
        "stability, all variants are considered."
    )
    parser.add_argument(
        "-a",
        "--actions",
        nargs="+",
        choices=['1', '2', '3', '4'],
        default=['1', '2', '3', '4'],
        help="Choice of action. (1: Collect simulation outputs, 2: Prepare inputs, 3: Whitening transformation, 4: Train a DiffNet)"
    )
    parser.add_argument(
        "-s",
        "--select",
        type=str,
        default='name C or name CA or name N',  # backbone atoms according to GROMACS' definition
        help="Strings for selecting atoms for trajectory alignment. The input string should follows" 
             "mdtraj atom selection syntax (e.g. name C or name CA or name CB or name N)."
    )
    parser.add_argument(
        "-r",
        "--res",
        type=int,
        help="The residue (1-based) of interest in the focused region for a split autoencoder."
    )
    parser.add_argument(
        "-d",
        "--dist",
        type=float,
        default=1.0, 
        help="The cutoff distance from the residue of interest, within which the residues will be "
             "classified in the focused region. (Default: 1.0 nm)"
    )
    parser.add_argument(
        "-c",
        "--cluster",
        default=False,
        action="store_true",
        help="Whether to find the cluster centroid and use it as the reference structure."
    )
    parser.add_argument(
        "-ref",
        "--ref",
        default=None,
        help="The filename of the reference structure pdb. If the argument is not specified, the first variant pdb will be used."
    )
    

    args_parse = parser.parse_args()

    return args_parse

def format_time(t):
    hh_mm_ss = str(datetime.timedelta(seconds=t)).split(':')
    hh, mm, ss = float(hh_mm_ss[0]), float(hh_mm_ss[1]), float(hh_mm_ss[2])
    if hh == 0:
        if mm == 0:
            t_str = f'{ss} second(s)'
        else:
            t_str = f'{mm:.0f} minute(s) {ss:.0f} second(s)'
    else:
        t_str = f'{hh:.0f} hour(s) {mm:.0f} minute(s) {ss:.0f} second(s)'

    return t_str

def fileno(file_or_fd):
    fd = getattr(file_or_fd, 'fileno', lambda: file_or_fd)()
    if not isinstance(fd, int):
        raise ValueError("Expected a file (`.fileno()`) or a file descriptor")
    return fd

@contextmanager
def stdout_redirected(to=os.devnull, stdout=None):
    if stdout is None:
        stdout = sys.stdout

    stdout_fd = fileno(stdout)
    # copy stdout_fd before it is overwritten
    # Note: `copied` is inheritable on Windows when duplicating a standard stream
    with os.fdopen(os.dup(stdout_fd), 'wb') as copied:
        stdout.flush()  # flush library buffers that dup2 knows nothing about
        try:
            os.dup2(fileno(to), stdout_fd)  # $ exec >&to
        except ValueError:  # filename
            with open(to, 'wb') as to_file:
                os.dup2(to_file.fileno(), stdout_fd)  # $ exec > to
        try:
            yield stdout # allow code to be run with the redirected stdout
        finally:
            # restore stdout to its previous value
            # Note: dup2 makes stdout_fd inheritable unconditionally
            stdout.flush()
            os.dup2(copied.fileno(), stdout_fd)  # $ exec >&copied 
    
def merged_stderr_stdout():  # $ exec 2>&1
    return stdout_redirected(to=sys.stdout, stdout=sys.stderr)

def logger(*args, **kwargs):
    print(*args, **kwargs)
    with open('results_run_diffnets.txt', "a") as f:
        print(file=f, *args, **kwargs)

def check_file(files):
    for i in files:
        if os.path.isfile(i) is False:
            print(f'Simulation file {i} not found!')
            sys.exit()
        
def compare_atom_sel(u1, u2, sel_1, sel_2):
    # Step 1: Compare the number of atoms
    if len(sel_1) != len(sel_2):
        raise ImproperlyConfigured(
            f'The number of selected atoms is different from that in the WT!')

    # Step 2: Compare atom types and residue types
    u1_top, u2_top = u1.topology.to_dataframe()[0], u2.topology.to_dataframe()[0]
    
    a_type_1 = [u1_top.iloc[i]['name'] for i in sel_1]
    a_type_2 = [u2_top.iloc[i]['name'] for i in sel_2]
    if a_type_1 != a_type_2:
        raise ImproperlyConfigured(
            f'The atom types of selected atoms are different from those in the WT!')
    
    r_type_1 = [u1_top.iloc[i]['resName'] for i in sel_1]
    r_type_2 = [u2_top.iloc[i]['resName'] for i in sel_2]
    if r_type_1 != r_type_2:
        raise ImproperlyConfigured(
            f'The residue types of selected atoms are different from those in the WT!')

def run_command(cmd):
    """This is only for more convenient memory profiling"""
    os.system(cmd)

def convert_memory_units(mem):
    """
    mem: RAM memory usage in MB (default units of memory_usage)
    """
    power = 1024
    mem *= power ** 2   # first convert to byte

    # Get the total available RAM memory
    tot_mem = psutil.virtual_memory().total   # in byte
    mem_percent = mem / tot_mem * 100  

    # Convert units
    n = 0
    power_labels = {0: '', 1: 'k', 2: 'M', 3: 'G', 4: 'T'}
    while mem > power:
        mem /= power
        n += 1
    mem_str = f'{mem: .2f} {power_labels[n]}B'

    return mem_str, mem_percent


if __name__ == "__main__":
    args = initialize()

    # Step 0: Setting things up
    logger(f'Command line: {" ".join(sys.argv)}')
    if args.cluster is True and args.ref is not None:
        raise ImproperlyConfigured(
            f'The arguments cluster and ref should not be specified at the same time!')
    if args.variants is None:
        if args.property == '1':  # dimerization propensity
            args.variants = [1, 9, 10, 13]
        elif args.property == '2':  # proteolytic stability
            args.variants = range(1, 14)

    if args.model == ['all']:
        args.model = ['4EYD', '4EY9', '4EY1', '3I3Z', '2MVC']
    
    pH_dict = {'4EYD': 'pH_8.0', '4EY9': 'pH_8.0', '4EY1': 'pH_7.9', '3I3Z': 'pH_6.9', '2MVC': 'pH_7.3'}
    Sys = [i.lower() for i in args.model]
    project_dir = '/ocean/projects/cts160011p/wehs7661/Glycoinsulin_project/'
    main_code = 'python3.6 /ocean/projects/cts160011p/wehs7661/GitHub_repo/diffnets/diffnets/cli/main.py'

    # Step 1: Create symbolic links for the simulation outputs (including pdb, xtc files).
    if '1' in args.actions:
        logger('\nStep 1: Creating symbolic links ...')
        logger('===================================')
        t1 = time.time()

        os.makedirs('data')
        for i in range(len(Sys)):
            for j in args.variants:
                if j == 1:  # wildtype
                    pdb_src = f'{project_dir}/wildtype_insulin/{args.model[i]}/{pH_dict[args.model[i]]}/Sol_ions/{Sys[i]}_ions.pdb'
                    xtc_src = f'{project_dir}/wildtype_insulin/{args.model[i]}/{pH_dict[args.model[i]]}/MD/{Sys[i]}_md.xtc'
                else:       # other variants
                    pdb_src = f'{project_dir}/glyco_insulin/2018_ACS_paper/{args.model[i]}_glycoforms/glycoform_{j}_ACS/Sol_ions/glycoform_{j}_ACS_ions.pdb'
                    xtc_src = f'{project_dir}/glyco_insulin/2018_ACS_paper/{args.model[i]}_glycoforms/glycoform_{j}_ACS/MD/glycoform_{j}_ACS_md.xtc'

                check_file([pdb_src, xtc_src])
                
                os.system(f'ln -s {pdb_src} data/{Sys[i]}_GF_{j}.pdb')
                os.system(f'mkdir data/{Sys[i]}_traj_{j} && ln -s {xtc_src} data/{Sys[i]}_traj_{j}/{Sys[i]}_GF_{j}.xtc')

        t2 = time.time()
        logger(f'Time elapsed: {format_time(t2 - t1)}')

    # Step 2: Create npy files as the inputs for DiffNets
    if '2' in args.actions:
        logger('\nStep 2: Creating inputs for DiffNets ...')
        logger('========================================')
        t1 = time.time()

        sim_dirs, pdb_fns, atom_sel, stride = [], [], [], []  # for whitening transformation (one element/list for each variant)
        u_list = []  # for comparing atom_sel
        for i in range(len(Sys)):
            for j in args.variants:
                sim_dirs.append(f'./data/{Sys[i]}_traj_{j}')
                pdb_fns.append(f'./data/{Sys[i]}_GF_{j}.pdb')

                u = md.load(os.readlink(f'./data/{Sys[i]}_GF_{j}.pdb'))
                u_list.append(u)

                atoms = list(u.top.select(args.select))  # 0-indexed
                atom_sel.append(atoms)
                
                if j != 1:
                    # Compare the selected atoms of variant j with the wildtype to make sure they are the same
                    compare_atom_sel(u_list[0], u_list[-1], atom_sel[0], atom_sel[-1])

                stride.append(10)

        close_inds = []  # for training (indices of atoms of interest RELATIVE TO master.pdb)
        # All variants should have the same list of indices given the same region of interest and close_inds only contains one list
        # Here we use first wildtype structures to define the region
        u = md.load(os.readlink(f'./data/{Sys[i]}_GF_1.pdb'))
        
        if args.res is not None:
            close_xyz_inds, non_close_xyz_inds = split_inds(u, args.res, args.dist)
            region_idx = [int(i) for i in close_xyz_inds[::3]/3]  # could include water/ions
        else:
            if args.property == '1':       # dimerization property 
                region_idx = u.top.select("resid 43 to 46")  # dimer interface B23-B26 (resid (0-based) 43 to 46)
            elif args.property == '2':     # proteolytic stability
                region_idx = u.top.select("resid 42 to 48")  # P3--P2' region B22-B28 (resid 42 to 48)
        
        for j in region_idx:
            if j in atoms:
                close_inds.append(atoms.index(j))

        np.save('./traj_dirs.npy', np.array(sim_dirs))
        np.save('./pdb_fns.npy', np.array(pdb_fns))
        np.save('./atom_sel.npy', np.array(atom_sel))
        np.save('./stride.npy', np.array(stride))
        np.save('./close_inds.npy', np.array(close_inds))

        t2 = time.time()
        logger(f'Time elapsed: {format_time(t2 - t1)}')

    # Step 3: Whitening transformation
    if '3' in args.actions:
        logger('\nStep 3: Performing whitening transformation ...')
        logger('===============================================')
        print(f'Total available memory: {psutil.virtual_memory().total}')
        t1 = time.time()

        with open('results_run_diffnets.txt', 'a') as f, stdout_redirected(f):
            with merged_stderr_stdout():
                # Below we run the command and get its maximum memory usage
                if args.ref is None:
                    if args.cluster is False:
                        cmd = f'{main_code} process ./traj_dirs.npy ./pdb_fns.npy ./whitened_data -aatom_sel.npy -sstride.npy'
                    else:
                        cmd = f'{main_code} process ./traj_dirs.npy ./pdb_fns.npy ./whitened_data -aatom_sel.npy -sstride.npy -c'
                else:
                    cmd = f'{main_code} process ./traj_dirs.npy ./pdb_fns.npy ./whitened_data -aatom_sel.npy -sstride.npy -r{args.ref}'
                mem = memory_usage((run_command, (cmd,)))
                max_mem, max_mem_percent = convert_memory_units(np.max(mem))

        t2 = time.time()

        logger(f'The maximum memory usage of data whitening was: {max_mem} ({max_mem_percent:.1f}%).')
        logger(f'Time elapsed: {format_time(t2 - t1)}')
        
    # Step 4: Training DiffNets
    if '4' in args.actions:
        logger('\nStep 4: Training DiffNets ...')
        logger('===============================')
        t1 = time.time()

        with open('results_run_diffnets.txt', 'a') as f, stdout_redirected(f):
            with merged_stderr_stdout():
                # Below we run the command and get its maximum memory usage
                cmd = f'{main_code} train config.yml'
                mem = memory_usage((run_command, (cmd,)))
                max_mem, max_mem_percent = convert_memory_units(np.max(mem))

        t2 = time.time()
        logger(f'The maximum memory usage of data training was: {max_mem} ({max_mem_percent:.1f}%).')
        logger(f'Time elapsed: {format_time(t2 - t1)}')

