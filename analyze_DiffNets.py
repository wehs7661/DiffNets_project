import os
import sys
import argparse
import numpy as np
import time
import datetime
import pickle
import natsort
import glob
import mdtraj as md 
import matplotlib.pyplot as plt
from matplotlib import rc
from collections import defaultdict
from contextlib import contextmanager
from argparse import RawTextHelpFormatter
from diffnets import utils
from diffnets import analysis


def initialize():
    parser = argparse.ArgumentParser(
        description="This code runs data analysis and some sanity check for a trained DiffNets.")
    parser.add_argument(
        "-d",
        "--diffnets",
        default="./split_sae_e10_lr0001_lat50_rep0_em",
        help="The trained DiffNets of interest."
    )
    parser.add_argument(
        "-a",
        "--actions",
        nargs="+",
        choices=['1', '2'],
        default=['1', '2'],
        help="Choice of action. (1: Data analysis, 2: Sanity checks)"
    )

    args_parse = parser.parse_args()

    return args_parse

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

if __name__ == "__main__":
    args = initialize()
    main_code = 'python3.6 /ocean/projects/cts160011p/wehs7661/GitHub_repo/diffnets/diffnets/cli/main.py'

    rc("font", **{"family": "sans-serif", "sans-serif": ["DejaVu Sans"], "size": 10})
    # Set the font used for MathJax - more on this later
    rc("mathtext", **{"default": "regular"})
    plt.rc("font", family="serif")

    logger(f'Command line: {" ".join(sys.argv)}')
    if '1' in args.actions:
        logger('\nPart 1: Perform standard data analysis for the trained DiffNets ...')
        logger('=====================================================================')
        t1 = time.time()

        stdout_fd = sys.stdout.fileno()
        with open('results_run_diffnets.txt', 'a') as f, stdout_redirected(f):
            with merged_stderr_stdout():
                os.system(f'{main_code} analyze ./whitened_data {args.diffnets}')

        t2 = time.time()
        logger(f'Time elapsed: {format_time(t2 - t1)}')

    if '2' in args.actions:
        logger('\nPart 2: Perform additional analysis for the trained DiffNets ...')
        logger('==================================================================')
        t1 = time.time()

        logger('Assessing the quality of trajectory alignment ...')
        xtc_files = natsort.natsorted(glob.glob('./whitened_data/aligned_xtcs/*.xtc'))
        top = './whitened_data/master.pdb'
        for i in xtc_files:
            traj = md.load(i, top=top)
            rmsd = md.rmsd(traj, md.load(top))
            logger(f'    The average RMSD of the trajectory {i.split("/")[-1]} w.r.t the reference is {np.mean(rmsd):.3f} nm.')

        logger('\nPerforming sanity checks ...')
        logger('    [ Check 1: Average reconstructed RMSD ]')
        rmsd = np.load(os.path.join(args.diffnets, "rmsd.npy"))  # nm
        logger(f'    ==> Result: The average RMSD is {np.mean(rmsd) * 10:.2f} angstrom.')
        plt.figure()
        plt.hist(rmsd * 10, bins=100)
        plt.xlabel('RMSD ($ \AA $)')
        plt.ylabel('Count')
        plt.title('The distribution of the reconstructed RMSD')
        plt.grid()
        plt.savefig('RMSD_distribution.png', dpi=600)

        logger('\n    [ Check 2: Output label distribution ]') 
        lab_fns = utils.get_fns(os.path.join(args.diffnets, "labels"), "*.npy")
        traj_d_path = os.path.join('./whitened_data', "traj_dict.pkl")
        traj_d = pickle.load(open(traj_d_path, 'rb'))
        lab_v = defaultdict(list)
        for key, item in traj_d.items():
            for traj_ind in range(item[0],item[1]):
                lab = np.load(lab_fns[traj_ind])
                lab_v[key].append(lab)

        plt.figure()
        leg_labels = []
        for k in traj_d.keys():
            t = np.concatenate(lab_v[k])
            n, x = np.histogram(t, range=(0, 1), bins=50)
            leg_labels.append(k.split("/")[-1])
            plt.plot(x[:-1], n, label=k)
        plt.xlabel('DiffNets output label')
        plt.ylabel('# of simulation frames')
        plt.legend()
        plt.savefig('label_plot.png', dpi=600)

        """
        logger('\n    [ Check 3: AUC calculation ]') 
        net_fn = os.path.join(args.diffnets, 'nn_best_polish.pkl')
        out_fn = 'auc_plot'  # acutally not active in original code of `calc_auc`
        cm = np.load('./whitened_data/cm.npy')
        xtc_dir = 'whitened_data/aligned_xtcs'
        """
        
        """
        logger('\nPlotting the loss functions as a function of time ...\n')
        loss = np.load(f'{args.diffnets}/all_loss_term_polish.npy')
        legends = ['L1 norm', 'MSE', 'BCE', 'Cov. penalty']
        n_epochs = len(loss[0])
        plt.figure()
        for i in range(len(loss)):
            plt.plot(np.arange(1, n_epochs + 1), loss[i], label='legends[0')
        plt.xlabel('Number of epochs')
        plt.ylabel('Loss function')
        plt.grid()
        plt.legend()
        plt.savefig('loss_epochs.png', dpi=600)
        """
        t2 = time.time()
        logger(f'Time elapsed: {format_time(t2 - t1)}')