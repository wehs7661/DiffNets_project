import os
import sys
import argparse
import numpy as np
import time
import datetime
import pickle
import natsort
import glob
import scipy.stats
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

def parse_pml(pml_file, top):
    """
    This function gets the atom pairs from rescorr-100.pml
    """
    f = open(pml_file, 'r')
    lines = f.readlines()
    f.close()

    ordinal = lambda n: "%d%s" % (n,"tsnrhtdd"[(n//10%10!=1)*(n%10<4)*n%10::4])
    pairs_A, pairs_B, atoms_A, atoms_B = [], [], [], []    # distances colored in red and blue
    for i in range(50):   # there should be 50 pairs of distances (6 lines for each pair)
        token = lines[6 * i].split('\n')[0].split(',')[-2:]     # the line containing "distance" 
        idx, atom = [], []   # The indices of the pair of atoms and mdtraj selections
        for j in token:
            res = int(j.split('resi')[-1].split('and')[0])   # resi in pymol is 1-based
            a_type = j.split('resi')[-1].split('and')[-1]
            selection = f'resi {res - 1} and' + a_type       # resi in mdtraj is 0-based
            idx.append(int(md.load(top).topology.select(selection)))
            atom.append(f'({a_type.split(" ")[-1]} of the {ordinal(res)} residue)')
    
        if 'red' in lines[6 * i + 1]:
            pairs_A.append(idx)
            atoms_A.append(atom)
        elif 'blue' in lines[6 * i + 1]:
            pairs_B.append(idx)
            atoms_B.append(atom)

    return pairs_A, pairs_B, atoms_A, atoms_B
                

if __name__ == "__main__":
    args = initialize()
    main_code = 'python3.6 /ocean/projects/cts160011p/wehs7661/GitHub_repo/diffnets/diffnets/cli/main.py'

    rc("font", **{"family": "sans-serif", "sans-serif": ["DejaVu Sans"], "size": 10})
    # Set the font used for MathJax - more on this later
    rc("mathtext", **{"default": "regular"})
    plt.rc("font", family="serif")

    logger(f'Command line: {" ".join(sys.argv)}')
    if '1' in args.actions:
        logger('\nPart 1: Standard data analysis for the trained DiffNets')
        logger('=========================================================')
        t1 = time.time()

        stdout_fd = sys.stdout.fileno()
        with open('results_run_diffnets.txt', 'a') as f, stdout_redirected(f):
            with merged_stderr_stdout():
                os.system(f'{main_code} analyze ./whitened_data {args.diffnets}')

        t2 = time.time()
        logger(f'Time elapsed: {format_time(t2 - t1)}')

    if '2' in args.actions:
        logger('\nPart 2: Additional analysis for the trained DiffNets')
        logger('======================================================')
        t1 = time.time()

        logger('Assessing the quality of trajectory alignment ...')
        top = './whitened_data/master.pdb'
        xtc_files = natsort.natsorted(glob.glob('./whitened_data/aligned_xtcs/*.xtc'))
        inds_files = natsort.natsorted(glob.glob('./whitened_data/indicators/*.npy'))
        inds = []
        for i in inds_files:
            inds.append(np.load(i)[0])
        trajs = []
        for i in xtc_files:
            traj = md.load(i, top=top)
            trajs.append(traj)
            rmsd = md.rmsd(traj, md.load(top))
            logger(f'    The average RMSD of the trajectory {i.split("/")[-1]} w.r.t the reference is {np.mean(rmsd):.3f} nm.')

        logger('\nCaculating distances between atoms involved in structural determinants for each trajectory ...')
        pairs_A, pairs_B, atoms_A, atoms_B = parse_pml(args.diffnets + '/rescorr-100.pml', top)
        
        trajs_0, trajs_1 = [], []    # 0-labeled trajs and 1-labeled trajs
        for i in range(len(trajs)):
            if inds[i] == 0:
                trajs_0.append(trajs[i])
            elif inds[i] == 1:
                trajs_1.append(trajs[i])
        
        trajs_0 = md.join(trajs_0)
        trajs_1 = md.join(trajs_1)
        
        n = 0    # number of inconsistent distances
        logger(f'    DiffNets predict the following {len(pairs_A)} distances should decrease as the label goes from 0 to 1:')  
        d_A0 = md.compute_distances(trajs_0, pairs_A)
        d_A1 = md.compute_distances(trajs_1, pairs_A)

        avg_d_A0 = np.mean(d_A0, axis=1)
        avg_d_A1 = np.mean(d_A1, axis=1)
        std_d_A0 = np.std(d_A0, axis=1)
        std_d_A1 = np.std(d_A1, axis=1)
        [t, p_value] = scipy.stats.ttest_ind(d_A0, d_A1, axis=1)  # p_value here is two-tailed
        
        # Note that in ttest_ind, H_0 indicates identical means of d_A0 and d_A1. We denote the p-value as p1.
        # In a two-tailed t test, H_0 would be avg_d_A1[i] = avg_d_A0[i] and H_1 would be avg_d_A1[i] != avg_d_A0[i].
        # (1) if p1 > 0.05, H_0 is accepted --> No significant difference between d_A0[i] and d_A1[i].
        # (2) If p1 < 0.05, H_0 is rejected --> We conclude that avg_d_A1[i] != avg_d_A0[i].
        
        # However, in our case here, we want to know if avg_d_A0[i] > avg_d_A1[i], not just avg_d_A1[i] != avg_d_A0[i], so we need an one-tailed t test.
        # That is, we have H_0: avg_d_A0[i] <= avg_d_A1[i], H_1: avg_d_A0[i] > avg_d_A1[i].
        # (1) If p1/2 > 0.05, H_0 is accepted --> We cannot conclude that avg_d_A0[i] > avg_d_A1[i], i.e. DiffNets prediction is consistent with the simulation.
        # (2) If p1/2 < 0.05, H_0 is rejected --> We accept H_1: avg_d_A0[i] > avg_d_A1[i]. DiffNets prediction is consistent with the simulation.

        # Reminder: H_0 is the claim we want to argue against, while H_1 is the claim we hope to be true. 
        # In our case here, we want to know if avg_d_A0[i] > avg_d_A1[i], so we have H_0: avg_d_A0[i] <= avg_d_A1[i], H_1: avg_d_A0[i] > avg_d_A1[i].
        # 


        # Note that in ttest_ind, H_0 indicates identical means of d_A0 and d_A1. We denote the p-value as p1.
        # In a two-sided t test, H_0 would be avg_d_A1[i] = avg_d_A0[i] and H_1 would be avg_d_A1[i] != avg_d_A0[i]
        # In a one-sided t test, H_0 stays the same, but there are two kinds of H_1: avg_d_A1[i] > avg_d_A0[i] or avg_d_A1[i] < avg_d_A0[i]
        # (1) If p1 > 0.05, H_0 is accepted. --> No significance difference between two samples.
        # (2) If p1 < 0.05, H_0 is rejected and the H_1 in the two sided test is accepted. --> Strong evidence that avg_d_A1[i] != avg_d_A0[i].
        # (3) If p1/2 < 0.05, H_0 is rejected and either kind of H_1 is accepted.
        # (3-1) If t > 0, then H_1: 
        # (3-2) If t < 0, then H_1: 



        # In either case of H_0: avg_d_A1[i] < avg_d_A0[i] or H_0: avg_d_A1[i] > avg_d_A0[i], the corresponding p-value, p2, is p1/2.
        # (1) If p2 = p1/2 > 0.05 and t >0 --> H_0: avg_d_A1[i] > avg_d_A0[i] rejected
        


        # (1) p1 > 0.05 --> H_0: avg_d_A1[i] = avg_d_A0[i] accepted (No significant statistical difference)
        
        
        # (2) p2 > 0.05 --> H_0: avg_d_A1[i] > avg_d_A0[i] accepted, or H_0: avg_d_A1[i] < avg_d_A0[i] accepted (there is significant difference)

        # (2) If p2 < 0.05 (H_0 rejected): 
        # 
        # In this case, H_0 is avg_d_A1[i] < avg_d_A0[i], so if the updated p-value is smaller than 0.05, we reject H_0.
        # That is, 
        
        # H_0 in our case should be "avg_d_A1[i] < avg_d_A0[i]", so we divide p_value by 2.
        # If the updated p_value is smaller than 0.05, we reject H_0 -> We can't conclude that avg_d_A1[i] < avg_d_A0[i].
        # In this case, that means we can't conclude that the DiffNets prediction is consistent with the simulation itself.
        # On the other hand, if the updated p_value is larger than 0.05 

        for i in range(len(pairs_A)): 
            logger(f'        Distance between {atoms_A[i][0]} and {atoms_A[i][1]}')
            logger(f'            The average distance of the 0-labeled variants: {avg_d_A0[i]:.3f} +/- {std_d_A0[i]:.3f} nm')      
            logger(f'            The average distance of the 1-labeled variants: {avg_d_A1[i]:.3f} +/- {std_d_A1[i]:.3f} nm')      
            logger(f'            The p-value is {p_value[i] / 2}.')
            logger(f'            The t-statistic is {t[i]}.')


            if avg_d_A1[i] < avg_d_A0[i]:  # H_0: avg_d_A1[i] < avg_d_A0[i] (consistent) 
                pass
                #logger('            Result: The prediction is CONSISTENT with the trajectory.') 
            else:
                n += 1
                #logger('            Result: The prediction is INCONSISTENT with the trajectory.') 

        logger(f'\n    DiffNets predict the following {len(pairs_B)} distances should increase as the label goes from 0 to 1:')  
        d_B0 = md.compute_distances(trajs_0, pairs_B)
        d_B1 = md.compute_distances(trajs_1, pairs_B)

        avg_d_B0 = np.mean(d_B0, axis=1)
        avg_d_B1 = np.mean(d_B1, axis=1)
        std_d_B0 = np.std(d_B0, axis=1)
        std_d_B1 = np.std(d_B1, axis=1)
        [t, p_value] = scipy.stats.ttest_ind(d_B0, d_B1, axis=1)  # H_0: identical mean
        
        for i in range(len(pairs_B)): 
            logger(f'        Distance between {atoms_B[i][0]} and {atoms_B[i][1]}')
            logger(f'            The average distance of the 0-labeled variants: {avg_d_B0[i]:.3f} +/- {std_d_B0[i]:.3f} nm')      
            logger(f'            The average distance of the 1-labeled variants: {avg_d_B1[i]:.3f} +/- {std_d_B1[i]:.3f} nm')      
            logger(f'            The p-value is {p_value[i] / 2}.')
            logger(f'            The t-statistic is {t[i]}.')
            if avg_d_B1[i] > avg_d_B0[i]:
                pass
                #logger('            Result: The prediction is CONSISTENT with the trajectory.') 
            else:
                n += 1
                #logger('            Result: The prediction is INCONSISTENT with the trajectory.') 
        #print(f'\n    Overall result: {n} out of 50 predictions made by DiffNets are inconsistent.')

        logger('\nPart 3: Sanity checks')
        logger('=======================')
        logger(' Check 1: Average reconstructed RMSD ]')
        rmsd = np.load(os.path.join(args.diffnets, "rmsd.npy"))  # nm
        logger(f'==> Result: The average RMSD is {np.mean(rmsd) * 10:.2f} angstrom.')
        plt.figure()
        plt.hist(rmsd * 10, bins=100)
        plt.xlabel('RMSD ($ \AA $)')
        plt.ylabel('Count')
        plt.title('The distribution of the reconstructed RMSD')
        plt.grid()
        plt.savefig('RMSD_distribution.png', dpi=600)

        logger('\n[ Check 2: Output label distribution ]') 
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
        logger('The figure has been generated.')

        """
        logger('\n[ Check 3: AUC calculation ]') 
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
        logger(f'\nTime elapsed: {format_time(t2 - t1)}')