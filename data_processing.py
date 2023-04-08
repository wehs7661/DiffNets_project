import os
import multiprocessing as mp
import functools
import glob
from .utils import *
import pickle
import time
import datetime
from collections import defaultdict

import numpy as np
import mdtraj as md
from scipy.linalg import inv, sqrtm
import torch

class ImproperlyConfigured(Exception):
    '''The given configuration is incomplete or otherwise not usable.'''
    pass

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

class ProcessTraj:
    """Process raw trajectory data to select a subset of atoms and align all
       frames to a reference pdb. Results in a directory structure that the
       training relies on.

    Parameters
    ----------
    traj_dir_paths : list of str's, required
        One string/path for each variant to a dir that contains ALL 
        trajectory files for that variant. **ORDER MATTERS** -- when
        training you will set a value "act_map" that depends on this 
        order.
    pdb_fn_paths : list of str's,  required
        One string/path for each variant to a dir that contains the
        starting pdb file. Variants must be in same order as traj_dir_paths.
    outdir : str
        Name of dir to output processed data to. This dir will be used as input
        during DiffNet training.
    atom_sel : str, or array-like, shape=(n_variants, n_inds)
               (default="name CA or name CB or name N or name C")
         If str, it should follow the selection syntax used in MDTraj. 
         e.g. pdb.top.select("name CA") - "name CA" would be appropriate.
         If list, there should be a list of indices for each variant since 
         choosing equivalent atoms may require different indexing for each 
         variant. 
     stride : integer (default=1)
         Subsample every nth data frame. Value of 1 means no subsampling.
    """

    def __init__(self,
                 traj_dir_paths,
                 pdb_fn_paths,
                 outdir,
                 method,
                 atom_sel=None,
                 stride=1,
                 pdb_ref=None):
        self.traj_dir_paths = traj_dir_paths
        self.pdb_fn_paths = pdb_fn_paths
        self.outdir = outdir
        self.xtc_dir = os.path.join(outdir, "aligned_xtcs")
        self.indicator_dir = os.path.join(outdir, "indicators")
        self.atom_sel = atom_sel
        if self.atom_sel is None:
            self.atom_sel = self.extract_default_inds()
        self.stride = stride
        self.master = self.make_master_pdb(method, pdb_ref)
        self.n_feats = 3*self.master.top.n_atoms
        self.ref_provided = False

    def extract_default_inds(self):
       
       pdb_lens = []
       glycines = [] 
       for fn in self.pdb_fn_paths:
           pdb =  md.load(fn)
           pdb_lens.append(pdb.top.n_residues)
           for res in pdb.top.residues:
               if res.name == "GLY":
                   glycines.append(res.index)
       glycines = np.unique(glycines)
       
       if len(np.unique(pdb_lens)) != 1:
            raise ImproperlyConfigured(
                f'Cannot use default index extraction unless all variant '
                 'pdbs have the same number of residues. Consider supplying '
                 'a custom atom_sel to choose equivalent atoms across different '
                 'variant pdbs.')

       var_inds = defaultdict(list)
       var_inds_list = []
       for fn in self.pdb_fn_paths:
           pdb =  md.load(fn)
           for res in pdb.top.residues:
               if res.index in glycines:
                   sele = "resid %s and (name CA or name C or name N)" % res.index
                   j = pdb.topology.select(sele)
               else:
                   sele = "resid %s and (name CA or name C or name CB or name N)" % res.index
                   j = pdb.topology.select(sele)
               var_inds[fn].append(j)
           var_inds_list.append(np.concatenate(var_inds[fn]))
       
       return var_inds_list

    def optimize_alignment(self, method):
        """
        This function tries to optimize trajectory alignment by one of the following methods
        (1) Perform cluster analysis for the combined trajectory and find the centroid
        (2) Directly find the structure that minimize the average alingment RMSD

        Parameters
        ----------
        method (str): 
            Two choices: "cluster" or "minimize".
        """
        inputs, traj_d = self.make_traj_list()
        trajs, n_frames = [], []

        for i in range(len(inputs)):
            if type(self.stride) == np.ndarray:
                t = md.load(inputs[i][0], top=inputs[i][1], stride=int(self.stride[i]))
            else:
                t = md.load(inputs[i][0], top=inputs[i][1], stride=int(self.stride))
        
            if isinstance(self.atom_sel, list) or type(self.atom_sel) == np.ndarray:
                inds = self.atom_sel[i]
            else:
                inds = master.top.select(self.atom_sel)

            t = t.atom_slice(inds)
            trajs.append(t)
            n_frames.append(t.n_frames)

        traj_tot = md.join(trajs)
        distances = np.empty((traj_tot.n_frames, traj_tot.n_frames))
        print(f'    Calculating pairwise RMSD values ({traj_tot.n_frames} x {traj_tot.n_frames}) ...')

        for i in range(traj_tot.n_frames):
            distances[i] = md.rmsd(traj_tot, traj_tot, i)

        if method == 'cluster':
            print('    Find the centroid of the cluster ...')
            beta = 1
            index = np.exp(-beta * distances / distances.std()).sum(axis=1).argmax()
        elif method == 'minimize':
            d_avg = []
            for i in range(traj_tot.n_frames):
                d_avg.append(np.mean(distances[i]))
            index = d_avg.index(min(d_avg))
            
        for i in range(1, len(inputs) + 1):
            if index - np.sum(n_frames[:i]) < 0:
                idx  = index - np.sum(n_frames[:i - 1])
                ordinal = lambda n: "%d%s" % (n,"tsnrhtdd"[(n//10%10!=1)*(n%10<4)*n%10::4])
                print(f'    The selected frame is the {ordinal(idx)} frame in the {ordinal(i)} trajectory.')
                break

        master = traj_tot[index]
    
        d = distances[index]
        print('    The average RMSD of each trajectory:')
        for i in range(len(inputs)):
            lower = int(np.sum(n_frames[:i]))
            upper = int(np.sum(n_frames[:i + 1]))
            print(f'        {inputs[i][0]}: {np.mean(d[lower:upper]):.3f} nm.')

        return master


    def make_master_pdb(self, method, pdb_ref):
        """Creates a reference pdb centered at the origin
        using the first variant pdb specified in self.pdb_fn_paths.
        If pdb_ref is specified, use it as master.pdb.
        """
        ## TODO: Add in a check that all pdbs have same number of atoms
        ## WTH: The number of atoms in each file in self.pdb_fn_paths actually do not need to be the same as long as the atoms in atom_sel are referring to the same set of atoms. In fact, this should be more practical since variants could easily have different numbers of atoms. 
        ## WTH: Comparison of atom_sel has been done externally in run_DiffNets.py
        
        # Step 1: Create master object
        if method == 'first_var':
            # print(f'\nNote: The reference structure master.pdb is taken from the first variant pdb {self.pdb_fn_paths[0]}.')
            pdb_fn = self.pdb_fn_paths[0]
            master = md.load(pdb_fn)
            if isinstance(self.atom_sel, list) or type(self.atom_sel)==np.ndarray:
                inds = self.atom_sel[0]
            else:   # The atom_sel should be a string (seems never used though ...)
                inds = master.top.select(self.atom_sel)
            master = master.atom_slice(inds)  # inds is a list, while master is a mdtraj.Trajectory object that select the atoms with indices inds.
        elif method == 'cluster':
            t1 = time.time()
            print('    Performing clustering analysis ...')
            master = self.optimize_alignment(method='cluster')
            t2 = time.time()
            print(f'   Time elapsed: {format_time(t2-t1)}')
        elif method == 'minimize':
            t1 = time.time()
            master = self.optimize_alignment(method='minimize')
            t2 = time.time()
            print(f'    Time elapsed: {format_time(t2-t1)}')
        elif method == 'config':
            if pdb_ref is not None:
                print(f'\nNote: The reference structure master.pdb is taken from {pdb_ref}.')
                # The provided pdb should be the same set of selected atomes as the one used
                # in all trajectories, so we don't slice master here.
                master = md.load(pdb_ref)
                self.ref_provided = True
            else:
                raise ImproperlyConfigured('Method "config" is used but no PDB/GRO structure was given.')

        # Step 3: Center the coordinates and save as master.pdb
        master.center_coordinates()
        master_fn = os.path.join(self.outdir, "master.pdb")
        mkdir(self.outdir)
        master.save(master_fn)
        return master

    def make_traj_list(self):
        """Makes a list of all variant trajectories where each item is a
        list that contains 1) a path to the trajectory, 2) a path to the 
        corresponding topology (pdb) file, 3) a trajectory number - from 
        0 to n where n is total number of trajectories, and 4) an integer
        to indicate which variant simulation the trajectory came from.
        """
        traj_num = 0
        inputs = []
        i = 0
        var_dirs = self.traj_dir_paths
        pdb_fns = self.pdb_fn_paths
        traj_d = {}
        for vd, fn in zip(var_dirs,pdb_fns):
            traj_fns = get_fns(vd, "*.xtc")
            traj_d[fn] = [traj_num, traj_num+len(traj_fns)] 
            
            for traj_fn in traj_fns:
                #i indicates which variant the traj came from -- used for training
                inputs.append((traj_fn, fn, traj_num, i))
                traj_num += 1
            i += 1
        return inputs, traj_d

    def _preprocess_traj(self,inputs):
        """Given inputs - a path to a trajectory, corresponding topology file,
        an output trajectory number, and an integer indicating which variant
        simulation a trajectory came from - process the trajectory to be
        stripped to a subset of atoms and aligned to a reference pdb. Write
        to file 1) the resulting .xtc trajectory file, 2) mean center of mass of
        each atom in the trajectory, and 3) an indicator array to indicate which
        variant each simulation frame came from.

        Returns
        -------
        n : int
            Number of simulation frames in the trajectory
        """
        traj_fn, top_fn, traj_num, var_ind = inputs
        
        # 1. Loading the trajectory
        print(f"\nProcessing traj {traj_num} ({traj_fn}) with top {top_fn} ...", end='')
        if type(self.stride) == np.ndarray:
            traj = md.load(traj_fn, top=top_fn, stride=self.stride[var_ind])
        else:
            traj = md.load(traj_fn, top=top_fn, stride=self.stride)
        print(f" --> {traj.n_frames} simulation frames!")
        
        # 2. Stripping the trajectory 
        print(f'    Stripping traj {traj_num} to a subset of the selected atoms ...')
        if isinstance(self.atom_sel, list) or type(self.atom_sel)==np.ndarray:
             inds = self.atom_sel[var_ind]
        else:   # The case that self.atom_sel is a string for mdtraj atom selection
             inds = traj.top.select(self.atom_sel)
        traj = traj.atom_slice(inds+5)

        # Compare the selected atoms of master.pdb and the trajectory (in case that the provided pdb_ref do not have correct atoms selected)
        if self.ref_provided is True:
            d1 = self.master.topology.to_dataframe()[0]
            d2 = traj.topology.to_dataframe()[0]
        
            if len(d1) != len(d2):
                raise ImproperlyConfigured(
                f'The provided reference PDB structure does not have the same number of atoms as the selected atoms used in the trajectory!')
        
            a1 = [d1.iloc[i]['name'] for i in range(len(d1))]
            a2 = [d2.iloc[i]['name'] for i in range(len(d2))]
            if a1 != a2:
                raise ImproperlyConfigured(
                f'The provided reference PDB structure does not have the same atom types as the ones in the trajectory!')

            r1 = [d1.iloc[i]['resName'] for i in range(len(d1))]
            r2 = [d2.iloc[i]['resName'] for i in range(len(d2))]
            if r1 != r2:
                raise ImproperlyConfigured(
                f'The provided reference PDB structure does not have the same residues as the ones in the trajectory!')

        # 3. Aligning the trajectory
        print(f'    Superposing traj {traj_num} to master.pdb ...')  # All the trajectores are superposed against master.pdb
        traj = traj.superpose(self.master, parallel=False)
        # rmsd = md.rmsd(traj, self.master, parallel=False)
        
        # 4. save traj and its center of mass
        print(f'    Saving the aligned trajectory of traj {traj_num} as {str(traj_num).zfill(6)}.xtc" ...')
        new_traj_fn = os.path.join(self.xtc_dir, str(traj_num).zfill(6) + ".xtc")  # ./whitened_data/aligned_xtcs/*.xtc (e.g. 000000.xtc, 000001.xtc, ...)
        traj.save(new_traj_fn)

        # 5. Getting/Saving CM
        n = len(traj)  # Number of frames in the trajectory
        print(f'    Calculating and saving CM of traj {traj_num} ...')
        # Note: traj.xyz.shape = (n_frames, n_atoms, 3) (.mean(axis=0) averages over all frames)
        cm = traj.xyz.astype(np.double).reshape((n, 3*traj.top.n_atoms)).mean(axis=0)
        new_cm_fn = os.path.join(self.xtc_dir, "cm" + str(traj_num).zfill(6) + ".npy")
        np.save(new_cm_fn, cm)
        
        # 6. Saving indicators
        print(f'    Saving indicators for traj {traj_num} ...')
        indicators = var_ind * np.ones(n)  # indicate which variant the trajectory is of
        indicators_fn = os.path.join(self.indicator_dir, str(traj_num).zfill(6) + ".npy")
        np.save(indicators_fn, indicators)
        
        return n

    def preprocess_traj(self,inputs):
        """Strip all trajectories to a subset of atoms and align to a
           reference pdb. Also, calculate and write out the mean center 
           of mass of all atoms across all trajectories. Will write out 
           new trajectory (.xtc files) and corresponding "inidcator" lists
           to indicate which variant simulation each data frame came from.

        Parameters
        ---------
        inputs : array-like, shape=(n_trajectories,4)
            For each trajectory there should be 1) path to trajectory,
            2) path to corresponding topology file, 3) output trajectory
            number, and 4) integer indicating which variant the trajectory
            came from. 
        """
        # If you use 20 cores to load in 20 trajectories at a time
        # make sure the node has enough memory for all 20 trajectories
        # or your job might stall without crashing :/
        n_cores = mp.cpu_count()
        print(f"\nNote: {n_cores} cores will be used.")
        pool = mp.Pool(processes=n_cores)  # use n_cores to run the process
        f = functools.partial(self._preprocess_traj)
        result = pool.map_async(f, inputs)
        result.wait()
        traj_lens = result.get()
        traj_lens = np.array(traj_lens, dtype=int)
        pool.close()

        traj_len_fn = os.path.join(self.outdir, "traj_lens.npy")
        np.save(traj_len_fn, traj_lens)
        traj_fns = get_fns(self.xtc_dir, "*.xtc")  # list of paths of the aligned xtc files
        cm_fns = get_fns(self.xtc_dir, "cm*.npy")  # list of paths of the cm00000*.npy files
        # n_traj = len(traj_fns)  # number of trajectories
        # print("\n Found %d trajectories" % n_traj)
        cm = np.zeros(self.n_feats, dtype=np.double)
        for i, cm_fn in enumerate(cm_fns):
            d = np.load(cm_fn)
            cm += traj_lens[i] * d
        cm /= traj_lens.sum()
        cm_fn = os.path.join(self.outdir, "cm.npy")
        np.save(cm_fn, cm)

    def traj2samples(self):
        """For every trajectory frame, write out a PyTorch tensor file,
        which will be used as input to the DiffNet"""

        traj_fns = get_fns(self.xtc_dir, "*.xtc")
        cm_fn = os.path.join(self.outdir, "cm.npy")
        cm = np.load(cm_fn)
        ex_dir = os.path.join(self.outdir,"data")  # /whitened_data/data
        i = 0
        for t in traj_fns:
            traj = md.load(t,top=self.master)
            data = traj.xyz.astype(np.double).reshape((len(traj),3*self.master.top.n_atoms))
            data -= cm
            for d in data:
                frame = torch.from_numpy(d).type(torch.FloatTensor)
                torch.save(frame,os.path.join(ex_dir,"ID-%s.pt" % i))
                i+=1

    def run(self):
        """Process raw trajectory data to select a subset of atoms and align all
       frames to a reference pdb. Results in a directory structure that the
       training relies on.
       """
        inputs, traj_d = self.make_traj_list()
        traj_d_path = os.path.join(self.outdir,"traj_dict.pkl")
        pickle.dump(traj_d, open(traj_d_path, "wb" )) 
        mkdir(self.xtc_dir)                        # ./whitened_data/aligned_xtcs
        mkdir(self.indicator_dir)                  # ./whitened_data/indicators
        mkdir(os.path.join(self.outdir,"data"))    # ./whitened_data/data
        self.preprocess_traj(inputs)
        self.traj2samples()

class WhitenTraj: 
    """Normalize the trajectories with a data whitening procedure [1] that
       removes covariance between atoms in trajectories.

       Parameters
       ---------
       data_dir : str
           Path to a directory that contains a topology file, a file with
           the mean center of mass of all atoms across all trajectories, 
           and a dir named "aligned_xtcs" with all aligned trajectories.

       References
       ----------
       [1] Wehmeyer C, Noé F. Time-lagged autoencoders: Deep learning of 
       slow collective variables for molecular kinetics. J Chem Phys. 2018. 
       doi:10.1063/1.5011399
    """
    def __init__(self,data_dir):
        self.data_dir = data_dir
        self.xtc_dir = os.path.join(self.data_dir,"aligned_xtcs")
        self.cm = np.load(os.path.join(self.data_dir,"cm.npy"))

    def get_c00(self, coords, cm, traj_num):
        """Calculates the covariance matrix.

        Parameters
        ----------
        coords : np.ndarray, shape=(n_frames,3*n_atoms)
            XYZ coordinates of a trajectory.
        cm : np.ndarray, shape=(3*n_atoms,)
            Avg. center of mass of each atom across all trajectories.
        traj_num : integer
            Used to name the covariance matrix we are going to write
            out for a trajectory.
        """
        coords -= cm
        #n_coords = coords.shape[0]
        #norm_const = 1.0 / n_coords
        #c00 = np.einsum('bi,bo->io', coords, coords)
        #matmul is faster
        c00 = np.matmul(coords.transpose(),coords) # cov[X, X]=E[(X-mu_x)(X-mu_x)^T], here is (X-mu_x)(X-mu_x)^T
        assert isinstance(c00.flat[0], np.double)
        np.save(os.path.join(self.xtc_dir, "cov"+traj_num+".npy"),c00)  # e.g. ./whitened_data/aligned_xtcs/cov000000.npy Note that these files will be deleted then after c00.npy is made.

    def _get_c00_xtc(self, xtc_fn, top, cm):
        """Reshape MDTraj Trajectory item into an array of XYZ
           coordinates and then call other function to calculate
           covariance matrix, c00.

        Parameters
        ----------
        xtc_fn : str
            Path to trajectory
        top : md.Trajectory object
            Topology corresponding to the trajectory
        cm : np.ndarray, shape=(3*n_atoms,)
            Avg. center of mass of each atom across all trajectories.

        Returns
        -------
        n : int
            Number of data frames in the trajectory
        """
        traj = md.load(xtc_fn, top=top)
        traj_num = xtc_fn.split("/")[-1].split(".")[0]  # e.g. 000000, 000001, ...
        n = len(traj)
        n_atoms = traj.top.n_atoms
        coords = traj.xyz.astype(np.double).reshape((n, 3 * n_atoms))
        self.get_c00(coords,cm,traj_num)
        return n

    def get_c00_xtc_list(self, xtc_fns, top, cm, n_cores):
        """Calculate the covariance matrix across all trajectories.

        Parameters
        ----------
        xtc_fn : list of str's
            Paths to trajectories.
        top : md.Trajectory object
            Topology corresponding to the trajectories
        cm : np.ndarray, shape=(3*n_atoms,)
            Avg. center of mass of each atom across all trajectories.
        n_cores : int
            Number of threads to parallelize task across.

        Returns
        -------
        c00 : np.ndarray, shape=(n_atoms*3,n_atoms*3)
            Covariance matrix across all trajectories
        """
        pool = mp.Pool(processes=n_cores)
        f = functools.partial(self._get_c00_xtc, top=top, cm=cm)
        result = pool.map_async(f, xtc_fns)
        result.wait()
        r = result.get()
        pool.close()        
        
        c00_fns = np.sort(glob.glob(os.path.join(self.xtc_dir, "cov*.npy")))
        c00 = np.sum(np.load(c00_fn) for c00_fn in c00_fns)
        c00 /= sum(r)
        assert isinstance(c00.flat[0], np.double)  # c00.flat[0] is just the first element in c00
        return c00

    def get_wuw_mats(self, c00):
        """Calculate whitening matrix and unwhitening matrix.
           Method adapted from deeptime (https://github.com/markovmodel/deeptime/blob/master/time-lagged-autoencoder/tae/utils.py)

        Parameters
        ----------
        c00 : np.ndarray, shape=(n_atoms*3,n_atoms*3)
            Covariance matrix

        Returns
        -------
        uwm : np.ndarray, shape=(n_atoms*3,n_atoms*3)
            unwhitening matrix
        wm : np.ndarray, shape=(n_atoms*3,n_atoms*3)
            whitening matrix
        """
        # Previous implementation
        # uwm = sqrtm(c00).real
        # wm = inv(uwm).real
        # return uwm, wm

        # Updated implementation
        # e, v = torch.symeig(torch.from_numpy(c00).double(), eigenvectors=True)
        # torch.symeig is deprecated in favor of torch.linalg.eigh()
        e, v = torch.linalg.eigh(torch.from_numpy(c00).double(), UPLO='U')
        
        # In valid covariance matrix the smallest eigenvalue should be positive
        # because the covariance matrix is a positive semidefinite matrix
        # https://stats.stackexchange.com/questions/52976/is-a-sample-covariance-matrix-always-symmetric-and-positive-definite
        assert torch.min(e) > 0.0
        d = torch.diag(1.0 / torch.sqrt(e))
        wm = torch.mm(torch.mm(v, d), v.t())
        return inv(wm.numpy()), wm.numpy()

    def apply_unwhitening(self, whitened, uwm, cm):
        """ Apply unwhitening to XYZ coordinates.

        Parameters
        ----------
        whitened : np.ndarray, shape=(n_frames,3*n_atoms)
            Whitened XYZ coordinates of a trajectory.
        uwm : np.ndarray, shape=(n_atoms*3,n_atoms*3)
            Unwhitening matrix
        cm : np.ndarray, shape=(3*n_atoms,)
            Avg. center of mass of each atom across all trajectories.

        Returns
        -------
        coords : np.ndarray, shape=(n_frames,3*n_atoms)
            XYZ coordinates of a trajectory.
        """
        # multiply each row in whitened by c00_sqrt
        coords = np.einsum('ij,aj->ai', uwm, whitened) # np.transpose(np.inner(uwm, whitened))
        coords += cm
        return coords

    def apply_whitening(self, coords, wm, cm):
        """ Apply whitening to XYZ coordinates.

        Parameters
        ----------
        coords : np.ndarray, shape=(n_frames,3*n_atoms)
            XYZ coordinates of a trajectory.
        wm : np.ndarray, shape=(n_atoms*3,n_atoms*3)
            whitening matrix
        cm : np.ndarray, shape=(3*n_atoms,)
            Avg. center of mass of each atom across all trajectories.

        Returns
        -------
        whitened : np.ndarray, shape=(n_frames,3*n_atoms)
            Whitened XYZ coordinates of a trajectory.
        """
        # multiply each row in coords by inv_c00
        whitened = np.einsum('ij,aj->ai', wm, coords)
        return whitened

    def _apply_whitening_xtc_fn(self, xtc_fn, top, outdir, wm, cm):
        """Apply data whitening to a trajectory file


        Parameters
        ----------
        xtc_fn : str
            Path to trajectory
        top : md.Trajectory object
            Topology corresponding to the trajectories
        outdir : str
            Directory to output whitened trajectory
        wm : np.ndarray, shape=(n_atoms*3,n_atoms*3)
            whitening matrix
        cm : np.ndarray, shape=(3*n_atoms,)
            Avg. center of mass of each atom across all trajectories.
        """
        print(f"Performing whitening transformation for {xtc_fn} ...")
        traj = md.load(xtc_fn, top=top)

        n = len(traj)
        n_atoms = traj.top.n_atoms
        coords = traj.xyz.reshape((n, 3 * n_atoms))
        coords -= cm
        whitened = self.apply_whitening(coords, wm, cm)
        dir, fn = os.path.split(xtc_fn)
        new_fn = os.path.join(outdir, fn)
        traj = md.Trajectory(whitened.reshape((n, n_atoms, 3)), top)
        traj.save(new_fn)

    def apply_whitening_xtc_dir(self,xtc_dir, top, wm, cm, n_cores, outdir):
        """Apply data whitening parallelized across many trajectories

        Parameters
        ----------
        xtc_dir : str
            Directory that contains the trajectories of interest
        top : md.Trajectory object
            Topology corresponding to the trajectories
        outdir : str
            Directory to output whitened trajectory
        wm : np.ndarray, shape=(n_atoms*3,n_atoms*3)
            whitening matrix
        cm : np.ndarray, shape=(3*n_atoms,)
            Avg. center of mass of each atom across all trajectories.
        n_cores : int
            Number of threads to parallelize task across.
        """
        xtc_fns = np.sort(glob.glob(os.path.join(xtc_dir, "*.xtc")))

        pool = mp.Pool(processes=n_cores)
        f = functools.partial(self._apply_whitening_xtc_fn, top=top, outdir=outdir, wm=wm, cm=cm)
        pool.map(f, xtc_fns)
        pool.close()

    def run(self):
        """Whiten existing processed trajectory data in self.data_dir
           to calculate and write out a covariance matrix (c00.npy), a
           whitening matrix (wm.npy) and an unwhitening matrix (uwm.npy).
        """
        outdir = self.data_dir
        whitened_dir = os.path.join(outdir,"whitened_xtcs")
        mkdir(whitened_dir)
        n_cores = mp.cpu_count()
        traj_fns = get_fns(self.xtc_dir, "*.xtc")  # aligned xtc files
        master = md.load(os.path.join(outdir,"master.pdb"))

        # 1. Calculate and save the covariance matrix
        print('Calculating and saving the covariance matrix ...')
        c00 = self.get_c00_xtc_list(traj_fns, master.top, self.cm, n_cores)
        c00_fn = os.path.join(outdir,"c00.npy")
        np.save(c00_fn, c00)

        c00_fns = np.sort(glob.glob(os.path.join(self.xtc_dir, "cov*.npy")))
        for fn in c00_fns:
            os.remove(fn)
            
        # 2. Calculate and save the whitening/unwhitening matrices
        print('Calculating and saving the whitening/unwhitening matrices ...')
        uwm, wm = self.get_wuw_mats(c00)
        uwm_fn = os.path.join(outdir, "uwm.npy")
        np.save(uwm_fn, uwm)
        wm_fn = os.path.join(outdir, "wm.npy")
        np.save(wm_fn, wm)
        #self.apply_whitening_xtc_dir(self.myNav.xtc_dir, master.top, wm, self.cm, n_cores, whitened_dir)
