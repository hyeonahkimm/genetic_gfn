import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import torch
from mol_mdp_ext import MolMDPExtended, BlockMoleculeDataExtended
import time
import threading
from tqdm import tqdm
from botorch.utils.multi_objective.hypervolume import Hypervolume

class Dataset:

    def __init__(self, args, bpath, oracle, device):
        self.test_split_rng = np.random.RandomState(142857)
        self.train_rng = np.random.RandomState(int(time.time()))
        self.train_mols = []
        self.test_mols = []
        self.all_mols = []
        self.train_mols_map = {}

        self.mdp = MolMDPExtended(bpath)
        self.mdp.post_init(device, args.proxy_repr_type, include_nblocks=args.include_nblocks)
        self.mdp.build_translation_table()
        if args.floatX == 'float64':
            self.mdp.floatX = torch.double
        else:
            self.mdp.floatX = torch.float
        self.mdp._cue_max_blocks = args.max_blocks
        self.max_blocks = args.max_blocks
        self.oracle = oracle
        self._device = device
        self.seen_molecules = set()
        self.stop_event = threading.Event()

        self.target_norm = [-8.6, 1.10]  # for dockerscore

        self.hypervolume = Hypervolume(ref_point=torch.zeros(len(args.objectives)))

    def load_h5(self, path, test_ratio=0.1, num_init_examples=None):
        import json
        columns = ["smiles", "dockscore","blockidxs", "slices", "jbonds", "stems"]
        store = pd.HDFStore(path, 'r')
        df = store.select('df')
        # Pandas has problem with calculating some stuff on float16
        df.dockscore = df.dockscore.astype("float64")
        for cl_mame in columns[2:]:
            df.loc[:, cl_mame] = df[cl_mame].apply(json.loads)

        test_idxs = self.test_split_rng.choice(
            len(df), int(test_ratio * len(df)), replace=False)

        split_bool = np.zeros(len(df), dtype=bool)
        split_bool[test_idxs] = True
        self.scores = []
        self.smis = []
        import pdb; pdb.set_trace()
        for i in tqdm(range(len(df))):
            m = BlockMoleculeDataExtended()
            for c in range(1, len(columns)):
                setattr(m, columns[c], df.iloc[i, c - 1])
            m.blocks = [self.mdp.block_mols[i] for i in m.blockidxs]
            if len(m.blocks) > self.max_blocks:
                continue
            m.numblocks = len(m.blocks)
            m.score = self.oracle.get_score([m])
            self.scores.append(m.score)
            self.smis.append(m.smiles)
            self.all_mols.append(m)
            if split_bool[i]: 
                self.test_mols.append(m)
            else:
                self.train_mols.append(m)
            if len(self.train_mols)+len(self.test_mols) >= num_init_examples:
                break
        store.close()

        print("Sampling initial {} molecules from all {} molecules...".format(
            num_init_examples, len(split_bool)))
        print(len(self.train_mols), 'train mols')
        print(len(self.test_mols), 'test mols')

    def r2r(self, dockscore=None, normscore=None):
        if dockscore is not None:
            normscore = 4-(min(0, dockscore) -
                           self.target_norm[0])/self.target_norm[1]
        normscore = max(0.1, normscore)
        return (normscore/1) ** 1

    def _get(self, i, dset):
        return [(dset[i], dset[i].score)]

    def sample(self, n):
        eidx = np.random.randint(0, len(self.train_mols), n)
        samples = sum((self._get(i, self.train_mols) for i in eidx), [])

        return zip(*samples)

    def sample2batch(self, mb):
        s, r = mb
        s = self.mdp.mols2batch([self.mdp.mol2repr(i) for i in s])
        r = torch.tensor(pd.DataFrame.from_dict(
            r).values, device=self._device).float()
        return (s, r)

    def iterset(self, n, mode):
        if mode == 'test':
            dset = self.test_mols
        elif mode == 'train':
            dset = self.train_mols

        N = len(dset)
        for i in range(int(np.ceil(N/n))):
            samples = sum((self._get(j, dset)
                          for j in range(i*n, min(N, (i+1)*n))), [])
            yield self.sample2batch(zip(*samples))

    def add_samples(self, batch):
        picked_mols, scores, picked_smis = batch

        for m in picked_mols:
            if np.random.uniform() < (1/10):
                self.test_mols.append(m)
            else:
                self.train_mols.append(m)
            self.all_mols.append(m)
            
        self.scores += scores
        self.smis += [smis[-1] for smis in picked_smis]
        
        self.stop_event.clear()

    def compute_hypervolume(self):
        scores = torch.tensor(pd.DataFrame.from_dict(self.scores).values)
        volume = self.hypervolume.compute(scores)

        return volume
    
    def start_samplers(self, n, mbsize):
        self.ready_events = [threading.Event() for i in range(n)]
        self.resume_events = [threading.Event() for i in range(n)]
        self.results = [None] * n
        def f(idx):
            while not self.stop_event.is_set():
                try:
                    self.results[idx] = self.sample2batch(self.sample(mbsize))
                except Exception as e:
                    print("Exception while sampling:")
                    print(e)
                    self.sampler_threads[idx].failed = True
                    self.sampler_threads[idx].exception = e
                    self.ready_events[idx].set()
                    break
                self.ready_events[idx].set()
                self.resume_events[idx].clear()
                self.resume_events[idx].wait()
        self.sampler_threads = [threading.Thread(target=f, args=(i,)) for i in range(n)]
        [setattr(i, 'failed', False) for i in self.sampler_threads]
        [i.start() for i in self.sampler_threads]
        round_robin_idx = [0]
        def get():
            while True:
                idx = round_robin_idx[0]
                round_robin_idx[0] = (round_robin_idx[0] + 1) % n
                if self.ready_events[idx].is_set():
                    r = self.results[idx]
                    self.ready_events[idx].clear()
                    self.resume_events[idx].set()
                    return r
                elif round_robin_idx[0] == 0:
                    time.sleep(0.001)
        return get

    def stop_samplers_and_join(self):
        self.stop_event.set()
        if hasattr(self, 'sampler_threads'):
          while any([i.is_alive() for i in self.sampler_threads]):
            [i.set() for i in self.resume_events]
            [i.join(0.05) for i in self.sampler_threads]