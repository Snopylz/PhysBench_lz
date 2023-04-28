import sys
sys.path.append("..")
from utils import *
from multiprocessing import Pool
import tqdm

cores = 8

df = pd.read_csv('PURE_dataset_index.csv')
files_pure = df['file']

df = pd.read_csv('UBFC_rPPG2_dataset_index.csv')
files_ubfc_rppg2 = df['file']

def cache(f, vid, overwrite=False):
    if overwrite or not os.path.exists(f'{tmp}/{hash(f)}.h5'):
        with h5py.File(f'{tmp}/{hash(f)}.h5', 'w') as _:
            boxes, landmarks = generate_vid_labels(vid)
            _.create_dataset('boxes', data=np.array(boxes), compression=0)
            _.create_dataset('landmarks', data=np.array(landmarks), compression=0)

def cache_pure(f):
    vid, bvp, ts = loader_pure(f)
    cache(loader_pure.base+f, vid)

def cache_ubfc_rppg2(f):
    vid, bvp, ts = loader_ubfc_rppg2(f)
    cache(loader_ubfc_rppg2.base+f, vid)

if __name__ == '__main__':
    with Pool(cores) as p:
        p.map(cache_ubfc_rppg2, files_ubfc_rppg2)
        p.map(cache_pure, files_pure)
    print('finished')