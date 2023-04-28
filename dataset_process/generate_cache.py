import sys
sys.path.append("..")
from utils import *
from multiprocessing import Pool
import tqdm

cores = 8

df = pd.read_csv('../datasets/CCNU_dataset_index.csv')
files_ccnu = df[(df['codec']=='MJPG')&(df['fold']>=0)]['file']

df = pd.read_csv('../datasets/PURE_dataset_index.csv')
files_pure = df['file']

df = pd.read_csv('../datasets/UBFC_rPPG2_dataset_index.csv')
files_ubfc_rppg2 = df['file']

def cache(f, vid, overwrite=False):
    if overwrite or not os.path.exists(f'{tmp}/{hash(f)}.h5'):
        with h5py.File(f'{tmp}/{hash(f)}.h5', 'w') as f:
            boxes, landmarks = generate_vid_labels(vid)
            f.create_dataset('boxes', data=np.array(boxes), compression=0)
            f.create_dataset('landmarks', data=np.array(landmarks), compression=0)

def cache_ccnu(f):
    vid, bvp, ts = loader_ccnu(f)
    cache(loader_ccnu.base+f, vid)

def cache_pure(f):
    vid, bvp, ts = loader_pure(f)
    cache(loader_pure.base+f, vid)

def cache_ubfc_rppg2(f):
    vid, bvp, ts = loader_ubfc_rppg2(f)
    cache(loader_ubfc_rppg2.base+f, vid)

if __name__ == '__main__':
    with Pool(cores) as p:
        #p.map(cache_ccnu, files_ccnu)
        #p.map(cache_pure, files_pure)
        p.map(cache_ubfc_rppg2, files_ubfc_rppg2)