import os
import pandas as pd
import heartpy as hp
import scipy.io as sio
import numpy as np
from scipy.signal import welch, butter, lfilter
import json

gold_file_path_list = []
orignal_seq_file_list = []

file_list = os.listdir("/home/dejavu/Code/PhysBench/deblur_rppg")

all_gold_list = []
all_origin_pred_list = []

def get_hr(y, sr=30, min=30, max=180):
    p, q = welch(y, sr, nfft=1e5/sr, nperseg=np.min((len(y)-1, 256)))
    return p[(p>min/60)&(p<max/60)][np.argmax(q[(p>min/60)&(p<max/60)])]*60

for _ in file_list:
    subject, subject_num = temp = _[:-4].split('_')
    orignal_seq_file_list.append(os.path.join("/home/dejavu/Code/PhysBench/deblur_rppg",_))
    gold_file_path_list.append(os.path.join("/big_data1/MMPD", "subject"+subject[1:], 
                                            subject+"_"+subject_num+".mat"))
# print(gold_file_path_list)
for file in orignal_seq_file_list:
    rppg_pred = pd.read_csv(file)[' BVP'].to_list()
    num = 450
    count = len(rppg_pred) // num
    for i in range(4):
        rppg_clip = rppg_pred[i*450 : i*450+450]
        all_origin_pred_list.append(get_hr(rppg_clip, 30))

for file in gold_file_path_list: 
    rppg_pred = sio.loadmat(file)['GT_ppg'][0]
    for i in range(4):
        rppg_clip = rppg_pred[i*450 : i*450+450]
        all_gold_list.append(get_hr(rppg_clip, 30))
        
with open('gold_data.json', 'w') as file:
    json.dump(all_gold_list, file)
    
with open('deblur_pred_data.json', 'w') as file:
    json.dump(all_origin_pred_list, file)