import os

tmp = "Z:/rppg/tmp_physbench"
if not os.path.exists(tmp):
    os.makedirs(tmp)

dataset_ccnu = "/N7T/RLAP"
dataset_mmpd = "C:/MMPD"
dataset_pure = "Z:/PURE"
dataset_ubfc_rppg2 = "C:/UBFC-rPPG/DATASET_2"
dataset_ubfc_phys = "C:/UBFC-PHYS"
dataset_scamps = 'C:/scamps_videos'
dataset_cohface = 'C:/cohface'

# Please first generate these data through dataset_process.ipynb.
test_set_CCNU = "Z:/rppg/ccnu_dataset_test.h5"
test_set_CCNU_rPPG = "Z:/rppg/ccnu_rppg_dataset_test.h5"
test_set_PURE = "Z:/rppg/pure_dataset.h5"
test_set_UBFC_rPPG2 = "Z:/rppg/ubfc_rppg2_dataset.h5"
test_set_UBFC_PHYS = "Z:/rppg/ubfc_phys_dataset.h5"
test_set_MMPD = "Z:/rppg/mmpd_dataset.h5"
test_set_COHFACE = "Z:/rppg/cohface_dataset.h5"