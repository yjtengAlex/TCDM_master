#!/bin/bash
mrconvert -fslgrad /data_new/tyj/low_data/102311_bvecs.txt /data_new/tyj/low_data/102311_bvals.txt  /data_new/tyj/low_data/102311_lr.nii.gz  result/mix/102311_datamix.mif -force
dwi2mask  result/mix/102311_datamix.mif /data_new/tyj/low_data/102311_lr_mask.nii.gz -force
dwi2response tournier  result/mix/102311_datamix.mif  result/response/102311_response.txt -force
dwi2fod csd  result/mix/102311_datamix.mif  result/response/102311_response.txt  result/fod/102311_b1000_90_CSDSH.nii.gz -mask /data_new/tyj/low_data/102311_lr_mask.nii.gz -force
sh2peaks  result/fod/102311_b1000_90_CSDSH.nii.gz  result/peak/102311_b1000_90_peak.nii.gz -force
# ========================================================================================================================
# 1 /20
mrconvert -fslgrad /data_new/tyj/low_data/102816_bvecs.txt /data_new/tyj/low_data/102816_bvals.txt  /data_new/tyj/low_data/102816_lr.nii.gz  result/mix/102816_datamix.mif -force
dwi2mask  result/mix/102816_datamix.mif /data_new/tyj/low_data/102816_lr_mask.nii.gz -force
dwi2response tournier  result/mix/102816_datamix.mif  result/response/102816_response.txt -force
dwi2fod csd  result/mix/102816_datamix.mif  result/response/102816_response.txt  result/fod/102816_b1000_90_CSDSH.nii.gz -mask /data_new/tyj/low_data/102816_lr_mask.nii.gz -force
sh2peaks  result/fod/102816_b1000_90_CSDSH.nii.gz  result/peak/102816_b1000_90_peak.nii.gz -force
# ========================================================================================================================
# 2 /20
mrconvert -fslgrad /data_new/tyj/low_data/104416_bvecs.txt /data_new/tyj/low_data/104416_bvals.txt  /data_new/tyj/low_data/104416_lr.nii.gz  result/mix/104416_datamix.mif -force
dwi2mask  result/mix/104416_datamix.mif /data_new/tyj/low_data/104416_lr_mask.nii.gz -force
dwi2response tournier  result/mix/104416_datamix.mif  result/response/104416_response.txt -force
dwi2fod csd  result/mix/104416_datamix.mif  result/response/104416_response.txt  result/fod/104416_b1000_90_CSDSH.nii.gz -mask /data_new/tyj/low_data/104416_lr_mask.nii.gz -force
sh2peaks  result/fod/104416_b1000_90_CSDSH.nii.gz  result/peak/104416_b1000_90_peak.nii.gz -force
# ========================================================================================================================
# 3 /20
mrconvert -fslgrad /data_new/tyj/low_data/105923_bvecs.txt /data_new/tyj/low_data/105923_bvals.txt  /data_new/tyj/low_data/105923_lr.nii.gz  result/mix/105923_datamix.mif -force
dwi2mask  result/mix/105923_datamix.mif /data_new/tyj/low_data/105923_lr_mask.nii.gz -force
dwi2response tournier  result/mix/105923_datamix.mif  result/response/105923_response.txt -force
dwi2fod csd  result/mix/105923_datamix.mif  result/response/105923_response.txt  result/fod/105923_b1000_90_CSDSH.nii.gz -mask /data_new/tyj/low_data/105923_lr_mask.nii.gz -force
sh2peaks  result/fod/105923_b1000_90_CSDSH.nii.gz  result/peak/105923_b1000_90_peak.nii.gz -force
# ========================================================================================================================
# 4 /20
mrconvert -fslgrad /data_new/tyj/low_data/108323_bvecs.txt /data_new/tyj/low_data/108323_bvals.txt  /data_new/tyj/low_data/108323_lr.nii.gz  result/mix/108323_datamix.mif -force
dwi2mask  result/mix/108323_datamix.mif /data_new/tyj/low_data/108323_lr_mask.nii.gz -force
dwi2response tournier  result/mix/108323_datamix.mif  result/response/108323_response.txt -force
dwi2fod csd  result/mix/108323_datamix.mif  result/response/108323_response.txt  result/fod/108323_b1000_90_CSDSH.nii.gz -mask /data_new/tyj/low_data/108323_lr_mask.nii.gz -force
sh2peaks  result/fod/108323_b1000_90_CSDSH.nii.gz  result/peak/108323_b1000_90_peak.nii.gz -force
# ========================================================================================================================
# 5 /20
mrconvert -fslgrad /data_new/tyj/low_data/685058_bvecs.txt /data_new/tyj/low_data/685058_bvals.txt  /data_new/tyj/low_data/685058_lr.nii.gz  result/mix/685058_datamix.mif -force
dwi2mask  result/mix/685058_datamix.mif /data_new/tyj/low_data/685058_lr_mask.nii.gz -force
dwi2response tournier  result/mix/685058_datamix.mif  result/response/685058_response.txt -force
dwi2fod csd  result/mix/685058_datamix.mif  result/response/685058_response.txt  result/fod/685058_b1000_90_CSDSH.nii.gz -mask /data_new/tyj/low_data/685058_lr_mask.nii.gz -force
sh2peaks  result/fod/685058_b1000_90_CSDSH.nii.gz  result/peak/685058_b1000_90_peak.nii.gz -force
# ========================================================================================================================
# 6 /20
mrconvert -fslgrad /data_new/tyj/low_data/748662_bvecs.txt /data_new/tyj/low_data/748662_bvals.txt  /data_new/tyj/low_data/748662_lr.nii.gz  result/mix/748662_datamix.mif -force
dwi2mask  result/mix/748662_datamix.mif /data_new/tyj/low_data/748662_lr_mask.nii.gz -force
dwi2response tournier  result/mix/748662_datamix.mif  result/response/748662_response.txt -force
dwi2fod csd  result/mix/748662_datamix.mif  result/response/748662_response.txt  result/fod/748662_b1000_90_CSDSH.nii.gz -mask /data_new/tyj/low_data/748662_lr_mask.nii.gz -force
sh2peaks  result/fod/748662_b1000_90_CSDSH.nii.gz  result/peak/748662_b1000_90_peak.nii.gz -force
# ========================================================================================================================
# 7 /20
mrconvert -fslgrad /data_new/tyj/low_data/751348_bvecs.txt /data_new/tyj/low_data/751348_bvals.txt  /data_new/tyj/low_data/751348_lr.nii.gz  result/mix/751348_datamix.mif -force
dwi2mask  result/mix/751348_datamix.mif /data_new/tyj/low_data/751348_lr_mask.nii.gz -force
dwi2response tournier  result/mix/751348_datamix.mif  result/response/751348_response.txt -force
dwi2fod csd  result/mix/751348_datamix.mif  result/response/751348_response.txt  result/fod/751348_b1000_90_CSDSH.nii.gz -mask /data_new/tyj/low_data/751348_lr_mask.nii.gz -force
sh2peaks  result/fod/751348_b1000_90_CSDSH.nii.gz  result/peak/751348_b1000_90_peak.nii.gz -force
# ========================================================================================================================
# 8 /20
mrconvert -fslgrad /data_new/tyj/low_data/756055_bvecs.txt /data_new/tyj/low_data/756055_bvals.txt  /data_new/tyj/low_data/756055_lr.nii.gz  result/mix/756055_datamix.mif -force
dwi2mask  result/mix/756055_datamix.mif /data_new/tyj/low_data/756055_lr_mask.nii.gz -force
dwi2response tournier  result/mix/756055_datamix.mif  result/response/756055_response.txt -force
dwi2fod csd  result/mix/756055_datamix.mif  result/response/756055_response.txt  result/fod/756055_b1000_90_CSDSH.nii.gz -mask /data_new/tyj/low_data/756055_lr_mask.nii.gz -force
sh2peaks  result/fod/756055_b1000_90_CSDSH.nii.gz  result/peak/756055_b1000_90_peak.nii.gz -force
# ========================================================================================================================
# 9 /20
mrconvert -fslgrad /data_new/tyj/low_data/761957_bvecs.txt /data_new/tyj/low_data/761957_bvals.txt  /data_new/tyj/low_data/761957_lr.nii.gz  result/mix/761957_datamix.mif -force
dwi2mask  result/mix/761957_datamix.mif /data_new/tyj/low_data/761957_lr_mask.nii.gz -force
dwi2response tournier  result/mix/761957_datamix.mif  result/response/761957_response.txt -force
dwi2fod csd  result/mix/761957_datamix.mif  result/response/761957_response.txt  result/fod/761957_b1000_90_CSDSH.nii.gz -mask /data_new/tyj/low_data/761957_lr_mask.nii.gz -force
sh2peaks  result/fod/761957_b1000_90_CSDSH.nii.gz  result/peak/761957_b1000_90_peak.nii.gz -force
# ========================================================================================================================
# 10 /20
mrconvert -fslgrad /data_new/tyj/low_data/833148_bvecs.txt /data_new/tyj/low_data/833148_bvals.txt  /data_new/tyj/low_data/833148_lr.nii.gz  result/mix/833148_datamix.mif -force
dwi2mask  result/mix/833148_datamix.mif /data_new/tyj/low_data/833148_lr_mask.nii.gz -force
dwi2response tournier  result/mix/833148_datamix.mif  result/response/833148_response.txt -force
dwi2fod csd  result/mix/833148_datamix.mif  result/response/833148_response.txt  result/fod/833148_b1000_90_CSDSH.nii.gz -mask /data_new/tyj/low_data/833148_lr_mask.nii.gz -force
sh2peaks  result/fod/833148_b1000_90_CSDSH.nii.gz  result/peak/833148_b1000_90_peak.nii.gz -force
# ========================================================================================================================
# 11 /20
mrconvert -fslgrad /data_new/tyj/low_data/837560_bvecs.txt /data_new/tyj/low_data/837560_bvals.txt  /data_new/tyj/low_data/837560_lr.nii.gz  result/mix/837560_datamix.mif -force
dwi2mask  result/mix/837560_datamix.mif /data_new/tyj/low_data/837560_lr_mask.nii.gz -force
dwi2response tournier  result/mix/837560_datamix.mif  result/response/837560_response.txt -force
dwi2fod csd  result/mix/837560_datamix.mif  result/response/837560_response.txt  result/fod/837560_b1000_90_CSDSH.nii.gz -mask /data_new/tyj/low_data/837560_lr_mask.nii.gz -force
sh2peaks  result/fod/837560_b1000_90_CSDSH.nii.gz  result/peak/837560_b1000_90_peak.nii.gz -force
# ========================================================================================================================
# 12 /20
mrconvert -fslgrad /data_new/tyj/low_data/845458_bvecs.txt /data_new/tyj/low_data/845458_bvals.txt  /data_new/tyj/low_data/845458_lr.nii.gz  result/mix/845458_datamix.mif -force
dwi2mask  result/mix/845458_datamix.mif /data_new/tyj/low_data/845458_lr_mask.nii.gz -force
dwi2response tournier  result/mix/845458_datamix.mif  result/response/845458_response.txt -force
dwi2fod csd  result/mix/845458_datamix.mif  result/response/845458_response.txt  result/fod/845458_b1000_90_CSDSH.nii.gz -mask /data_new/tyj/low_data/845458_lr_mask.nii.gz -force
sh2peaks  result/fod/845458_b1000_90_CSDSH.nii.gz  result/peak/845458_b1000_90_peak.nii.gz -force
# ========================================================================================================================
# 13 /20
mrconvert -fslgrad /data_new/tyj/low_data/896778_bvecs.txt /data_new/tyj/low_data/896778_bvals.txt  /data_new/tyj/low_data/896778_lr.nii.gz  result/mix/896778_datamix.mif -force
dwi2mask  result/mix/896778_datamix.mif /data_new/tyj/low_data/896778_lr_mask.nii.gz -force
dwi2response tournier  result/mix/896778_datamix.mif  result/response/896778_response.txt -force
dwi2fod csd  result/mix/896778_datamix.mif  result/response/896778_response.txt  result/fod/896778_b1000_90_CSDSH.nii.gz -mask /data_new/tyj/low_data/896778_lr_mask.nii.gz -force
sh2peaks  result/fod/896778_b1000_90_CSDSH.nii.gz  result/peak/896778_b1000_90_peak.nii.gz -force
# ========================================================================================================================
# 14 /20
mrconvert -fslgrad /data_new/tyj/low_data/898176_bvecs.txt /data_new/tyj/low_data/898176_bvals.txt  /data_new/tyj/low_data/898176_lr.nii.gz  result/mix/898176_datamix.mif -force
dwi2mask  result/mix/898176_datamix.mif /data_new/tyj/low_data/898176_lr_mask.nii.gz -force
dwi2response tournier  result/mix/898176_datamix.mif  result/response/898176_response.txt -force
dwi2fod csd  result/mix/898176_datamix.mif  result/response/898176_response.txt  result/fod/898176_b1000_90_CSDSH.nii.gz -mask /data_new/tyj/low_data/898176_lr_mask.nii.gz -force
sh2peaks  result/fod/898176_b1000_90_CSDSH.nii.gz  result/peak/898176_b1000_90_peak.nii.gz -force
# ========================================================================================================================
# 15 /20
mrconvert -fslgrad /data_new/tyj/low_data/901038_bvecs.txt /data_new/tyj/low_data/901038_bvals.txt  /data_new/tyj/low_data/901038_lr.nii.gz  result/mix/901038_datamix.mif -force
dwi2mask  result/mix/901038_datamix.mif /data_new/tyj/low_data/901038_lr_mask.nii.gz -force
dwi2response tournier  result/mix/901038_datamix.mif  result/response/901038_response.txt -force
dwi2fod csd  result/mix/901038_datamix.mif  result/response/901038_response.txt  result/fod/901038_b1000_90_CSDSH.nii.gz -mask /data_new/tyj/low_data/901038_lr_mask.nii.gz -force
sh2peaks  result/fod/901038_b1000_90_CSDSH.nii.gz  result/peak/901038_b1000_90_peak.nii.gz -force
# ========================================================================================================================
# 16 /20
mrconvert -fslgrad /data_new/tyj/low_data/901442_bvecs.txt /data_new/tyj/low_data/901442_bvals.txt  /data_new/tyj/low_data/901442_lr.nii.gz  result/mix/901442_datamix.mif -force
dwi2mask  result/mix/901442_datamix.mif /data_new/tyj/low_data/901442_lr_mask.nii.gz -force
dwi2response tournier  result/mix/901442_datamix.mif  result/response/901442_response.txt -force
dwi2fod csd  result/mix/901442_datamix.mif  result/response/901442_response.txt  result/fod/901442_b1000_90_CSDSH.nii.gz -mask /data_new/tyj/low_data/901442_lr_mask.nii.gz -force
sh2peaks  result/fod/901442_b1000_90_CSDSH.nii.gz  result/peak/901442_b1000_90_peak.nii.gz -force
# ========================================================================================================================
# 17 /20
mrconvert -fslgrad /data_new/tyj/low_data/979984_bvecs.txt /data_new/tyj/low_data/979984_bvals.txt  /data_new/tyj/low_data/979984_lr.nii.gz  result/mix/979984_datamix.mif -force
dwi2mask  result/mix/979984_datamix.mif /data_new/tyj/low_data/979984_lr_mask.nii.gz -force
dwi2response tournier  result/mix/979984_datamix.mif  result/response/979984_response.txt -force
dwi2fod csd  result/mix/979984_datamix.mif  result/response/979984_response.txt  result/fod/979984_b1000_90_CSDSH.nii.gz -mask /data_new/tyj/low_data/979984_lr_mask.nii.gz -force
sh2peaks  result/fod/979984_b1000_90_CSDSH.nii.gz  result/peak/979984_b1000_90_peak.nii.gz -force
# ========================================================================================================================
# 18 /20
mrconvert -fslgrad /data_new/tyj/low_data/984472_bvecs.txt /data_new/tyj/low_data/984472_bvals.txt  /data_new/tyj/low_data/984472_lr.nii.gz  result/mix/984472_datamix.mif -force
dwi2mask  result/mix/984472_datamix.mif /data_new/tyj/low_data/984472_lr_mask.nii.gz -force
dwi2response tournier  result/mix/984472_datamix.mif  result/response/984472_response.txt -force
dwi2fod csd  result/mix/984472_datamix.mif  result/response/984472_response.txt  result/fod/984472_b1000_90_CSDSH.nii.gz -mask /data_new/tyj/low_data/984472_lr_mask.nii.gz -force
sh2peaks  result/fod/984472_b1000_90_CSDSH.nii.gz  result/peak/984472_b1000_90_peak.nii.gz -force
# ========================================================================================================================
# 19 /20
mrconvert -fslgrad /data_new/tyj/low_data/991267_bvecs.txt /data_new/tyj/low_data/991267_bvals.txt  /data_new/tyj/low_data/991267_lr.nii.gz  result/mix/991267_datamix.mif -force
dwi2mask  result/mix/991267_datamix.mif /data_new/tyj/low_data/991267_lr_mask.nii.gz -force
dwi2response tournier  result/mix/991267_datamix.mif  result/response/991267_response.txt -force
dwi2fod csd  result/mix/991267_datamix.mif  result/response/991267_response.txt  result/fod/991267_b1000_90_CSDSH.nii.gz -mask /data_new/tyj/low_data/991267_lr_mask.nii.gz -force
sh2peaks  result/fod/991267_b1000_90_CSDSH.nii.gz  result/peak/991267_b1000_90_peak.nii.gz -force
# ========================================================================================================================
# 20 /20
tckgen result/fod/102311_b1000_90_CSDSH.nii.gz result/tck/102311_tck.tck -algorithm iFOD2 -select 100000 -seed_image /data_new/tyj/low_data/102311_lr_mask.nii.gz -force
tckgen result/fod/102816_b1000_90_CSDSH.nii.gz result/tck/102816_tck.tck -algorithm iFOD2 -select 100000 -seed_image /data_new/tyj/low_data/102816_lr_mask.nii.gz -force
tckgen result/fod/104416_b1000_90_CSDSH.nii.gz result/tck/104416_tck.tck -algorithm iFOD2 -select 100000 -seed_image /data_new/tyj/low_data/104416_lr_mask.nii.gz -force
tckgen result/fod/105923_b1000_90_CSDSH.nii.gz result/tck/105923_tck.tck -algorithm iFOD2 -select 100000 -seed_image /data_new/tyj/low_data/105923_lr_mask.nii.gz -force
tckgen result/fod/108323_b1000_90_CSDSH.nii.gz result/tck/108323_tck.tck -algorithm iFOD2 -select 100000 -seed_image /data_new/tyj/low_data/108323_lr_mask.nii.gz -force
tckgen result/fod/685058_b1000_90_CSDSH.nii.gz result/tck/685058_tck.tck -algorithm iFOD2 -select 100000 -seed_image /data_new/tyj/low_data/685058_lr_mask.nii.gz -force
tckgen result/fod/748662_b1000_90_CSDSH.nii.gz result/tck/748662_tck.tck -algorithm iFOD2 -select 100000 -seed_image /data_new/tyj/low_data/748662_lr_mask.nii.gz -force
tckgen result/fod/751348_b1000_90_CSDSH.nii.gz result/tck/751348_tck.tck -algorithm iFOD2 -select 100000 -seed_image /data_new/tyj/low_data/751348_lr_mask.nii.gz -force
tckgen result/fod/756055_b1000_90_CSDSH.nii.gz result/tck/756055_tck.tck -algorithm iFOD2 -select 100000 -seed_image /data_new/tyj/low_data/756055_lr_mask.nii.gz -force
tckgen result/fod/761957_b1000_90_CSDSH.nii.gz result/tck/761957_tck.tck -algorithm iFOD2 -select 100000 -seed_image /data_new/tyj/low_data/761957_lr_mask.nii.gz -force
tckgen result/fod/833148_b1000_90_CSDSH.nii.gz result/tck/833148_tck.tck -algorithm iFOD2 -select 100000 -seed_image /data_new/tyj/low_data/833148_lr_mask.nii.gz -force
tckgen result/fod/837560_b1000_90_CSDSH.nii.gz result/tck/837560_tck.tck -algorithm iFOD2 -select 100000 -seed_image /data_new/tyj/low_data/837560_lr_mask.nii.gz -force
tckgen result/fod/845458_b1000_90_CSDSH.nii.gz result/tck/845458_tck.tck -algorithm iFOD2 -select 100000 -seed_image /data_new/tyj/low_data/845458_lr_mask.nii.gz -force
tckgen result/fod/896778_b1000_90_CSDSH.nii.gz result/tck/896778_tck.tck -algorithm iFOD2 -select 100000 -seed_image /data_new/tyj/low_data/896778_lr_mask.nii.gz -force
tckgen result/fod/898176_b1000_90_CSDSH.nii.gz result/tck/898176_tck.tck -algorithm iFOD2 -select 100000 -seed_image /data_new/tyj/low_data/898176_lr_mask.nii.gz -force
tckgen result/fod/901038_b1000_90_CSDSH.nii.gz result/tck/901038_tck.tck -algorithm iFOD2 -select 100000 -seed_image /data_new/tyj/low_data/901038_lr_mask.nii.gz -force
tckgen result/fod/901442_b1000_90_CSDSH.nii.gz result/tck/901442_tck.tck -algorithm iFOD2 -select 100000 -seed_image /data_new/tyj/low_data/901442_lr_mask.nii.gz -force
tckgen result/fod/979984_b1000_90_CSDSH.nii.gz result/tck/979984_tck.tck -algorithm iFOD2 -select 100000 -seed_image /data_new/tyj/low_data/979984_lr_mask.nii.gz -force
tckgen result/fod/984472_b1000_90_CSDSH.nii.gz result/tck/984472_tck.tck -algorithm iFOD2 -select 100000 -seed_image /data_new/tyj/low_data/984472_lr_mask.nii.gz -force
tckgen result/fod/991267_b1000_90_CSDSH.nii.gz result/tck/991267_tck.tck -algorithm iFOD2 -select 100000 -seed_image /data_new/tyj/low_data/991267_lr_mask.nii.gz -force
