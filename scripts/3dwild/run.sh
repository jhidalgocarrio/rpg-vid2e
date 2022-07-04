#! /bin/bash
conda activate rpg_Super-SlowMo
export MPII_ROOT=/media/dani/data/3dwild/mpi_inf_3dhp
export VID2E_ROOT=/home/dani/code/catkin_ws/src/rpg_vid2e
python $VID2E_ROOT/scripts/3dwild/main.py \
--file $MPII_ROOT/S1/Seq1/imageSequence/video_0.avi \
--fps 25 \
--upsample_frames \
--upsample_factor 3 \
--upsample_device cuda:0 \
--n_upsample_frames -1 \
--downsample_factor 4

conda deactivate
python /home/dani/code/catkin_ws/src/rpg_vid2e/scripts/3dwild/main.py \
--file /media/dani/data/3dwild/mpi_inf_3dhp/S1/Seq1/imageSequence/video_0.avi \
--fps 25 \
--upsample_factor 3 \
--convert_events \
--contrast_threshold_pos 0.1 \
--contrast_threshold_neg 0.1 \
--log_eps 0.001 \
--use_log_image \
--renderer_preprocess_gaussian_blur 0 \
--renderer_preprocess_median_blur 0 \
