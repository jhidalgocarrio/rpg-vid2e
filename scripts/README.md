# Convert Video Dataset to Event Dataset
## Requirements
* ROS
* PyTorch >= 1.0

## Installation
    
    git clone --recursive git@github.com:uzh-rpg/rpg_vid2e.git

### Download example dataset
    
    wget rpg.ifi.uzh.ch/datasets/vid2e_example_dataset.zip
    unzip vid2e_example_datset.zip
    rm vid2e_example_dataset.zip

## Run
### Convert Rosbag Dataset to Pandas Dataset

    python scripts/convert_rosbags_to_pandas_dataset.py \
    --event_topic /cam0/events \
    --image_topic /cam0/image_raw \
    --rendering_topic /dvs_rendering \
    --dataset_root rosbag_dataset/ \
    --output_root . \
    --ros_version melodic
    
#### Visualize Sample from Pandas Dataset
    
    python scripts/visualize_pandas_sample.py \
    --dataset_root dataset/ \
    --label garfield \
    --idx 1 \
    --ignore renderings

### Generate High Frame Rate Video
#### Download Model and Super SlowMo

download the checkpoint for the Super-SlowMo model [here](https://drive.google.com/open?id=1IvobLDbRiBgZr3ryCRrWL8xDbMZ-KnpF).

#### Generate High Frame Rate Video Dataset From Pandas Dataset

    python -W ignore scripts/upsample_pandas_dataset.py \
    --dataset_root dataset/ \
    --file_prefix cam0_image \
    --upsampling_factor 3 \
    --device cuda:0 --batch_size 3

## Learning
First create a dataset split from one of the datasets before

    python scripts/prepare_dataset_split.py \
    --dataset_root dataset/ \
    --split 0.5 0.25 0.25 \
    --output_root . 
    
Start learning

    mkdir log
    python -W ignore scripts/flying_events_learning/main.py \
    --dataset_root dataset/ \
    --split_folder split_train_0.5_val_0.25_test_0.25/ \
    --log_dir log/log_1 \
    --device cuda:0 \
    --batch_size 4
    
## Generate Events from Video
In order to generate frames from video, ESIM has to read them from a folder structure. So for his reason, we need to extract
the frames from the pandas dataframes.
    
    python scripts/extract_frames.py \
    --dataset_root dataset/ \
    --num_workers 4 \
    --prefix cam0_image_raw
    
With these extracted frames we can use ESIM to generate events
    
    python scripts/convert_frames_to_events.py \
    --dataset_root dataset/ \
    --num_workers 1 \
    --prefix extracted_cam0_image_raw
    

    