"""
Convert Libero Object HDF5 dataset to LeRobot format.

This script converts HDF5 files from the libero_object dataset to LeRobot format.
The HDF5 files contain multiple demonstrations with observations and actions.

Usage:
uv run convert_libero_object_hdf5.py --data_dir /home/ubuntu/openpi/dataset/libero_object

The resulting dataset will be saved to the $HF_LEROBOT_HOME directory.
"""

import os
import shutil
import h5py
import numpy as np
from pathlib import Path
from PIL import Image
import tyro

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

REPO_NAME = "april529/libero_object"  # Name of the output dataset, consistent with original script
OUTPUT_DIR = Path("/home/ubuntu/openpi/dataset/libero_object_lerobot")  # Custom output directory


def resize_image(image_array, target_size=(256, 256)):
    """Resize image from 128x128 to 256x256"""
    if len(image_array.shape) == 3:
        # Single image
        img = Image.fromarray(image_array)
        img_resized = img.resize(target_size, Image.LANCZOS)
        return np.array(img_resized)
    else:
        # Batch of images
        resized_images = []
        for img_array in image_array:
            img = Image.fromarray(img_array)
            img_resized = img.resize(target_size, Image.LANCZOS)
            resized_images.append(np.array(img_resized))
        return np.array(resized_images)


def extract_task_from_filename(filename):
    """Extract task description from filename"""
    # Remove .hdf5 extension and replace underscores with spaces
    task = filename.replace('.hdf5', '').replace('_demo', '').replace('_', ' ')
    return task


def main(data_dir: str, *, push_to_hub: bool = False):
    data_path = Path(data_dir)
    
    # Clean up any existing dataset in the output directory
    output_path = OUTPUT_DIR / REPO_NAME
    if output_path.exists():
        shutil.rmtree(output_path)

    # Create LeRobot dataset with proper features
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        root=OUTPUT_DIR,
        robot_type="panda",
        fps=10,
        features={
            "image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image", 
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (9,),  # robot_states has 9 dimensions
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # Process all HDF5 files in the directory
    hdf5_files = list(data_path.glob("*.hdf5"))
    print(f"Found {len(hdf5_files)} HDF5 files to process")
    
    for hdf5_file in hdf5_files:
        print(f"Processing {hdf5_file.name}...")
        task_description = extract_task_from_filename(hdf5_file.name)
        
        with h5py.File(hdf5_file, 'r') as f:
            # Get all demo keys (demo_0, demo_1, etc.)
            demo_keys = [key for key in f['data'].keys() if key.startswith('demo_')]
            
            for demo_key in demo_keys:
                demo_data = f['data'][demo_key]
                
                # Get the length of the episode
                episode_length = len(demo_data['actions'])
                
                # Extract data for this episode
                actions = demo_data['actions'][:]
                agentview_images = demo_data['obs']['agentview_rgb'][:]
                wrist_images = demo_data['obs']['eye_in_hand_rgb'][:]
                robot_states = demo_data['robot_states'][:]
                
                # Resize images from 128x128 to 256x256
                agentview_resized = resize_image(agentview_images)
                wrist_resized = resize_image(wrist_images)
                
                # Add frames to the dataset
                for i in range(episode_length):
                    dataset.add_frame({
                        "image": agentview_resized[i],
                        "wrist_image": wrist_resized[i], 
                        "state": robot_states[i].astype(np.float32),
                        "actions": actions[i].astype(np.float32),
                        "task": task_description,
                    })
                
                # Save the episode
                dataset.save_episode()
                print(f"  Saved episode {demo_key} with {episode_length} frames")

    print(f"Conversion complete! Dataset saved to {output_path}")
    
    # Optionally push to the Hugging Face Hub
    if push_to_hub:
        dataset.push_to_hub(
            tags=["libero", "panda", "hdf5"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )
        print("Dataset pushed to Hugging Face Hub!")


if __name__ == "__main__":
    tyro.cli(main)