
import glob
import json
import os
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
import time

import src.frame_extract as frame_extract
from unicodedata import name

import cv2

# TODO: adapt to new annotation database.

# padding of leading zeros on video names
VIDEO_NAME_ZERO_PADDING = 4
# padding of leading zeros on static images names
IMG_NAME_ZERO_PADDING = 5
# Static Paths
ROOT_DIR = Path(".").parent.resolve()
# Video prefix
VIDEO_PREFIX = "video_"
# Directory of cropped images
CROPPED_DIR = "cropped"
# how much padding (%) to leave around bounding box when cropping images
BORDER_PADDING = 0.3

def get_frames(video_dirpath, video_name, output_dir_path):
    video_path = str(video_dirpath / video_name) + ".mp4"
    frame_extract.video_to_frames(video_path=video_path, frames_dir=output_dir_path, overwrite=True, every=1, chunk_size=50)

def get_and_crop_images(jaad_db, jaad_obj):
    
    
    
    counter = 0
    time_tracker = 0
    for video_name in jaad_db.keys():
        t0 = time.time()
        # set required paths
        image_dir = Path(jaad_obj._images_path) / video_name
        # if image_dir doesnt exist, create it
        Path.mkdir(image_dir, exist_ok=True, parents=True)    
        
        crops_metadata_dict = dict()
        
        # if crops metadata file exists, that means that the crop images
        # already exists. Don't repeat the process again in that case.
        if os.path.exists(crops_metadata_filepath):
            print(f"\nCrops metadata for {video_name} already exists, skipping process")
        else:
            get_frames(jaad_obj._clips_path, video_name, image_dir)
            with open(metadata_filepath, "r") as f:
                data = json.load(f)

            tracks = list(data.keys())

            for track in tracks:
                crops_metadata_dict[track] = dict()
                # DIR to save cropped images
                cropped_im_dir = OP_PROCESSING_DIR / video_name / CROPPED_DIR / track
                Path.mkdir(cropped_im_dir, exist_ok=True, parents=True)

                frames = list(data[track].keys())

                # for each frame theres an image, let's get the image full path for the frame and crop it using the bbox, save it
                # TODO read crop box coordinates from jaad_database
                for frame in frames:
                    print(frame, end='\r', flush=True)
                    frame_filename = str(frame).zfill(IMG_NAME_ZERO_PADDING) + ".png"
                    frame_filepath = image_dir / frame_filename

                    frame_data = data[track][frame]

                    img = cv2.imread(str(frame_filepath))

                    (left, top, right, bottom) = (
                        int(float(frame_data["xtl"])),
                        int(float(frame_data["ytl"])),
                        int(float(frame_data["xbr"])),
                        int(float(frame_data["ybr"])),
                    )

                    left = max(0, left - int(BORDER_PADDING * (right - left)))
                    right = min(1920, right + int(BORDER_PADDING * (right - left)))
                    top = max(0, top - int(BORDER_PADDING * (bottom - top)))
                    bottom = min(1080, bottom + int(BORDER_PADDING * (bottom - top)))

                    crops_metadata_dict[track][frame] = {"left": left, "top": top, "right": right, "bottom": bottom}

                    # crop image
                    cropped_img = img[top:bottom, left:right]
                    (height, width, filters) = cropped_img.shape

                    cv2.imwrite(str(cropped_im_dir / f"{int(frame):0{IMG_NAME_ZERO_PADDING}d}.png"), cropped_img)
            print('\n')
            # delete folder with all the full images to save space
            shutil.rmtree(str(image_dir), ignore_errors=True)
            
            
            with open(crops_metadata_filepath, "w") as f:
                json.dump(crops_metadata_dict, f, indent=4)
            
        counter += 1
        time_diff = time.time() - t0
        time_tracker += time_diff
        mean_time = time_tracker / counter
        print("\n")
        print("Elapsed time: ", time_diff, " seconds")
        print("Time remaining: ", mean_time * (len(queue_path) - counter), " seconds")
        print("Processed ", counter, " out of ", len(queue_path), " videos")
        print("\n")

def infer_clip(openpose_root_dir, output_dir_path, selected_videos, opt_flags):
    """%cd ../../openpose/
    
        !./build/examples/openpose/openpose.bin --image_dir ../pedestrians/openpose_processing/video_0001/cropped/0_1_2b --write_json ../pedestrians/openpose_processing/video_0001/output/json --write_images ../pedestrians/openpose_processing/video_0001/output/images --display 0 
    """
    def build_command(path_to_clip_as_image_folder, flags):
        full_command_list = list()
        
        # path to executable
        bin_path_execute = "./build/examples/openpose/openpose.bin"
        full_command_list.append(bin_path_execute)
        
        # flags to pass to command
        image_args =  ["--image_dir", path_to_clip_as_image_folder]
        
        # add image_args to full_command_list
        full_command_list += image_args
        
        # add flags
        for flag, value in flags.items():

            if value != None:
                full_command_list += ["--" + flag, value]
            else:
                full_command_list += ["--" + flag]
        
        return full_command_list
    
    def run_command(opt_flags=opt_flags, selected_videos=selected_videos):
        paths = get_crop_paths()
        
        os.chdir(openpose_root_dir)
        
        for path in paths:
            video_name, track = get_data_from_path(path)
            if video_name in selected_videos:
            
                built_in_flags = {
                    "write_json": output_dir_path + "/json/" + video_name + "/" + track,
                    "write_images": output_dir_path + "/images/" + video_name + "/" + track,
                }    
                
                Path.mkdir(Path(built_in_flags["write_json"]), exist_ok=True, parents=True)
                Path.mkdir(Path(built_in_flags["write_images"]), exist_ok=True, parents=True)
                
                
                flags = {**built_in_flags, **opt_flags}
                print(f"Inferring {video_name}")
                command_list = build_command(path, flags)
                print("Commands:", command_list)
                subprocess.run(command_list)
                print(f"Inferred {video_name}")
            else:
                print("Skipping ", video_name)
    
    run_command()
        
    
    
    
