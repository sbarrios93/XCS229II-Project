from __future__ import annotations
import glob
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from unicodedata import name
import shutil
import sys

import cv2
from importlib_metadata import pathlib

# padding of leading zeros on video names
VIDEO_NAME_ZERO_PADDING = 4
# padding of leading zeros on static images names
IMG_NAME_ZERO_PADDING = 5
# Static Paths
ROOT_DIR = Path(".").parent.resolve()
DATA_DIR = ROOT_DIR / "data"
ANNOTATIONS_DIR = DATA_DIR / "annotations"

## JAAD Clips path
JAAD_CLIPS_DIR = DATA_DIR / "JAAD_clips"

# static image dir
IMG_DIR = DATA_DIR / "images"
# OPENPOSE_PROCESSING_DIR
OP_PROCESSING_DIR = ROOT_DIR / "openpose_processing"
# Make dir if it doesnt exist, don't complain if it does
OP_PROCESSING_DIR.mkdir(exist_ok=True, parents=True)
# BBOX_METADATA FILENAME
BBOX_METADATA_FILENAME = "bbox_metadata.json"
# Metadata of Crops Filename
CROPS_METADATA_FILENAME = "crops_metadata.json"

# Video prefix
VIDEO_PREFIX = "video_"

# Directory of cropped images
CROPPED_DIR = "cropped"

# how much padding (%) to leave around bounding box when cropping images
BORDER_PADDING = 0.3




def get_queue(
    range_stop=346,
    range_start=0,
    video_prefix=VIDEO_PREFIX,
    video_zero_padding=VIDEO_NAME_ZERO_PADDING,
    annotations_dir=ANNOTATIONS_DIR,
):
    """
    Returns a dict of video names in the form of
    video_name: path to annotation file
    """

    # create list of videos to process using JAAD file formatting video_000\d.mp4
    queue_list = [video_prefix + str(i + 1).zfill(video_zero_padding) for i in range(range_start, range_stop)]

    # get list of all annotation files
    xml_files = glob.glob(str(annotations_dir / "*.xml"))

    # get list of files that appear in queue
    return {
        (xml.split("/")[-1]).split(".")[0]: xml for xml in xml_files if (xml.split("/")[-1]).split(".")[0] in queue_list
    }


def get_annotations():
    def parse_tracks(filepath):
        # here will go the data that will output to json
        tracks_dict = dict()

        # run xml parser
        tree = ET.parse(filepath)
        root = tree.getroot()

        # find all tracks in the xml file that contain the label pedestrian
        tracks = [t for t in root.findall("track") if t.attrib.get("label", None) == "pedestrian"]

        # for each track, find all the bounding boxes and their metadata "items", we also need the id which
        # is the id of each pedestrian
        for i, track in enumerate(tracks):
            # the id is inside the box, so we need to retrieve it later, we start with id=None
            # to check later if id = None, else set it on the first iteration
            id = None
            boxes = track.findall(".//box")
            for box in boxes:
                if id is None:
                    id = box.findall(".//attribute/[@name='id']")[0].text
                    tracks_dict[id] = {}
                occlusion = box.findall(".//attribute/[@name='occlusion']")[0].text
                cross = box.findall(".//attribute/[@name='cross']")[0].text
                items = dict(box.items())
                items["occlusion"] = occlusion
                items["cross"] = cross
                frame = items["frame"]
                tracks_dict[id][frame] = items
        return tracks_dict

    def save_bbox_metadata(path, video_name, name, data):

        path_dir = path / video_name
        filepath_dir = path_dir / name

        # make dir if it doesnt exist, don't complain if it does
        Path.mkdir(path_dir, exist_ok=True, parents=True)

        with open(filepath_dir, "w") as f:
            f.write(json.dumps(data, indent=4))
        return 1

    # start the loop, save the metadata
    def get_save_annotations(queue_range=346):
        queue_path=get_queue(queue_range)
        for video_name, filepath in queue_path.items():
            try:
                data = parse_tracks(filepath)
                save_bbox_metadata(
                    path=OP_PROCESSING_DIR, video_name=video_name, name=BBOX_METADATA_FILENAME, data=data
                )
            except Exception as e:
                print(e)
                print(f"Error processing annotation for {video_name}")
                continue

    get_save_annotations()

def get_frames(video_dirpath, video_name, output_dir_path):
    video_path = str(video_dirpath / video_name) + ".mp4"
    video = cv2.VideoCapture(video_path)
    ok, frame = video.read()
    count = 0
    while ok:
        output_path = str(output_dir_path / f"{count:0{IMG_NAME_ZERO_PADDING}d}.png")
        cv2.imwrite(output_path, frame)
        print('WRITTEN FRAME:',count, end='\r', flush=True)
        count+=1
        ok, frame = video.read()
    video.release()
    print('Done! \n')

def get_and_crop_images(queue_range=346):
    
    queue_path=get_queue(queue_range)
    
    for video_name in queue_path.keys():
        # set required paths
        image_dir = IMG_DIR / video_name
        metadata_filepath = OP_PROCESSING_DIR / video_name / BBOX_METADATA_FILENAME
    
        # if image_dir doesnt exist, create it
        Path.mkdir(image_dir, exist_ok=True, parents=True)    
        

        get_frames(JAAD_CLIPS_DIR, video_name, image_dir)

        
        crops_metadata_filepath = OP_PROCESSING_DIR / video_name / CROPS_METADATA_FILENAME
        crops_metadata_dict = dict()

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

                cv2.imwrite(str(cropped_im_dir / f"{frame}.png"), cropped_img)
        print('\n')
        # delete folder with all the full images to save space
        shutil.rmtree(str(image_dir), ignore_errors=True)
           
        
        with open(crops_metadata_filepath, "w") as f:
            json.dump(crops_metadata_dict, f, indent=4)
