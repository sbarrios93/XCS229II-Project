import os
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
import time
import os

import itertools
from src import frame_extract
import yaml

import cv2


class SkeletonPipeline:
    def __init__(self, jaad_db, **opts):
        self.jaad_db = jaad_db
        self.jaad_object = jaad_db.jaad_object
        self.opts = opts

        self._clips_paths = Path(self.jaad_object._clips_path)
        self._images_path = Path(self.jaad_object._images_path)
        self._jaad_path = Path(self.jaad_object._jaad_path)
        self._processed_dirpath = Path(self.jaad_db.processed_dirpath)

        # Load Config File
        self.config_file = "config.yaml"
        self.config_path = Path.absolute(Path(self.config_file))
        self.config = yaml.load(self.config_path.read_text(), Loader=yaml.SafeLoader)
        self.root_path = Path.absolute(Path("."))

        self.params = dict(
            video_name_zero_padding=4,
            img_name_zero_padding=5,
            video_prefix="video_",
            cropped_folder_name="cropped",
            keypoints_folder_name="keypoints",
            openpose_dir=None,
        )
        assert all(k in self.params for k in self.opts.keys()), "Wrong option(s)."
        self.params.update(self.opts)
        self.params.update(self.config)

        assert self.params["openpose_dir"] is not None, "Openpose directory not specified."

    def extract_frames(self, video_name, output_dir_path):
        video_dirpath = self._clips_paths
        video_path = str(video_dirpath / video_name) + ".mp4"
        frame_extract.video_to_frames(
            video_path=video_path,
            frames_dir=output_dir_path,
            overwrite=True,
            every=1,
            chunk_size=50,
        )

    def _get_num_frames(self, video_name):
        return self.jaad_db.db[video_name]["num_frames"]

    def _get_crop_path(self, video_name, pid):
        return Path(self._processed_dirpath) / self.params["cropped_folder_name"] / video_name / pid

    def _get_single_video_images_path(self, video_name):
        return Path(self._images_path) / video_name

    def _get_keypoints_path(self, video_name, pid, type_):
        if type_ == "image":
            return self._processed_dirpath / self.params["keypoints_folder_name"] / video_name / "images" / pid
        elif type_ == "json":
            return self._processed_dirpath / self.params["keypoints_folder_name"] / video_name / "json" / pid

    def _run_crop_pipeline(self, video_name, pid):
        frames_path = self._get_single_video_images_path(video_name)
        cropped_path = self._get_crop_path(video_name, pid)

        assert Path.exists(frames_path), f"Frames path for video {video_name} does not exist."

        # for easier access
        ped_annotations = self.jaad_db.db[video_name]["ped_annotations"]
        frames = ped_annotations[pid]["frames"]
        cropped_boxes = ped_annotations[pid]["cropped_box"]

        for frame in frames:
            frame_ix = frames.index(frame)
            print("Cropping frame {} of {}".format(frame, pid), end="\r", flush=True)

            frame_filename = str(frame).zfill(self.params["img_name_zero_padding"]) + ".png"
            frame_filepath = frames_path / frame_filename
            cropped_filepath = cropped_path / frame_filename

            img = cv2.imread(str(frame_filepath))

            [left, top, right, bottom] = cropped_boxes[frame_ix]

            cropped_img = img[top:bottom, left:right]
            cv2.imwrite(str(cropped_filepath), cropped_img)

        print("\n")
        return None

    def _build_openpose_command(self, flags):
        full_command_list = list()

        # executable
        bin_path_execute = "./build/examples/openpose/openpose.bin"  # path to executable
        full_command_list.append(bin_path_execute)

        # add flags
        for flag, value in flags.items():
            if value != None:
                full_command_list += ["--" + flag, value]
            else:
                full_command_list += ["--" + flag]
        return full_command_list

    def _load_opt_flags(self):
        # load flags from flags.yaml
        opt_flags_path = Path(self.root_path) / "flags.yaml"
        assert Path.exists(opt_flags_path), "flags.yaml not found."
        return yaml.load(opt_flags_path.read_text(), Loader=yaml.SafeLoader)

    def _cropped_image_loader(self, cropped_image_dir, temp_folder="temp"):
        cropped_images = cropped_image_dir.glob("*.png")

        Path.mkdir(cropped_image_dir / temp_folder, parents=True, exist_ok=True)

        while True:
            files = list(itertools.islice(cropped_images, 20))
            if not files:
                break
            for file in files:
                shutil.copy(str(file), str(cropped_image_dir / temp_folder))

    def _run_inference_pipeline(self, video_name, pid, temp_folder="temp"):
        cropped_image_dir = self._get_crop_path(video_name, pid)
        temp_image_dir = cropped_image_dir / temp_folder
        Path.mkdir(temp_image_dir, parents=True, exist_ok=True)

        cropped_images = cropped_image_dir.glob("*.png")

        built_in_flags = {
            "image_dir": str(Path.absolute(temp_image_dir)),
            "write_json": str(Path.absolute(self._get_keypoints_path(video_name, pid, "json"))),
            "write_images": str(Path.absolute(self._get_keypoints_path(video_name, pid, "image"))),
        }

        Path.mkdir(Path(built_in_flags["write_json"]), exist_ok=True, parents=True)
        Path.mkdir(Path(built_in_flags["write_images"]), exist_ok=True, parents=True)

        opt_flags = self._load_opt_flags()

        all_flags = {**built_in_flags, **opt_flags}
        command_list = self._build_openpose_command(flags=all_flags)
        print("Commands:", command_list)
        while True:
            files = list(itertools.islice(cropped_images))
            if not files:
                print(f"Inferred {video_name}")
                shutil.rmtree(str(temp_image_dir))
                return None
            for file in files:
                shutil.copy(str(file), str(cropped_image_dir / temp_folder))
            os.chdir(self.params["openpose_dir"])
            subprocess.run(command_list)
            os.chdir(self.root_path)
            for f in files:
                os.remove(str(temp_image_dir / f.name))

    def _run_extraction_pipeline(self, video_image_dir, video_name):
        # Check if we already have the extracted frames or we need to extract them
        # if image_dir doesnt exist, create it, extract frames
        if not video_image_dir.exists():
            print("Extracting frames for video {}".format(video_name))
            Path.mkdir(video_image_dir, exist_ok=True, parents=True)
            self.extract_frames(video_name, video_image_dir)
        else:
            # if image_dir exists, check if we have the correct number of frames in it
            # get length of frames for video
            frame_count = self._get_num_frames(video_name)
            # get number of frames already in folder
            frame_count_in_folder = len(list(Path(video_image_dir).rglob("*.png")))
            if frame_count > frame_count_in_folder:
                print("Frames for video {} are missing. Extracting frames.".format(video_name))
                shutil.rmtree(video_image_dir, ignore_errors=False, onerror=None)
                Path.mkdir(video_image_dir, exist_ok=True, parents=True)
                self.extract_frames(video_name, video_image_dir)
            else:
                print("Frames for video {} are already extracted.".format(video_name))
        return None

    def _single_video_pipeline_constructor(self, video_name, **opts):

        params = {"cropping": True, "keypoints": True, "force": True}
        assert all(k in params for k in opts.keys()), "Wrong option(s)."
        params.update(opts)

        video_pipeline = {"inf": {}, "crop": {}, "ext": False}

        # set variable for easier access
        ped_annotations_dict = self.jaad_db.db[video_name]["ped_annotations"]

        for pid in ped_annotations_dict.keys():

            video_pipeline["inf"][pid] = False
            video_pipeline["crop"][pid] = False

            if "b" in pid:  # only do this on pedestrians with annotations (string ending in b)

                # SECTION KEYPOINTS
                if params["keypoints"]:  # are we supposed to run the keypoints pipeline?
                    if (len(ped_annotations_dict[pid].get("skeleton_keypoints", [])) < 10) or (
                        params["force"]
                    ):  # do we have less than 10 keypoints? or are we forced to run?
                        video_pipeline["inf"][pid] = True
                #!SECTION

        pid_to_infer = [k for k, v in video_pipeline["inf"].items() if v]  # get list of pids that need to be inferred

        for pid in pid_to_infer:
            # SECTION CROPPING
            if params["cropping"]:  # are we supposed to run the cropping pipeline?
                cropped_path = self._get_crop_path(video_name, pid)
                if not cropped_path.exists():  # are we missing the cropped images?
                    Path.mkdir(cropped_path, parents=True, exist_ok=True)
                    video_pipeline["crop"][pid] = True
                else:
                    num_crop_boxes = len(list(cropped_path.rglob("*.png")))
                    if (num_crop_boxes < 10) or (
                        params["force"]
                    ):  # do we have less than 10 cropped boxes? or are we forced to run?
                        video_pipeline["crop"][pid] = True
            # !SECTION
            if video_pipeline["crop"][pid]:
                video_pipeline["ext"] = True

        return video_pipeline

    def prepare_images(self, run_cropping=True, run_keypoints=True, force=False):

        params = {"cropping": run_cropping, "keypoints": run_keypoints, "force": force}

        print("Preparing images with the following parameters:")
        for item in params.items():
            print(item)

        # if we passed skip_cropping as false, we must have the cropped bounding boxes
        if run_cropping:
            assert (
                self.jaad_db.cropped_run == True
            ), "Can't crop images, cropped bounding boxes have not been extracted. Please run `add_cropped_box` first."

        counter = 0
        time_tracker = 0
        for video_name in self.jaad_db.db.keys():
            t0 = time.time()  # start timer
            # set required paths
            video_image_dir = Path(self._images_path) / video_name

            video_pipeline = self._single_video_pipeline_constructor(video_name, **params)

            crop_pid_list = [k for k, v in video_pipeline["crop"].items() if v]
            infer_pid_list = [k for k, v in video_pipeline["inf"].items() if v]

            if video_pipeline["ext"]:
                self._run_extraction_pipeline(video_image_dir, video_name)
            else:
                print("Extraction for video {} not required.".format(video_name))
            for pid in crop_pid_list:
                self._run_crop_pipeline(video_name, pid)
            for pid in infer_pid_list:
                self._run_inference_pipeline(video_name, pid)
            counter += 1
            time_diff = time.time() - t0
            time_tracker += time_diff
            mean_time = time_tracker / counter
            video_queue_length = len(self.jaad_db.db.keys())
            print("\n")
            print("Elapsed time: ", time_diff, " seconds")
            print("Time remaining: ", mean_time * video_queue_length - counter, " seconds")
            print("Processed ", counter, " out of ", video_queue_length, " videos")
            print("\n")
