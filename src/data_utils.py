import pickle
import typing
import os
import numpy as np
import yaml
import json
import glob


class JaadDatabase:
    def __init__(
        self,
        jaad_object: object,
        filename: typing.Union[str, bytes, os.PathLike] = "jaad_database.pkl",
        regen=False,
        processed_dirpath="data/processed",
    ) -> None:
        self.jaad_object = jaad_object
        self.filename = filename
        self.regen = regen
        self.original_filepath = os.path.join(self.jaad_object.cache_path, self.filename)
        self.processed_dirpath = processed_dirpath
        self.processed_filepath = os.path.join(self.processed_dirpath, self.filename)
        self.db = None
        self.cropped_run = False

    def run_database_generator(self) -> dict:
        """
        Generate or retrieve a database from the jaad_object.
        """

        if not self.regen:
            try:
                with open(self.original_filepath, "rb") as handle:
                    self.db = pickle.load(handle)
                return self.db
            except FileNotFoundError:
                print("Could not find the database file, generating it now...")
            self.db = self.jaad_object.generate_database()
            return self.db
        else:
            if os.path.exists(self.file_path):
                print("Previous version of database found, deleting old version.")
                os.remove(self.file_path)
                print("Generating database file")
            self.db = self.jaad_object.generate_database()
            return self.db

    def _save_database(self, verbose=True):
        """
        Saves the database to a pickle file in the processed folder .
        """

        if not os.path.exists(self.processed_dirpath):
            os.makedirs(self.processed_dirpath)
        with open(self.processed_filepath, "wb") as handle:
            pickle.dump(self.db, handle)
            if verbose:
                print(f"Database saved to {self.processed_filepath}")

    def _get_bbox(self, video_name: str, pid: str, return_type: str = "array"):
        """get bbox from the database

        Args:
            video_name (str): video name
            pid (str): pedestrian id
            return_type (str, optional): return either array or list. Defaults to 'array'.
        """
        if return_type == "array":
            return np.array(self.db[video_name]["ped_annotations"][pid]["bbox"])
        elif return_type == "list":
            return self.db[video_name]["ped_annotations"][pid]["bbox"]
        else:
            raise ValueError(f"Invalid return_type: {return_type}")
    
    def _get_cropped_box(self, video_name: str, pid: str, return_type: str = "array"):
        """get cropped box from the database
        
        Args:
            video_name (str): video name
            pid (str): pedestrian id
            return_type (str, optional): return either array or list. Defaults to 'array'.
        """
        if not self.cropped_run:
            print("Cropped box not added to database, running add_cropped_bbox() first.")
            self.add_cropped_bbox()
        
        if return_type == "array":
            return np.array(self.db[video_name]["ped_annotations"][pid]["cropped_box"])
        elif return_type == "list":
            return self.db[video_name]["ped_annotations"][pid]["cropped_box"]
        else:
            raise ValueError(f"Invalid return_type: {return_type}")
        
        
    def _expand_bbox(self, arr: np.ndarray, padding: float = 0.3) -> list:
        """
        Structure of the array
        [[xtl, ytl, xbr, ybr], [xtl, ytl, xbr, ybr], ...]
        """
        arr[:, 0] -= (arr[:, 2] - arr[:, 0]) * padding
        arr[:, 1] -= (arr[:, 3] - arr[:, 1]) * padding
        arr[:, 2] += (arr[:, 2] - arr[:, 0]) * padding
        arr[:, 3] += (arr[:, 3] - arr[:, 1]) * padding

        np.clip(arr, 0, None, out=arr)
        np.clip(arr[:, 2], None, 1920, out=arr[:, 2])
        np.clip(arr[0, 3], None, 1080, out=arr[:, 3])

        arr = np.floor(arr).astype(int)
        return arr.tolist()

    def add_cropped_bbox(self, **opts) -> None:
        """
        For each bbox in the jaad_database, include the dimensions of the cropped box
        including the optional padding
        """

        params = {
            "padding": 0.3,
        }
        assert all(k in params for k in opts.keys()), "Wrong option(s)."
        params.update(opts)

        for video_name in self.db:
            for pid in self.db[video_name]["ped_annotations"]:
                bbox_arr = self._get_bbox(video_name, pid, return_type="array")
                self.db[video_name]["ped_annotations"][pid]["cropped_box"] = self._expand_bbox(
                    bbox_arr, params["padding"]
                )
        self.cropped_run = True
        self._save_database()

    def _calculate_keypoints(
        self,
        cropped_box: typing.Union[list, np.ndarray],
        keypoints_arr: np.ndarray,
    ):
        """

        Returns:

        """
        # check that params types are valid:
        assert isinstance(cropped_box, (np.ndarray, list)), "cropped_box must be either a list or a numpy array."
        assert isinstance(keypoints_arr, np.ndarray), "keypoints_arr must be a numpy array."

        # reshape keypoints arr
        keypoints_arr = keypoints_arr.reshape(-1, 3)

        # array has each row as [x, y, confidence]. We need to separate it into [x, y] and [confidence]
        coordinates_arr = keypoints_arr[:, :2]
        confidence_arr = keypoints_arr[:, 2]

        # multiply by (coordinates_arr > 0) to only sum along the non-zero elements
        transformed_coordinates = cropped_box[:2] * (coordinates_arr > 0) + coordinates_arr

        return np.concatenate([transformed_coordinates, confidence_arr[:, np.newaxis]], axis=1)
    
    def _select_best_skeleton(self, skeleton_candidates: list,  bbox):
    
        assert isinstance(skeleton_candidates, list), "skeleton_candidates must be a list."
        
        scores = np.array([])
        for candidate in skeleton_candidates:
            score = np.sum(((candidate[:, :2]>=bbox[:2]) & (candidate[:, :2]<=bbox[2:])).all(1))
            scores = np.append(scores, score)
        
        # choose the best skeleton
        return skeleton_candidates[np.argmax(scores)]
            
    def _parse_keypoints(
        self,
        video_keypoint_filepath,
        video_name,
        pid,
        people_key="people",
        keypoints_subkey="pose_keypoints_2d",
        frame_id: int = None,
    ):

        # check if pass pathname actually exists
        assert os.path.exists(video_keypoint_filepath), "Keypoint file does not exist."

        # load file
        with open(video_keypoint_filepath, "rb") as f:
            content = json.load(f)

        # check that the key 'people' exists in the json file
        assert people_key in content, f"Key {people_key} does not exist in the json file."
        
        # if we didn't pass an integer as frame id, then we need to get it from the file name
        if frame_id is None:
            frame_id = int(video_keypoint_filepath.split("/")[-1].split("_")[0])
        assert isinstance(frame_id, int), "frame_id must be an integer."
        
        # because the frames might not start from 0 (if a person appears after only some frames), we need to get the index of this frame
        frame_ix = self.db[video_name]['ped_annotations'][pid]['frames'].index(frame_id)

        # the way we are bulding the algorithm is with running skeleton inference over cropped areas of the video. Even so, it may be the case that people overlap in a cropped area, making the openpose algorithm output more than one skeleton. In that case, we will have more than one skeleton per person. We need to define which one is the correct.

        # let's check if there's one more than one skeleton per person
        skeleton_count = len(content[people_key])
        
        # get all cropped_boxes
        cropped_boxes = self._get_cropped_box(video_name, pid, return_type="array")

        if skeleton_count == 1:
            # if there's only one skeleton per person, we can just use the first one
            # first convert to np.array
            skeleton_array = np.array(content[people_key][0][keypoints_subkey])
            # run the _calculate_keypoints function
            return self._calculate_keypoints(cropped_boxes[frame_ix], skeleton_array)
        
        elif skeleton_count > 1:
            bboxes = self._get_bbox(video_name, pid, return_type="array")
            skeleton_candidates = []
            candidate_scores = []
            for skeleton in content[people_key]:
                skeleton_array = np.array(skeleton[keypoints_subkey]) # to np.array
                fitted_skeleton = self._calculate_keypoints(cropped_boxes[frame_ix], skeleton_array) # run function
                skeleton_candidates.append(fitted_skeleton) # append candidate to list
                return self._select_best_skeleton(skeleton_candidates, bboxes[frame_ix]) # run function
        else:
             return None 

    def append_keypoints(self, keypoint_dir="keypoints", **opts):
        """
         
        """

        # set optional parameters
        params = {"people_key": "people", "keypoints_subkey": "pose_keypoints_2d"}
        assert all(k in params for k in opts.keys()), "Wrong option(s)."  # check that params exist
        params.update(opts)

        # build keypoint_path, if it doesn't exist, throw an error
        self.keypoint_path = os.path.join(self.processed_dirpath, keypoint_dir)
        if not os.path.exists(self.keypoint_path):
            raise FileNotFoundError(self.keypoint_path)

        # for each video in the database, add the keypoints to the database
        for video_name in self.db:
            for pid in self.db[video_name]["ped_annotations"]:
                video_keypoint_filepath = os.path.join(self.keypoint_path,video_name, "json" , pid)
                if "b" not in pid:
                    continue
                if not os.path.exists(video_keypoint_filepath):
                    # raise FileNotFoundError(video_keypoint_filepath)
                    print(f"{video_keypoint_filepath} does not exist.")
                    continue
                processed_keypoints = []
                
                # get all the keypoint files in the directory
                keypoint_files = glob.glob(video_keypoint_filepath + "/*.json")
                # sort the keypoint files by frame id
                sorted_keypoint_files = sorted(keypoint_files)
                
                for file in sorted_keypoint_files:
                    processed_keypoints.append(self._parse_keypoints(file, video_name, pid))
                self.db[video_name]["ped_annotations"][pid]["skeleton_keypoints"] = processed_keypoints
