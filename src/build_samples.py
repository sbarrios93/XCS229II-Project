import os
import pickle
import sys
import typing
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import scipy
import yaml
from data.jaad.jaad_data import JAAD
from numpy.lib.stride_tricks import sliding_window_view
from scipy import signal
from tqdm import tqdm
import numpy as np
from src.data_utils import JaadDatabase


class BuildSamples(JaadDatabase):
    def __init__(
        self,
        jaad_object: object,
        filename: typing.Union[str, bytes, os.PathLike] = "jaad_database.pkl",
        regen=False,
        processed_dirpath="data/processed",
    ) -> None:
        super().__init__(jaad_object, filename=filename, regen=regen, processed_dirpath=processed_dirpath)

        self.db = self.read_from_pickle("data/processed/jaad_database.pkl")
        self.sequence_dict = None

    def _generate_raw_sequence(self, image_set="all", **opts):
        """
        Generates pedestrian tracks
        :param image_set: the split set to produce for. Options are train, test, val.
        :param opts:
                'fstride': Frequency of sampling from the data.
                'sample_type': Whether to use 'all' pedestrian annotations or the ones
                                    with 'beh'avior only.
                'subset': The subset of data annotations to use. Options are: 'default': Includes high resolution and
                                                                                        high visibility videos
                                                                        'high_visibility': Only videos with high
                                                                                            visibility (include low
                                                                                            resolution videos)
                                                                        'all': Uses all videos
                'height_rng': The height range of pedestrians to use.
                'squarify_ratio': The width/height ratio of bounding boxes. A value between (0,1]. 0 the original
                                        ratio is used.
                'data_split_type': How to split the data. Options: 'default', predefined sets, 'random', randomly split the data,
                                        and 'kfold', k-fold data split (NOTE: only train/test splits).
                'seq_type': Sequence type to generate. Options: 'trajectory', generates tracks, 'crossing', generates
                                tracks up to 'crossing_point', 'intention' generates tracks similar to human experiments
                'min_track_size': Min track length allowable.
                'random_params: Parameters for random data split generation. (see _get_random_pedestrian_ids)
                'kfold_params: Parameters for kfold split generation. (see _get_kfold_pedestrian_ids)
        :return: Sequence data in the form
        ---------------------------
        keys = dict_keys(['image', 'pid', 'bbox', 'center', 'occlusion', 'intent'])

        {'image': [['../data/jaad/images/video_0001/00000.png',
                    '../data/jaad/images/video_0001/00001.png',
                    '../data/jaad/images/video_0001/00002.png',
                    '../data/jaad/images/video_0001/00003.png',
                    '../data/jaad/images/video_0001/00004.png',...
        'pid': [[['0_1_2b'],
                ['0_1_2b'],
                ['0_1_2b'],
                ['0_1_2b'],
                ['0_1_2b'],
                ['0_1_2b'],
                ['0_1_2b'],
                ['0_1_2b'],
                ['0_1_2b'],
                ['0_1_2b'],
                ['0_1_2b'],
                ['0_1_2b'],
                ['0_1_2b'],...
        ---------------------------
        """
        params = {
            "fstride": 1,
            "sample_type": "all",  # 'beh'
            "subset": "default",
            "height_rng": [0, float("inf")],
            "squarify_ratio": 0,
            "data_split_type": "default",  # kfold, random, default
            "seq_type": "intention",
            "min_track_size": 15,
            "random_params": {"ratios": None, "val_data": True, "regen_data": False},
            "kfold_params": {"num_folds": 5, "fold": 1},
        }
        assert all(k in params for k in opts.keys()), "Wrong option(s)." "Choose one of the following: {}".format(
            list(params.keys())
        )
        params.update(opts)
        return self.jaad_object.generate_data_trajectory_sequence(image_set, **params)

    def _raw_sequence_transformer(self, sequence_data: dict, window, **opts):
        """
        Parses a single sequence from _generate_raw_sequence (by calling generate_sequence_samples)
        :param sequence_data: The sequence data to parse. Comes in the form dict(image, pid, intent)
        :param window: The window size to use
        """

        # unpack dict(image, pid)
        image_list, pid_list, intent = sequence_data["image"], sequence_data["pid"], sequence_data["intent"]

        # PID is the same along the array, then we just need one
        pid = pid_list[0][0]

        # same for video_name
        video_name = image_list[0].split("/")[-2]

        # get rolling windows
        sliding_windows_image_array = sliding_window_view(image_list, window)
        sliding_windows_intent_array = sliding_window_view(intent, window, 0)

        sliding_window_frame_id = [
            pd.Series(arr).str.extract(r"(\d{5})").astype(int).to_numpy().flatten()
            for arr in sliding_windows_image_array
        ]  # array([['../data/jaad/images/video_0001/00000.png', '../data/jaad/images/video_0001/00001.png' => [array([0, 1, 2, 3, 4...]), array([1, 2, 3, 4, 5...])....

        # frames of the pid
        pid_frames = self.db[video_name]["ped_annotations"][pid]["frames"]
        skeletons = self.db[video_name]["ped_annotations"][pid]["skeleton_keypoints"]

        # apply savgol filter
        skeletons = self._apply_savgol_filter(skeletons, window_size=11, polyorder=3)

        array_with_sequences = []
        for array_ix, array in enumerate(
            sliding_window_frame_id
        ):  # an array contain the ids of the frames for that sliding window
            skeletons_single_sequence = []
            confidence_single_sequence = []
            for ix in array:
                try:
                    frame_skeleton = skeletons[pid_frames.index(ix), :, :]
                except IndexError:
                    frame_skeleton = None
                if (frame_skeleton is None) or (np.isnan(frame_skeleton).all()):
                    break
                skeletons_single_sequence.append(frame_skeleton[:, :2])
                confidence_single_sequence.append(frame_skeleton[:, 2])
            assert all(
                len(lst["frame_sequence"]) == len(array) for lst in array_with_sequences
            ), "Skeleton length does not match array length"
            if frame_skeleton is not None:
                data = {
                    "video_name": video_name,
                    "pid": pid,
                    "frame_sequence": array,
                    "skeleton_sequence": skeletons_single_sequence,
                    "confidence": confidence_single_sequence,
                    "intent": sliding_windows_intent_array[array_ix].flatten(),
                }
                array_with_sequences.append(data)

        return array_with_sequences

    def _apply_savgol_filter(self, skeletons, window_size=11, polyorder=3):
        """
        Applies Savgol filter to the array
        :param skeletons: The array of frames containing skeletons
        :param window_size: The window size
        :param polyorder: The polynomial order
        :return: The filtered array
        """
        for i, _arr in enumerate(skeletons):
            if _arr is None:
                _arr = np.empty((25, 3))
                _arr[:] = np.NaN
            skeletons[i] = _arr

        stack = np.vstack(skeletons).reshape(-1, 75)
        for i in range(stack.shape[1]):
            stack[:, i] = signal.savgol_filter(stack[:, i], window_size, polyorder)

        stack = stack.reshape(-1, 25, 3)
        return stack

    def _parse_raw_sequence(self, sequence_data, **opts):
        """

        Args:
            sequence_data (dict): sequence_data from _generate_raw_sequence

        Returns:
            dict: In the shape of:
            [
               [
                   {
                       'video_name': 'video_0001',
                        'pid': '0_1_2b',
                        'frame_sequence': array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14]), -> single array
                        'skeleton_sequence': [array([[1447.6762,  675.5998],...array([[...]])] -> multiple arrays, one per frame of frame_sequence.
                        'confidence': [array([0.796439, 0.76417...],...array([[0.....]])], -> multiple arrays, one per frame of frame_sequence,
                        'intent': array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) -> single array, one element per frame
                    },

                    {
                        'video_name': 'video_0001',
                        'pid': '0_1_2b',
                        'frame_sequence': array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15]), -> NOTE this is what change between elements of the list !NOTE
                        'skeleton_sequence': [array([[1450.5655,  675.1043],...
                    },
                    ...
                ] NOTE this whole list was just for video_0001 and pedestrian_id 0_1_2b...,
                [
                    NEW LIST FOR video_0001 and pedestrian_id 0_1_3b
                ],
                [
                    NEW LIST FOR video_0002 and pedestrian_id ...
                ],
        """

        params = {}
        params.update(opts)

        # if visualize_inner_func is True, we want to print the sequence data to look at it
        print("+" * 30, "Print raw sequence data", sequence_data, "+" * 30, sep="\n") if params[
            "visualize_inner_func"
        ] else None

        sequence_dict = {
            "sequences_count": 0,
            "pos_sequences": 0,
            "sequences": [],
        }

        print("-" * 70, "Parsing sequence data", sep="\n")
        for ith_sample in tqdm(range(len(sequence_data["image"]))):
            sample_data = {
                "image": sequence_data["image"][ith_sample],
                "pid": sequence_data["pid"][ith_sample],
                "intent": sequence_data["intent"][ith_sample],
            }
            parsed_sample = self._raw_sequence_transformer(sequence_data=sample_data, window=params["min_track_size"])
            sequence_dict["sequences"].append(parsed_sample)
            sequence_dict["sequences_count"] += len(parsed_sample)
            for i in parsed_sample:
                if i["intent"][0] == 1:
                    sequence_dict["pos_sequences"] += 1

        return sequence_dict

    def _transform_features(self, sequence_dict):
        # TODO this
        raise NotImplementedError("Transforming features not implemented yet")

    def generate_sequence_samples(
        self, image_set="all", window=15, visualize_inner_func=False, save=True, regen=False, **opts
    ):

        """
        Calls _generate_raw_sequence to generate sequence data.
        :param image_set: the split set to produce for. Options are train, test, val.

        :return: Sequence data
        """
        params = {
            "sample_type": "beh",
            "height_rng": [60, float("inf")],
            "min_track_size": window,
        }
        assert all(k in params for k in opts.keys()), "Wrong option(s)." "Choose one of the following: {}".format(
            list(params.keys())
        )

        # SECTION: LOAD FILE OR RUN FUNCTIONS FOR PARSED SEQUENCES DICT
        # load sample_sequence_dict if exist and regen is false
        sample_sequence_dict_path = Path(self.processed_dirpath) / "sample_sequence_dict.pkl"
        print(f"Function running with parameter regen: {regen}")
        if not regen:
            if sample_sequence_dict_path.exists():
                print(f"Loading saved file from {str(sample_sequence_dict_path)}")
                with open(str(sample_sequence_dict_path), "rb") as file:
                    parsed_sequences = pickle.load(file)
                return parsed_sequences
            else:
                print(f"No previous pickle file found on {str(sample_sequence_dict_path)}", "Generating...", sep="\n")

        if regen:  # Print notice nto user
            print(f"Forcing regeneration of {sample_sequence_dict_path}")

        # run functions
        sequence_data = self._generate_raw_sequence(image_set, **params)
        parsed_sequences = self._parse_raw_sequence(
            sequence_data=sequence_data, **params, visualize_inner_func=visualize_inner_func
        )
        if save:
            with open("data/processed/sample_sequence_dict.pkl", "wb") as file:
                pickle.dump(parsed_sequences, file)
        # !SECTION

        # FIXME: This is not the final return value
        return parsed_sequences  # return attribute


class BodyBuilder:
    def __init__(self, arr: np.ndarray) -> None:
        self.arr = arr

        self.parts = {}
        self.body_yaml_path = "data/helper_data/body_25b_parts.yaml"
        self.body_yaml_dict = self._load_body_yaml_dict()

        self.body_parts_coordinates = self._map_to_dict(self.arr, self.body_yaml_dict)

    def _load_body_yaml_dict(self):
        return yaml.load(Path.read_text(Path(self.body_yaml_path)), Loader=yaml.SafeLoader)

    def _map_to_dict(self, arr, body_yaml_dict):
        parts = {}
        for k, v in body_yaml_dict.items():
            parts[v] = arr[k, :]
        return parts

    def _compute_length_body(self):
        # Length of head
        self.length_Neck_HeadTop = scipy.spatial.distance.euclidean(
            self.body_parts_coordinates["Neck"], self.body_parts_coordinates["HeadTop"]
        )
        self.length_Neck_LEar = scipy.spatial.distance.euclidean(
            self.body_parts_coordinates["Neck"], self.body_parts_coordinates["LEar"]
        )
        self.length_Neck_REar = scipy.spatial.distance.euclidean(
            self.body_parts_coordinates["Neck"], self.body_parts_coordinates["REar"]
        )
        self.length_Neck_LEye = scipy.spatial.distance.euclidean(
            self.body_parts_coordinates["Neck"], self.body_parts_coordinates["LEye"]
        )
        self.length_Neck_REye = scipy.spatial.distance.euclidean(
            self.body_parts_coordinates["Neck"], self.body_parts_coordinates["REye"]
        )
        self.length_Nose_LEar = scipy.spatial.distance.euclidean(
            self.body_parts_coordinates["Nose"], self.body_parts_coordinates["LEar"]
        )
        self.length_Nose_REar = scipy.spatial.distance.euclidean(
            self.body_parts_coordinates["Nose"], self.body_parts_coordinates["REar"]
        )
        self.length_Nose_LEye = scipy.spatial.distance.euclidean(
            self.body_parts_coordinates["Nose"], self.body_parts_coordinates["LEye"]
        )
        self.length_Nose_REye = scipy.spatial.distance.euclidean(
            self.body_parts_coordinates["Nose"], self.body_parts_coordinates["REye"]
        )
        self.length_head = np.maximum.reduce(
            [
                self.length_Neck_HeadTop,
                self.length_Neck_LEar,
                self.length_Neck_REar,
                self.length_Neck_LEye,
                self.length_Neck_REye,
                self.length_Nose_LEar,
                self.length_Nose_REar,
                self.length_Nose_LEye,
                self.length_Nose_REye,
            ]
        )
        # Length of torso
        self.length_Neck_LHip = scipy.spatial.distance.euclidean(
            self.body_parts_coordinates["Neck"], self.body_parts_coordinates["LHip"]
        )
        self.length_Neck_RHip = scipy.spatial.distance.euclidean(
            self.body_parts_coordinates["Neck"], self.body_parts_coordinates["RHip"]
        )
        self.length_torso = np.maximum(self.length_Neck_LHip, self.length_Neck_RHip)

        # Length of right leg
        self.length_leg_right = scipy.spatial.distance.euclidean(
            self.body_parts_coordinates["RHip"], self.body_parts_coordinates["RKnee"]
        ) + scipy.spatial.distance.euclidean(
            self.body_parts_coordinates["RKnee"], self.body_parts_coordinates["RAnkle"]
        )

        # Length of left leg
        self.length_leg_left = scipy.spatial.distance.euclidean(
            self.body_parts_coordinates["LHip"], self.body_parts_coordinates["LKnee"]
        ) + scipy.spatial.distance.euclidean(
            self.body_parts_coordinates["LKnee"], self.body_parts_coordinates["LAnkle"]
        )

        # Length of leg
        self.length_leg = np.maximum(
            self.length_leg_right, self.length_leg_left
        )

        # Length of body
        self.length_body = self.length_head + self.length_torso + self.length_leg

        # Check all samples have length_body of 0
        assert (self.length_body.astype(int)) > 0, "Length of body is 0"


builder = BuildSamples(jaad_object=JAAD("data/jaad"))
data = builder.generate_sequence_samples()
A = BodyBuilder(data["sequences"][0][0]["skeleton_sequence"][0])
A._compute_length_body()
print(A.length_body)
