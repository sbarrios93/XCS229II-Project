import os
import pickle
from sre_constants import SUCCESS
import sys
import typing
from pathlib import Path
from itertools import repeat
import glob
from unittest import skip
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from itertools import combinations

import numpy as np
import pandas as pd
import scipy
import yaml
from data.jaad.jaad_data import JAAD
from numpy.lib.stride_tricks import sliding_window_view
from scipy import signal
from tqdm import tqdm
from multiprocessing import Pool

from src.data_utils import JaadDatabase

def run(args):
    feature_array = pd.DataFrame()
    range_list, sequence_dict = args # unpack the tuple
    for n in range_list:
        for j in range(len(sequence_dict[n])):
            n_pid_seq = sequence_dict[n][j]
            print(f"Processing {n_pid_seq['video_name']}, {n_pid_seq['pid']}")
            feature_dict = {}
            feature_counter = 0
            for i, skeleton in enumerate(n_pid_seq['skeleton_sequence']):
                body = BodyBuilder(skeleton)
                frame = n_pid_seq['frame_sequence'][i]
                # angle combinations
                comb_names = body.get_angle_combinations()[2]
                comb_values = body.get_angle_combinations()[0]
                angles_dict = dict(zip(comb_names, comb_values))
                for k, v in angles_dict.items():
                    # [feature_0]_angle_LShoulder_RShoulder_LWrist_frame_0
                    feature_dict[f"[feature-{feature_counter}]_ANGLE_[{k[0]}-{k[1]}{k[2]}]_[{n_pid_seq['video_name']}]_{n_pid_seq['pid']}_[frame-{frame}]"] = v
                    feature_counter += 1

                ## cosine features
                cosine_dict = body.get_cosine_features()
                for k, v in cosine_dict.items():
                    feature_dict[f"[feature-{feature_counter}]_COSINE_[{k}]_[{n_pid_seq['video_name']}]_{n_pid_seq['pid']}_[frame-{frame}]"] = v
                    feature_counter += 1

                ## position relative to frame features
                position_dict = dict(zip(body.get_position_features().index, body.get_position_features().values))
                for k, v in position_dict.items():
                    feature_dict[f"[feature-{feature_counter}]_POSITION-ON-FRAME-X_[{k}]_[{n_pid_seq['video_name']}]_{n_pid_seq['pid']}_[{n_pid_seq['video_name']}]_{n_pid_seq['pid']}_[frame-{frame}]"] = v[0]
                    feature_counter += 1
                    feature_dict[f"[feature-{feature_counter}]_POSITION_ON_FRAME-Y_[{k}]_[{n_pid_seq['video_name']}]_{n_pid_seq['pid']}_[frame-{frame}]"] = v[1]
                    feature_counter += 1

                ## normalized_keypoints
                normalized_keypoint_dict = dict(zip(body.get_normalized_body_parts_df().index, body.get_normalized_body_parts_df().values))

                for k, v in normalized_keypoint_dict.items():
                    feature_dict[f"[feature-{feature_counter}]_NORMALIZED-KEYPOINT-X_[{k}]_[{n_pid_seq['video_name']}]_{n_pid_seq['pid']}_[frame-{frame}]"] = v[0]
                    feature_counter += 1
                    feature_dict[f"[feature-{feature_counter}]_NORMALIZED-KEYPOINT-Y_[{k}]_[{n_pid_seq['video_name']}]_{n_pid_seq['pid']}_[frame-{frame}]"] = v[1]
                    feature_counter += 1
            try:
                feature_dict[f"target_[{n_pid_seq['video_name']}]_{n_pid_seq['pid']}_[frame-{frame}]"] = int(n_pid_seq['intent'][0])
            except:
                pass
            try:
                feature_array = pd.concat([feature_array, pd.DataFrame.from_dict(feature_dict, orient='index')])
            except NameError:
                feature_array = pd.DataFrame.from_dict(feature_dict, orient='index')

    if not feature_array.empty:
        feature_array.columns = ['val']
        feature_array.to_parquet(f"data/processed/pd_feature_array/pandas_checkpoint_{str(min(range_list))}-{str(max(range_list))}.parquet")

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
        fail_count = 0
        success_count = 0

        params = {
            "savgol_window_size": 11,
            "savgol_polyorder": 3,
        }
        params.update(opts)

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
        skeletons = self._apply_savgol_filter(
            skeletons, window_size=params["savgol_window_size"], polyorder=params["savgol_polyorder"]
        )

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
                if len(skeletons_single_sequence) ==  window:
                    data = {
                        "video_name": video_name,
                        "pid": pid,
                        "frame_sequence": array,
                        "skeleton_sequence": skeletons_single_sequence,
                        "confidence": confidence_single_sequence,
                        "intent": sliding_windows_intent_array[array_ix].flatten(),
                    }
                    array_with_sequences.append(data)
                    success_count += 1
                else: # not enough frames
                    # print(f"Not enough frames for sequence on {video_name}, {pid}, {array}")
                    fail_count += 1

        try:
            tqdm.write(str(fail_count/(success_count+fail_count)))
        except:
            tqdm.write(str(100))

        return array_with_sequences

    def _interpolate_array_column(self, arr):
        """
        Interpolates a numpy array column
        :param arr: Numpy array
        :return: Numpy array with interpolated columns
        """
        args_where_zero = np.where(arr == 0)
        arr_copy = arr.copy()

        series = pd.Series(arr_copy)
        series[series == 0] = np.NaN

        series = series.interpolate().interpolate(method="bfill")
        arr_copy[args_where_zero] = series.iloc[args_where_zero].values

        return arr_copy

    def _apply_savgol_filter(self, skeletons, window_size=11, polyorder=3, interpolate=True):
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

            # interpolate where the values are 0
            if interpolate and np.any(stack[:, i] == 0):
                stack[:, i] = self._interpolate_array_column(stack[:, i])

            stack[:, i] = signal.savgol_filter(stack[:, i], window_size, polyorder, mode="nearest")  # run the filter

            # fix getting negative values
            zero_negative_value_indices = (np.where(stack[:, i] <= 0))[0]
            # if there are negative value, check previous value, if its positive use it, if its negative, set ot 0.
            # If there is no previous value, use next value with same procedure.
            for arg in zero_negative_value_indices:
                try:
                    if stack[arg - 1, i] > 0:
                        stack[arg, i] = stack[arg - 1, i]
                    else:
                        stack[arg, i] = 0
                except IndexError:
                    if stack[arg + 1, i] > 0:
                        stack[arg, i] = stack[arg + 1, i]
                    else:
                        stack[arg, i] = 0

            non_nan_array = stack[:, i][~np.isnan(stack[:, i])]
            if not np.isnan(non_nan_array).all():
                assert non_nan_array.min() >= 0, "Negative values in the array"

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
            parsed_sample = self._raw_sequence_transformer(
                sequence_data=sample_data,
                window=params["min_track_size"],
                savgol_window_size=params["savgol_window_size"],
                savgol_polyorder=params["savgol_polyorder"],
            )
            sequence_dict["sequences"].append(parsed_sample)
            sequence_dict["sequences_count"] += len(parsed_sample)
            for i in parsed_sample:
                if i["intent"][0] == 1:
                    sequence_dict["pos_sequences"] += 1

        return sequence_dict

    def _split_list(self, a, n):
        k, m = divmod(len(a), n)
        return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

    def _make_features(self, sequence_dict):
        """
        Each sequence is a list of dictionaries, each dictionary is a frame.
        Each sequence represents a pedestrian id and its video. E.g. sequence_dict[0] refers to pedestrian id 0_1_2b and video_0001. sequence_dict[1] refers to pedestrian id 0_1_3b and video_0001. Each dictionary on a sequence contains the name of the video, pid, a frame sequence (list of frames), a skeleton sequence (list of skeletons), the intent (array) and the confidence(array). Each skeleton is a list of 25 2D points, each point is a list of x, y coordinates.

        The next dictionary on the sequence will be the same but for the next frame sequence. For example dictionary[0] will have the frame sequence [0, 1....,14], dictionary[1] will have the frame sequence [1, 2....,15], dictionary[2] will have the frame sequence [2, 3....,16], and so on.
        """


        num_workers = os.cpu_count()-4

        list_range = list(range(len(sequence_dict)))

        # skip list_id that hasa already been processed
        skip_list_id = []
        for filepath in glob.glob('data/processed/pd_feature_array/*.parquet'):
            ids = filepath.split('/')[-1].split('_')[-1].split('.parquet')[0]
            ids = ids.split('-')
            id_range = [int(ids[0]), int(ids[1])+1]
            id_list_range = list(range(id_range[0], id_range[1]))
            skip_list_id += id_list_range




        # assert there are no duplicates in skip list. This would mean we have duplicate ids in our files
        assert len(skip_list_id) == len(set(skip_list_id)), f"Duplicate ids in skip list (Duplicate ids in files) {[x for n, x in enumerate(skip_list_id) if x in skip_list_id[:n]]}"

        list_range = [i for i in list_range if i not in skip_list_id]

        splits = min(len(list_range), 200)

        lists_to_work_on = list(self._split_list(list_range, splits))
        args = ((lists_to_work_on[i], sequence_dict) for i in range(splits))

        with Pool(processes=num_workers) as pool:
            progress_bar = tqdm(total=len(lists_to_work_on))
            print("mapping ...")
            results = tqdm(pool.imap(run, args), total=len(lists_to_work_on))
            print("running ...")
            tuple(results)  # fetch the lazy results
            print("done")



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
                features = self._make_features(sequence_dict=parsed_sequences['sequences'])
                return features
            else:
                print(f"No previous pickle file found on {str(sample_sequence_dict_path)}", "Generating...", sep="\n")

        if regen:  # Print notice nto user
            print(f"Forcing regeneration of {sample_sequence_dict_path}")

        # run functions
        sequence_data = self._generate_raw_sequence(image_set, **params)
        parsed_sequences = self._parse_raw_sequence(
            sequence_data=sequence_data,
            **params,
            visualize_inner_func=visualize_inner_func,
            savgol_window_size=7,
            savgol_polyorder=3,
        )

        if save:
            with open("data/processed/sample_sequence_dict.pkl", "wb") as file:
                pickle.dump(parsed_sequences, file)
        # !SECTION
        features = self._make_features(sequence_dict=parsed_sequences['sequences'])


        return features  # return attribute


class BodyBuilder:
    def __init__(self, arr: np.ndarray) -> None:
        self.arr = arr

        self.body_yaml_path = "data/helper_data/body_25b_parts.yaml"
        self.body_yaml_dict = self._load_body_yaml_dict()

        self.body_parts_coordinates = self._map_to_dict(self.arr, self.body_yaml_dict)

        self.length_body = None

        self._compute_length_body()

        # dataframe option
        self.body_parts_df = pd.DataFrame(self.body_parts_coordinates, index=["x", "y"]).T

        self.center_of_gravity = self._compute_center_of_gravity()

        # normalize coordinates using the body length
        self.normalized_body_parts_df = (self.body_parts_df - self.center_of_gravity) / self.length_body


    def _load_body_yaml_dict(self):
        return yaml.load(Path.read_text(Path(self.body_yaml_path)), Loader=yaml.SafeLoader)

    def _map_to_dict(self, arr, body_yaml_dict):
        parts = {}
        for k, v in body_yaml_dict.items():
            parts[v] = arr[k, :]
        return parts

    def _compute_center_of_gravity(self):
        return self.body_parts_df.sum() / len(self.body_parts_df)

    def _try_euclidean(self, a, b):
        try:
            return scipy.spatial.distance.euclidean(a, b)
        except ValueError:
            return 0

    def _compute_length_body(self):
        # Length of head
        self.length_Neck_HeadTop = self._try_euclidean(
            self.body_parts_coordinates["Neck"], self.body_parts_coordinates["HeadTop"]
        )
        self.length_Neck_LEar = self._try_euclidean(
            self.body_parts_coordinates["Neck"], self.body_parts_coordinates["LEar"]
        )
        self.length_Neck_REar = self._try_euclidean(
            self.body_parts_coordinates["Neck"], self.body_parts_coordinates["REar"]
        )
        self.length_Neck_LEye = self._try_euclidean(
            self.body_parts_coordinates["Neck"], self.body_parts_coordinates["LEye"]
        )
        self.length_Neck_REye = self._try_euclidean(
            self.body_parts_coordinates["Neck"], self.body_parts_coordinates["REye"]
        )
        self.length_Nose_LEar = self._try_euclidean(
            self.body_parts_coordinates["Nose"], self.body_parts_coordinates["LEar"]
        )
        self.length_Nose_REar = self._try_euclidean(
            self.body_parts_coordinates["Nose"], self.body_parts_coordinates["REar"]
        )
        self.length_Nose_LEye = self._try_euclidean(
            self.body_parts_coordinates["Nose"], self.body_parts_coordinates["LEye"]
        )
        self.length_Nose_REye = self._try_euclidean(
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
        self.length_Neck_LHip = self._try_euclidean(
            self.body_parts_coordinates["Neck"], self.body_parts_coordinates["LHip"]
        )
        self.length_Neck_RHip = self._try_euclidean(
            self.body_parts_coordinates["Neck"], self.body_parts_coordinates["RHip"]
        )
        self.length_torso = np.maximum(self.length_Neck_LHip, self.length_Neck_RHip)

        # Length of right leg
        self.length_leg_right = self._try_euclidean(
            self.body_parts_coordinates["RHip"], self.body_parts_coordinates["RKnee"]
        ) + self._try_euclidean(
            self.body_parts_coordinates["RKnee"], self.body_parts_coordinates["RAnkle"]
        )

        # Length of left leg
        self.length_leg_left = self._try_euclidean(
            self.body_parts_coordinates["LHip"], self.body_parts_coordinates["LKnee"]
        ) + self._try_euclidean(
            self.body_parts_coordinates["LKnee"], self.body_parts_coordinates["LAnkle"]
        )

        # Length of leg
        self.length_leg = np.maximum(self.length_leg_right, self.length_leg_left)

        # Length of body
        self.length_body = self.length_head + self.length_torso + self.length_leg

        # Check all samples have length_body of 0
        assert (self.length_body.astype(int)) > 0, "Length of body is 0"

    def get_normalized_body_parts_df(self):
        return self.normalized_body_parts_df

    def get_body_parts_df(self):
        return self.body_parts_df

    def get_body_parts_coordinates(self):
        return self.body_parts_coordinates

    def get_length_body(self):
        return self.length_body

    def get_length_head(self):
        return self.length_head

    def get_length_torso(self):
        return self.length_torso

    def get_length_leg_right(self):
        return self.length_leg_right

    def get_length_leg_left(self):
        return self.length_leg_left

    def get_length_leg(self):
        return self.length_leg

    def get_length_body_parts(self):
        return self.length_body

    def get_euclidean_normalized_matrix(self):
        return pd.DataFrame(
            scipy.spatial.distance.cdist(self.normalized_body_parts_df, self.normalized_body_parts_df),
            columns=self.normalized_body_parts_df.index,
            index=self.normalized_body_parts_df.index,
        )

    def get_cosine_normalized_matrix(self):
        return pd.DataFrame(
            scipy.spatial.distance.cdist(self.normalized_body_parts_df, self.normalized_body_parts_df, metric="cosine"),
            columns=self.normalized_body_parts_df.index,
            index=self.normalized_body_parts_df.index,
        )

    def get_cosine_matrix(self):
        return pd.DataFrame(
            scipy.spatial.distance.cdist(self.body_parts_df, self.body_parts_df, metric="cosine"),
            columns=self.body_parts_df.index,
            index=self.body_parts_df.index,
        )

    def _calculate_angle(self, p1, p2, p3):
        """
        Calculate angle between three points
        """
        v1 = p2 - p1
        v2 = p3 - p1
        return np.rad2deg(np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))))

    def get_angle_combinations(self):
        n_features = len(self.normalized_body_parts_df)
        # Avoid this features:  Nose LEye REye LEar REar LElbow RElbow Neck HeadTop LBigToe LSmallToe LHeel RBigToe RSmallToe RHeel
        avoid_features = [0, 1, 2, 3, 4, 7, 8, 17, 18, 19, 20, 21, 22, 23, 24]

        length_array = [i for i in range(n_features) if i not in avoid_features]

        combs = list(combinations(length_array, 3))

        angles = []

        for trio in combs:
            angle = self._calculate_angle(
                self.normalized_body_parts_df.iloc[trio[0]],
                self.normalized_body_parts_df.iloc[trio[1]],
                self.normalized_body_parts_df.iloc[trio[2]],
            )
            angles.append(angle)

        named_combs = []
        for comb in combs:
            named_combs.append(
                (self.body_yaml_dict[comb[0]], self.body_yaml_dict[comb[1]], self.body_yaml_dict[comb[2]])
            )

        return np.array(angles), combs, named_combs

    def get_cosine_features(self):
        """
        Extract cosine features from the cosine matrix. We just need one triangle and not include diagonal elements. Also the features of the face I don't think are useful.
        """
        avoid_features = [0, 1, 2, 3, 4, 17, 18]  # ['Nose', 'LEye', 'REye', 'LEar', 'REar', 'Neck', 'HeadTop']
        avoid_features = [self.body_yaml_dict[i] for i in avoid_features]

        value_dict = {}
        for i in self.get_cosine_normalized_matrix().index:
            for j in self.get_cosine_normalized_matrix().columns:
                if i != j:
                    if i not in avoid_features and j not in avoid_features:
                        if f"{i}-{j}" not in value_dict.keys() and f"{j}-{i}" not in value_dict.keys():
                            value_dict[f"{i}-{j}"] = self.get_cosine_normalized_matrix().loc[i, j]
        return value_dict

    def get_euclidean_features(self):
        """
        Extract cosine features from the cosine matrix. We just need one triangle and not include diagonal elements. Also the features of the face I don't think are useful.
        """
        avoid_features = [0, 1, 2, 3, 4, 17, 18]  # ['Nose', 'LEye', 'REye', 'LEar', 'REar', 'Neck', 'HeadTop']
        avoid_features = [self.body_yaml_dict[i] for i in avoid_features]

        counter = 0
        value_dict = {}
        for i in self.get_euclidean_normalized_matrix()().index:
            for j in self.get_euclidean_normalized_matrix()().columns:
                if i != j:
                    if i not in avoid_features and j not in avoid_features:
                        if f"{i}-{j}" not in value_dict.keys() and f"{j}-{i}" not in value_dict.keys():
                            counter += 1
                            print(counter, f"{i}-{j}")
                            value_dict[f"{i}-{j}"] = self.get_euclidean_normalized_matrix()().loc[i, j]
        return value_dict

    def get_position_features(self):
        """
        Extract position from each body part coordinate in relation with the frame size (1920x1080)
        """
        return self.get_body_parts_df() / np.array([1920, 1080])

    def save_dict(self, dict_to_save, file_name):
        dir_path = Path('data/input_features')
        Path.mkdir(dir_path, parents=True, exist_ok=True)

        with open(f"{dir_path}/{file_name}.pkl", "wb") as f:
            pickle.dump(dict_to_save, f)

        print(f"{file_name}.pkl saved")


    def save_numpy(self, numpy_to_save, file_name, dir_path='data/input_features'):
        dir_path = Path(dir_path)
        Path.mkdir(dir_path, parents=True, exist_ok=True)

        np.save(f"{dir_path}/{file_name}.npy", numpy_to_save)

        print (f"{file_name}.npy saved")


if __name__ == '__main__':
    builder = BuildSamples(jaad_object=JAAD("data/jaad"))
    data = builder.generate_sequence_samples(regen=False)