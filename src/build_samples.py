import glob
import json
import os
import pickle
import typing
import pandas as pd
from pathlib import Path
import sys

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import sklearn
import yaml

from data.jaad.jaad_data import JAAD

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
        :return: Sequence data
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

    def _get_frames_with_no_skeleton(self, arr, video_name, pid):
        """
        Returns the indices of the frames with no skeleton
        :param arr: The array of frames
        :param video_name: The video name
        :param pid: The pid
        :return: The indices of the frames with no skeleton
        """
        frames_with_no_skeleton = []
        frames = self.db[video_name]['ped_annotations'][pid]['frames']
        skeletons = self.db[video_name]['ped_annotations'][pid]['skeleton_keypoints']
        for frame in arr:
            skel = skeletons[frames.index(frame)]
            if skel is None:
                frames_with_no_skeleton.append(frame)

        return frames_with_no_skeleton


    def _parse_sequence_data(self, sequence_data: dict, window, **opts):
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
            pd.Series(arr).str.extract("(\d{5})").astype(int).to_numpy().flatten() for arr in sliding_windows_image_array
        ]  # array([['../data/jaad/images/video_0001/00000.png', '../data/jaad/images/video_0001/00001.png' => [array([0, 1, 2, 3, 4...]), array([1, 2, 3, 4, 5...])....

        # frames of the pid
        pid_frames = self.db[video_name]['ped_annotations'][pid]['frames']
        skeletons = self.db[video_name]['ped_annotations'][pid]['skeleton_keypoints']

        array_with_sequences = []
        for array_ix, array in enumerate(sliding_window_frame_id):  # an array contain the ids of the frames for that sliding window
            skeletons_single_sequence = []
            confidence_single_sequence = []
            for ix in array:
                try:
                    frame_skeleton = skeletons[pid_frames.index(ix)]
                except IndexError:
                    frame_skeleton = None
                if frame_skeleton is None:
                    break
                skeletons_single_sequence.append(frame_skeleton[:,:2])
                confidence_single_sequence.append(frame_skeleton[:,2])
            assert all(len(lst['frame_sequence']) == len(array) for lst in array_with_sequences), "Skeleton length does not match array length"
            if frame_skeleton is not None:
                data = {
                    'video_name': video_name,
                    'pid': pid,
                    'frame_sequence': array,
                    'skeleton_sequence': skeletons_single_sequence,
                    'confidence': confidence_single_sequence,
                    'intent': sliding_windows_intent_array[array_ix].flatten()

                }
                array_with_sequences.append(data)


        return array_with_sequences

    def generate_sequence_samples(self, image_set="all", window=15, visualize_inner_func=False, **opts):

        """
        Calls _generate_raw_sequence to generate sequence data.
        :param image_set: the split set to produce for. Options are train, test, val.

        -----
        the data returned by the inner function _generate_raw_sequence is as follows:

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

        -----

        :return: Sequence data
        """

        params = {
            "sample_type": "beh",
            "height_rng": [60, float("inf")],
            "min_track_size": window,
        }
        params.update(opts)

        sequence_data = self._generate_raw_sequence(image_set, **params)

        # if visualize_inner_func is True, we want to print the sequence data to look at it
        print("+" * 30, "Print raw sequence data", sequence_data, "+" * 30, sep="\n") if visualize_inner_func else None

        return sequence_data


count=0
count_pos = 0
jaad = JAAD('data/jaad')
jaad_db = JaadDatabase(jaad_object=jaad)
db = jaad_db.read_from_pickle('data/processed/jaad_database.pkl')
samples = BuildSamples(jaad)
samples_ = samples._generate_raw_sequence(sample_type='beh', height_rng=[60, float("inf")], min_track_size=15)
data = {
    'sequences_count': 0,
    'pos_sequences': 0,
    'sequences': []
}
for i in range(len(samples_['image'])):
    d = {"image": samples_['image'][i], 'pid': samples_['pid'][i], 'intent': samples_['intent'][i]}
    w = samples._parse_sequence_data(sequence_data=d, window=15)
    data['sequences'].append(w)
    data['sequences_count'] += len(w)
    for j in w:
        if j['intent'][0] == 1:
            data['pos_sequences'] += 1
    if i % 10 == 0:
        print(i)

 def euclidean_dist(a, b):
          # This function calculates the euclidean distance between 2 point in 2-D coordinates
          # if one of two points is (0,0), dist = 0
          # a, b: input array with dimension: m, 2
          # m: number of samples
          # 2: x and y coordinate
          try:
              if (a.shape[1] == 2 and a.shape == b.shape):
                  # check if element of a and b is (0,0)
                  bol_a = (a[:,0] != 0).astype(int)
                  bol_b = (b[:,0] != 0).astype(int)
                  dist = np.linalg.norm(a-b, axis=1)
                  return((dist*bol_a*bol_b).reshape(a.shape[0],1))
          except:
              print("[Error]: Check dimension of input vector")
              return 0


with open('data/processed/jaad_sequences_data.pkl', 'wb') as f:
    pickle.dump(data, f)