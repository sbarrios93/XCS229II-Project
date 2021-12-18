# %%

import glob
import json
import os
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
import time
import sys
import os
import yaml


import cv2
sys.path.append(os.path.dirname(os.getcwd()))
os.chdir('/Users/seba/local-projects/pedestrians')


from src import data_utils, frame_extract, skeleton_pipeline
from data.jaad.jaad_data import JAAD


# %%
jaad_dir = os.path.join('data/jaad')


jaad = JAAD(jaad_dir)

jaad_db = data_utils.JaadDatabase(jaad)
jaad_db.run_database_generator()

jaad_db.add_cropped_bbox()

# %%
pipeline = skeleton_pipeline.SkeletonPipeline(jaad_db)
pipeline.prepare_images()

# %%



