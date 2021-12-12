import os
from data.jaad_data import JAAD

imdb = JAAD(data_path = os.environ['PEDESTRIANS'] + '/data')
imdb.extract_and_save_images()