from data.jaad.jaad_data import JAAD
from src import data_utils, skeleton_pipeline

# %%
jaad_dir = "data/jaad"

jaad = JAAD(jaad_dir)

jaad_db = data_utils.JaadDatabase(jaad)
jaad_db.run_database_generator()

jaad_db.add_cropped_bbox()

jaad_db.append_keypoints()

# %%
pipeline = skeleton_pipeline.SkeletonPipeline(jaad_db)
pipeline.prepare_images()

# %%
jaad_db.append_keypoints()