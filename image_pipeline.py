from data.jaad.jaad_data import JAAD
from src import data_utils, skeleton_pipeline
import argparse

RUN_INFERENCE = False
parser = argparse.ArgumentParser(description="Image pipeline")

# take as argument the value RUN_INFERENCE as --run-inference
parser.add_argument("--run-inference", dest="run_inference", action="store_true", help="Run inference")
args = parser.parse_args()

# %%
jaad_dir = "data/jaad"

jaad = JAAD(jaad_dir)

jaad_db = data_utils.JaadDatabase(jaad)
jaad_db.run_database_generator()

jaad_db.add_cropped_bbox()

jaad_db.append_keypoints()


if args.run_inference or RUN_INFERENCE:
    print("Running inference because", f"{'--run-inference, = True' * args.run_inference}", f"{'RUN_INFERENCE = True' * RUN_INFERENCE}")
    pipeline = skeleton_pipeline.SkeletonPipeline(jaad_db)
    pipeline.prepare_images()
    jaad_db.append_keypoints()
