from src import skeleton_pipeline
import yaml
import argparse
import os

# Instantiate the parser
parser = argparse.ArgumentParser(description='Infer skeleton of images. Usage: python main.py openpose_dir output_dir')

parser.add_argument('openpose_dir', type=str, help='Path to openpose directory', default=os.environ['OPENPOSE'])
parser.add_argument('output_dir', type=str, help='Path to output directory')
parser.add_argument('--skip-annotations', dest='skip_annotations', action='store_true', help='Skip annotation step')
parser.add_argument('--skip-cropping', dest='skip_cropping', action='store_true', help='Skip cropping step')
args = parser.parse_args()



# read each line of document selected_videos and add them to list
selected_videos = []
with open('selected_videos', 'r') as f:
    for line in f:
        selected_videos.append(line.strip())

# load flags from flags.yaml
with open('flags.yaml', 'r') as f:
    opt_flags = yaml.safe_load(f)

opt_flags = [f for f in opt_flags if f not None]


def run_pipeline(openpose_dir = args.openpose_dir, output_dir = args.output_dir,skip_annotations=args.skip_annotations, skip_cropping=args.skip_cropping, selected_videos=selected_videos, opt_flags=opt_flags):

    if not skip_annotations:
        skeleton_pipeline.get_annotations()
    if not skip_cropping:
        skeleton_pipeline.get_and_crop_images()
    skeleton_pipeline.infer_clip(openpose_dir, output_dir, selected_videos, opt_flags)
    
if __name__ == '__main__':
    run_pipeline()