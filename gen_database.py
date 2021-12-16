from data.jaad import jaad_data
from src import data_utils


jaad = jaad_data.JAAD('data/jaad')

jaad_db = data_utils.JaadDatabase(jaad)
jaad_db.run_database_generator()

jaad_db.add_cropped_bbox()
jaad_db.append_keypoints()




