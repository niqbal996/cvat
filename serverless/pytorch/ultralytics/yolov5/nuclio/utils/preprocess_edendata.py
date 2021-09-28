
import torch
import os
import shutil
from pathlib import Path
# Name of the dataset downloaded from Eden Library
path = "/home/naeem/git/yolov5/eden_data"

# Define the source path
SOURCE_IMAGES_PATH = str(
    Path(Path.cwd()).parents[0].joinpath("eden_library_datasets").joinpath(path)
)

# Copying configuration files
shutil.copy("broc.names", "./eden_data")
shutil.copy("broc.yaml", "./eden_data")
shutil.copy("train.txt", "./eden_data")
shutil.copy("val.txt", "./eden_data")

IMAGES_PATH = "eden_data/images/"
LABELS_PATH = "eden_data/labels/"

os.makedirs(IMAGES_PATH + "train", exist_ok=True)
os.makedirs(IMAGES_PATH + "val", exist_ok=True)
os.makedirs(LABELS_PATH + "train", exist_ok=True)
os.makedirs(LABELS_PATH + "val", exist_ok=True)

conf_files = ["train.txt", "val.txt"]

print("Creating folder structure...")

for conf_file in conf_files:
    with open(conf_file, "r") as reader:
        # Create the yolov5/eden_data/images/train path
        img_dst_dir = IMAGES_PATH + conf_file.split(".")[0]
        # Create the yolov5/eden_data/images/val path
        lab_dst_dir = LABELS_PATH + conf_file.split(".")[0]
        # Read and print the entire file line by line
        line = reader.readline()
        im_files = os.listdir(SOURCE_IMAGES_PATH)
        while line != "":
            aux_im = line.split("/")[-1]
            for im_file in im_files:
                if im_file.strip() == aux_im.strip():
                    img_source = SOURCE_IMAGES_PATH + os.path.sep + im_file.strip()
                    shutil.copy(img_source, img_dst_dir)
                    image_annotation = (
                        SOURCE_IMAGES_PATH
                        + os.path.sep
                        + im_file.strip().split(".")[0]
                        + ".txt"
                    )
                    shutil.copy(image_annotation, lab_dst_dir)
            line = reader.readline()

print("Process finished correctly")