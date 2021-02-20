# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
from detectron2.structures import BoxMode

setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random, torch

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.data.datasets.nuimages import load_nuimages_dicts
from detectron2.structures import instances
from nuimages_inference import transform_instance_to_dict

root_path = '/mnt/disk1/nuImages/'


categories = ["animal",
              "flat.driveable_surface",
               "human.pedestrian.adult",
               "human.pedestrian.child",
               "human.pedestrian.construction_worker",
               "human.pedestrian.personal_mobility",
               "human.pedestrian.police_officer",
               "human.pedestrian.stroller",
               "human.pedestrian.wheelchair",
               "movable_object.barrier",
               "movable_object.debris",
               "movable_object.pushable_pullable",
               "movable_object.trafficcone",
               "static_object.bicycle_rack",
               "vehicle.bicycle",
               "vehicle.bus.bendy",
               "vehicle.bus.rigid",
               "vehicle.car",
               "vehicle.construction",
               "vehicle.ego",
               "vehicle.emergency.ambulance",
               "vehicle.emergency.police",
               "vehicle.motorcycle",
               "vehicle.trailer",
               "vehicle.truck"]
categories = ['human.pedestrian',
              'vehicle.car',
              'vehicle.bus',
              'vehicle.truck',
              'vehicle.cycle',
              'vehicle.cycle.withrider']
dataset = 'nuimages_mini'
# version = 'v1.0-train'
dataset = 'nuimages'
version = 'v1.0-mini'
get_dicts = lambda p=root_path, c=categories: load_nuimages_dicts(path=p, version=version, categories=c)
DatasetCatalog.register(dataset, get_dicts)
MetadataCatalog.get(dataset).thing_classes = categories
MetadataCatalog.get(dataset).evaluator_type = "coco"


dataset_dicts = load_nuimages_dicts(root_path,version,categories)
print(MetadataCatalog.get(dataset))
print(dataset_dicts)

for d in dataset_dicts:
    img = cv2.imread(d["file_name"])
    print(d)
    #print(MetadataCatalog.get(dataset))
    if len(d['annotations']) == 0:
        continue
    visualizer = Visualizer(img[:, :, ::-1], MetadataCatalog.get(dataset), scale=1.2)
    out = visualizer.draw_dataset_dict(d)
    cv2.imshow("Demo",out.get_image()[:, :, ::-1])
    cv2.waitKey(0)