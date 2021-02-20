import numpy as np
from tqdm import tqdm
from nuimages.nuimages import NuImages
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from nuscenes.utils.geometry_utils import view_points, transform_matrix
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import  BoxVisibility
from pycocotools import mask
from detectron2.utils.visualizer import Visualizer
from detectron2.structures import Boxes, BoxMode, PolygonMasks
from detectron2.utils.file_io import PathManager

from .. import DatasetCatalog, MetadataCatalog

dataset = 'nuimages_train'
version = 'v1.0-train'
root_path = '/mnt/disk1/nuImages_test/'

categories = ['human.pedestrian',
              'vehicle.car',
              'vehicle.bus',
              'vehicle.truck',
              'vehicle.cycle',
              'vehicle.cycle.with_rider']
full_categories = ["animal",
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
categories_mapping = [[2,3,4,5,6,7,8],
                      [17,18,20,21],
                      [15,16],
                      [23,24],
                      [14,22]]

def convert_categories(cid,categories_mapping):
    for i in range(len(categories_mapping)):
        if cid in categories_mapping[i]:
            return i
    return None

def load_nuimages_dicts(path, version, categories = categories):
    assert (path[-1] == "/"), "Insert '/' in the end of path"
    nuim = NuImages(dataroot='/mnt/disk1/nuImages_test', version=version, verbose=True, lazy=True)

    if categories == None:
        categories = [data["name"] for data in nuim.category]
    assert (isinstance(categories, list)), "Categories type must be list"

    dataset_dicts = []
    nuim.load_tables(['object_ann', 'sample_data', 'category', 'attribute'])

    for idx in tqdm(range(0, len(nuim.sample))):
        data = nuim.sample_data[idx]
        # if only want CAM_FRONT, uncomment this 2 line
        # if not (data['filename'][:17] =="sweeps/CAM_FRONT/" or data['filename'][:18] =="samples/CAM_FRONT/"):
        #     continue
        record = {}
        record["file_name"] = path + data["filename"]
        record["image_id"] = idx
        record["height"] = data["height"]
        record["width"] = data["width"]
        objs = []
        if data['is_key_frame']:

            objects = []
            for i in nuim.object_ann:
                if i['sample_data_token']==nuim.sample_data[idx]['token']:
                    objects.append(i)
            _, segs = nuim.get_segmentation(data['token'])
            objnum=1
            for object in objects:
                seg = (segs == objnum)
                objnum += 1
                seg = seg.astype('uint8')
                for j in range(len(nuim.category)):
                    if nuim.category[j]['token'] == object['category_token']:
                        catid = j
                        break
                catid = convert_categories(catid,categories_mapping)
                if catid == None:
                    continue
                if catid == 4 and len(object['attribute_tokens']) > 0:
                    if object['attribute_tokens'][0] == nuim.attribute[0]['token']:
                        catid = 5
                obj = {
                    "bbox": object['bbox'],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "category_id": catid,
                    "iscrowd": 0,
                    "segmentation": mask.encode(np.asarray(seg, order="F"))
                }
                objs.append(obj)
        record["annotations"] = objs
        if len(objs) > 0:
            dataset_dicts.append(record)
    return dataset_dicts



get_dicts = lambda p = root_path, c = categories: load_nuimages_dicts(path=p,version = version, categories=c)
DatasetCatalog.register(dataset,get_dicts)
MetadataCatalog.get(dataset).thing_classes = categories
MetadataCatalog.get(dataset).evaluator_type = "coco"
