# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
from detectron2.structures import BoxMode


# import some common libraries
import numpy as np
import os, json, cv2, random, torch

from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.data.datasets.nuimages import load_nuimages_dicts
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
from detectron2.structures import instances


import pycocotools.mask as cocomask

root_path = '/mnt/disk1/nuImages_test/'

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
              'vehicle.cycle.with_rider']
dataset = 'nuimages_mini'
version = 'v1.0-mini'
get_dicts = lambda p=root_path, c=categories: load_nuimages_dicts(path=p, version=version, categories=c)
DatasetCatalog.register(dataset, get_dicts)
MetadataCatalog.get(dataset).thing_classes = categories
MetadataCatalog.get(dataset).evaluator_type = "coco"

dataset_dicts = load_nuimages_dicts(root_path, version, categories)

def transform_instance_to_dict(instances):
    scores = instances.scores.numpy()
    pred_classes = instances.pred_classes.numpy()
    pred_boxes = instances.pred_boxes.tensor.numpy()
    pred_masks = instances.pred_masks.numpy()
    pred_masks_encode = []
    for objid in range(len(scores)):
        mask_encode = cocomask.encode(np.asfortranarray(pred_masks[objid]))
        pred_masks_encode.append(mask_encode)
    #pred_classes = convert_anno_to_nuimages(pred_classes)
    # dict= {
    #     'n_obj': len(scores),
    #     'scores': scores,
    #     'pred_classes': convert_anno_to_nuimages(pred_classes),
    #     'pred_boxes': pred_boxes,
    #     'pred_masks': pred_masks_encode
    # }
    dict = {'n_obj': 0,
        'scores': [],
        'pred_classes': [],
        'pred_boxes': [],
        'pred_masks': []}
    index = 0
    new_instances = instances[0:0]
    for i in range(len(scores)):
        if pred_classes[i] < len(categories):
            dict['n_obj']+=1
            dict['scores'].append(scores[i])
            dict['pred_classes'].append(pred_classes[i])
            dict['pred_boxes'].append(pred_boxes[i])
            dict['pred_masks'].append(pred_masks_encode[i])
            new_instances = instances.cat([new_instances,instances[i]])
        # else:
        #     if i == len(scores)-1:
        #         instances = instances[:i-1]
        #     else:
        #         instances = instances.cat([instances[0:i],instances[i+1:]])
    #print(new_instances)
    return new_instances,dict
i = 0
for d in dataset_dicts:
    img = cv2.imread(d["file_name"])

    if len(d['annotations']) == 0:
        continue
    visualizer = Visualizer(img[:, :, ::-1], MetadataCatalog.get(dataset), scale=1.2)
    out = visualizer.draw_dataset_dict(d)

    # get inference:
    cfg = get_cfg()
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    #cfg.merge_from_file(model_zoo.get_config_file("NuImages-RCNN-FPN.yaml"))
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    model_path = '/home/jinghui/detectron2/tools/output'
    cfg.MODEL.WEIGHTS =os.path.join(model_path, "model_final.pth")
    #cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)
    infer_filename = d["file_name"]
    nuimage_data_root = '/mnt/disk1/nuImages/mini/samples/'
    infer_path = '/mnt/disk1/nuImages_test/nuimage_mini_result/vis' + infer_filename[32:]

    # seq_names = sorted(os.listdir(nuimage_data_root))
    # seq_path = os.path.join(nuimage_data_root, seq)
    # seq_path0 = os.path.join(seq_path)
    # image_names0 = sorted(os.listdir(seq_path0))
    #
    # im_path = os.path.join(seq_path0, image_name)

    im = cv2.imread(infer_filename)

    outputs = predictor(im)
    instances = outputs["instances"].to("cpu")
    infer_img, _= transform_instance_to_dict(instances)
    #infer_img = cv2.imread(infer_path)
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(dataset), scale=1.2)
    infer_out = v.draw_instance_predictions(infer_img)


    infer_img = infer_out.get_image()[:, :, ::-1]
    gt_img = out.get_image()[:, :, ::-1]


    # cv2.imshow("Orignal", im)
    # cv2.waitKey(0)
    # cv2.imshow("Ground Truth", out.get_image()[:, :, ::-1])
    # cv2.waitKey(0)
    # cv2.imshow("Test Result", infer_out.get_image()[:, :, ::-1])


    Hori = np.concatenate((gt_img,infer_img), axis=0)
    i+=1
    savefile_name = infer_filename[40:]
    i = 0
    while savefile_name[i] != '/':
        i+= 1
    cv2.imwrite('/mnt/disk1/nuImages_test/mini_result%s.png' % savefile_name[i:-4],Hori)
    print('/mnt/disk1/nuImages_test/mini_result%s.png' % savefile_name[i+40:-4])
    #cv2.imshow("GT & inference",Hori)
    #cv2.waitKey(0)
