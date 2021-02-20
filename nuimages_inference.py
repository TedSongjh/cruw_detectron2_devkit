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


import pycocotools.mask as cocomask

seq = ['2019_09_29_ONRD002_CS_NORMAL']
cruw_data_root = os.path.join('/mnt/disk1/CRUW/CRUW_MINI/sequences/',seq[0])
#cruw_infer_root = '/mnt/disk1/CRUW/CRUW_MINI/maskrcnn_results'
nuimage_data_root = '/mnt/disk1/nuImages/mini/samples/'
nuimage_infer_root = '/mnt/disk1/nuImages_test/nuimage_mini_result'
model_path = '/home/jinghui/detectron2/tools/output'
model_name = 'nuimage_coco'
dataset = 'nuimages_mini'


categories = ['human.pedestrian',
              'vehicle.car',
              'vehicle.bus',
              'vehicle.truck',
              'vehicle.cycle',
              'vehicle.cycle.withrider']
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

def visualize(im, instance, save_path=None):
    # We can use `Visualizer` to draw the predictions on the image.
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(dataset), scale=1.2)
    out = v.draw_instance_predictions(instance)
    if save_path is None:
        cv2.imshow("Demo", out.get_image()[:, :, ::-1])
        cv2.waitKey(0)
    else:
        cv2.imwrite(save_path, out.get_image()[:, :, ::-1])



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


def get_nuimages_dicts(img_dir):
    json_file = os.path.join(img_dir, "via_region_data.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    for idx, v in enumerate(imgs_anns.values()):
        record = {}

        filename = os.path.join(img_dir, v["filename"])
        height, width = cv2.imread(filename).shape[:2]

        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        annos = v["regions"]
        objs = []
        for _, anno in annos.items():
            assert not anno["region_attributes"]
            anno = anno["shape_attributes"]
            px = anno["all_points_x"]
            py = anno["all_points_y"]
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": 0,
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts




def write_det_txt(save_path, output_dict):
    with open(save_path, 'w') as f:
        for frameid in range(len(output_dict['IMAGES_0'])):
            dets_frame_dict = output_dict['IMAGES_0'][frameid]
            for objid in range(dets_frame_dict['n_obj']):
                mask_encode = dets_frame_dict['pred_masks'][objid]
                det_str = "%d %s %s %.4f %.2f %.2f %.2f %.2f %d %d %s\n" % \
                          (frameid, 'IMAGES_0', categories[dets_frame_dict['pred_classes'][objid]],
                           dets_frame_dict['scores'][objid],
                           dets_frame_dict['pred_boxes'][objid][0], dets_frame_dict['pred_boxes'][objid][1],
                           dets_frame_dict['pred_boxes'][objid][2], dets_frame_dict['pred_boxes'][objid][3],
                           mask_encode['size'][0], mask_encode['size'][1], mask_encode['counts'])
                f.write(det_str)

            # if len(output_dict['IMAGES_1']) != 0:
            #     dets_frame_dict = output_dict['IMAGES_1'][frameid]
            #     for objid in range(dets_frame_dict['n_obj']):
            #         mask_encode = dets_frame_dict['pred_masks'][objid]
            #         det_str = "%d %s %s %.4f %.2f %.2f %.2f %.2f %d %d %s\n" % \
            #                   (frameid, 'IMAGES_0', categories[dets_frame_dict['pred_classes'][objid]],
            #                    dets_frame_dict['scores'][objid],
            #                    dets_frame_dict['pred_boxes'][objid][0], dets_frame_dict['pred_boxes'][objid][1],
            #                    dets_frame_dict['pred_boxes'][objid][2], dets_frame_dict['pred_boxes'][objid][3],
            #                    mask_encode['size'][0], mask_encode['size'][1], mask_encode['counts'])
            #         f.write(det_str)


if __name__ == '__main__':

    cfg = get_cfg()
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file("NuImages-RCNN-FPN.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Use current final model
    cfg.MODEL.WEIGHTS =os.path.join(model_path, "model_final.pth")
    predictor = DefaultPredictor(cfg)

    seq_names = sorted(os.listdir(nuimage_data_root))
    for seq in seq_names:
        output_dict = {
            'IMAGES_0': []
        }
        seq_path = os.path.join(nuimage_data_root, seq)

        seq_path0 = os.path.join(seq_path)
        image_names0 = sorted(os.listdir(seq_path0))



        for image_name in image_names0:
            im_path = os.path.join(seq_path0, image_name)
            print("Inferring %s" % im_path)
            im = cv2.imread(im_path)
            # cv2.imshow('test',im)
            # cv2.waitKey(0)
            outputs = predictor(im)
            instances = outputs["instances"].to("cpu")
            #print(outputs)
            # print(outputs["instances"].pred_boxes)
            instances,det_dict = transform_instance_to_dict(instances)

            # print(det_dict)
            output_dict['IMAGES_0'].append(det_dict)

            save_path = os.path.join(nuimage_infer_root, 'vis', seq, image_name)
            if not os.path.exists(os.path.join(nuimage_infer_root, 'vis', seq)):
                os.makedirs(os.path.join(nuimage_infer_root, 'vis', seq))
            visualize(im, instances, save_path)



        save_path = os.path.join(nuimage_infer_root, 'txts', seq + '.txt')
        if not os.path.exists(os.path.join(nuimage_infer_root, 'txts')):
            os.makedirs(os.path.join(nuimage_infer_root, 'txts'))
        write_det_txt(save_path, output_dict)

