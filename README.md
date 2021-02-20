# cruw_detectron2_devkit
Object detection kit base on detectron2 to provide result in CRUW dataset format, train base on transformed NuImages dataset


## Introduction
This project is aiming for provide ground truth for [CRUW dataset](https://www.cruwdataset.org/introduction) built by Information Processing Lab @UWECE.This devkit  provide image base Mask RCNN groud truth result as benchmark in cruw format. The object detection base on [Detectron2](https://github.com/facebookresearch/detectron2) is the main part for this project. And use transformed [nuImages](https://www.nuscenes.org/nuimages) dataset to pretrain the benchmark model.

## Installation

To install Dectron2:
```
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

This devkit is used in this project to load nuImage dataset.
To use nuScenes devkit:
```
pip install nuscenes-devkit
```

Put [NuImages-RCNN-FPN.yaml](https://github.com/TedSongjh/cruw_detectron2_devkit/blob/main/NuImages-RCNN-FPN.yaml) Config file in folder ```/detectron2/configs```

Put [nuimages_loader.py](https://github.com/TedSongjh/cruw_detectron2_devkit/blob/main/nuimages_loader.py) dataset transformer in folder ```/detectron2/detectron2/data/datasets```

Put [nuimages_inference.py](https://github.com/TedSongjh/cruw_detectron2_devkit/blob/main/nuimages_inference.py), [visual&inference.py](https://github.com/TedSongjh/cruw_detectron2_devkit/blob/main/visual%26inference.py), [visualize_nuImages.py](https://github.com/TedSongjh/cruw_detectron2_devkit/blob/main/visualize_nuImages.py) in root folder of detectron2
## nuImages Data loader
**1. Convert nuImages dataset to CRUW dataset format**

Use the nuScence build in devkit to load nuImages dataset and convert the categories use a mapping function, read all the relational data and transfer the metadata as a dict. For the segmantation part, the orignal segmantation format is a single map with category IDs for each instance, convert the segmantation to each map per object, which can help with futher fusion in objects.And also, in nuImages, the cyclist are not seperated into different kind of pedestrain, but we want to merge the cyclist and the vehicle.cycle, so read the attribution annotation and the bicycle with rider will be train as different category.

The categories mapping from nuImages to CRUW is:
nuImages Category | CRUW Category
------------ | -------------
animal	|	-
human.pedestrian.adult	|	human.pedestrian
human.pedestrian.child	|	human.pedestrian
human.pedestrian.construction_worker	|	human.pedestrian
human.pedestrian.personal_mobility	|	human.pedestrian
human.pedestrian.police_officer	|	human.pedestrian
human.pedestrian.stroller	|	human.pedestrian
human.pedestrian.wheelchair	|	human.pedestrian
movable_object.barrier	|	-
movable_object.debris	|	-
movable_object.pushable_pullable	|	-
movable_object.trafficcone	|	-
static_object.bicycle_rack	|	-
vehicle.bicycle(without attribute: without_rider)	|	vehicle.cycle
vehicle.bicycle(without attribute: with_rider)	|	vehicle.cycle.withrider
vehicle.bus.bendy	|	vehicle.bus
vehicle.bus.rigid	|	vehicle.bus
vehicle.car	|	vehicle.car
vehicle.construction	|	vehicle.car
vehicle.emergency.ambulance	|	vehicle.car
vehicle.emergency.police	|	vehicle.car
vehicle.motorcycle(without attribute: without_rider)	|	vehicle.cycle
vehicle.motorcycle(with attribute: with_rider)	|	vehicle.cycle.withrider
vehicle.trailer	|	vehicle.truck
vehicle.truck	|	vehicle.truck
flat.drivable_surface	|	-
flat.ego	|	-

**2. Use Custom datasets on Detectron2**

After made the dataset reader, register the nuimages_test and nuimages_train dataset and metadata. use COCO InstanceSegmentation evaluator in the following part, and convert the nuImages format to CRUW dataset format, by changing object information schema, segmantation map to bitmask and bounding box format. This part is in [nuimages_loader.py](https://github.com/TedSongjh/cruw_detectron2_devkit/blob/main/nuimages_loader.py). Change dataset and version name to register other dataset. 
```
dataset = 'nuimages_train'
version = 'v1.0-train'
root_path = '/mnt/disk1/nuImages_test/'
get_dicts = lambda p = root_path, c = categories: load_nuimages_dicts(path=p,version = version, categories=c)
DatasetCatalog.register(dataset,get_dicts)
MetadataCatalog.get(dataset).thing_classes = categories
MetadataCatalog.get(dataset).evaluator_type = "coco"
```


**3.Train nuImages use Mask R-CNN**

Train on nuImages v1.0-train dataset
First, change dataset and version in [nuimages_loader.py](https://github.com/TedSongjh/cruw_detectron2_devkit/blob/main/nuimages_loader.py) to
```
dataset = 'nuimages_train'
version = 'v1.0-train'
```


To train the dataset on Detectron2 useing ResNet FPN backbone. 

```
./detectron2/tools/train_net.py   --config-file ../configs/NuImages-RCNN-FPN.yaml   --num-gpus 1 SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025
```

The detail of this archetecuture can be found in [NuImages-RCNN-FPN.yaml](https://github.com/TedSongjh/cruw_detectron2_devkit/blob/main/NuImages-RCNN-FPN.yaml)

Train from last model (in this case is model_final.pth)
```
./detectron2/tools/train_net.py --num-gpus 1  --config-file ../configs/NuImages-RCNN-FPN.yaml MODEL.WEIGHTS ~/detectron2/tools/output-1/model_final.pth SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025
```

**4.Evaluation on nuImages val dataset**

First, change dataset and version in [nuimages_loader.py](https://github.com/TedSongjh/cruw_detectron2_devkit/blob/main/nuimages_loader.py) to
```
dataset = 'nuimages_val'
version = 'v1.0-val'
```
Then run eval-only command and set model weights to the last checkpoint (in this case is model_final.pth)
```
./detectron2/tools/train_net.py    --config-file ../configs/NuImages-RCNN-FPN.yaml   --eval-only MODEL.WEIGHTS ~/detectron2/tools/output/model_final.pth
```

## Inference tools
There are three inference tools to visulize result

**1. Visulize nuImages groud truth**

run [visualize_nuImages.py](https://github.com/TedSongjh/cruw_detectron2_devkit/blob/main/visualize_nuImages.py)

**2. Inference on own dataset**

run [nuimages_inference.py](https://github.com/TedSongjh/cruw_detectron2_devkit/blob/main/nuimages_inference.py) to save inference result in folder

**3. Inference on nuImages dataset and compare to ground truth**

run [visual&inference.py](https://github.com/TedSongjh/cruw_detectron2_devkit/blob/main/visual%26inference.py) to save both the groud truth and inference result









