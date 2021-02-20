# cruw_detectron2_devkit
Object detection kit base on detectron2 to provide result in CRUW dataset format, train base on transformed NuImages dataset


## Introduction
This project is aiming for provide ground truth for [CRUW dataset](https://www.cruwdataset.org/introduction) built by Information Processing Lab @UWECE.This devkit  provide image base Mask RCNN groud truth result as benchmark in cruw format. The object detection base on Detectron2 is the main part for this project. And use transformed [nuImages](https://www.nuscenes.org/nuimages) dataset to pretrain the benchmark model.

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

## nuImages Data loader
**1. Convert nuImages dataset to CRUW dataset format**
I use the nuScence build in devkit to load nuImages dataset and convert the categories use a mapping function, read all the relational data and transfer the metadata as a dict. For the segmantation part, the orignal segmantation format is a single map with category IDs for each instance, I convert the segmantation to each map per object, which can help me with futher fusion in objects.

And also, in nuImages, the cyclist are not seperated into different kind of pedestrain, but we want to merge the cyclist and the vehicle.cycle, so I read the attribution annotation and the bicycle with rider will be train as different category. After the training, I can relate the human.pedestrian with the vehicle.cycle.withrider by identify the corss part in bounding box and segmantation, and merge these bounding box and segmantation together. Same idea will be implement on vehicle.trailer and vehicle.car. But we don't have enough trailer object to train for now, this part will be added in future.

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

After made the dataset reader, register the nuimages_test and nuimages_train dataset and metadata in [builtin.py](https://github.com/TedSongjh/CSE599-fianl-project/blob/main/builtin.py). Because the nuImages don't have built in evaluator. I choose to use COCO InstanceSegmentation evaluator in the following part, so I load these two dataset by COCO format. So I have to convert the CRUW dataset format to COCO format, by changing object information schema, segmantation map to bitmask and bounding box format. Also, because CRUW dataset sensor setup only have dual camera facing front, I filter out all the samples facing other direction in nuImages. This part is also in [nuimages.py](https://github.com/TedSongjh/CSE599-fianl-project/blob/main/nuimages.py).The instances detail information is in the chart below.

**3.Train nuImages use Mask R-CNN**
Train on nuImages train-1.0 dataset
First, change dataset and version in [nuimages.py](https://github.com/TedSongjh/CSE599-fianl-project/blob/main/nuimages.py) to
```
dataset = 'nuimages_train'
version = 'v1.0-train'
```


To train the dataset on Detectron2 useing ResNet FPN backbone. 

```
./detectron2/tools/train_net.py   --config-file ../configs/NuImages-RCNN-FPN.yaml   --num-gpus 1 SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025
```

The detail of this archetecuture can be found in [NuImages-RCNN-FPN.yaml](https://github.com/TedSongjh/CSE599-fianl-project/blob/main/configs/NuImages-RCNN-FPN.yaml)
Train from last model (in this case is model_final.pth)
```
./detectron2/tools/train_net.py --num-gpus 1  --config-file ../configs/NuImages-RCNN-FPN.yaml MODEL.WEIGHTS ~/detectron2/tools/output-1/model_final.pth SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025
```

**4.Evaluation on nuImages val dataset
First, change dataset and version in [nuimages.py](https://github.com/TedSongjh/CSE599-fianl-project/blob/main/nuimages.py) to
```
dataset = 'nuimages_val'
version = 'v1.0-val'
```
Then run eval-only command and set model weights to the last checkpoint (in this case is model_final.pth)
```
./detectron2/tools/train_net.py    --config-file ../configs/NuImages-RCNN-FPN.yaml   --eval-only MODEL.WEIGHTS ~/detectron2/tools/output/model_final.pth
```

## Inference



