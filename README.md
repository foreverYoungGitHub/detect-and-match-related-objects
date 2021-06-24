
## Detecting and Matching Related Objects with One Proposal Multiple Predictions

This repo is the implementation for [Detecting and Matching Related Objects with One Proposal Multiple Predictions](https://openaccess.thecvf.com/content/CVPR2021W/CVSports/html/Liu_Detecting_and_Matching_Related_Objects_With_One_Proposal_Multiple_Predictions_CVPRW_2021_paper.html) in CVPR 2021. 

In this paper, we propose a simple yet efficient way to detect and match players and related objects at once without extra cost, by considering an implicit association for prediction of multiple objects through the same proposal box. 

![Illustration of the proposed network architecture for one proposal multiple prediction, applied to the ice hockey application.](images/player+stick-detection.png)

This repo cantains:
- Dataset generation for the proposed [COCO +Torso dataset](./COCO_Torso)
- The [model implementation](./model) & [inference code](./demo.py)

## Demo

To visual validate the result for you can use [demo.py](./demo.py) to test the model in sample images.

```bash
usage: demo.py [-h] -m {fpn,fpn+mp} -c CHECKPOINT -i DEMO_FILE [-d]

optional arguments:
  -h, --help            show this help message and exit
  -m {fpn,fpn+mp}, --model {fpn,fpn+mp}
                        the model name
  -c CHECKPOINT, --checkpoint CHECKPOINT
                        optional checkpoint file
  -i DEMO_FILE, --demo-file DEMO_FILE
                        the address of the demo file
  -d, --display         whether display the detection result
```

For example, 
```bash
# for regular detector
python -m demo -m fpn -c checkpoints/coco+torso/FPN_ResNet18_COCO_Torso_easy.pth -i images/COCO_val2014_000000022935.jpg

# for proposed detector with multiple prediction and matching
python -m demo -m fpn+mp -c checkpoints/coco+torso/FPN_MP_ResNet18_COCO_Torso_easy.pth -i images/COCO_val2014_000000022935.jpg
```

## Sample Results

|                   original image                  |                        normal detector                       |                        proposed detector                       |
|:-------------------------------------------------:|:------------------------------------------------------------:|:--------------------------------------------------------------:|
| ![original](images/COCO_val2014_000000022935.jpg) | ![original](images/COCO_val2014_000000022935_fpn_result.jpg) | ![original](images/COCO_val2014_000000022935_fpnmp_result.jpg) |
| ![original](images/COCO_val2014_000000398203.jpg) | ![original](images/COCO_val2014_000000398203_fpn_result.jpg) | ![original](images/COCO_val2014_000000398203_fpnmp_result.jpg) |

## Citation

```
@inproceedings{liu2021detecting,
  title={Detecting and Matching Related Objects with One Proposal Multiple Predictions},
  author={Liu, Yang and Hafemann, Luiz G and Jamieson, Michael and Javan, Mehrsan},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={4520--4527},
  year={2021}
}
```