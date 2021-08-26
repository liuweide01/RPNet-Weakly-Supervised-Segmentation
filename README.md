<<<<<<< HEAD
# Cross-Image-Region-Mining-with-Region-Prototypical-Network-for-Weakly-Supervised-Segmentation
=======
# Cross-Image Region Mining with Region Prototypical Network for Weakly Supervised Segmentation
The code of:

Cross-Image Region Mining with Region
Prototypical Network for Weakly Supervised
Segmentation, Weide Liu, Xiangfei Kong, Tzu-Yi Hung, Guosheng Lin, [[Paper]](https://arxiv.org/abs/2108.07413)

## Citation
If you find the code useful, please consider citing our paper using the following BibTeX entry.
```
@article{liu2021cross,
  title={Cross-Image Region Mining with Region Prototypical Network for Weakly Supervised Segmentation},
  author={Liu, Weide and Kong, Xiangfei and Hung, Tzu-Yi and Lin, Guosheng},
  journal={arXiv preprint arXiv:2108.07413},
  year={2021}
}
```

## Prerequisite
* Python 3.7, PyTorch 1.1.0, and more in requirements.txt
* PASCAL VOC 2012 devkit and COCO 2014
* NVIDIA GPU with more than 1024MB of memory

## Usage

#### Install python dependencies
```
pip install -r requirements.txt
```
#### Download PASCAL VOC 2012 devkit
* Follow instructions in http://host.robots.ox.ac.uk/pascal/VOC/voc2012/#devkit

#### Download COCO 2014 devkit
* Follow instructions in https://cocodataset.org/#home

#### Run run_sample.py or make your own script
```
python run_sample.py
```
* You can either mannually edit the file, or specify commandline arguments.

#### Train DeepLab with the generated pseudo labels


## Related Repositories
This project is build based on IRN: https://github.com/jiwoon-ahn/irn.
Many thanks to their greak work! 

## TO DO
* Training code for MS-COCO
* Code refactoring
>>>>>>> Add all my files
