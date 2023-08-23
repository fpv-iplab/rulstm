# What Would You Expect? Anticipating Egocentric Actions with Rolling-Unrolling LSTMs and Modality Attention
See the quickstart here 👉 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fpv-iplab/rulstm/blob/master/RULSTM/Rolling-Unrolling-LSTM-Quickstart.ipynb)

This repository hosts the code related to the following papers:

Antonino Furnari and Giovanni Maria Farinella, Rolling-Unrolling LSTMs for Action Anticipation from First-Person Video. IEEE Transactions on Pattern Analysis and Machine Intelligence (PAMI). 2020. [Download](http://arxiv.org/pdf/2005.02190.pdf)

Antonino Furnari and Giovanni Maria Farinella, What Would You Expect? Anticipating Egocentric Actions with Rolling-Unrolling LSTMs and Modality Attention. International Conference on Computer Vision, 2019. [Download](https://arxiv.org/pdf/1905.09035.pdf)

Please also see the project web page at [http://iplab.dmi.unict.it/rulstm](http://iplab.dmi.unict.it/rulstm).

If you use the code/models hosted in this repository, please cite the following papers:

```
@article{furnari2020rulstm,
  author = {Antonino Furnari and Giovanni Maria Farinella},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence (PAMI)},
  title = {Rolling-Unrolling LSTMs for Action Anticipation from First-Person Video},
  year = {2020}
}
```

```
@inproceedings{furnari2019rulstm, 
  title = { What Would You Expect? Anticipating Egocentric Actions with Rolling-Unrolling LSTMs and Modality Attention. }, 
  author = { Antonino Furnari and Giovanni Maria Farinella },
  year = { 2019 },
  booktitle = { International Conference on Computer Vision (ICCV) },
}
```
## Updates:
 * **23/08/2023** A quickstart notebook is available [here](RULSTM/Rolling-Unrolling-LSTM-Quickstart.ipynb). You can also open it directly in Colab clicking on the badge: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fpv-iplab/rulstm/blob/master/RULSTM/Rolling-Unrolling-LSTM-Quickstart.ipynb)
 * **28/06/2021** We are now providing object detections on all frames of EPIC-KITCHENS-100. Please see this README (below) for more information;
 * **11/01/2021** We have updated the archive providing the EGTEA Gaze+ pre-extracted features. Please see this README (below) for more information;
 * **01/10/2020** We are now sharing the rgb/flow/obj EPIC-KITCHENS-100 features and pre-trained models used to report baseline results in the [Rescaling Egocentric Vision](https://arxiv.org/abs/2006.13256) paper;
 * **04/05/2020** We have now published an extended version of this work on PAMI. Please check the text above for the updated references;
 * **23/03/2020** We are now providing pre-extracted features for EGTEA Gaze+. See README for more information;
 * **11/10/2019** We are now also providing TSN and object-based features extracted for **each frame of EPIC-KITCHENS**. They can be downloaded using the `download_data_full.sh` script rather than `download_data.sh`;
 * **23/10/2019** Added some scripts to show how to extract features from videos. The scripts can be found under `FEATEXT` and are documented in this README. 

## Overview
This repository provides the following components:
 * The official PyTorch implementation of the proposed Rolling-Unrolling LSTM approach, including Sequence-Completion Pre-Training and Modality ATTention (MATT);
 * A program to train, validate and test the proposed method on the EPIC-KITCHENS-55 and EPIC-KITCHENS-100 datasets;
 * Pre-extracted features for EPIC-KITCHENS-55 and EPIC-KITCHENS-100. Specifically, we include:
   * RGB features: extracted from RGB iamges using a BNInception CNN trained for the task of *egocentric action recognition* using [Temporal Segment Networks](https://github.com/yjxiong/tsn-pytorch);
   * Flow features: similar to RGB features, but extracted with a BNInception CNN trained on optical flow;
   * OBJ features: object-based features obtained by running a Faster R-CNN object detector trained on EPIC-KITCHENS-55;
 * The checkpoints of the RGB/Flow/OBJ/Fusion models trained for both tasks: egocentric action anticipation and early action recognition;
 * The checkpoints of the TSN models (to be used with the [official PyTorch implementation of TSN](https://github.com/yjxiong/tsn-pytorch));
 * The checkpoint of the Faster R-CNN object detector trained on EPIC-KITCHENS-55;
 * The training/validation split used for the experiments. Note that the TSN and Faster R-CNN models have been trained on the training set of this split.

Please, refer to the paper for more technical details. The following sections document the released material.

## RU-LSTM Implementation and main training/validation/test program
The provided implementation and training/validation/test program can be found in the `RULSTM` directory. In order to proceed to training, it is necessary to retrieve the pre-extracted features from our website. To save space and bandwidth, we provide features extracted only on the subset of frames used for the experiments (we sampled frames at about 4fps - please see the paper). These features are sufficient to train/validate/test the methods on the whole EPIC-KITCHENS-55 dataset following the settings reported in the paper.

### Requirements
To run the code, you will need a Python3 interpreter and some libraries (including PyTorch). 

#### Anaconda
An Anaconda environment file with a minimal set of requirements is provided in `environment.yml`. If you are using Anaconda, you can create a suitable environment with:

`conda env create -f environment.yml`

To activate the environment, type:

`conda activate rulstm`

#### Pip
If you are not using Anaconda, we provide a list of libraries in `requirements.txt`. You can install these libraries with:

`pip install -r requirements.txt`

### Dataset, training/validaiton splits, and features
We provide CSVs for training/validation/and testing on EPIC-KITCHENS-55 in the `data/ek55` directory. A brief description of each csv follows:
 * `actions.csv`: maps action ids to (verb,noun) pairs;
 * `EPIC_many_shot_nouns.csv`: contains the list of many shot nouns for class-aware metrics (please refer to [the EPIC-KITCHENS-55 paper](https://arxiv.org/abs/1804.02748) for more details);
 * `EPIC_many_shot_verbs.csv`: similar to the previous one, but related to verbs;
 * `test_seen.csv`: contains the timestamps (expressed in number of frames) of the "seen" test set (S1);
 * `test_unseen.csv`: contains the timestamps (expressed in number of frames) of the "unseen" test set (S2);
 * `training.csv`: contains annotations for the training set in our training/validation split;
 * `validation.csv`: contains annotations for the validation set in our training/validation split;
 * `training_videos.csv`: contains the list of training videos in our training/validation split;
 * `validation_videos.csv`: contains the list of validation videos in our training/validation split;
We also provide CSVs for training/validation/testing on EPIC-KITCHENS-100 in the `data/ek100` directory. 

Training and validation CSVs report the following columns:
 * Annotation ID;
 * Video name (without extension);
 * Start frame;
 * End frame;
 * Verb ID;
 * Noun ID;
 * Action ID.
 
The test CSVs do not report the last three columns since test annotations are not public. These CSVs are provided to allow producing predicitons in JSON format to be submitted to the leaderboard.

Please note that time-stamps are reported in terms of frame numbers in the csvs. This has been done by assuming a fixed framerate of 30fps. Since the original videos have been collected a different framerates, we first converted all videos to 30fps using ffmpeg.

We provide pre-extracted features. The features are stored in LMDB datasets. To download them, run the following commands:

 * EPIC-KITCHENS-55: `./scripts/download_data_ek55.sh`;

Alternatively, you can download features extracted from each frame by using the script:

 * EPIC-KITCHENS-55: `./scripts/download_data_ek55_full.sh`;
 * EPIC-KITCHENS-100: `./scripts/download_data_ek100_full.sh`;

Please note that this download is significantly heavier and that it is not required to run the training with default parameters on EPIC-KITCHENS-55.

This should populate three directories `data/ek{55|100}/rgb`, `data/ek{55|100}/flow`, `data/ek{55|100}/obj` with the LMDB datasets.

### Trainining
Models can be trained using the `main.py` program. For instance, to train the RGB branch for the action anticipation task, use the following commands:

#### EPIC-KITCHENS-55
 * `mkdir models/`
 * `python main.py train data/ek55 models/ek55 --modality rgb --task anticipation --sequence_completion`
 * `python main.py train data/ek55 models/ek55 --modality rgb --task anticipation`

#### EPIC-KITCHENS-100
 * `mkdir models/`
 * `python main.py train data/ek100 models/ek100 --modality rgb --task anticipation --sequence_completion --num_class 3806 --mt5r`
 * `python main.py train data/ek100 models/ek100 --modality rgb --task anticipation --num_class 3806 --mt5r`

This will first pre-train using sequence completion, then fine-tune to the main anticipation task. All models will be stored in the `models/ek{55|100}` directory.

Optionally, a `--visdom` flag can be passed to the training program in order to enable loggin using visdom. To allow this, it is necessary to install visdom with:

`pip install visdom`

And run it with:

`python -m visdom.server`

Similar commands can be used to train all models. The following scripts contain all commands required to train the models for egocentric action anticipation and early action recognition:
 * `scripts/train_anticipation_ek{55|100}.sh`;
 * `scripts/train_recognition_ek55.sh`.

### Validation
The anticipation models can be validated using the following commands:

#### Action Anticipation

##### EPIC-KITCHENS-55
 * RGB branch: `python main.py validate data/ek55 models/ek55 --modality rgb --task anticipation`;
 * Optical Flow branch: `python main.py validate data/ek55 models/ek55 --modality flow --task anticipation`;
 * Object branch: `python main.py validate data/ek55 models/ek55 --modality obj --task anticipation --feat_in 352`;
 * Complete architecture with MATT: `python main.py validate data/ek55 models/ek55 --modality fusion --task anticipation`.

##### EPIC-KITCHENS-100
 * RGB branch: `python main.py validate data/ek100 models/ek100 --modality rgb --task anticipation --num_class 3806 --mt5r -ek100`;
 * Optical Flow branch: `python main.py validate data/ek100 models/ek100 --modality flow --task anticipation --num_class 3806 --mt5r -ek100`;
 * Object branch: `python main.py validate data/ek100 models/ek100 --modality obj --task anticipation --feat_in 352 --num_class 3806 --mt5r -ek100`;
 * Complete architecture with MATT: `python main.py validate data/ek100 models/ek100 --modality fusion --task anticipation --num_class 3806 --mt5r -ek100`.

These instructions will evaluate the models using the official measures of the EPIC-KITCHENS-100 dataset for the action anticipation challenge.

###### Validation Jsons
You can produce validation jsons as follows:

 * `mkdir -p jsons/ek100`;
 * Anticipation: `python main.py validate_json data/ek100 models/ek100 --modality fusion --task anticipation --json_directory jsons/ek100 --ek100 --num_class 3806 --mt5r`;
 * Early recognition: `python main.py validate_json data/ek100 models/ek100 --modality fusion --task early_recognition --json_directory jsons/ek100 -ek100 --num_class 3806 --mt5r`.

#### Early Action Recognition
Similarly, for early action recognition:

#### EPIC-KITCHENS-55
 * RGB branch: `python main.py validate data models --modality rgb --task early_recognition`;
 * Optical Flow branch: `python main.py validate data models --modality flow --task early_recognition`;
 * Object branch: `python main.py validate data models --modality obj --task early_recognition --feat_in 352`;
 * Late fusion model: `python main.py validate data models --modality fusion --task early_recognition`.

### Test
 The `main.py` program also allows to run the models on the EPIC-KITCHENS-55 and EPIC-KITCHENS-100 test sets and produce jsons to be sent to the leaderboard (see [http://epic-kitchens.github.io/](http://epic-kitchens.github.io/)). To test models, you can use the following commands:

#### EPIC-KITCHENS-55
 * `mkdir -p jsons/ek55`;
 * Anticipation: `python main.py test data/ek55 models/ek55 --modality fusion --task anticipation --json_directory jsons/ek55`;
 * Early recognition: `python main.py test data/ek55 models/ek55 --modality fusion --task early_recognition --json_directory jsons/ek55`.

#### EPIC-KITCHENS-100
 * `mkdir -p jsons/ek100`;
 * Anticipation: `python main.py test data/ek100 models/ek100 --modality fusion --task anticipation --json_directory jsons/ek100 --ek100 --num_class 3806 --mt5r`;
 * Early recognition: `python main.py test data/ek100 models/ek100 --modality fusion --task early_recognition --json_directory jsons/ek100 -ek100 --num_class 3806 --mt5r`.

### Pretrained Models

#### EPIC-KITCHENS-55
We provide the official checkpoints used to report the results on EPIC-KITCHENS-55 in our ICCV paper. These can be downloaded using the script:

`./script/download_models_ek55.sh`

The models will be downloaded in `models/ek55`. You can test the model and obtain the results reported in the paper using the same `main.py` program. For instance:

`python main.py test data/ek55 models/ek55 --modality fusion --task anticipation --json_directory jsons`

#### EPIC-KITCHENS-100
We provide the checkpoints used to report the results in the EPIC-KITCHENS-100 paper (https://arxiv.org/abs/2006.13256). These can be downloaded using the script:

`./script/download_models_ek100.sh`

The models will be downloaded in `models/ek100`. You can produce the validation and test jsons replicating the results of the paper as follows:

 * `python main.py test data/ek100 models/ek100 --modality fusion --task anticipation --json_directory jsons --ek100 --mt5r`
 * `python main.py validate_json data/ek100 models/ek100 --modality fusion --task anticipation --json_directory jsons --ek100 --mt5r`

## TSN models
Can be downloaded from the following URLs:

### EPIC-KITCHENS-55
 * RGB: `http://iplab.dmi.unict.it/sharing/rulstm/TSN-rgb.pth.tar`;
 * Flow: `http://iplab.dmi.unict.it/sharing/rulstm/TSN-flow.pth.tar`.
 
### EPIC-KITCHENS-100
 * RGB: `http://iplab.dmi.unict.it/sharing/rulstm/TSN-rgb-ek100.pth.tar`;
 * Flow: `http://iplab.dmi.unict.it/sharing/rulstm/TSN-flow-ek100.pth.tar`.

## Faster-RCNN Model Trained on EPIC-KITCHENS-55
We release the Faster-RCNN object detector trained on EPIC-KITCHENS-55 that we used for our experiments. The detector has been trained using the [detectron](https://github.com/facebookresearch/Detectron) library. The `yaml` configuration file used to train the model is available in the `FasterRCNN` directory of this repository. The weights can be downloaded from [this link](http://iplab.dmi.unict.it/rulstm/downloads/ek18-2gpu-e2e-faster-rcnn-R-101-FPN_1x.pkl).

### Usage
Make sure the detectron library is installed and available in the system path. A good idea might be to use a docker container. Please refer to [https://github.com/facebookresearch/Detectron/blob/master/INSTALL.md](https://github.com/facebookresearch/Detectron/blob/master/INSTALL.md) for more details.

Sample usage:
 * clone the repository with `git clone https://github.com/antoninofurnari/rulstm.git`;
 * move to the FasterRCNN directory with `cd rulstm/FasterRCNN/`;
 * download the weights with `curl -o weights/ek18-2gpu-e2e-faster-rcnn-R-101-FPN_1x.pkl http://iplab.dmi.unict.it/rulstm/downloads/ek18-2gpu-e2e-faster-rcnn-R-101-FPN_1x.pkl`;
 * run the sample detection script on a video with `python tools/detect_video.py --cfg config/ek18-2gpu-e2e-faster-rcnn-R-101-FPN_1x.yaml --wts weights/ek18-2gpu-e2e-faster-rcnn-R-101-FPN_1x.pkl path/to/video.mp4`.
 
A new file `path/to/video.mp4_detections.npy` will be created. The file will contain a list of arrays reporting the coordinates of the objects detected in each frame of the video. Specifically, the detections of a given frame will be contained in a tensor of shape `N x 6`, where:
 * `N` is the number of object detected in the frame;
 * the first column contains the IDs of the detected objects;
 * the fifth column contains the detection confidence scores;
 * columns 2-5 contain the coordinates of the detected bounding boxes in the format: `[xmin, ymin, xmax, ymax]`.
Please refer to [https://github.com/epic-kitchens/annotations/blob/master/EPIC_noun_classes.csv](https://github.com/epic-kitchens/annotations/blob/master/EPIC_noun_classes.csv) for the list of object ids.

## Feature Extraction
A few example scripts showing how we performed feature extraction from video, can be found in the `FEATEXT` directory. 

To extract features using the TSN models, it is necessary to install the `pretrainedmodels` package through `pip install pretrainedmodels`.

To run the examples follow these steps:
 * `cd FEATEXT`;
 * `./scripts/download_models.sh`;
 * Extract sample data with `tar xvf data.tar`. This will extract a few files in the `data` folder. These include:
   * RGB frames. In general, these can be extracted using programs such as `ffmpeg`;
   * Optical flows. These have been extracted using the TVL1 algorithm. Different programs can be used to extract the optical flow son GPU. For instance, https://github.com/feichtenhofer/gpu_flow;
   * Object bounding boxes. These can be extracted using the Faster R-CNN model provided with this repository (see the `detect_video.py` script in `FasterRCNN/tools`;
 * Create the destination directory for the features with `mkdir features`;
 * Run the example scripts:
   * `python extract_sample_rgb.py`;
   * `python extract_sample_flow.py`;
   * `python extract_sample_obj.py`;
 * The scripts will create sample LMDB dbs in `features`;
 
## EGTEA Gaze+ Pre-Extracted Features
We provide the EGTEA Gaze+ features used for the experiments (see paper for the details) at [https://iplab.dmi.unict.it/sharing/rulstm/features/egtea.zip](https://iplab.dmi.unict.it/sharing/rulstm/features/egtea.zip). The features have been extracted using three different TSN models trained following the official splits proposed by the authors of EGTEA Gaze+ (see [http://cbs.ic.gatech.edu/fpv/](http://cbs.ic.gatech.edu/fpv/)). The annotations formatted in a way to be directly usable with this repository can be found in `RULSTM/data/egtea`.

**Note**: a previous version of the zip file contained the following LMDB databases:
 * `TSN-C_3_egtea_action_CE_flow_model_best_fcfull_hd`;
 * `TSN-C_3_egtea_action_CE_rgb_model_best_fcfull_hd`;   
 * `TSN-C_3_egtea_action_CE_s1_rgb_model_best_fcfull_hd`;
 * `TSN-C_3_egtea_action_CE_s1_flow_model_best_fcfull_hd`; 
 * `TSN-C_3_egtea_action_CE_s2_rgb_model_best_fcfull_hd`;
 * `TSN-C_3_egtea_action_CE_s2_flow_model_best_fcfull_hd`;
 * `TSN-C_3_egtea_action_CE_s3_rgb_model_best_fcfull_hd`;
 * `TSN-C_3_egtea_action_CE_s3_flow_model_best_fcfull_hd`.
 
The first two databases had been included **by mistake and should be ignored**, instead, the remaining six databases should be used for the experiments when the standard evaluation protocol based on three splits is adopted. The following paragraph explains in detail how they have been created:
 * `TSN-C_3_egtea_action_CE_s1_rgb_model_best_fcfull_hd`: features extracted using an RGB TSN model training using s2 and s3 as training set;
 * `TSN-C_3_egtea_action_CE_s1_flow_model_best_fcfull_hd`: features extracted using a Flow TSN model training using s2 and s3 as training set; 
 * `TSN-C_3_egtea_action_CE_s2_rgb_model_best_fcfull_hd`: features extracted using an RGB TSN model training using s1 and s3 as training set;
 * `TSN-C_3_egtea_action_CE_s2_flow_model_best_fcfull_hd`: features extracted using a Flow TSN model training using s1 and s3 as training set;
 * `TSN-C_3_egtea_action_CE_s3_rgb_model_best_fcfull_hd`: features extracted using an RGB TSN model training using s1 and s2 as training set;
 * `TSN-C_3_egtea_action_CE_s3_flow_model_best_fcfull_hd`: features extracted using a Flow TSN model training using s1 and s2 as training set.

An updated version of the zip file including only the correct databases is available at [https://iplab.dmi.unict.it/sharing/rulstm/features/egtea.zip](https://iplab.dmi.unict.it/sharing/rulstm/features/egtea.zip). 

## Object detections on EPIC-KITCHENS-100
We provide object detections obtained on each frame of EPIC-KITCHENS-100. The detections have been obtained by running the Faster RCNN model trained on EPIC-KITCHENS-55 described above and included in this repository. You can download a zip file containing all detections through this link: [https://iplab.dmi.unict.it/sharing/rulstm/detected_objects.zip](https://iplab.dmi.unict.it/sharing/rulstm/detected_objects.zip).

**Note** These detections are a superset of the ones used for the original experiments on EPIC-KITCHENS-55. If you are experimenting with EK-55, you can just discard the extra videos not belonging to EK-55.

The zip file contains a `npy` file for each video in EPIC-KITCHENS-100. For examle:

```
P01_01.MP4_detections.npy
P01_02.MP4_detections.npy
P01_03.MP4_detections.npy
P01_04.MP4_detections.npy
P01_05.MP4_detections.npy
P01_06.MP4_detections.npy
...
```

Each file contains all object detections obtained in the video referenced in the filename. You can load these `npy` files as in this example code:

```python
import numpy as np
data=np.load('P04_101.MP4_detections.npy', allow_pickle=True, encoding='latin1')
```

`data` will be a 1-dimensional numpy ndarray containing `n` entries, where `n` is the number of frames in the video. The `n-th` entry of the dataframe will be an array of shape `m \times 6` where, `m` is the number of objects detected in the frame. The six columns contain respectively:
 * The class id. Please note that the background class is specified as `0`, so it is necessary to subtract `1` in order to match the noun class IDs reported in https://github.com/epic-kitchens/epic-kitchens-55-annotations/blob/master/EPIC_noun_classes.csv;
 * The `x1`, `y1`, `x2`, `y2` bounding box coordinates;
 * The detection confidence score.

The following example code separates class ids, box coordinates and confidence scores:

```python
object_classes = data[:,0]-1
object_boxes = data[:,1:5]
detection_scores = data[:,-1]
```

## Related Works
 * A. Furnari, S. Battiato, K. Grauman, G. M. Farinella, Next-Active-Object Prediction from Egocentric Videos, Journal of Visual Communication and Image Representation, 2017. [Download](https://arxiv.org/pdf/1904.05250.pdf);
 * A. Furnari, S. Battiato, G. M. Farinella (2018). Leveraging Uncertainty to Rethink Loss Functions and Evaluation Measures for Egocentric Action Anticipation . In International Workshop on Egocentric Perception, Interaction and Computing (EPIC) in conjunction with ECCV . [code](https://github.com/fpv-iplab/action-anticipation-losses);
 * More related works at [http://iplab.dmi.unict.it/fpv/](http://iplab.dmi.unict.it/fpv/) and [https://github.com/fpv-iplab](https://github.com/fpv-iplab).
