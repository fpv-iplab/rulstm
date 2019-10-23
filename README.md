# What Would You Expect? Anticipating Egocentric Actions with Rolling-Unrolling LSTMs and Modality Attention
This repository hosts the code related to the following paper:

Antonino Furnari and Giovanni Maria Farinella, What Would You Expect? Anticipating Egocentric Actions with Rolling-Unrolling LSTMs and Modality Attention. International Conference on Computer Vision, 2019. [Download](https://arxiv.org/pdf/1905.09035.pdf)

Please also see the project web page at [http://iplab.dmi.unict.it/rulstm](http://iplab.dmi.unict.it/rulstm).

If you use the code/models hosted in this repository, please cite the following paper:

```
@inproceedings{furnari2019rulstm, 
  title = { What Would You Expect? Anticipating Egocentric Actions with Rolling-Unrolling LSTMs and Modality Attention. }, 
  author = { Antonino Furnari and Giovanni Maria Farinella },
  year = { 2019 },
  booktitle = { International Conference on Computer Vision (ICCV) },
}
```
## Updates:
 * **11/10/2019** We are now also providing TSN and object-based features extracted for **each frame of EPIC-KITCHENS**. They can be downloaded using the `download_data_full.sh` script rather than `download_data.sh`;
 * **23/10/2019** Added some scripts to show how to extract features from videos. The scripts can be found under `FEATEXT` and are documented in this README. 
## Overview
This repository provides the following components:
 * The official PyTorch implementation of the proposed Rolling-Unrolling LSTM approach, including Sequence-Completion Pre-Training and Modality ATTention (MATT);
 * A program to train, validate and test the proposed method on the EPIC-Kitchens dataset;
 * Pre-extracted features for a subset of frames of EPIC-Kitchens. Specifically, we include:
   * RGB features: extracted from RGB iamges using a BNInception CNN trained for the task of *egocentric action recognition* using [Temporal Segment Networks](https://github.com/yjxiong/tsn-pytorch);
   * Flow features: similar to RGB features, but extracted with a BNInception CNN trained on optical flow;
   * OBJ features: object-based features obtained by running a Faster R-CNN object detector trained on EPIC-Kitchens;
 * The checkpoints of the RGB/Flow/OBJ/Fusion models trained for both tasks: egocentric action anticipation and early action recognition;
 * The checkpoints of the TSN models (to be used with the [official PyTorch implementation of TSN](https://github.com/yjxiong/tsn-pytorch));
 * The checkpoint of the Faster R-CNN object detector trained on EPIC-Kitchens;
 * The training/validation split used for the experiments. Note that the TSN and Faster R-CNN models have been trained on the training set of this split.

Please, refer to the paper for more technical details. The following sections document the released material.

## RU-LSTM Implementation and main training/validaiton/test program
The provided implementation and training/validation/test program can be found in the `RULSTM` directory. In order to proceed to training, it is necessary to retrieve the pre-extracted features from our website. To save space and bandwidth, we provide features extracted only on the subset of frames used for the experiments (we sampled frames at about 4fps - please see the paper). These features are sufficient to train/validate/test the methods on the whole EPIC-Kitchens dataset following the settings reported in the paper.

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
We provide CSV for training/validation/and testing on EPIC-Kitchens in the `data` directory. A brief description of each csv follows:
 * `actions.csv`: maps action ids to (verb,noun) pairs;
 * `EPIC_many_shot_nouns.csv`: contains the list of many shot nouns for class-aware metrics (please refer to [the EPIC-Kitchens paper](https://arxiv.org/abs/1804.02748) for more details);
 * `EPIC_many_shot_verbs.csv`: similar to the previous one, but related to verbs;
 * `test_seen.csv`: contains the timestamps (expressed in number of frames) of the "seen" test set (S1);
 * `test_unseen.csv`: contains the timestamps (expressed in number of frames) of the "unseen" test set (S2);
 * `training.csv`: contains annotations for the training set in our training/validation split;
 * `validation.csv`: contains annotations for the validation set in our training/validation split;
 * `training_videos.csv`: contains the list of training videos in our training/validation split;
 * `validation_videos.csv`: contains the list of validation videos in our training/validation split;

 Please note that time-stamps are reported in terms of frame numbers in the csvs. This has been done by assuming a fixed framerate of 30fps. Since the original videos have been collected a different framerates, we first converted all videos to 30fps using ffmpeg.

We also provide pre-extracted features. The features are stored in LMDB datasets. To download them, run the following command:

`./scripts/download_data.sh`

Alternatively, you can download features extracted from each frame by using the script:

`./scripts/download_data_full.sh`

Please note that this download is significantly heavier and that it is not required to run the training with default parameters.

This should populate three directories `data/rgb`, `data/flow`, `data/obj` with the LMDB datasets.

### Trainining
Models can be trained using the `main.py` program. For instance, to train the RGB branch for the action anticipation task, use the following commands:

 * `mkdir models`
 * `python main.py train data models --modality rgb --task anticipation --sequence_completion`
 * `python main.py train data models --modality rgb --task anticipation`

This will first pre-train using sequence completion, then fine-tune to the main anticipation task. All models will be stored in the `models` directory.

Optionally, a `--visdom` flag can be passed to the training program in order to enable loggin using visdom. To allow this, it is necessary to install visdom with:

`pip install visdom`

And run it with:

`python -m visdom.server`

Similar commands can be used to train all models. The following scripts contain all commands required to train the models for egocentric action anticipation and early action recognition:
 * `scripts/train_anticipation.sh`;
 * `scripts/train_recognition.sh`.

### Validation
The anticipation models can be validated using the following commands:
 * RGB branch: `python main.py validate data models --modality rgb --task anticipation`;
 * Optical Flow branch: `python main.py validate data models --modality flow --task anticipation`;
 * Object branch: `python main.py validate data models --modality obj --task anticipation --feat_in 352`;
 * Complete architecture with MATT: `python main.py validate data models --modality fusion --task anticipation`.

Similarly, for early action recognition:
* RGB branch: `python main.py validate data models --modality rgb --task early_recognition`;
 * Optical Flow branch: `python main.py validate data models --modality flow --task early_recognition`;
 * Object branch: `python main.py validate data models --modality obj --task early_recognition --feat_in 352`;
 * Late fusion model: `python main.py validate data models --modality fusion --task early_recognition`.

 ### Test
 The `main.py` program also allows to run the models on the EPIC-Kitchens test sets and produce jsons to be sent to the leaderboard (see [http://epic-kitchens.github.io/](http://epic-kitchens.github.io/)). To test models, you can use the following commands:

  * `mkdir jsons`;
  * Anticipation: `python main.py test data models --modality fusion --task anticipation --json_directory jsons`;
  * Early recognition: `python main.py test data models --modality fusion --task early_recognition --json_directory jsons`.

### ICCV Models
We also provide the official checkpoints used to report the results in our ICCV paper. These can be downloaded using the script:

`./script/download_models.sh`

The models will be downloaded in `iccv_models`. You can test the model and obtain the results reported in the paper using the same `main.py` program. For instance:

`python main.py test data iccv_models --modality fusion --task anticipation --json_directory jsons`

## TSN models
Can be downloaded from the following URLs:
 * RGB: `http://iplab.dmi.unict.it/rulstm/downloads/TSN-rgb.pth.tar`;
 * Flow: `http://iplab.dmi.unict.it/rulstm/downloads/TSN-flow.pth.tar`.

## Faster-RCNN Model Trained on EPIC-Kitchens
We release the Faster-RCNN object detector trained on EPIC-Kitchens that we used for our experiments. The detector has been trained using the [detectron](https://github.com/facebookresearch/Detectron) library. The `yaml` configuration file used to train the model is available in the `FasterRCNN` directory of this repository. The weights can be downloaded from [this link](http://iplab.dmi.unict.it/rulstm/downloads/ek18-2gpu-e2e-faster-rcnn-R-101-FPN_1x.pkl).

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

## Related Works
 * A. Furnari, S. Battiato, K. Grauman, G. M. Farinella, Next-Active-Object Prediction from Egocentric Videos, Journal of Visual Communication and Image Representation, 2017. [Download](https://arxiv.org/pdf/1904.05250.pdf);
 * A. Furnari, S. Battiato, G. M. Farinella (2018). Leveraging Uncertainty to Rethink Loss Functions and Evaluation Measures for Egocentric Action Anticipation . In International Workshop on Egocentric Perception, Interaction and Computing (EPIC) in conjunction with ECCV . [code](https://github.com/fpv-iplab/action-anticipation-losses);
 * More related works at [http://iplab.dmi.unict.it/fpv/](http://iplab.dmi.unict.it/fpv/) and [https://github.com/fpv-iplab](https://github.com/fpv-iplab).
