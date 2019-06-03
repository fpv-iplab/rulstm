# Code for the paper: What Would You Expect? Anticipating Egocentric Actions with Rolling-Unrolling LSTMs and Modality Attention
This repository hosts the code related to the following paper:

A. Furnari, G. M. Farinella, What Would You Expect? Anticipating Egocentric Actions with Rolling-Unrolling LSTMs and Modality Attention. [Download](https://arxiv.org/pdf/1905.09035.pdf)

```
@article{furnari2019rulstm, 
  title = { What Would You Expect? Anticipating Egocentric Actions with Rolling-Unrolling LSTMs and Modality Attention. }, 
  author = { Antonino Furnari and Giovanni Maria Farinella },
  year = { 2019 },
  journal = { arXiv preprint arXiv:1905.09035 },
}
```
**The code is not fully available yet. Please stay tuned, it will be released soon. More information at the project web page: [http://iplab.dmi.unict.it/rulstm/](http://iplab.dmi.unict.it/rulstm/)**

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
