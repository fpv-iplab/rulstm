""" Implements a dataset object which allows to read representations from LMDB datasets in a multi-modal fashion
The dataset can sample frames for both the anticipation and early recognition tasks."""

import numpy as np
import lmdb
from tqdm import tqdm
from torch.utils import data
import pandas as pd

def read_representations(frames, env, tran=None):
    """ Reads a set of representations, given their frame names and an LMDB environment.
    Applies a transformation to the features if provided"""
    features = []
    # for each frame
    for f in frames:
        # read the current frame
        with env.begin() as e:
            dd = e.get(f.strip().encode('utf-8'))
        if dd is None:
            print(f)
        # convert to numpy array
        data = np.frombuffer(dd, 'float32')
        # append to list
        features.append(data)
    # convert list to numpy array
    features=np.array(features)
    # apply transform if provided
    if tran:
        features=tran(features)
    return features

def read_data(frames, env, tran=None):
    """A wrapper form read_representations to handle loading from more environments.
    This is used for multimodal data loading (e.g., RGB + Flow)"""
    # if env is a list
    if isinstance(env, list):
        # read the representations from all environments
        l = [read_representations(frames, e, tran) for e in env]
        return l
    else:
        # otherwise, just read the representations
        return read_representations(frames, env, tran)

class SequenceDataset(data.Dataset):
    def __init__(self, path_to_lmdb, path_to_csv, label_type = 'action',
                time_step = 0.25, sequence_length = 14, fps = 30,
                img_tmpl = "frame_{:010d}.jpg",
                transform = None,
                challenge = False,
                past_features = True,
                action_samples = None):
        """
            Inputs:
                path_to_lmdb: path to the folder containing the LMDB dataset
                path_to_csv: path to training/validation csv
                label_type: which label to return (verb, noun, or action)
                time_step: in seconds
                sequence_length: in time steps
                fps: framerate
                img_tmpl: image template to load the features
                tranform: transformation to apply to each sample
                challenge: allows to load csvs containing only time-stamp for the challenge
                past_features: if past features should be returned
                action_samples: number of frames to be evenly sampled from each action
        """

        # read the csv file
        if challenge:
            self.annotations = pd.read_csv(path_to_csv, header=None, names=['video','start','end'])
        else:
            self.annotations = pd.read_csv(path_to_csv, header=None, names=['video','start','end','verb','noun','action'])

        
        self.challenge=challenge
        self.path_to_lmdb = path_to_lmdb
        self.time_step = time_step
        self.past_features = past_features
        self.action_samples = action_samples
        self.fps=fps
        self.transform = transform
        self.label_type = label_type
        self.sequence_length = sequence_length
        self.img_tmpl = img_tmpl
        self.action_samples = action_samples
        
        # initialize some lists
        self.ids = [] # action ids
        self.discarded_ids = [] # list of ids discarded (e.g., if there were no enough frames before the beginning of the action
        self.discarded_labels = [] # list of labels discarded (e.g., if there were no enough frames before the beginning of the action
        self.past_frames = [] # names of frames sampled before each action
        self.action_frames = [] # names of frames sampled from each action
        self.labels = [] # labels of each action
        
        # populate them
        self.__populate_lists()

        # if a list to datasets has been provided, load all of them
        if isinstance(self.path_to_lmdb, list):
            self.env = [lmdb.open(l, readonly=True, lock=False) for l in self.path_to_lmdb]
        else:
            # otherwise, just load the single LMDB dataset
            self.env = lmdb.open(self.path_to_lmdb, readonly=True, lock=False)

    def __get_frames(self, frames, video):
        """ format file names using the image template """
        frames = np.array(list(map(lambda x: video+"_"+self.img_tmpl.format(x), frames)))
        return frames
    
    def __populate_lists(self):
        """ Samples a sequence for each action and populates the lists. """
        for _, a in tqdm(self.annotations.iterrows(), 'Populating Dataset', total = len(self.annotations)):

            # sample frames before the beginning of the action
            frames = self.__sample_frames_past(a.start)

            if self.action_samples:
                # sample frames from the action
                # to sample n frames, we first sample n+1 frames with linspace, then discard the first one
                action_frames = np.linspace(a.start, a.end, self.action_samples+1, dtype=int)[1:]

            # check if there were enough frames before the beginning of the action
            if frames.min()>=1: #if the smaller frame is at least 1, the sequence is valid
                self.past_frames.append(self.__get_frames(frames, a.video))
                self.ids.append(a.name)
                # handle whether a list of labels is required (e.g., [verb, noun]), rather than a single action
                if isinstance(self.label_type, list):
                    if self.challenge: # if sampling for the challenge, there are no labels, just add -1
                        self.labels.append(-1)
                    else:
                        # otherwise get the required labels
                        self.labels.append(a[self.label_type].values.astype(int))
                else: #single label version
                    if self.challenge:
                        self.labels.append(-1)
                    else:
                        self.labels.append(a[self.label_type])
                if self.action_samples:
                    self.action_frames.append(self.__get_frames(action_frames, a.video))
            else:
                #if the sequence is invalid, do nothing, but add the id to the discarded_ids list
                self.discarded_ids.append(a.name)
                if isinstance(self.label_type, list):
                    if self.challenge: # if sampling for the challenge, there are no labels, just add -1
                        self.discarded_labels.append(-1)
                    else:
                        # otherwise get the required labels
                        self.discarded_labels.append(a[self.label_type].values.astype(int))
                else: #single label version
                    if self.challenge:
                        self.discarded_labels.append(-1)
                    else:
                        self.discarded_labels.append(a[self.label_type])

    def __sample_frames_past(self, point):
        """Samples frames before the beginning of the action "point" """
        # generate the relative timestamps, depending on the requested sequence_length
        # e.g., 2.  , 1.75, 1.5 , 1.25, 1.  , 0.75, 0.5 , 0.25
        # in this case "2" means, sample 2s before the beginning of the action
        time_stamps = np.arange(self.time_step,self.time_step*(self.sequence_length+1),self.time_step)[::-1]
        
        # compute the time stamp corresponding to the beginning of the action
        end_time_stamp = point/self.fps 

        # subtract time stamps to the timestamp of the last frame
        time_stamps = end_time_stamp-time_stamps

        # convert timestamps to frames
        # use floor to be sure to consider the last frame before the timestamp (important for anticipation!)
        # and never sample any frame after that time stamp 
        frames = np.floor(time_stamps*self.fps).astype(int)
        
        # sometimes there are not enough frames before the beginning of the action
        # in this case, we just pad the sequence with the first frame
        # this is done by replacing all frames smaller than 1
        # with the first frame of the sequence
        if frames.max()>=1:
            frames[frames<1]=frames[frames>=1].min()

        return frames

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        """ sample a given sequence """
        # get past frames
        past_frames = self.past_frames[index]

        if self.action_samples:
            # get action frames
            action_frames = self.action_frames[index]

        # return a dictionary containing the id of the current sequence
        # this is useful to produce the jsons for the challenge
        out = {'id':self.ids[index]}

        if self.past_features:
            # read representations for past frames
            out['past_features'] = read_data(past_frames, self.env, self.transform)

        # get the label of the current sequence
        label = self.labels[index]
        out['label'] = label

        if self.action_samples:
            # read representations for the action samples
            out['action_features'] = read_data(action_frames, self.env, self.transform)

        return out

