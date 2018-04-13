import numpy as np
import os

from PIL import Image
from .UsefulDataset import UsefulDataset
from .data_utils import npy_loader, pil_loader, _find_classes, _finds_inputs_and_targets

class PredictFolderDataset(UsefulDataset):
    def __init__(self,
                 root,
                 input_regex='*',
                 input_transform=None,
                 input_loader='pil',
                 exclusion_file=None):
        """
        Dataset class for loading out-of-memory data that is more geared toward prediction data loading (where ground truth is not available). \n

        Arguments
        ---------
        :param root: string\n
            path to main directory\n

        :param input_regex: string (default is any valid image file)\n
            regular expression to find input images\n
            e.g. if all your inputs have the word 'input',
            you'd enter something like input_regex='*input*'

        :param input_transform: torch transform\n
            transform to apply to each input before returning

        :param input_loader: string in `{'npy', 'pil'} or callable  (default: pil)\n
            defines how to load input samples from file\n
            if a function is provided, it should take in a file path as input and return the loaded sample.

        :param exclusion_file: string\n
            list of files to exclude when enumerating all files.\n
            The list must be a full path relative to the root parameter
        """

        # call the super constructor first, then set our own parameters
        super().__init__()

        if input_loader == 'npy':
            input_loader = npy_loader
        elif input_loader == 'pil':
            input_loader = pil_loader
        self.file_loader = input_loader

        root = os.path.expanduser(root)

        # this returns (optionally) partitioned data but since we're not interested in partitioning, we just use the first partition (which contains everything)
        data, _ = _finds_inputs_and_targets(root, class_mode='path', input_regex=input_regex, exclusion_file=exclusion_file)

        if len(data) == 0:
            raise (RuntimeError('Found 0 data items in subfolders of: %s' % root))
        else:
            print('Found %i data items' % len(data))

        self.root = os.path.expanduser(root)
        self.input_transform = input_transform
        self.data = data

    def __getitem__(self, index):
        # get paths
        input_path, target_path = self.data[index]

        # load samples into memory
        input_sample = self.file_loader(input_path)  # do generic data read

        # apply transform
        if self.input_transform is not None:
            input_sample = self.input_transform(input_sample)

        return input_sample, target_path

    def __len__(self):
        return len(self.data)

    def getdata(self):
        return self.data

    def getmeta_data(self):
        meta = {'num_inputs': self.num_inputs,  # these are hardcoded for the fit module to work
                'num_targets': self.num_targets,
                'input_transform': self.input_transform,
                'file_loader': self.file_loader
                }
        return meta