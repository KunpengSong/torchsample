
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from .datasetz.usefuldataset import UsefulDataset

import random
# import torch.utils.data

import os
import os.path
import warnings
import fnmatch

import numpy as np
import pandas as pd
import torch as th

from . import transforms

try:
    from PIL import Image
except:
    warnings.warn('Cant import PIL.. Cant load PIL images')

def pil_loader(path, color_space=''):
    try:
        if color_space.lower() == 'rgb':
            return Image.open(path).convert('RGB')
        elif color_space.lower() == 'l':
            return Image.open(path).convert('L')
        elif color_space.lower() == '1' or color_space.lower() == 'binary':
            return Image.open(path).convert('1')
        else:
            return Image.open(path)
    except OSError:
        print("!!!  Could not read path: " + path)
        exit(2)

def npy_loader(path, color_space=None):     # color space is unused here
    return np.load(path)



class FolderDataset(UsefulDataset):

    def __init__(self, 
                 root,
                 class_mode='label',
                 input_regex='*',
                 rel_target_root='',
                 target_prefix='',
                 target_postfix='',
                 target_extension='png',
                 transform=None, 
                 target_transform=None,
                 co_transform=None,
                 apply_co_transform_first=True,
                 file_loader='pil',
                 exclusion_file=None,
                 target_index_map={255:1}):
        """
        Dataset class for loading out-of-memory data.\n
        The rel_target_root parameter is used for image segmentation cases
            Typically the structure will look like the following\n
            |- root (aka training images)\n
            |  - dir1\n
            |  - dir2\n
            |- masks (aka label images)\n
            |  - dir1\n
            |  - dir2\n

        Arguments
        ---------
        :param root: string\n
            path to main directory\n

        :param class_mode: string in `{'label', 'image'}`\n
            type of target sample to look for and return\n
            `label` = return class folder as target\n
            `image` = return another image as target (determined by optional target_prefix/postfix)\n
                NOTE: if class_mode == 'image', in addition to input, you must also provide rel_target_root,
                target_prefix or target_postfix (in any combination).

        :param input_regex: string (default is any valid image file)\n
            regular expression to find input images\n
            e.g. if all your inputs have the word 'input',
            you'd enter something like input_regex='*input*'

        :param rel_target_root: string (default is Nothing)\n
            root of directory where to look for target images RELATIVE to the root dir (first arg)

        :param target_prefix: string (default is Nothing)\n
            prefix to use (if any) when trying to locate the matching target

        :param target_postfix: string\n
            postfix to use (if any) when trying to locate the matching target

        :param transform: torch transform\n
            transform to apply to input sample individually

        :param target_transform: torch transform\n
            transform to apply to target sample individually

        :param co_transform: torch transform\n
            transform to apply to both the input and the target

        :param apply_co_transform_first: bool\n
            whether to apply the co-transform before or after individual transforms (default: True = before)

        :param file_loader: string in `{'npy', 'pil'} or callable  (default: pil)\n
            defines how to load samples from file\n
            if a function is provided, it should take in a file path as input and return the loaded sample.

        :param exclusion_file: string\n
            list of files to exclude when enumerating all files.\n
            The list must be a full path relative to the root parameter

        :param target_index_map: dict (defaults to binary mask: {255:1})\n
            a dictionary that maps pixel values in the image to classes to be recognized.\n
            Used in conjunction with 'image' class_mode to produce a label for semantic segmentation
            For semantic segmentation this is required so the default is a binary mask. However, if you want to turn off
            this feature off then specify target_index_map=None
        """

        if file_loader == 'npy':
            file_loader = npy_loader
        elif file_loader == 'pil':
            file_loader = pil_loader
        self.file_loader = file_loader

        root = os.path.expanduser(root)

        classes, class_to_idx = _find_classes(root)
        data, _ = _finds_inputs_and_targets(root, class_mode=class_mode, class_to_idx=class_to_idx, input_regex=input_regex,
                                            rel_target_root=rel_target_root, target_prefix=target_prefix, target_postfix=target_postfix,
                                            target_extension=target_extension, exclusion_file=exclusion_file)

        if len(data) == 0:
            raise(RuntimeError('Found 0 images in subfolders of: %s' % root))
        else:
            print('Found %i images' % len(data))

        self.root = os.path.expanduser(root)
        self.data = data
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.co_transform = co_transform
        self.apply_co_transform_first = apply_co_transform_first
        self.target_index_map = target_index_map
        
        self.class_mode = class_mode

    def __getitem__(self, index):
        # get paths
        input_sample, target_sample = self.data[index]

        if len(self.classes) == 1 and self.class_mode == 'image':          # this is a binary segmentation map
            target_sample = self.file_loader(target_sample, color_space='L')
        else:
            target_sample = self.file_loader(target_sample)
        # load samples into memory
        if self.class_mode == 'image':          # images get special treatment because we have to transform the mask into class values
            input_sample = self.file_loader(input_sample, color_space='rgb')
            # if we're dealing with image masks, we need to change the underlying pixels
            if self.target_index_map:
                target_sample = np.array(target_sample)     # convert to np
                for k, v in self.target_index_map.items():
                    target_sample[target_sample == k] = v   # replace pixels with class values
                target_sample = Image.fromarray(target_sample.astype(np.float32))   # convert back to image
        else:
            input_sample = self.file_loader(input_sample)       # do generic data read

    # def __getitem__(self, index):
    #     img_path, mask_path = self.data[index]
    #     img, mask = Image.open(img_path).convert('RGB'), Image.open(mask_path)
    #
    #     mask = np.array(mask)
    #     mask_copy = mask.copy()
    #     for k, v in self.target_index_map.items():
    #         mask_copy[mask == k] = v
    #     mask = Image.fromarray(mask_copy.astype(np.uint8))
    #
    #     if self.co_transform is not None:
    #         img, mask = self.co_transform(img, mask)
    #     if self.transform is not None:
    #         img = self.transform(img)
    #     if self.target_transform is not None:
    #         mask = self.target_transform(mask)
    #
    #     return img, mask

        
        # apply transforms
        if self.apply_co_transform_first and self.co_transform is not None:
            input_sample, target_sample = self.co_transform(input_sample, target_sample)
        if self.transform is not None:
            input_sample = self.transform(input_sample)
        if self.target_transform is not None:
            target_sample = self.target_transform(target_sample)
        if not self.apply_co_transform_first and self.co_transform is not None:
            input_sample, target_sample = self.co_transform(input_sample, target_sample)

        return input_sample, target_sample

    def __len__(self):
        return len(self.data)

    def getdata(self):
        return self.data

    def getmeta_data(self):
        meta = {'transform': self.transform,
                'target_transform': self.target_transform,
                'co_transform': self.co_transform,
                'class_to_idx': self.class_to_idx,
                'class_mode': self.class_mode,
                'classes': self.classes,
                'file_loader': self.file_loader,
                'apply_co_transform_first': self.apply_co_transform_first,
                'target_index_map': self.target_index_map
                }
        return meta

class ClonedFolderDataset(FolderDataset):
    # TODO: Explore to load everything generically
    #   -  https://stackoverflow.com/questions/19305296/multiple-constructors-in-python-using-inheritance

    def __init__(self, data, meta_data, **kwargs):
        """
        Dataset that can be initialized with a dictionary of internal parameters

        Arguments
        ---------
        :param data: list
            list of data on which the dataset operates

        :param meta_data: dict
            parameters that correspond to the target dataset's attributes
        """

        if len(data) == 0:
            raise (RuntimeError('No data provided'))
        else:
            print('Initializing with %i data items' % len(data))

        self.data = data

        # Source: https://stackoverflow.com/questions/2466191/set-attributes-from-dictionary-in-python
        # generic way of initializing the object
        for key in meta_data:
            setattr(self, key, meta_data[key])
        for key in kwargs:
            setattr(self, key, kwargs[key])

        # self.class_mode = meta_data['class_mode']
        # self.class_to_idx = meta_data['class_to_idx']
        # self.transform = meta_data['transform']
        # self.target_transform = meta_data['target_transform']
        # self.co_transform = meta_data['co_transform']
        # self.apply_co_transform_first = meta_data['apply_co_transform_first']
        # self.file_loader = meta_data['file_loader']

    def __getitem__(self, index):
        return super().__getitem__(index)

    def __len__(self):
        return super().__len__()

    def __getdata__(self):
        return super().__getdata__()

    def __getmeta_data__(self):
        return super().__getmeta_data__()


class BaseDataset(object):
    """An abstract class representing a Dataset.

    All other datasets should subclass it. All subclasses should override
    ``__len__``, that provides the size of the dataset, and ``__getitem__``,
    supporting integer indexing in range from 0 to len(self) exclusive.
    """

    def __len__(self):
        return len(self.inputs) if not isinstance(self.inputs, (tuple,list)) else len(self.inputs[0])

    def add_input_transform(self, transform, add_to_front=True, idx=None):
        if idx is None:
            idx = np.arange(len(self.num_inputs))
        elif not is_tuple_or_list(idx):
            idx = [idx]

        if add_to_front:
            for i in idx:
                self.input_transform[i] = transforms.Compose([transform, self.input_transform[i]])
        else:
            for i in idx:
                self.input_transform[i] = transforms.Compose([self.input_transform[i], transform])

    def add_target_transform(self, transform, add_to_front=True, idx=None):
        if idx is None:
            idx = np.arange(len(self.num_targets))
        elif not is_tuple_or_list(idx):
            idx = [idx]

        if add_to_front:
            for i in idx:
                self.target_transform[i] = transforms.Compose([transform, self.target_transform[i]])
        else:
            for i in idx:
                self.target_transform[i] = transforms.Compose([self.target_transform[i], transform])

    def add_co_transform(self, transform, add_to_front=True, idx=None):
        if idx is None:
            idx = np.arange(len(self.min_inputs_or_targets))
        elif not is_tuple_or_list(idx):
            idx = [idx]

        if add_to_front:
            for i in idx:
                self.co_transform[i] = transforms.Compose([transform, self.co_transform[i]])
        else:
            for i in idx:
                self.co_transform[i] = transforms.Compose([self.co_transform[i], transform])

    def load(self, num_samples=None, load_range=None):
        """
        Load all data or a subset of the data into actual memory.
        For instance, if the inputs are paths to image files, then this
        function will actually load those images.

        Arguments
        ---------
        num_samples : integer (optional)
            number of samples to load. if None, will load all
        load_range : numpy array of integers (optional)
            the index range of images to load
            e.g. np.arange(4) loads the first 4 inputs+targets
        """
        def _parse_shape(x):
            if isinstance(x, (list,tuple)):
                return (len(x),)
            elif isinstance(x, th.Tensor):
                return x.size()
            else:
                return (1,)

        if num_samples is None and load_range is None:
            num_samples = len(self)
            load_range = np.arange(num_samples)
        elif num_samples is None and load_range is not None:
            num_samples = len(load_range)
        elif num_samples is not None and load_range is None:
            load_range = np.arange(num_samples)


        if self.has_target:
            for enum_idx, sample_idx in enumerate(load_range):
                input_sample, target_sample = self.__getitem__(sample_idx)

                if enum_idx == 0:
                    if self.num_inputs == 1:
                        _shape = [len(load_range)] + list(_parse_shape(input_sample))
                        inputs = np.empty(_shape)
                    else:
                        inputs = []
                        for i in range(self.num_inputs):
                            _shape = [len(load_range)] + list(_parse_shape(input_sample[i]))
                            inputs.append(np.empty(_shape))
                        #inputs = [np.empty((len(load_range), *_parse_shape(input_sample[i]))) for i in range(self.num_inputs)]

                    if self.num_targets == 1:
                        _shape = [len(load_range)] + list(_parse_shape(target_sample))
                        targets = np.empty(_shape)
                        #targets = np.empty((len(load_range), *_parse_shape(target_sample)))
                    else:
                        targets = []
                        for i in range(self.num_targets):
                            _shape = [len(load_range)] + list(_parse_shape(target_sample[i]))
                            targets.append(np.empty(_shape))
                        #targets = [np.empty((len(load_range), *_parse_shape(target_sample[i]))) for i in range(self.num_targets)]

                if self.num_inputs == 1:
                    inputs[enum_idx] = input_sample
                else:
                    for i in range(self.num_inputs):
                        inputs[i][enum_idx] = input_sample[i]

                if self.num_targets == 1:
                    targets[enum_idx] = target_sample
                else:
                    for i in range(self.num_targets):
                        targets[i][enum_idx] = target_sample[i]

            return inputs, targets
        else:
            for enum_idx, sample_idx in enumerate(load_range):
                input_sample = self.__getitem__(sample_idx)

                if enum_idx == 0:
                    if self.num_inputs == 1:
                        _shape = [len(load_range)] + list(_parse_shape(input_sample))
                        inputs = np.empty(_shape)
                        #inputs = np.empty((len(load_range), *_parse_shape(input_sample)))
                    else:
                        inputs = []
                        for i in range(self.num_inputs):
                            _shape = [len(load_range)] + list(_parse_shape(input_sample[i]))
                            inputs.append(np.empty(_shape))
                        #inputs = [np.empty((len(load_range), *_parse_shape(input_sample[i]))) for i in range(self.num_inputs)]

                if self.num_inputs == 1:
                    inputs[enum_idx] = input_sample
                else:
                    for i in range(self.num_inputs):
                        inputs[i][enum_idx] = input_sample[i]

            return inputs

    def fit_transforms(self):
        """
        Make a single pass through the entire dataset in order to fit
        any parameters of the transforms which require the entire dataset.
        e.g. StandardScaler() requires mean and std for the entire dataset.

        If you dont call this fit function, then transforms which require properties
        of the entire dataset will just work at the batch level.
        e.g. StandardScaler() will normalize each batch by the specific batch mean/std
        """
        it_fit = hasattr(self.input_transform, 'update_fit')
        tt_fit = hasattr(self.target_transform, 'update_fit')
        ct_fit = hasattr(self.co_transform, 'update_fit')
        if it_fit or tt_fit or ct_fit:
            for sample_idx in range(len(self)):
                if hasattr(self, 'input_loader'):
                    x = self.input_loader(self.inputs[sample_idx])
                else:
                    x = self.inputs[sample_idx]
                if it_fit:
                    self.input_transform.update_fit(x)
                if self.has_target:
                    if hasattr(self, 'target_loader'):
                        y = self.target_loader(self.targets[sample_idx])
                    else:
                        y = self.targets[sample_idx]
                if tt_fit:
                    self.target_transform.update_fit(y)
                if ct_fit:
                    self.co_transform.update_fit(x,y)


def _process_array_argument(x):
    if not is_tuple_or_list(x):
        x = [x]
    return x


class TensorDataset(BaseDataset):

    def __init__(self,
                 inputs,
                 targets=None,
                 input_transform=None, 
                 target_transform=None,
                 co_transform=None):
        """
        Dataset class for loading in-memory data.

        Arguments
        ---------
        inputs: numpy array

        targets : numpy array

        input_transform : class with __call__ function implemented
            transform to apply to input sample individually

        target_transform : class with __call__ function implemented
            transform to apply to target sample individually

        co_transform : class with __call__ function implemented
            transform to apply to both input and target sample simultaneously

        """
        self.inputs = _process_array_argument(inputs)
        self.num_inputs = len(self.inputs)
        self.input_return_processor = _return_first_element_of_list if self.num_inputs==1 else _pass_through

        if targets is None:
            self.has_target = False
        else:
            self.targets = _process_array_argument(targets)
            self.num_targets = len(self.targets)
            self.target_return_processor = _return_first_element_of_list if self.num_targets==1 else _pass_through
            self.min_inputs_or_targets = min(self.num_inputs, self.num_targets)
            self.has_target = True

        self.input_transform = _process_transform_argument(input_transform, self.num_inputs)
        if self.has_target:
            self.target_transform = _process_transform_argument(target_transform, self.num_targets)
            self.co_transform = _process_co_transform_argument(co_transform, self.num_inputs, self.num_targets)

    def __getitem__(self, index):
        """
        Index the dataset and return the input + target
        """
        input_sample = [self.input_transform[i](self.inputs[i][index]) for i in range(self.num_inputs)]

        if self.has_target:
            target_sample = [self.target_transform[i](self.targets[i][index]) for i in range(self.num_targets)]
            #for i in range(self.min_inputs_or_targets):
            #    input_sample[i], target_sample[i] = self.co_transform[i](input_sample[i], target_sample[i])

            return self.input_return_processor(input_sample), self.target_return_processor(target_sample)
        else:
            return self.input_return_processor(input_sample)


def default_file_reader(x):
    if isinstance(x, str):
        if x.endswith('.npy'):
            x = npy_loader(x)
        else:
            try:
                x = pil_loader(x, color_space='RGB')
            except:
                raise ValueError('File Format is not supported')
    #else:
        #raise ValueError('x should be string, but got %s' % type(x))
    return x

def is_tuple_or_list(x):
    return isinstance(x, (tuple,list))

def _process_transform_argument(tform, num_inputs):
    tform = tform if tform is not None else _pass_through
    if is_tuple_or_list(tform):
        if len(tform) != num_inputs:
            raise Exception('If transform is list, must provide one transform for each input')
        tform = [t if t is not None else _pass_through for t in tform]
    else:
        tform = [tform] * num_inputs
    return tform

def _process_co_transform_argument(tform, num_inputs, num_targets):
    tform = tform if tform is not None else _multi_arg_pass_through
    if is_tuple_or_list(tform):
        if len(tform) != num_inputs:
            raise Exception('If transform is list, must provide one transform for each input')
        tform = [t if t is not None else _multi_arg_pass_through for t in tform]
    else:
        tform = [tform] * min(num_inputs, num_targets)
    return tform

def _process_csv_argument(csv):
    if isinstance(csv, str):
        df = pd.read_csv(csv)
    elif isinstance(csv, pd.DataFrame):
        df = csv
    else:
        raise ValueError('csv argument must be string or dataframe')
    return df

def _select_dataframe_columns(df, cols):
    if isinstance(cols[0], str):
        inputs = df.loc[:,cols].values
    elif isinstance(cols[0], int):
        inputs = df.iloc[:,cols].values
    else:
        raise ValueError('Provided columns should be string column names or integer column indices')
    return inputs

def _process_cols_argument(cols):
    if isinstance(cols, tuple):
        cols = list(cols)
    return cols

def _return_first_element_of_list(x):
    return x[0]

def _pass_through(x):
    return x

def _multi_arg_pass_through(*x):
    return x


class CSVDataset(BaseDataset):

    def __init__(self,
                 csv,
                 input_cols=[0],
                 target_cols=[1],
                 input_transform=None,
                 target_transform=None,
                 co_transform=None):
        """
        Initialize a Dataset from a CSV file/dataframe. This does NOT
        actually load the data into memory if the CSV contains filepaths.

        Arguments
        ---------
        csv : string or pandas.DataFrame
            if string, should be a path to a .csv file which
            can be loaded as a pandas dataframe

        input_cols : int/list of ints, or string/list of strings
            which columns to use as input arrays.
            If int(s), should be column indicies
            If str(s), should be column names

        target_cols : int/list of ints, or string/list of strings
            which columns to use as input arrays.
            If int(s), should be column indicies
            If str(s), should be column names

        input_transform : class which implements a __call__ method
            tranform(s) to apply to inputs during runtime loading

        target_tranform : class which implements a __call__ method
            transform(s) to apply to targets during runtime loading

        co_transform : class which implements a __call__ method
            transform(s) to apply to both inputs and targets simultaneously
            during runtime loading
        """
        self.input_cols = _process_cols_argument(input_cols)
        self.target_cols = _process_cols_argument(target_cols)

        self.df = _process_csv_argument(csv)

        self.inputs = _select_dataframe_columns(self.df, input_cols)
        self.num_inputs = self.inputs.shape[1]
        self.input_return_processor = _return_first_element_of_list if self.num_inputs==1 else _pass_through

        if target_cols is None:
            self.num_targets = 0
            self.has_target = False
        else:
            self.targets = _select_dataframe_columns(self.df, target_cols)
            self.num_targets = self.targets.shape[1]
            self.target_return_processor = _return_first_element_of_list if self.num_targets==1 else _pass_through
            self.has_target = True
            self.min_inputs_or_targets = min(self.num_inputs, self.num_targets)

        self.input_loader = default_file_reader
        self.target_loader = default_file_reader

        self.input_transform = _process_transform_argument(input_transform, self.num_inputs)
        if self.has_target:
            self.target_transform = _process_transform_argument(target_transform, self.num_targets)
            self.co_transform = _process_co_transform_argument(co_transform, self.num_inputs, self.num_targets)

    def __getitem__(self, index):
        """
        Index the dataset and return the input + target
        """
        input_sample = [self.input_transform[i](self.input_loader(self.inputs[index, i])) for i in range(self.num_inputs)]

        if self.has_target:
            target_sample = [self.target_transform[i](self.target_loader(self.targets[index, i])) for i in range(self.num_targets)]
            for i in range(self.min_inputs_or_targets):
                input_sample[i], input_sample[i] = self.co_transform[i](input_sample[i], target_sample[i])

            return self.input_return_processor(input_sample), self.target_return_processor(target_sample)
        else:
            return self.input_return_processor(input_sample)

    def split_by_column(self, col):
        """
        Split this dataset object into multiple dataset objects based on
        the unique factors of the given column. The number of returned
        datasets will be equal to the number of unique values in the given
        column. The transforms and original dataframe will all be transferred
        to the new datasets

        Useful for splitting a dataset into train/val/test datasets.

        Arguments
        ---------
        col : integer or string
            which column to split the data on.
            if int, should be column index
            if str, should be column name

        Returns
        -------
        - list of new datasets with transforms copied
        """
        if isinstance(col, int):
            split_vals = self.df.iloc[:,col].values.flatten()

            new_df_list = []
            for unique_split_val in np.unique(split_vals):
                new_df = self.df[:][self.df.iloc[:,col]==unique_split_val]
                new_df_list.append(new_df)
        elif isinstance(col, str):
            split_vals = self.df.loc[:,col].values.flatten()

            new_df_list = []
            for unique_split_val in np.unique(split_vals):
                new_df = self.df[:][self.df.loc[:,col]==unique_split_val]
                new_df_list.append(new_df)
        else:
            raise ValueError('col argument not valid - must be column name or index')

        new_datasets = []
        for new_df in new_df_list:
            new_dataset = self.copy(new_df)
            new_datasets.append(new_dataset)

        return new_datasets

    def train_test_split(self, train_size):
        if train_size < 1:
            train_size = int(train_size * len(self))

        train_indices = np.random.choice(len(self), train_size, replace=False)
        test_indices = np.array([i for i in range(len(self)) if i not in train_indices])

        train_df = self.df.iloc[train_indices,:]
        test_df = self.df.iloc[test_indices,:]

        train_dataset = self.copy(train_df)
        test_dataset = self.copy(test_df)

        return train_dataset, test_dataset

    def copy(self, df=None):
        if df is None:
            df = self.df

        return CSVDataset(df,
                          input_cols=self.input_cols,
                          target_cols=self.target_cols,
                          input_transform=self.input_transform,
                          target_transform=self.target_transform,
                          co_transform=self.co_transform)


def random_split_dataset(orig_dataset, splitRatio=0.8, random_seed=None):
    '''
    Randomly split the given dataset into two datasets based on the provided ratio

    :param orig_dataset: UsefulDataset
        dataset to split (of type torchsample.datasetz.UsefulDataset)

    :param splitRatio: float
        ratio to use when splitting the data

    :param random_seed: int
        random seed for replicability of results

    :return: tuple of split Useful
    '''
    random.seed(a=random_seed)

    # not cloning the dictionary at this point... maybe it should be?
    orig_dict = orig_dataset.getmeta_data()
    part1 = []
    part2 = []


    for i, item in enumerate(orig_dataset.getdata()):
        if random.random() < splitRatio:
            part1.append(item)
        else:
            part2.append(item)

    return ClonedFolderDataset(part1, orig_dict), ClonedFolderDataset(part2, orig_dict)

def _find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def _is_image_file(filename):
    IMG_EXTENSIONS = [
        '.jpg', '.JPG', '.jpeg', '.JPEG',
        '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.npy'
    ]
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def _finds_inputs_and_targets(root, class_mode, class_to_idx=None, input_regex='*',
                              rel_target_root='', target_prefix='', target_postfix='', target_extension='png',
                              splitRatio=1.0, random_seed=None, exclusion_file=None):
    """
    Map a dataset from a root folder. Optionally, split the dataset randomly into two partitions (e.g. train and val)

    :param root: string\n
        root dir to scan
    :param class_mode: string in `{'label', 'image'}`\n
        whether to return a label or an image as target
    :param class_to_idx: list\n
        classes to map to indices
    :param input_regex: string (default: *)\n
        regex to apply to scanned input entries
    :param rel_target_root: string\n
        relative target root to scan (if any)
    :param target_prefix: string\n
        prefix to use (if any) when trying to locate the matching target
    :param target_postfix: string\n
        postfix to use (if any) when trying to locate the matching target
    :param splitRatio: float\n
        if set to 0.0 < splitRatio < 1.0 the function will return two datasetz
    :param random_seed: int (default: None)\n
        you can control replicability of the split by explicitly setting the random seed
    :param exclusion_file: string (default: None)\n
        list of files (one per line) to exclude when enumerating all files\n
        The list must contain paths relative to the root parameter\n
        each line may include the filename and additional comma-separated metadata, in which case the first item will be considered the path itself and the rest will be ignored

    :return: partition1 (list of (input, target)), partition2 (list of (input, target))
    """
    if class_mode is not 'image' and class_mode is not 'label':
        raise ValueError('class_mode must be one of: label, image')

    if class_mode == 'image' and rel_target_root == '' and target_prefix == '' and target_postfix == '':
            raise ValueError('must provide either rel_target_root or a value for target prefix/postfix when class_mode is set to: image')

    ## Handle exclusion list, if any
    exclusion_list = set()
    if exclusion_file:
        with open(exclusion_file, 'r') as exclfile:
            for line in exclfile:
                exclusion_list.add(line.split(',')[0])

    trainlist_inputs = []
    trainlist_targets = []
    vallist_inputs = []
    vallist_targets = []
    icount = 0
    for subdir in sorted(os.listdir(root)):
        d = os.path.join(root, subdir)
        if not os.path.isdir(d):
            continue

        for rootz, _, fnames in sorted(os.walk(d)):
            for fname in fnames:
                if _is_image_file(fname):
                    if fnmatch.fnmatch(fname, input_regex):
                        icount = icount + 1

                        # enforce random split
                        if random.random() < splitRatio:
                            inputs = trainlist_inputs
                            targets = trainlist_targets
                        else:
                            inputs = vallist_inputs
                            targets = vallist_targets

                        if not os.path.join(subdir,fname) in exclusion_list:        # exclude any undesired files
                            path = os.path.join(rootz, fname)
                            inputs.append(path)
                            if class_mode == 'label':
                                targets.append(class_to_idx[subdir])
                            elif class_mode == 'image':
                                name_vs_ext = fname.rsplit('.', 1)
                                target_fname = os.path.join(root, rel_target_root, subdir, target_prefix + name_vs_ext[0] + target_postfix + '.' + target_extension)
                                if os.path.exists(target_fname):
                                    targets.append(target_fname)
                                else:
                                    raise ValueError('Could not locate file: ' + target_fname + ' corresponding to input: ' + path)
    if class_mode is None:
        return trainlist_inputs, vallist_inputs
    else:
        assert len(trainlist_inputs) == len(trainlist_targets) and len(vallist_inputs) == len(vallist_targets)
        print("Total processed: %i    Train-list: %i items   Val-list: %i items    Exclusion-list: %i items" % (icount, len(trainlist_inputs), len(vallist_inputs), len(exclusion_list)))
        return list(zip(trainlist_inputs, trainlist_targets)), list(zip(vallist_inputs, vallist_targets))
