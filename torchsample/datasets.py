
from __future__ import absolute_import

from .datasetz.usefuldataset import UsefulDataset

import random
import torch.utils.data

import os
import os.path
import warnings
import fnmatch

import numpy as np

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


class TensorDataset(torch.utils.data.Dataset):

    def __init__(self,
                 input_tensor,
                 target_tensor=None,
                 input_transform=None, 
                 target_transform=None,
                 co_transform=None):
        """
        Dataset class for loading in-memory data.

        Arguments
        ---------
        input_tensor : torch tensor

        target_tensor : torch tensor

        transform : torch transform
            transform to apply to input sample individually

        target_transform : torch transform
            transform to apply to target sample individually
        """
        self.inputs = input_tensor
        self.targets = target_tensor
        if target_tensor is None:
            self.has_target = False
        else:
            self.has_target = True
        self.input_transform = input_transform
        self.target_transform = target_transform
        self.co_transform = co_transform

    def __getitem__(self, index):
        """Return a (transformed) input and target sample from an integer index"""
        # get paths
        input_sample = self.inputs[index]
        if self.has_target:
            target_sample = self.targets[index]

        # apply transforms
        if self.input_transform is not None:
            input_sample = self.input_transform(input_sample)
        if self.has_target and self.target_transform is not None:
            target_sample = self.target_transform(target_sample)
        if self.has_target and self.co_transform is not None:
            input_sample, target_sample = self.co_transform(input_sample, target_sample)

        if self.has_target:
            return input_sample, target_sample
        else:
            return input_sample

    def __len__(self):
        """Number of samples"""
        return self.inputs.size(0)


class CSVDataset(torch.utils.data.Dataset):

    def __init__(self,
                 filepath,
                 input_transform=None,
                 target_transform=None,
                 co_transform=None):
        pass


class MultiTensorDataset(torch.utils.data.Dataset):

    def __init__(self,
                 input_tensors,
                 target_tensors=None,
                 input_transform=None, 
                 target_transform=None,
                 co_transform=None):
        """
        Sample multiple input/target tensors at once

        Example:
        >>> import torch
        >>> from torch.utils.data import DataLoader
        >>> x1 = torch.ones(100,5)
        >>> x2 = torch.zeros(100,10)
        >>> y = torch.ones(100,1)*10
        >>> dataset = MultiTensorDataset([x1,x2], None)
        >>> loader = DataLoader(dataset, sampler=SequentialSampler(x1.size(0)), batch_size=5)
        >>> loader_iter = iter(loader)
        >>> x,y = next(loader_iter)
        """
        if not isinstance(input_tensors, list):
            input_tensors = [input_tensors]
        self.inputs = input_tensors
        self.num_inputs = len(input_tensors)

        if not isinstance(target_tensors, list) and target_tensors is not None:
            target_tensors = [target_tensors]
        self.targets = target_tensors

        if target_tensors is None:
            self.has_target = False
        else:
            self.has_target = True
            self.num_targets = len(target_tensors)

        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """Return a (transformed) input and target sample from an integer index"""
        # get paths
        input_samples = [self.inputs[i][index] for i in range(self.num_inputs)]
        if self.has_target:
            target_samples = [self.targets[i][index] for i in range(self.num_targets)]

        # apply transforms
        if self.input_transform is not None:
            input_samples = [self.input_transform(input_samples[i]) for i in range(self.num_inputs)]
        if self.has_target and self.target_transform is not None:
            target_samples = [self.target_transform(target_samples[i]) for i in range(self.num_targets)]

        if self.has_target:
            return input_samples, target_samples
        else:
            return [input_samples]

    def __len__(self):
        """Number of samples"""
        return self.inputs.size(0)


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
