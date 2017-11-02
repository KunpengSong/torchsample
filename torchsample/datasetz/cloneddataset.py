# from torchsample.torchsample.datasets import FolderDataset

# class ClonedFolderDataset(FolderDataset):
#     # TODO: Explore to load everything generically
#     #   -  https://stackoverflow.com/questions/19305296/multiple-constructors-in-python-using-inheritance
#
#     def __init__(self, data, *meta_data, **kwargs):
#         """
#         Dataset that can be initialized with a dictionary of internal parameters
#
#         Arguments
#         ---------
#         :param data: list
#             list of data on which the dataset operates
#
#         :param meta_data: dict
#             parameters that correspond to the target dataset's attributes
#         """
#
#         if len(data) == 0:
#             raise (RuntimeError('No data provided'))
#         else:
#             print('Initializing with %i data items' % len(data))
#
#         self.data = data
#
#         # Source: https://stackoverflow.com/questions/2466191/set-attributes-from-dictionary-in-python
#         # generic way of initializing the object
#         for dictionary in meta_data:
#             for key in dictionary:
#                 setattr(self, key, dictionary[key])
#         for key in kwargs:
#             setattr(self, key, kwargs[key])
#
#         # self.class_mode = meta_data['class_mode']
#         # self.class_to_idx = meta_data['class_to_idx']
#         # self.transform = meta_data['transform']
#         # self.target_transform = meta_data['target_transform']
#         # self.co_transform = meta_data['co_transform']
#         # self.apply_co_transform_first = meta_data['apply_co_transform_first']
#         # self.file_loader = meta_data['file_loader']
#
#     def __getitem__(self, index):
#         return super().__getitem__(index)
#
#     def __len__(self):
#         return super().__len__()
#
#     def getdata(self):
#         return super().getdata()
#
#     def getmeta_data(self):
#         return super().getmeta_data()