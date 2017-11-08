import torch.utils.data.dataset as ds

class UsefulDataset(ds.Dataset):
    '''
    A torch.utils.data.Dataset class with additional useful functions.
    '''

    def __init__(self):
        self.num_inputs = 1         # these are hardcoded for the fit module to work
        self.num_targets = 1        # these are hardcoded for the fit module to work

    def __len__(self):
        super().__len__()

    def __getitem__(self, idx):
        super().__getitem__(idx)

    def getdata(self):
        '''
        Data that the Dataset class operates on. Typically iterable/list of tuple(label,target)

        :return: iterable
        '''
        pass

    def getmeta_data(self):
        '''
        Additional data to return that might be useful to consumer. Typically a dict()

        :return: dict(any)
        '''
        pass