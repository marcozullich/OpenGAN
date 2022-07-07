import torch

class BasicDataset(torch.utils.data.Dataset):
    '''
    A simple dataset wrapping a container of images without labels.
    '''
    def __init__(self, data, transform=None):
        '''
        Parameters:
        -----------
        data: a generic container of tensors
        transform: a pipeline of transformations to apply to the data
        '''
        self.data = data
        self.current_set_len = len(data)
        self.transform = transform
        
    def __len__(self):        
        return self.current_set_len
    
    def __getitem__(self, idx):
        curdata = self.data[idx]
        if self.transform is not None:
            return self.transform(curdata)
        return curdata