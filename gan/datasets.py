import torch
from typing import Collection
from torchvision import transforms as T

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

def get_random_data(shape:Collection[int], seed:int):
    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)
    data = torch.rand(shape, generator=generator)
    data = T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))(data)
    return data