import torchvision
from torchvision import transforms as T

TRANSFORM_TYPE = {
    "bare":T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
}

def get_dataset(root:str, transform_type:str="bare", **kwargs):
    '''
    Get the punches dataset

    Parameters
    ------------
    root: the base folder from which to load the data. It must contain subfolders, one for each category, each subfolder containing all the images for the corresponding category.
    transform_type: the type of transforms to apply to the data when sampled. Available options: "bare". Default: "bare"
    **kwargs: additional args to pass on to ImageFolder

    Returns
    ------------
    a torch.utils.data.Dataset of punches
    '''
    transform = TRANSFORM_TYPE[transform_type]
    dataset = torchvision.datasets.ImageFolder(
        root=root,
        transform=transform,
        **kwargs
    )
    return dataset
