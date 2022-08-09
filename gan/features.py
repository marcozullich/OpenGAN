from turtle import back
import torch
import torchvision
from torch.utils.data import DataLoader
import os



def get_backbone(class_, params_path:str, **kwargs):
    '''
    Instantiate the backbone for obtaining the features.

    Parameters
    ------------
    class_: a class inheriting from torch.nn.Module specifying the model to be instantiated
    params_path: the path to the state_dict containing the pretrained params for the backbone
    layer_features: the layer from which the features need to be extracted
    kwargs: possible args to pass on to the class_ constructor

    Returns
    ------------
    an instantiated pretrained model ready for extracting the features
    '''
    net_class = getattr(torchvision.models, class_)
    net = net_class(**kwargs)
    net.load_state_dict(torch.load(params_path))
    return net

def extract_features(backbone:torch.nn.Module, feats_module:str, dataset:torch.utils.data.Dataset, batch_size:int, ites:int=1, **kwargs):
    '''
    Get the features from the specified dataset.

    Parameters
    ------------
    backbone: a pretrained torch.nn.Module used for obtaining the features
    feats_module: the layer (module) from which extract the features
    dataset: a torch.utils.data.Dataset containing the datapoints to extract features out of
    batch_size: the batch size used for evaluating the features. None equates to len(dataset)
    ites: number of iterations to repeat the feature extraction. Useful for data augmentation if dataset has some probabilistic transforms.
    kwargs: possible args to pass on to DataLoader

    Returns
    ------------
    an instantiated pretrained model ready for extracting the features
    '''
    features = []
    def feature_extractor(module, input_, output):
        features.append(output.cpu())
    getattr(backbone, feats_module).register_forward_hook(feature_extractor)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, **kwargs)
    backbone.eval()
    backbone.cuda()
    for i in range(ites):
        print(f"Processing iteration {i+1}/{ites}...")
        with torch.no_grad():
            for j, data in enumerate(loader):
                print(f"Processing batch {j+1}/{len(loader)}")
                if isinstance(data, (tuple, list)):
                    data = data[0] #disregard labels in this process
                data = data.cuda()
                _ = backbone(data)
    return torch.cat(features, dim=0)

def get_features(path_features:str, force_calculation:bool, dataset:str, backbone_class, backbone_params:str, batch_size:int, num_classes:int=None):
    '''
    Loads or calculates features produced by the backbone by evaluating the images contained in a specified dataset.

    Params:
    -------------
    path_features: the path where the features are stored (in case of load) or to be saved (in case of calculation).
    force_calculation: forces the (re)calculation of the features even if path_features already exists. For a first calculation of features + saving, set force_calculation to True.
    dataset: the dataset.
    backbone_class: the module class for the network.
    backbone_params: the path to the parameters of the backbone to load.
    batch_size: the batch size which will be used for passing the raw data to the backbone to produce the features.
    num_classes: the number of classes in the dataset. If None, it will be inferred from the dataset.

    Returns:
    -------------
    a torch Tensor containing the features
    '''
    if (load_path:=path_features) is not None and (not force_calculation):
        assert os.path.isfile(load_path), f"The specified loading path for the training features {load_path} is not a file"
        features = torch.load(load_path)
    else:
        num_classes = num_classes if num_classes is not None else len(dataset.classes)
        backbone = get_backbone(backbone_class, backbone_params, num_classes=num_classes)
        features = extract_features(backbone, "layer4", dataset, batch_size)
        if (save_path:=path_features) is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(features, save_path)
            print(f"Train features saved at {save_path}")
    return features
