import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import pandas as pd

def get_outputs(
    discriminator: torch.nn.Module,
    testloader:DataLoader
):
    outputs = []
    device = next(discriminator.parameters()).device
    discriminator.eval()
    with torch.no_grad():
        for data in testloader:
            outputs.append(discriminator(data.to(device)))
    return torch.cat(outputs)

def plot_hist(
    outputs_train:torch.Tensor,
    outputs_ood:torch.Tensor,
    outputs_test:torch.Tensor,
    outputs_crops:torch.Tensor,
    outputs_random:torch.Tensor,
    save_path:str,
    title:str=""
):  
    if (folder:=os.path.dirname(save_path)) != "":
        os.makedirs(folder, exist_ok=True)
    if outputs_train is not None:
        plt.hist(outputs_train.detach().cpu().numpy(), label="train data", density=True, alpha=0.5)
    if outputs_ood is not None:
        plt.hist(outputs_ood.detach().cpu().numpy(), label="OOD data", density=True, alpha=0.5)
    if outputs_test is not None:
        plt.hist(outputs_test.detach().cpu().numpy(), label="validation data", density=True, alpha=0.5)
    if outputs_crops is not None:
        plt.hist(outputs_crops.detach().cpu().numpy(), label="random crops", density=True, alpha=0.5)
    if outputs_random is not None:
        plt.hist(outputs_random.detach().cpu().numpy(), label="random data", density=True, alpha=0.5)
    plt.legend(loc='upper right')
    plt.title(title)

    if os.path.splitext(save_path)[1] != ".png":
        save_path += ".png"
    plt.savefig(save_path)

def get_performance(
    outputs_valid:torch.Tensor,
    outputs_ood:torch.Tensor,
    outputs_randcrops:torch.Tensor,
    increment:float=.05
) -> pd.DataFrame:
    performance = {"threshold": [], "valid": [], "ood": [], "randcrops": []}
    for i in torch.linspace(increment, 1.0 - increment, int(1.0 / increment)):
        performance["threshold"].append(i.item())
        performance["valid"].append(len(outputs_valid[outputs_valid>i])/len(outputs_valid))
        performance["ood"].append(len(outputs_ood[outputs_ood<=i])/len(outputs_ood))
        performance["randcrops"].append(len(outputs_randcrops[outputs_randcrops<=i])/len(outputs_randcrops))
    return pd.DataFrame(performance)
        

# def get_performance(
#     outputs_train:torch.Tensor,
#     outputs_ood:torch.Tensor,
#     outputs_test:torch.Tensor,
#     threshold_ood:float,
#     threshold_true:float,
#     verbose:bool=True
# ):
#     def threshold_series(
#         series:torch.Tensor,
#         threshold_ood:float,
#         threshold_true:float,
#     ):
#         def do_threshold(
#             series:torch.Tensor,
#             thresh:float
#         ):
#             return series[series>=thresh]
#         if series is not None:
#             s_ood = do_threshold(series, threshold_ood)
#             if threshold_true is not None:
#                 s_true = do_threshold(series, threshold_true)
#                 s_ood += s_true
#             return s_ood
#         return None
    
#     def calc_performance(thr_series:torch.Tensor):
#         return {k:[l:=len(thr_series[thr_series==k]), l/len(thr_series)] for k in thr_series.unique()}
    
#     def print_performance(perf:dict, title:str):
#         if perf is not None:
#             string = f"{title} || "
#             for i, (k, p) in enumerate(perf.values()):
#                 string += f"{k}: {p}"
#                 if i+1 < len(perf):
#                     string += ", "
#         print(string)

#     if threshold_true is not None:
#         assert threshold_ood < threshold_true, f"threshold_ood should be strictly less than threshold_true. Found {threshold_ood} and {threshold_true}"
#     thr_train = threshold_series(outputs_train, threshold_ood, threshold_true)
#     perf_train = calc_performance(thr_train) if perf_train is not None else None
#     if verbose:
#         print_performance(perf_train, "train")
#     thr_test = threshold_series(outputs_test, threshold_ood, threshold_true)
#     perf_test = calc_performance(thr_test) if perf_train is not None else None
#     if verbose:
#         print_performance(perf_test, "test")
#     thr_ood = threshold_series(outputs_ood, threshold_ood, threshold_true)
#     perf_ood = calc_performance(thr_ood) if perf_train is not None else None
#     if verbose:
#         print_performance(perf_ood, "ood")
#     return perf_train, perf_test, perf_ood




    