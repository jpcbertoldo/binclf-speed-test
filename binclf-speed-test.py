#!/usr/bin/env python
# coding: utf-8
"""
Test which variation of FPR computation is faster.

Tested with python version `3.10.12` and environment in `requirements.txt`.

Examples: 
python binclf-speed-test.py --resolution 128 --num_images 100 --num_thresholds 1000 --seed 0 --algorithm numpy_numba --device cpu --mode perimg
python binclf-speed-test.py --resolution 128 --num_images 100 --num_thresholds 1000 --seed 0 --algorithm numpy_itertools --device cpu --mode set
python binclf-speed-test.py --resolution 128 --num_images 100 --num_thresholds 1000 --seed 0 --algorithm torchmetrics_unique_values --device cpu --mode set
"""

from __future__ import annotations

import argparse
import itertools
import multiprocessing
import os
import socket
import timeit
from pathlib import Path

import numba
import numpy as np
import pandas as pd
import scipy as sp
import scipy.stats
import torch
from numpy import ndarray
from torch import Tensor
from torch.nn import functional as F  # noqa: N812

# # setup
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

SAVE_DIR = Path("./.cache/binclf-speed-test")
SAVE_DIR.mkdir(exist_ok=True, parents=True)

# get the hostname
HOSTNAME = socket.gethostname()

# get the number of cpus
NUM_CPUS = multiprocessing.cpu_count()


# # data

def get_data_synthetic(nimgs: int, res: int, rng) -> (Tensor, Tensor):
    """ (nb. images, resolution, random number generator) -> (masks, asmaps) """
    
    # proportion of anomalous images; between 25% and 75%, uniformly distributed
    num_anom_images = int(rng.uniform(0.25, 0.75) * nimgs)
    num_norm_images = nimgs - num_anom_images
    
    # number of anomalous pixels; between .03% and 10% of the image, log-uniformely distributed
    anomsizes = 10 ** rng.uniform(np.log10(3e-4), np.log10(1e-1), num_anom_images)
    
    masks_norm = np.zeros((num_norm_images, res, res), dtype=int)
    masks_anom = (rng.uniform(0, 1, (num_anom_images, res, res)) >= anomsizes[..., None, None]).astype(int)
    masks = np.concatenate([masks_anom, masks_norm], axis=0)
    
    norm_pixels_distrib = sp.stats.chi2(df=1)
    asmaps_norm = norm_pixels_distrib.rvs((num_norm_images, res, res), random_state=rng)
    
    asmaps_anom = norm_pixels_distrib.rvs((num_anom_images, res, res), random_state=rng)
    for idx in range(num_anom_images):
        anom_df = rng.integers(1, 10)
        anom_pixel_scores = rng.chisquare(df=anom_df, size=(res, res))
        msk = masks_anom[idx]
        asmaps_anom[idx][msk] = anom_pixel_scores[msk]
    
    asmaps = np.concatenate([asmaps_anom, asmaps_norm], axis=0)     
    
    return torch.from_numpy(masks).to(torch.int32), torch.from_numpy(asmaps).to(torch.float32)


# # torchmetrics vectorized

# copied from `torchmetrics.functional.classification.precision_recall_curve`
# in version 1.1.0
# source permalink
# https://github.com/Lightning-AI/torchmetrics/blob/eda473f52cb9cf3ea2aa5079f04db0b34ac7f96b/src/torchmetrics/functional/classification/precision_recall_curve.py

def torchmetrics_vectorized(
    preds: Tensor,
    target: Tensor,
    thresholds: Tensor,
) -> Tensor:
    """
    source permalink
    https://github.com/Lightning-AI/torchmetrics/blob/eda473f52cb9cf3ea2aa5079f04db0b34ac7f96b/src/torchmetrics/functional/classification/precision_recall_curve.py#L205
    
    simplification
    the original uses `torchmetrics.utilities.data._bincount`, fallsback to `torch.bincount` when in one of the following cases:         
        - MPS devices
        - deterministic mode on GPU.
    
    Here we just use `torch.bincount` directly.
    """
    len_t = len(thresholds)
    preds_t = (preds.unsqueeze(-1) >= thresholds.unsqueeze(0)).long()  # num_samples x num_thresholds
    unique_mapping = preds_t + 2 * target.long().unsqueeze(-1) + 4 * torch.arange(len_t, device=target.device)
    
    # SIMPLIFICATION HERE
    # bins = bincount(unique_mapping.flatten(), minlength=4 * len_t)
    
    bins = torch.bincount(unique_mapping.flatten(), minlength=4 * len_t)
    
    return bins.reshape(len_t, 2, 2)


# # torchmetrics loop

# copied from `torchmetrics.functional.classification.precision_recall_curve`
# in version 1.1.0
# source permalink
# https://github.com/Lightning-AI/torchmetrics/blob/eda473f52cb9cf3ea2aa5079f04db0b34ac7f96b/src/torchmetrics/functional/classification/precision_recall_curve.py

def torchmetrics_loop(
    preds: Tensor,
    target: Tensor,
    thresholds: Tensor,
) -> Tensor:
    """Return the multi-threshold confusion matrix to calculate the pr-curve with.

    This implementation loops over thresholds and is more memory-efficient than
    `_binary_precision_recall_curve_update_vectorized`. However, it is slowwer for small
    numbers of samples (up to 50k).
    """
    len_t = len(thresholds)
    target = target == 1
    confmat = thresholds.new_empty((len_t, 2, 2), dtype=torch.int64)
    # Iterate one threshold at a time to conserve memory
    for i in range(len_t):
        preds_t = preds >= thresholds[i]
        confmat[i, 1, 1] = (target & preds_t).sum()
        confmat[i, 0, 1] = ((~target) & preds_t).sum()
        confmat[i, 1, 0] = (target & (~preds_t)).sum()
    confmat[:, 0, 0] = len(preds_t) - confmat[:, 0, 1] - confmat[:, 1, 0] - confmat[:, 1, 1]
    return confmat


# # numpy + itertools

def _numpy_itertools(scoremap: ndarray, mask: ndarray, thresholds: ndarray):
    """ 
    thresholds assumed to be sorted!!!
    """
    if not mask.dtype == bool:
        raise ValueError("mask must be boolean")
    
    num_th = len(thresholds)

    # POSITIVES
    scores_pos = scoremap[mask]
    # the sorting is very important for the algorithm to work and the speedup
    scores_pos = np.sort(scores_pos)
    # start counting with lowes th, so everything is predicted as positive
    # this variable is updated in the loop
    num_pos = current_count_tp = scores_pos.size

    tps = np.empty((num_th), dtype=np.int64)
    
    # NEGATIVES
    # same thing but for the negative samples
    scores_neg = scoremap[~mask]
    scores_neg = np.sort(scores_neg)
    num_neg = current_count_fp = scores_neg.size

    fps = np.empty((num_th), dtype=np.int64)
    
    # it will progressively drop the scores that are below the current th
    for thidx, th in enumerate(thresholds):
        
        # < becasue it is the same as ~(>=)
        num_drop = sum(1 for _ in itertools.takewhile(lambda x: x < th, scores_pos))
        scores_pos = scores_pos[num_drop:]
        current_count_tp -= num_drop
        tps[thidx] = current_count_tp
        
        # same with the negatives 
        num_drop = sum(1 for _ in itertools.takewhile(lambda x: x < th, scores_neg))
        scores_neg = scores_neg[num_drop:]
        current_count_fp -= num_drop
        fps[thidx] = current_count_fp
    
    fns = num_pos * np.ones((num_th,), dtype=np.int64) - tps
    tns = num_neg * np.ones((num_th,), dtype=np.int64) - fps
    
    # sequence of dimensions is (thresholds, true class, predicted class)
    # so `tps` is `confmat[:, 1, 1]`, `fps` is `confmat[:, 0, 1]`, etc.
    return np.stack([
        np.stack([tns, fps], axis=-1),
        np.stack([fns, tps], axis=-1),
    ], axis=-1).transpose(0, 2, 1)
    

numpy_itertools = np.vectorize(_numpy_itertools, signature="(n),(n),(k)->(k,2,2)")


# # numpy + numba

@numba.jit(nopython=True)
def _numpy_numba(scoremap: ndarray, mask: ndarray, thresholds: ndarray):
    """ 
    thresholds assumed to be sorted!!!
    """
    
    num_th = len(thresholds)

    # POSITIVES
    scores_pos = scoremap[mask]
    # the sorting is very important for the algorithm to work and the speedup
    scores_pos = np.sort(scores_pos)
    # start counting with lowes th, so everything is predicted as positive
    # this variable is updated in the loop
    num_pos = current_count_tp = len(scores_pos)

    tps = np.empty((num_th,), dtype=np.int64)
    
    # NEGATIVES
    # same thing but for the negative samples
    scores_neg = scoremap[~mask]
    scores_neg = np.sort(scores_neg)
    num_neg = current_count_fp = len(scores_neg)

    fps = np.empty((num_th,), dtype=np.int64)
    
    # it will progressively drop the scores that are below the current th
    for thidx, th in enumerate(thresholds):
        
        # < becasue it is the same as ~(>=)
        num_drop = 0
        num_scores = len(scores_pos)
        while num_drop < num_scores and scores_pos[num_drop] < th:  # ! scores_pos !
            num_drop += 1
        # ---
        scores_pos = scores_pos[num_drop:]
        current_count_tp -= num_drop
        tps[thidx] = current_count_tp
        
        # same with the negatives 
        num_drop = 0
        num_scores = len(scores_neg)
        while num_drop < num_scores and scores_neg[num_drop] < th:  # ! scores_neg !
            num_drop += 1
        # ---
        scores_neg = scores_neg[num_drop:]
        current_count_fp -= num_drop
        fps[thidx] = current_count_fp
    
    fns = num_pos * np.ones((num_th,), dtype=np.int64) - tps
    tns = num_neg * np.ones((num_th,), dtype=np.int64) - fps
    
    # sequence of dimensions is (thresholds, true class, predicted class)
    # so `tps` is `confmat[:, 1, 1]`, `fps` is `confmat[:, 0, 1]`, etc.
    return np.stack((
        np.stack((tns, fps), axis=-1),
        np.stack((fns, tps), axis=-1),
    ), axis=-1).transpose(0, 2, 1)


numpy_numba = np.vectorize(_numpy_numba, signature="(n),(n),(k)->(k,2,2)")


# # numpy + numba + parallel

@numba.jit(nopython=True, parallel=True)
def numpy_numba_parallel(scoremaps: ndarray, masks: ndarray, thresholds: ndarray):
    num_imgs = scoremaps.shape[0]
    num_th = len(thresholds)
    ret = np.empty((num_imgs, num_th, 2, 2), dtype=np.int64)
    for imgidx in numba.prange(num_imgs):
        scoremap = scoremaps[imgidx]
        mask = masks[imgidx]
        ret[imgidx] = _numpy_numba(scoremap, mask, thresholds)
    return ret


# # torchmetrics unique values

def torchmetrics_unique_values(preds: Tensor, target: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """
    # copied from `torchmetrics.functional.classification.precision_recall_curve._binary_clf_curve`
    # in version 1.1.0
    # source permalink
    # https://github.com/Lightning-AI/torchmetrics/blob/eda473f52cb9cf3ea2aa5079f04db0b34ac7f96b/src/torchmetrics/functional/classification/precision_recall_curve.py#L28
    
    simplifications:
        - no option `sample_weights` ==> as if `sample_weights=None`
        - assumed `pos_label=1`
    """
    with torch.no_grad():
        # remove class dimension if necessary
        if preds.ndim > target.ndim:
            preds = preds[:, 0]
        desc_score_indices = torch.argsort(preds, descending=True)
        preds = preds[desc_score_indices]
        target = target[desc_score_indices]
        distinct_value_indices = torch.where(preds[1:] - preds[:-1])[0]
        threshold_idxs = F.pad(distinct_value_indices, [0, 1], value=target.size(0) - 1)
        target = (target == 1).to(torch.long)
        # simpllified to torch.cumsum() instead of the adhoc version
        tps = torch.cumsum(target * 1, dim=0)[threshold_idxs]
        fps = 1 + threshold_idxs - tps
        return fps, tps, preds[threshold_idxs]


# # test function

# ## validate args

def _validate_args(
    anomaly_score_maps: Tensor, 
    masks: Tensor,
    thresholds_minmax: Tensor,
    num_thresholds: int, 
):
    
    thresholds_minmax = torch.as_tensor(thresholds_minmax)
    
    if thresholds_minmax.ndim != 1 or thresholds_minmax.shape[0] != 2:
        raise ValueError(
            f"Expected argument `thresholds_minmax` to be a 1D tensor with 2 elements, but got {thresholds_minmax.shape}"
        )
        
    if thresholds_minmax[0] >= thresholds_minmax[1]:
        raise ValueError(
            f"Expected argument `thresholds_minmax` to be strictly increasing, but got {thresholds_minmax}"
        )
    
    # *** validate args ***
    
    if not isinstance(num_thresholds, int):
        raise ValueError(
            f"Expected argument `num_thresholds` to be an integer, but got {type(num_thresholds)}"
        )
    
    if num_thresholds < 2:
        raise ValueError(
            f"If argument `num_thresholds` is an integer, expected it to be larger than 1, but got {num_thresholds}"
        )
    
    if masks.is_floating_point():
        raise ValueError(
            "Expected argument `masks` to be an int or long tensor with ground truth labels"
            f" but got tensor with dtype {masks.dtype}"
        )

    if not anomaly_score_maps.is_floating_point():
        raise ValueError(
            "Expected argument `anomaly_score_maps` to be an floating tensor with anomaly scores,"
            f" but got tensor with dtype {anomaly_score_maps.dtype}"
        )
        
    # *** ***
    
    if not((anomaly_score_maps >= thresholds_minmax[0]).all() and (anomaly_score_maps <= thresholds_minmax[1]).all()):
        raise ValueError(
            "Expected argument `anomaly_score_maps` to be between in the range of" 
            f"`thresholds_minmax` = ({thresholds_minmax[0]}, {thresholds_minmax[1]}),"
        )


# ## func

ALGORITHMS = {
    "torchmetrics_loop": torchmetrics_loop,
    "torchmetrics_vectorized": torchmetrics_vectorized,
    "numpy_itertools": numpy_itertools,
    "numpy_numba": numpy_numba,   
    "numpy_numba_parallel": numpy_numba_parallel,
    "torchmetrics_unique_values": torchmetrics_unique_values,
}
        

def perimg_binclf_curve(
    anomaly_score_maps: Tensor, 
    masks: Tensor,
    thresholds_minmax: Tensor,
    num_thresholds: int, 
    algorithm: str,
):
    _validate_args(
        anomaly_score_maps=anomaly_score_maps, 
        masks=masks,
        thresholds_minmax=thresholds_minmax,
        num_thresholds=num_thresholds, 
    )
    
    try:
        algorithm_function = ALGORITHMS[algorithm]
    
    except KeyError as ex:
        raise ValueError(
            f"Algorithm `{algorithm}` not found. "
            f"Available algorithms are: {list(ALGORITHMS.keys())}"
        ) from ex

    # *** format ***
    th_min, th_max = thresholds_minmax
    anomaly_score_maps = anomaly_score_maps.flatten(1)
    masks = masks.flatten(1)

    # adjust thresholds
    # `preds.sigmoid()` not necessary, `anomaly_score_maps` are in R+
    # thresholds are linearly spaced between min and max
    thresholds = torch.linspace(
        start=th_min, 
        end=th_max, 
        steps=num_thresholds, 
        device=anomaly_score_maps.device,
    )
    
    # *** update() ***
    
    if algorithm in ("torchmetrics_loop", "torchmetrics_vectorized"):
        return torch.stack([
            algorithm_function(asmap, mask, thresholds)
            for asmap, mask in zip(anomaly_score_maps, masks)
        ], dim=0)
        
    if algorithm in ("numpy_itertools", "numpy_numba", "numpy_numba_parallel"):
        return torch.from_numpy(algorithm_function(
            anomaly_score_maps.cpu().numpy(),
            masks.cpu().numpy().astype(bool),
            thresholds.cpu().numpy(),
        )).to(anomaly_score_maps.device)
        
    if algorithm == "torchmetrics_unique_values":
        # not necessary to return anything
        [
            algorithm_function(asmap, mask)
            for asmap, mask in zip(anomaly_score_maps, masks)
        ]
        return
    
    raise NotImplementedError(f"Algorithm `{algorithm}` not implemented.")


def set_binclf_curve(
    anomaly_score_maps: Tensor, 
    masks: Tensor,
    thresholds_minmax: Tensor,
    num_thresholds: int, 
    algorithm: str,
):
    _validate_args(
        anomaly_score_maps=anomaly_score_maps, 
        masks=masks,
        thresholds_minmax=thresholds_minmax,
        num_thresholds=num_thresholds, 
    )
    
    try:
        algorithm_function = ALGORITHMS[algorithm]
    
    except KeyError as ex:
        raise ValueError(
            f"Algorithm `{algorithm}` not found. "
            f"Available algorithms are: {list(ALGORITHMS.keys())}"
        ) from ex

    # *** format ***
    th_min, th_max = thresholds_minmax
    anomaly_score_maps = anomaly_score_maps.flatten(1)
    masks = masks.flatten(1)

    # adjust thresholds
    # `preds.sigmoid()` not necessary, `anomaly_score_maps` are in R+
    # thresholds are linearly spaced between min and max
    thresholds = torch.linspace(
        start=th_min, 
        end=th_max, 
        steps=num_thresholds, 
        device=anomaly_score_maps.device,
    )
    
    # *** update() ***
    
    if algorithm in ("torchmetrics_loop", "torchmetrics_vectorized"):
        return algorithm_function(anomaly_score_maps.flatten(0), masks.flatten(0), thresholds)
        
    if algorithm in ("numpy_itertools", "numpy_numba", "numpy_numba_parallel"):
        return torch.from_numpy(algorithm_function(
            anomaly_score_maps.flatten(0).cpu().numpy(),
            masks.flatten(0).cpu().numpy().astype(bool),
            thresholds.cpu().numpy(),
        )).to(anomaly_score_maps.device)
    if algorithm == "torchmetrics_unique_values":
        # not necessary to return anything
        algorithm_function(anomaly_score_maps.flatten(0), masks.flatten(0))
        return
        
    raise NotImplementedError(f"Algorithm `{algorithm}` not implemented.")


# # cli args
parser = argparse.ArgumentParser("binclf-speed-test")
parser.add_argument("--resolution", type=int, required=True)
parser.add_argument("--num_images", type=int, required=True)
parser.add_argument("--num_thresholds", type=int, required=True)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--algorithm", type=str, required=True, choices=list(ALGORITHMS.keys()))
parser.add_argument("--device", type=str, required=True, choices=["cpu", "cuda"])
parser.add_argument("--mode", type=str, choices=["perimg", "set"], default="perimg")
cliargs = parser.parse_args()
print(f"{cliargs=}")

# # main (speed test)
resolution = cliargs.resolution
num_images = cliargs.num_images
num_thresholds = cliargs.num_thresholds
seed = cliargs.seed
algorithm = cliargs.algorithm
device = cliargs.device
mode = cliargs.mode

NUM_REPEATS = 5
NUM_EXEC_TIMEIT = 1

rng = np.random.default_rng(seed)

print("generating data...")
masks, asmaps = get_data_synthetic(num_images, resolution, rng)
masks = masks.to(device)
asmaps = asmaps.to(device)
minmax = (asmaps.min(), asmaps.max())

# compilation calls
if algorithm in ('numpy_numba', 'numpy_numba_parallel'):
    print("compiling...")
    _ = perimg_binclf_curve(asmaps, masks, minmax, num_thresholds, algorithm)

if mode == "perimg":
    command = "perimg_binclf_curve(asmaps, masks, minmax, num_thresholds, algorithm)"
elif mode == "set":
    command = "set_binclf_curve(asmaps, masks, minmax, num_thresholds, algorithm)"
else:
    raise ValueError(f"Unknown mode `{mode}`")

print("executing...")
times = timeit.Timer(
    command,
    globals=globals(),
).repeat(NUM_REPEATS, NUM_EXEC_TIMEIT)
records = [
    dict(
        resolution=resolution,
        num_images=num_images,
        num_thresholds=num_thresholds,
        seed=seed,
        algorithm=algorithm,
        device=device,
        mode=mode,
        hostname=HOSTNAME,
        num_cpus=NUM_CPUS,
        seconds=(seconds / NUM_EXEC_TIMEIT),
    )
    for seconds in times
]

print("saving...")
df = pd.DataFrame.from_records(records)
filename = "::".join([
    f"resolution={resolution}",
    f"num_images={num_images}",
    f"num_thresholds={num_thresholds}",
    f"seed={seed}",
    f"algorithm={algorithm}",
    f"device={device}",
    f"hostname={HOSTNAME}",
    f"num_cpus={NUM_CPUS}",
    f"mode={mode}",
])
filepath = SAVE_DIR / f"{filename}.csv"
df.to_csv(filepath, index=False)
