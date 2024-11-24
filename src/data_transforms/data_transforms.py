import torch
import numpy as np
import random
from typing import Tuple, List


def random_jitter(
    events: torch.Tensor,
    target: torch.Tensor,
    scale_blasts: float = 0.1,
    scale_healthy: float = 0.1,
    shuffle: bool = True,
    other_tensors: List[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Random pertubation of cells.
    Has to be applied before label smoothing.
    """

    blast_idx = target == 1.0
    healthy_idx = target == 0.0

    blasts = events[blast_idx, :]
    healthy = events[healthy_idx, :]

    # Perturb according to the scale parameters
    blasts_perturbed = torch.randn(blasts.shape) * scale_blasts + blasts
    healthy_perturbed = torch.randn(healthy.shape) * scale_healthy + healthy

    # concatenate blasts and healthy
    cells = torch.cat([blasts_perturbed, healthy_perturbed], dim=0)
    labels = torch.cat(
        [torch.ones(blasts.shape[0]), torch.zeros(healthy.shape[0])], dim=0
    )
    if other_tensors is not None:
        for idx, other_t in enumerate(other_tensors):
            other_tensors[idx] = torch.cat(
                (other_t[blast_idx], other_t[healthy_idx]), dim=0
            )

    if shuffle:
        shuffle_idx = torch.randperm(cells.shape[0])
        cells, labels = cells[shuffle_idx, :], labels[shuffle_idx]
        transformed_other_tensors = []
        if other_tensors is not None:
            for other_t in other_tensors:
                transformed_other_tensors.append(other_t[shuffle_idx])

    return cells, labels, transformed_other_tensors


def random_sample(
    events: torch.Tensor,
    target: torch.Tensor,
    cell_count: int,
    shuffle: bool = True,
    other_tensors: List[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    sample randomly  cell_count number of cells.
    """

    # random sample cells double if cell_count bigger than events in sample
    if cell_count > events.shape[0]:
        n_missing_cells = cell_count - events.shape[0]
        if events.shape[0] == 0:
            print("Something wrong with the events, there are none!?")
        upsample_idx = np.random.choice(
            events.shape[0], size=n_missing_cells, replace=True
        )
        upsample_events = events[upsample_idx]
        upsample_target = target[upsample_idx]
        events = torch.cat((events, upsample_events), dim=0)
        target = torch.cat((target, upsample_target), dim=0)

        if other_tensors is not None:
            for idx, other_t in enumerate(other_tensors):
                upsample_clusters = other_t[upsample_idx]
                other_tensors[idx] = torch.cat((other_t, upsample_clusters), dim=0)

    if shuffle:
        shuffle_idx = torch.randperm(events.shape[0])
        events, target = events[shuffle_idx, :], target[shuffle_idx]
        transformed_other_tensors = []
        if other_tensors is not None:
            for other_t in other_tensors:
                transformed_other_tensors.append(other_t[shuffle_idx])

    cells = events[:cell_count, :]
    labels = target[:cell_count]
    transformed_other_tensors_sampled = []
    if other_tensors is not None:
        for other_t in transformed_other_tensors:
            transformed_other_tensors_sampled.append(other_t[:cell_count])

    return cells, labels, transformed_other_tensors_sampled


def resample(
    events: torch.Tensor,
    target: torch.Tensor,
    scale_blasts: float = 0.1,
    scale_healthy: float = 0.1,
    cell_count: int = None,
    cutoff: int = 1000,
    target_blast_ratio: List[float] = None,
    ignore_extreme_blast_ratios: bool = False,
    target_blast_ratio_scale: float = None,
    shuffle: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perturbes blasts with gaussian noise.
    cell_count determines the number of cells in the resulting sample.
    target_blast_ratio determines the ratio of blast cells to healthy cells in the resulting sample.
    It is a list of lower and upper bound, whee with unifrom dist a ratio value inbetween is chosen.
    If ignore_extreme_blast_ratios is true use true blast ratio r if r is not in the interval given
    by target_blast_ratio.
    target_blast_ratio_scale is an alternative way of augmentation where the new blast ratio is
    drawn from a normal distribution with mean true_blast_ratio and scale target_blast_ratio_scale.
    cutoff gives the threshold of min number of cells that are needed in the sample so the blast ratio is adjusted.
    if we have less blasts than cutoff, we keep the original ratio, otherwise we use r sampled from target_blast_ratio.
    """
    if cell_count is None:
        cell_count = events.shape[0]

    # random sample cells double if cell_count bigger than events in sample
    if cell_count > events.shape[0]:
        n_missing_cells = cell_count - events.shape[0]
        upsample_idx = np.random.choice(
            events.shape[0], size=n_missing_cells, replace=True
        )
        upsample_events = events[upsample_idx]
        upsample_target = target[upsample_idx]
        events = torch.cat((events, upsample_events), dim=0)
        target = torch.cat((target, upsample_target), dim=0)

    # if target_blast_ratio is None keep existing ratio
    assert not (
        target_blast_ratio and target_blast_ratio_scale
    ), "either target_blast_ratio or target_blast_ratio_scale must be None."
    assert not (
        ignore_extreme_blast_ratios and not target_blast_ratio
    ), "you set ignore_extreme_blast_ratios to true without setting target_blast_ratio."

    if shuffle:
        shuffle_idx = torch.randperm(events.shape[0])
        events, target = events[shuffle_idx], target[shuffle_idx]

    r = target.sum() / target.shape[0]
    if target_blast_ratio:
        if ignore_extreme_blast_ratios:
            if r >= min(target_blast_ratio) and r <= max(target_blast_ratio):
                r = random.uniform(*target_blast_ratio)
        else:
            r = random.uniform(*target_blast_ratio)
        r = torch.tensor(r)
    elif target_blast_ratio_scale:
        r = torch.normal(r, target_blast_ratio_scale)
    r = torch.clamp(r, min=0.0, max=1.0)

    blast_idx = target > 0.5
    healthy_idx = target < 0.5
    # todo assertion that target values are not 0.5

    if blast_idx.sum() > cutoff:

        blasts = events[[blast_idx]]
        healthy = events[[healthy_idx]]

        # Select a subset of cells with the appropriate ratio
        max_blast_idx = int(cell_count * r)

        # for samples with too few samples sample multiple times from the same blast cells
        num_blasts = blasts.shape[0]
        if max_blast_idx > num_blasts:
            num_repeats_blasts = (max_blast_idx // num_blasts) + 1
            blasts = blasts.repeat((num_repeats_blasts, 1))
        blasts = blasts[:max_blast_idx]

        max_healthy_idx = cell_count - max_blast_idx
        num_healthy = healthy.shape[0]
        if max_healthy_idx > num_healthy:
            num_repeats_healthy = (max_healthy_idx // num_healthy) + 1
            healthy = healthy.repeat((num_repeats_healthy, 1))
        healthy = healthy[:max_healthy_idx]

        # Perturb according to the scale parameters
        blasts_perturbed = torch.randn(blasts.shape) * scale_blasts + blasts
        healthy_perturbed = torch.randn(healthy.shape) * scale_healthy + healthy

        # concatenate blasts and healthy
        cells = torch.cat([blasts_perturbed, healthy_perturbed], dim=0)
        labels = torch.cat(
            [torch.ones(blasts.shape[0]), torch.zeros(healthy.shape[0])], dim=0
        )

    else:
        healthy = events[:cell_count]
        healthy_perturbed = torch.randn(healthy.shape) * scale_healthy + healthy

        cells = healthy_perturbed
        labels = target[:cell_count]

    if shuffle:
        shuffle_idx = torch.randperm(cells.shape[0])
        cells, labels = cells[shuffle_idx], labels[shuffle_idx]

    return cells, labels


def clean_spillover(
    events: torch.Tensor,
    target: torch.Tensor,
    clean_percentage: float = 0.001,
    other_tensors: List[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    clean spillover events
    """

    max_val = 4.5
    min_val = 0.0
    thr = (max_val - min_val) * clean_percentage
    minThr = min_val + thr
    maxThr = max_val - thr

    mask = (events < minThr) | (events > maxThr)
    remove_event = torch.any(mask, dim=1)

    transformed_other_tensors = []
    if other_tensors is not None:
        for other_t in other_tensors:
            transformed_other_tensors.append(other_t[~remove_event])

    return events[~remove_event, :], target[~remove_event], transformed_other_tensors


def label_smoothing(
    events: torch.Tensor,
    target: torch.Tensor,
    label_smoothing_eps: float = 0.1,
    n_classes: int = 2,
    other_tensors: List[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    target = target * (1 - label_smoothing_eps) + label_smoothing_eps / n_classes
    return events, target, other_tensors


class TransformStack:

    def __init__(
        self, stack: list, dataset_type: str, execution_probability: float = 1.0
    ):
        self.stack = stack
        self.ex_prob = execution_probability
        self.dataset_type = dataset_type
        assert self.dataset_type in ["train", "test", "eval"]

    def __call__(self, events, target, other_tensors: List[torch.Tensor] = None):
        for transform in self.stack:
            if transform.__name__ in ["min_max_scale_cols", "clean_spillover"]:
                events, target, other_tensors = transform(
                    events, target, other_tensors=other_tensors
                )
            elif (
                self.dataset_type == "train" and random.random() < self.ex_prob
            ):  # random.random in [0,1)
                events, target, other_tensors = transform(
                    events, target, other_tensors=other_tensors
                )

        return events, target, other_tensors
