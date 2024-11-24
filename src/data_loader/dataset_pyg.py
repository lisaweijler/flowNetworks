import torch
from pathlib import Path
from typing import Dict
from torch_geometric.data import Data
import torch_geometric.transforms as T

from collections.abc import Mapping
from typing import Any, List, Sequence

import torch.utils.data
from torch.utils.data.dataloader import default_collate

from torch_geometric.data import Batch
from torch_geometric.data.data import BaseData
from torch_geometric.typing import TensorFrame, torch_frame


from src.utils.datastructures import FlowSampleConfig
from src.data_transforms.data_transforms import TransformStack
from src.utils.markercollection import MarkerCollection
from .dataset_base import FlowDataset


class Collater:
    def __init__(self):
        pass

    def __call__(self, batch: List[Any]) -> Any:
        elem = batch[0]
        if isinstance(elem, BaseData):
            return Batch.from_data_list(
                batch,
                follow_batch=None,
                exclude_keys=None,
            )
        elif isinstance(elem, torch.Tensor):
            return default_collate(batch)
        elif isinstance(elem, TensorFrame):
            return torch_frame.cat(batch, dim=0)
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float)
        elif isinstance(elem, int):
            return torch.tensor(batch)
        elif isinstance(elem, str):
            return batch
        elif isinstance(elem, Mapping):
            return {key: self([data[key] for data in batch]) for key in elem}
        elif isinstance(elem, tuple) and hasattr(elem, "_fields"):
            return type(elem)(*(self(s) for s in zip(*batch)))
        elif isinstance(elem, Sequence) and not isinstance(elem, str):
            return [self(s) for s in zip(*batch)]

        raise TypeError(f"DataLoader found invalid type: '{type(elem)}'")


class PYGFlowDataset(FlowDataset):
    """
    Flow cytometry pytorch dataset. Processing preloaded files saved as pkl.
    """

    def __init__(
        self,
        preload_dir: Path,
        data_splits_dir: Path,
        dataset_type: str,
        enable_standardization: bool,
        flowsample_config: FlowSampleConfig,
        transform: TransformStack,
    ):

        super().__init__(
            preload_dir,
            data_splits_dir,
            dataset_type,
            enable_standardization,
            flowsample_config,
            transform,
        )
        self.pyg_transform = T.Compose([T.ToUndirected()])

    def __getitem__(self, idx) -> Dict:
        filepath = self.filepaths[idx]
        spl = self.fcm_sample_type.load_from_pickle_files(
            Path(self.preload_dir),
            Path(filepath).with_suffix("").name,
            self.flowsample_config,
            apply_marker_filter=True,
            apply_gate_filter=True,
        )

        n_events = len(spl.events.index)
        marker = spl.get_marker_list()
        name = spl.name
        marker = MarkerCollection.renameMarkers_after_preload(
            marker
        )  # in case markerdict has changed after preloading - but actually should not be necessary
        data = torch.tensor(spl.events.values)
        target_values = spl.get_class_labels()
        target = (
            target_values
            if isinstance(target_values, torch.Tensor)
            else torch.tensor(target_values.values)
        )

        if self.enable_standardization:
            data = self.strdhandler.standardize_tensor(data)

        name = spl.name
        other_tensors = []

        if self.transform is not None:
            data, target, other_tensors = self.transform(
                data, target, other_tensors
            )  # only applies transform for train data except minmaxscale (see transformstack class)

        pyg_data = Data(x=data.float(), y=target.float())

        pyg_data.marker = marker
        pyg_data.name = name
        pyg_data.n_events = n_events
        pyg_data.filepath = str(filepath)

        return pyg_data
