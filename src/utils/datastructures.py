from dataclasses import dataclass, field, asdict
from typing import List, OrderedDict
from pathlib import Path
from xmlrpc.client import Boolean
from numpy import inf
from collections import namedtuple

import logging


Metric = namedtuple("Metric", ["idx", "name", "function"])
Sample = namedtuple("Sample", ["idx", "name", "data", "labels"])


@dataclass
class LoggingConfig:
    # file_log_handler_path : Path = Path("")
    enable_stream_log_handler: bool = True
    enable_file_log_handler: bool = False
    stream_log_handler_level: int = logging.DEBUG
    file_log_handler_level: int = logging.ERROR


@dataclass
class GenericConfig:
    module: str  # e.g name of object/function for function handle
    args: tuple = tuple()
    kwargs: dict = field(default_factory=dict)  # e.g args for function handle


@dataclass
class FlowSampleConfig:
    markers: List[str]  # marker to use todo: if empty use all of them
    filter_gates: List[str]  # events not in this gates are dismissed  - OR connection
    classes_dict: OrderedDict  # labels to be used in prediction: f. e. {"blast": 1, "granulocytes": 2, "other": 0}, it uses the order given in the config.. make sure to think of correct order otherwise blasts can end up being granulocytes..
    sample_type: str = "src.data_loader.flow_sample.FlowSampleFile"
    kwargs: dict = None


@dataclass
class TransformsConfig:
    transforms_list: List[GenericConfig]
    execution_probability: float


@dataclass
class DataLoaderConfig:
    data_splits_dir: Path
    preloaded_data_dir: Path
    batch_size: int
    num_workers: int
    flow_sample_config: FlowSampleConfig
    shuffle: bool = False
    pin_memory: bool = False
    transforms_config: TransformsConfig = None
    enable_standardization: bool = False


@dataclass
class WandBConfig:
    project_name: str
    team: str
    notes: str = None
    tags: List[str] = field(default_factory=List)
    group: str = None
    enabled: bool = True


@dataclass
class MonitorConfig:
    criterion: str
    data_type: str
    metric_name: str


@dataclass
class TrainerConfig:
    epochs: int
    save_period: int
    monitor_config: MonitorConfig = None
    early_stop: int = inf
    knn_k: int = 10  # only used for the graph stuff
    remove_marker: bool = False


@dataclass
class VisualizationConfig:
    panel: List[List[str]]
    min_fig_size: int = 6
    n_points: int = 10000
    plot_attention: bool = True


@dataclass
class PlotConfig:
    config_name: str
    vis_config: VisualizationConfig
    data_loader_config: DataLoaderConfig
    save_dir: Path
    gpu_id: int  # dummy to be able to reuse config parser
    data_type: str  # eval, test, train
    resume_path: Path = None  # dummy to be able to reuse config parser
    figures_save_dir: Path = None  # dummy to be able to reuse config parser
    output_save_dir: Path = None  # dummy to be able to reuse config parser


# used
@dataclass
class SupervisedConfig:
    config_name: str
    logging_config: LoggingConfig
    data_loader_config: DataLoaderConfig
    supervised_model: GenericConfig
    optimizer: GenericConfig
    loss: GenericConfig
    train_metrics_list: List[
        GenericConfig
    ]  # function handle config is generic -> param change with type
    test_metrics_list: List[GenericConfig]
    trainer_config: TrainerConfig
    output_save_dir: Path  # gets added in config parser basically root / timestanmp , either test output or train output
    vis_config: VisualizationConfig
    resume_path: (
        Path  # getts added in config parser form flag arguments of command line..
    )
    gpu_id: int  # getts added in config parser form flag arguments of command line..
    figures_save_dir: (
        Path  # getts added in config parser form flag arguments of command line..
    )
    lr_scheduler: GenericConfig = None
    wandb_config: WandBConfig = None
    do_eval: bool = False
    pretrained_model_path: Path = None

    def dict(self):
        return {k: str(v) for k, v in asdict(self).items()}
