import argparse
import torch
from torch.utils.data import DataLoader

from src.data_loader.dataset_pyg import PYGFlowDataset, Collater
from src.trainer.supervised_pyg_trainer import Trainer
from src.utils.configparser import ConfigParser
from src.utils.datastructures import SupervisedConfig
from src.utils.dynamictypeloader import init_obj, init_ftn
from src.utils.loggingmanager import LoggingManager
from src.utils.wandb_logger import WandBLogger

MAX_NUM_THREADS = 6
torch.set_num_threads(MAX_NUM_THREADS)
import os

# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["WANDB_SERVICE_WAIT"] = "300"
os.environ["WANDB_INIT_TIMEOUT"] = "600"


def train(config: SupervisedConfig, wandb_logger=None):

    # python logger
    logging_manager = LoggingManager(config.logging_config)
    logging_manager.register_handlers(
        name="train", save_path=config.output_save_dir / "log.txt"
    )
    logger = logging_manager.get_logger_by_name(name="train")

    # create dataset and loader
    logger.info("-" * 20 + "Creating data_loader instance..")
    train_data = PYGFlowDataset.init_from_config(
        config.data_loader_config, dataset_type="train"
    )

    train_dataloader = DataLoader(
        train_data,
        batch_size=config.data_loader_config.batch_size,
        num_workers=config.data_loader_config.num_workers,
        shuffle=config.data_loader_config.shuffle,
        pin_memory=config.data_loader_config.pin_memory,
        collate_fn=Collater(),
    )
    eval_data = None
    if config.do_eval:
        eval_data = PYGFlowDataset.init_from_config(
            config.data_loader_config, dataset_type="eval"
        )
        eval_dataloader = DataLoader(
            eval_data,
            batch_size=1,
            num_workers=1,
            shuffle=False,
            pin_memory=False,
            collate_fn=Collater(),
        )

    logger.info("-" * 20 + "Done!")

    # model loading
    logger.info("-" * 20 + "Loading your hot shit model..")
    model = init_obj(config.supervised_model)

    logger.info(model)
    logger.info("-" * 20 + "Done!")

    # load pretrained model
    if config.pretrained_model_path is not None:
        logger.info(
            "Loading pretrained model weights: {} ...".format(
                str(config.pretrained_model_path)
            )
        )
        checkpoint = torch.load(config.pretrained_model_path, map_location="cpu")
        pretrained_dict = checkpoint["state_dict"]
        model_dict = model.state_dict()

        for key in pretrained_dict.keys():
            if key in model_dict.keys():
                model_dict[key] = pretrained_dict[key]

        model.load_state_dict(model_dict)

    # push model to gpu and get trainable params
    model.to(f"cuda:{config.gpu_id}")
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())

    # loss, metrics, optimizer, learning rate scheduler
    logger.info("-" * 20 + "Initializing loss, metrics, optimizer, lr scheduler..")
    loss_ftn = init_ftn(config.loss)
    metric_ftns = [init_ftn(config_met) for config_met in config.train_metrics_list]
    optimizer = init_obj(config.optimizer, trainable_params)
    lr_scheduler = init_obj(
        config.lr_scheduler, optimizer
    )  # returns none if obj. to be initialized (e.g. lr_scheduler) is none
    logger.info("-" * 20 + "Done!")

    # WandB logger
    if wandb_logger is None:
        logger.info("-" * 20 + "Initializing W&B Logger..")
        wandb_logger = WandBLogger(
            config.wandb_config, model, run_name=config.config_name, run_config=config
        )
        logger.info("-" * 20 + "Done!")
        dispose = True
    else:
        wandb_logger.watch(model)
        dispose = False

    # get the trainer class
    logger.info("-" * 20 + "Initializing your marvellous trainer..")

    trainer = Trainer(
        model,
        loss_ftn,
        metric_ftns,
        optimizer,
        config.trainer_config,
        wandb_logger,
        config.output_save_dir,
        config.gpu_id,
        train_dataloader,
        eval_data_loader=eval_dataloader,
        lr_scheduler=lr_scheduler,
        resume_path=config.resume_path,
    )

    logger.info("-" * 20 + "Done!")

    trainer.train()
    if dispose:
        trainer.dispose()  # finish WandBLogger


if __name__ == "__main__":

    args = argparse.ArgumentParser(description="Training ")
    configParser = ConfigParser(mode="train")
    config_default = "path_to_default_config/config.json"
    args.add_argument(
        "-c",
        "--config",
        default=config_default,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d", "--device", default="0", type=str, help="index of which GPU to use"
    )

    config = configParser.parse_config_from_args(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_id)
    config.gpu_id = 0
    train(config)
