from typing import List
import torch
from tqdm import tqdm
from collections import namedtuple
from torch_geometric.nn import knn_graph
import torch_geometric.transforms as T

from src.trainer.base_trainer import BaseTrainer
from src.utils.utils import MetricTracker
from src.utils.vis import mrd_plot


Log_plot = namedtuple("Log_plot", ["name", "figure"])


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
        self,
        model,
        loss_ftn,
        metric_ftns,
        optimizer,
        config,
        wandb_logger,
        training_save_dir,
        gpu_id,
        data_loader,
        eval_data_loader=None,
        lr_scheduler=None,
        resume_path=None,
    ):

        self.config = config

        # used in baseclass so written as property to not have to pass it over
        self._model = model
        self._optimizer = optimizer
        super().__init__(
            model,
            optimizer,
            config,
            wandb_logger,
            training_save_dir,
            gpu_id,
            resume_path=resume_path,
        )

        self.model.to(self.device)

        self.loss_ftn = loss_ftn
        self.metric_ftns = metric_ftns

        # Data
        self.data_loader = data_loader
        self.valid_data_loader = eval_data_loader

        self.len_epoch = len(self.data_loader)
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler

        # Metrics for training and validation
        self.train_metrics = MetricTracker(
            "loss", *[m.__name__ for m in self.metric_ftns]
        )
        self.valid_metrics = MetricTracker(
            "loss", *[m.__name__ for m in self.metric_ftns]
        )

        self.valid_freq = 5
        self.transform = T.Compose([T.ToUndirected()])  # , T.ToSparseTensor()])

    @property
    def model(self):
        return self._model

    @property
    def optimizer(self):
        return self._optimizer

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch. Starts by 1 not 0
        :return: A log that contains average loss and metric in this epoch.
        """

        self.model.train(True)
        self.train_metrics.reset()

        log_plots = []

        for batch_idx, batch in tqdm(
            enumerate(self.data_loader), desc="train", total=len(self.data_loader)
        ):
            self.optimizer.zero_grad()
            x, y, batch_ids = (
                batch.x.to(self.device),
                batch.y.to(self.device).unsqueeze(-1),
                batch.batch.to(self.device),
            )
            # edge_index = batch.adj_t.to(self.device)
            # edge_index = batch.edge_index.to(self.device)
            batch_size = torch.max(batch_ids) + 1
            # batch_size = torch.max(batch.batch)
            batch.edge_index = knn_graph(
                x, k=self.config.knn_k, batch=batch_ids, loop=False
            )
            # batch = self.transform(batch)
            if self.config.remove_marker:

                prediction = self.model(
                    x[:, [0, 1, 2, 3, 4, 7, 9, 10]], batch.edge_index, batch_size
                )  # apply model

            else:
                prediction = self.model(x, batch.edge_index, batch_size)  # apply model

            loss = self.loss_ftn(prediction, y)

            # backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2.5)
            self.optimizer.step()

            self.train_metrics.update("loss", loss.item(), n=batch_size)

            for met in self.metric_ftns:
                self.train_metrics.update(
                    met.__name__,
                    met(prediction.detach().cpu(), y.detach().cpu()),
                    n=batch_size,
                )

            if batch_idx == self.len_epoch:
                break

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        log, median_log = self.train_metrics.result()

        return log, median_log, log_plots

    def _val_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """

        self.model.eval()
        self.valid_metrics.reset()

        filenames = []
        log_plots = []

        if self.do_validation:
            with torch.no_grad():
                if epoch % self.valid_freq == 0 or epoch == self.epochs or epoch == 1:
                    for batch_idx, batch in tqdm(
                        enumerate(self.valid_data_loader),
                        desc="eval",
                        total=len(self.valid_data_loader),
                    ):
                        # Load data
                        x, target, batch_ids = (
                            batch.x.to(self.device),
                            batch.y.to(self.device).unsqueeze(-1),
                            batch.batch.to(self.device),
                        )
                        batch.edge_index = knn_graph(
                            x, k=self.config.knn_k, batch=batch_ids, loop=False
                        )  # .to(self.device)
                        # batch = self.transform(batch)
                        batch_size = torch.max(batch_ids) + 1
                        if self.config.remove_marker:

                            prediction = self.model(
                                x[:, [0, 1, 2, 3, 4, 7, 9, 10]],
                                batch.edge_index,
                                batch_size,
                            )  # apply model

                        else:
                            prediction = self.model(
                                x, batch.edge_index, batch_size
                            )  # apply model

                        filenames.append(
                            batch.name[0]
                        )  # because it is a batch so we have a list of names

                        loss = self.loss_ftn(prediction, target)

                        self.valid_metrics.update("loss", loss.item())
                        for met in self.metric_ftns:
                            self.valid_metrics.update(
                                met.__name__,
                                met(prediction.detach().cpu(), target.detach().cpu()),
                            )

                    # MRD figure
                    log_plots.append(self._log_mrd_plot(self.valid_metrics, filenames))
        log, median_log = self.valid_metrics.result()
        return log, median_log, log_plots

    def _log_mrd_plot(self, metrics: MetricTracker, filenames: List[str]) -> Log_plot:

        # MRD figure
        metric_data = metrics.data()
        mrd_fig = mrd_plot(
            mrd_list_gt=metric_data["mrd_gt"],
            mrd_list_pred=metric_data["mrd_pred"],
            f1_score=metric_data["f1_score"],
            filenames=filenames,
        )

        return Log_plot("MRD", mrd_fig)

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.data_loader, "n_samples"):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
