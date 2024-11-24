import argparse
import json
import torch
from tqdm import tqdm
from typing import List
from torch.utils.data import DataLoader
from pathlib import Path
from collections import namedtuple
import pandas as pd
import wandb


from src.data_loader.dataset_base import FlowDataset
from src.utils.utils import MetricTracker
from src.utils.configparser import ConfigParser
from src.utils.dynamictypeloader import init_obj, init_ftn
from src.utils.datastructures import SupervisedConfig
from src.utils.loggingmanager import LoggingManager
from src.utils.wandb_logger import WandBLogger
from src.utils.vis import PanelPlotTargetVSPrediction, mrd_plot



Log_plot = namedtuple('Log_plot', ['name', 'figure'])


def results_to_file(metrics: MetricTracker, 
                    file_list: List[str],
                    n_events_list: List[int], 
                    config_name: str,
                    model_path: str,
                    out_dir: Path) -> str:



    metric_names= metrics.get_metric_names()
    metric_data = metrics.data()
    metric_avg, metric_median = metrics.result()

    # header: general info and mean/ med results
    header =  f"// config_name: {config_name}\n"
    header += f"// model path:  {model_path}\n\n"
    for m_name in metric_names:
        header += f"// mean-{m_name:{25}}: {metric_avg[m_name]}\n"
        header += f"// median-{m_name:{23}}: {metric_median[m_name]}\n"
    header += "\n"


    # columns of file-wise results
    cols = "# experiment, label, total, " + ", ".join(metric_names) + "\n"

    # data of file-wise results
    log_list = []
    for idx, file in enumerate(file_list):
        metric_result_list = [f'{metric_data[m][idx]}' for m in metric_names]
        log_list.append(f'{file}, unknown, ' + str(n_events_list[idx]) + ', '+ ', '.join(metric_result_list))

    # write results to file
    with open(str(out_dir / (config_name + '_test_results.txt')), 'w') as text_file:
        text_file.write(header)
        text_file.write(cols)
        for l in log_list:
            text_file.write(l+'\n')

    return header


def test(config: SupervisedConfig, wandb_logger=None):

    # python logger
    logging_manager = LoggingManager(config.logging_config)
    logging_manager.register_handlers(name='test', 
                                      save_path=config.output_save_dir/'log.txt')
    logger = logging_manager.get_logger_by_name(name='test')

    # dataloader 
    logger.info('-'*20 + "Creating data_loader instance..")
    test_data = FlowDataset.init_from_config(config.data_loader_config, 
                                             dataset_type='test')
    test_dataloader = DataLoader(test_data, 
                                 batch_size=1)
    logger.info('-'*20 + 'Done!')

    # init loss and metrics
    logger.info('-'*20 + "Initializing loss and metrics.." )
    loss_ftn = init_obj(config.loss)
    metric_ftns = [init_ftn(config_met) for config_met in config.test_metrics_list]
    metrics_tracker = MetricTracker('loss', *[m.__name__ for m in metric_ftns])
    logger.info('-'*20 +'Done!')

    # model loading
    logger.info('-'*20 + "Initializing your hot shit model architecture..")
    model = init_obj(config.supervised_model)


    logger.info(model) 
    logger.info('-'*20 + 'Done!')



    #WandB logger
    if wandb_logger is None:
        logger.info('-'*20 + "Initializing W&B Logger.." )
        wandb_logger = WandBLogger(config.wandb_config, model, run_config=config)
        logger.info('-'*20 +'Done!')

    # loading check point/trained model parameters
    logger.info('-'*20 + "Loading checkpoint: {} ...".format(config.resume_path))
    checkpoint = torch.load(config.resume_path)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)

    # prepare model for testing
    logger.info('-'*20 + "Prepare model for testing..")
    device = torch.device(f'cuda:{config.gpu_id}' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    logger.info('-'*20 + 'Done!')

    # Begin testing
    logger.info('-'*20 + 'Firing up testing process..')
    file_list = []
    n_events_list = []
    filenames = []
    with torch.no_grad():
        for sample_idx, sample in tqdm(enumerate(test_dataloader), desc='test', total=len(test_dataloader)):

            data = sample['data'].to(device)
            target = sample['target'].float().to(device).unsqueeze(-1)
            name = sample['name'][0]
            
            filepath = sample['filepath'][0]
            file_list.append(filepath)
            filenames.append(name)

            n_events_list.append(int(sample['n_events'].item()))


            output, feats = model(data)

            
            loss = loss_ftn(output, target, feats)
            
            metrics_tracker.update('loss', loss.item())
            for met in metric_ftns:
                metrics_tracker.update(met.__name__, met(output, target))

            plot = PanelPlotTargetVSPrediction(config.figures_save_dir / f"{name}.png",
                                            name, 
                                            data.squeeze(0).cpu().detach().numpy(),
                                            target.squeeze(0).cpu().detach().numpy(),
                                            output.squeeze(0).cpu().detach().numpy(),
                                            config.data_loader_config.flow_sample_config.markers,
                                            config.vis_config.panel,
                                            config.vis_config.min_fig_size,
                                            config.vis_config.n_points)
            plot.generatePlotFile()

                
    

    log_plots = []
    # write results to wandb
    log, median_log = metrics_tracker.result()

    metric_data = metrics_tracker.data()

    mrd_fig = mrd_plot(mrd_list_gt=metric_data['mrd_gt'], mrd_list_pred=metric_data['mrd_pred'], f1_score=metric_data['f1_score'], filenames=filenames)
    log_plots.append(Log_plot("MRD", mrd_fig))

    result_table = pd.DataFrame.from_dict({'filenames': filenames, **metric_data})

    result_log = {'test/result_table': wandb.Table(dataframe=result_table)}
    for k,v in log.items():
        result_log.update({'test/' + k: v, 'test/median-' + k: median_log[k]})
    for plot in log_plots:
        result_log.update({"test/" + plot.name: plot.figure})

    wandb_logger.log(result_log)


    # write results to .txt file
    results_summary = results_to_file(metrics_tracker, 
                                    file_list,
                                    n_events_list, 
                                    config.config_name,
                                    config.resume_path,
                                    config.output_save_dir)

    logger.info(results_summary)


    wandb_logger.finish()





if __name__ == "__main__":
    args = argparse.ArgumentParser(description='Testing')
    configParser = ConfigParser(mode="test")
    config_default = "path_to_default_config/config.json"
    args.add_argument('-c', '--config', default=config_default, type=str, # should be mandatory 
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default="your_trained_model.pth", type=str, # should be mandatory
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default="0", type=str,
                      help='indices of GPUs to enable (default: all)')
    config = configParser.parse_config_from_args(args)


    test(config)