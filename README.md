# flowNetworks
Official Pytorch implementation of __On the importance of local and global feature
learning for automated measurable residual
disease detection in flow cytometry data__ accepted at ICPR 2024.

The paper provides an overview of various models suitable for processing single cell flow cytometry data ranging from simple MLP over attention- and graph-based models to hybrid versions. It proposes several adaptations to the [flowformer model](https://github.com/mwoedlinger/flowformer) leading to improved performances. 


This code is intended to foster collaboration and provide a framework for easy plug and play with different models for single cell flow cytometry data. Contributions and enhancements of the codebase are more than welcome.



---
## Setup 🛠
Clone the repository, create environment and install the required packages as follows:
```bash
git clone git@github.com:lisaweijler/flowNetworks.git 
cd flowNetworks 
conda create --name flownetworks python=3.10
conda activate flownetworks
conda install pytorch=2.5.1 torchvision=0.20.1 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install torch-geometric==2.6.1 
pip install pyg-lib==0.4.0 torch-cluster==1.6.3 torch-scatter==2.1.2 torch-sparse==0.6.18 -f https://data.pyg.org/whl/torch-2.5.1+cu121.html
conda install pandas=2.2.3 wandb=0.18.7 plotly=5.24.1 matplotlib=3.9.2 einops=0.8.0

```

Note: `pip install torch-geometric` might give dependency issues with `sympy`. If this happens simply install the required version e.g. `pip install sympy==1.13.1` and reinstall torch-geometric if necessary.

---


## Run flowNetworks 🚀
In this section you will find information on how to use this repository.  

### General information ⚙️
* **Config files**  In the folder `config_templates` a templated for training/testing the GIN-ST-FPS model is provided. This can be used as a starting point for your own experiments and adjusted as needed. 
* **Data** This project works with preloaded flow cytometry samples saved via pickle as pandas dataframes. To preload (including compensation, transformation and scaling) the package [flowmepy](https://pypi.org/project/flowmepy/) was used. The config files expect a "preloaded_data_dir" argument, where the path to the folder with the preloaded samples is specified. The path "data_splits_dir" should lead to a folder containing three *.txt files (train.txt and eval.txt are needed during training, while test.txt is needed during testing), where every line contains the path to a FCM file (.xml or .analysis). Those file paths in the *.txt files are used to load the correct files from the "preloaded_data_dir". In the folder `data_splits` an example is given. The vie14, bln and bue data from our work can be downloaded from [here](https://flowrepository.org/id/FR-FCM-ZYVT).


### Training 🚀
Example command for training a model in this framework:
```
python train_pyg.py --config config_templates/GIN-ST-FPS.json --device 0
```
For the pointnet models use the `train_pointnet.py`and `test_pointnet.py`, whenever pytorch-geometric is involved the `train_pyg.py`and `test_pyg.py`, and else the `train.py`and `test.py` files.

### Testing 📊
For testing you can use the same config as for training and specify the model to use via the command line and the --resume flag.
Example command for testing:
```
python test_pyg.py --config config_templates/GIN-ST-FPS.json --device 0 --resume path/to/your/trained/model.pth
```