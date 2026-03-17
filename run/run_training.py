import yaml
import itertools
import torch
from pathlib import Path
from datetime import datetime

from preprocess.dataset import build_dataset
from preprocess.normaliser import minmaxNormaliser
from models.CSDI.main_model import CSDI_spectra
from torch.utils.data import DataLoader
from train.train import train_model
from eval.eval import eval_model



def load_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config



def get_normaliser(x):
    normaliser = minmaxNormaliser()
    normaliser.fit(x)
    return normaliser 



def load_model(model_config, data_config, model_state_dict):
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CSDI_spectra(
        config=model_config,
        device=device,
        data_config=data_config
    )

    model.load_state_dict(model_state_dict)

    print("Model loaded")

    return model


    
def run_training(
    train_dataset,
    valid_dataset,
    batch_size,
    device,
    normaliser,
    model_config: dict, 
    data_config: dict, 
    save_path: Path,
    ):
    
    '''
    train_dataset:
        Torch dataset object containing training data
    valid_dataset:
        Torch dataset object containing validation data
    batch_size ( int ):
        batch size
    device:
        Torch device
    normaliser:
        Object used to normalise data 
    model_config: ( dict ) 
        Model and training hyperparameters.
    data_config: ( dict )
        Used during training. Controls amount of noise and probability of a conditional observation vanishing.
    save_path: ( Path )
        Path to directory which stores the output. 
    '''
    
    #Dataloaders
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size, shuffle=False) 
    
    #Model Creation:
    model = CSDI_spectra(config = model_config, device=device, data_config = data_config)
    
    # Train and evaluate simulated data
    train_model(model, model_config, train_loader, valid_loader, save_path, valid_epoch_interval=2, inference_interval=3)
    eval_model(model, valid_loader, normaliser, num_samp=20, save_path=save_path)
   
   
   
    
model_config = load_config("model_config.yaml")
data_config = load_config('data_config.yaml')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#run_training()
