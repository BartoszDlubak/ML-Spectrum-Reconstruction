import yaml
import torch
from pathlib import Path
import numpy as np

from preprocess.normaliser import minmaxNormaliser
from models.CSDI.main_model import CSDI_spectra
from eval.eval import eval_model
from torch.utils.data import DataLoader

seed = 42

torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

    
    
def load_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config



def get_normaliser(x):
    normaliser = minmaxNormaliser()
    normaliser.fit(x)
    return normaliser 


    
def get_torch_config(torch_config_path, device):
    
    torch_dict = torch.load(torch_config_path, map_location=device)

    model_config = torch_dict["config"]
    model_state_dict = torch_dict["model_state_dict"]
    return model_state_dict, model_config
      
      
    
# Run data
def run_eval(
    test_dataset,
    batch_size,
    device,
    normaliser,
    model_config: dict, 
    data_config: dict, 
    save_path: Path,
    ):
    
    '''
    test_dataset:
        Torch dataset object containing testing data
    batch_size ( int ):
        Batch size
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
    test_loader = DataLoader(test_dataset, batch_size, shuffle=True)
    
    #Model Creation:
    model = CSDI_spectra(config = model_config, device=device, data_config = data_config)
    
    # Train and evaluate simulated data
    eval_model(model, test_loader, normaliser, num_samp=20, save_path=save_path)
   
   

torch_config_path = ''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_config = load_config('data_config.yaml')
model_state_dict, model_config = get_torch_config(torch_config_path, device)

#run_eval()