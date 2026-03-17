import numpy as np
import torch
import torch.nn as nn
from .diff_models import diff_CSDI

def _set_missing(cond_mask: torch.Tensor, level: float = 0.0):
    """
    Randomly masks elements of cond_mask with probability `level`.
    """
    rand_mask = (torch.rand_like(cond_mask.float()) > level).int()
    cond_mask = cond_mask * rand_mask
    return cond_mask

def _add_log_noise(observed_data: torch.Tensor, cond_mask: torch.Tensor, level: float = 0.0):
    """
    Adds small multiplicative noise in log-space to `observed_data` where `cond_mask` == 1.
    """
    noise = level * torch.randn_like(observed_data)
    observed_data = observed_data.clone()  # avoid in-place modification
    observed_data[cond_mask == 1] += noise[cond_mask == 1]
    return observed_data



class CSDI_base_spectra(nn.Module):
    """
    Description:
    
    Base model which combines the forward diffusion process, backwards diffuson process (from diff_models.py), training and data imputation.
    
    Parameters:
    
    target_dim (int): number of features
    config (.yaml): contains all model hyper-parameters
    device (torch.device): device on which the program is run
    """
    
    def __init__(self, target_dim, config, data_config, device):
        super().__init__()
        
        self.device = device
        self.target_dim = target_dim
        
        self.emb_time_dim = config["model"]["timeemb"]
        self.emb_feature_dim = config["model"]["featureemb"]
        self.is_unconditional = config["model"]["is_unconditional"]
        self.target_strategy = config["model"]["target_strategy"]
        self.data_config = data_config

        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim
        if self.is_unconditional == False:
            self.emb_total_dim += 1  # for conditional mask
        self.embed_layer = nn.Embedding(
            num_embeddings=self.target_dim, embedding_dim=self.emb_feature_dim
        ).to(self.device)

        config_diff = config["diffusion"]
        config_diff["side_dim"] = self.emb_total_dim
        self.regularization = config["train"]["regularization"]

        input_dim = 1 if self.is_unconditional == True else 2
        
        #importing neural network for back process
        self.diffmodel = diff_CSDI(config_diff, input_dim).to(self.device)

        # parameters for diffusion model and the schedule for beta
        self.num_steps = config_diff["num_steps"]
        
        if config_diff["schedule"] == "quad":
            self.beta = np.linspace(
                config_diff["beta_start"] ** 0.5, config_diff["beta_end"] ** 0.5, self.num_steps
            ) ** 2
        elif config_diff["schedule"] == "linear":
            self.beta = np.linspace(
                config_diff["beta_start"], config_diff["beta_end"], self.num_steps
            )
        elif config_diff["schedule"] == "cosine":
            steps = self.num_steps

            
            def cosine_alpha_bar(t, T, stretch = 1.5):
                t_scaled = (t / T) ** stretch
                min_val = 1e-3
                return (1-min_val) * (np.cos((t_scaled) * (np.pi / 2))) ** 2 + min_val

            self.beta = np.array([
                1 - (cosine_alpha_bar(t + 1, steps) / cosine_alpha_bar(t, steps)) #,s
                for t in range(steps)
            ])

        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat)
        
        self.alpha_hat = torch.tensor(self.alpha_hat, dtype=torch.float32, device=self.device)
        self.alpha = torch.tensor(self.alpha, dtype=torch.float32, device=self.device)
        self.beta = torch.tensor(self.beta, dtype=torch.float32, device=self.device)
        self.alpha_unsqd = self.alpha.unsqueeze(1).unsqueeze(1)

    #time embedding; 128 is unrelated to size of dataset
    def time_embedding(self, pos, d_model=128):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe
    
    # test_pattern mask 
    def get_test_pattern_mask(self, observed_mask, test_pattern_mask):
        return observed_mask * test_pattern_mask
    
    # preparing inputs for the back process
    def get_side_info(self, observed_tp, cond_mask):
        B, K, L = cond_mask.shape

        time_embed = self.time_embedding(observed_tp, self.emb_time_dim)  # (B,L,emb)
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, K, -1)
        feature_embed = self.embed_layer(torch.arange(self.target_dim).to(self.device)) # (K,emb)
        feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)

        side_info = torch.cat([time_embed, feature_embed], dim=-1)  # (B,L,K,*)
        side_info = side_info.permute(0, 3, 2, 1)  # (B,*,K,L)

        if self.is_unconditional == False:
            side_mask = cond_mask.unsqueeze(1)  # (B,1,K,L)
            side_info = torch.cat([side_info, side_mask], dim=1)

        return side_info
       
    def calc_loss_valid(
        self, observed_data, cond_mask, observed_mask, side_info, is_train #focus_weights
    ):
        loss_sum = 0
        for t in range(self.num_steps):  # calculate loss for all t
            loss = self.calc_loss(
                observed_data, cond_mask, observed_mask, side_info, is_train, set_t=t #focus_weights
            )
            loss_sum += loss.detach()
        return loss_sum / self.num_steps
    
    # calculating loss: MSE of predicted noise and true noise added at time step t
    def calc_loss(
        self, observed_data, cond_mask, observed_mask, side_info, is_train, set_t=-1 #focus_weights
    ):
        B, K, L = observed_data.shape
        observed_data = _add_log_noise(observed_data, cond_mask, level= self.data_config['quality']['noise_level'])
        cond_mask = _set_missing(cond_mask, level = self.data_config['quality']['missing_level'])
        if is_train != 1:  # for validation
            t = (torch.ones(B) * set_t).long().to(self.device)
        else:
            t = torch.randint(0, self.num_steps, [B]).to(self.device)
        current_alpha = self.alpha_unsqd[t]  # (B,1,1)
        noise = torch.randn_like(observed_data)
        noisy_data = (current_alpha ** 0.5) * observed_data + (1.0 - current_alpha) ** 0.5 * noise
        
        total_input = self.set_input_to_diffmodel(noisy_data, observed_data, cond_mask)

        predicted = self.diffmodel(total_input, side_info, t)  # (B,K,L)

        target_mask = observed_mask - cond_mask
        residual = (noise - predicted) * target_mask 
        
        # residual = residual * focus_weights
        reg_diff = (predicted[:,:,:-1] - predicted[:,:,1:]) * target_mask[:,:,1:]
        
        reg_loss = self.regularization * (reg_diff ** 2).sum()
        default_loss = (residual ** 2).sum()
        # focus_loss = 0.2 * (residual[..., 66:174] ** 2).sum()
        focus_loss = 0
        num_eval = target_mask.sum()        
        
        loss = (default_loss + focus_loss + reg_loss) / (num_eval if num_eval > 0 else 1)
        return loss

    # seperates observed data into conditional and imputation targets
    def set_input_to_diffmodel(self, noisy_data, observed_data, cond_mask):
        if self.is_unconditional == True:
            total_input = noisy_data.unsqueeze(1)  # (B,1,K,L)
        else:
            cond_obs = (cond_mask * observed_data).unsqueeze(1)
            noisy_target = ((1 - cond_mask) * noisy_data).unsqueeze(1)
            total_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)

        return total_input
    
    
    # converts noise into the target data
    def impute(self, observed_data, observed_mask, cond_mask, side_info, n_samples):
        B, K, L = observed_data.shape

        imputed_samples = torch.zeros(B, n_samples, K, L).to(self.device)
        timesteps = torch.arange(self.num_steps, device=self.device)

        coeff1 = 1 / self.alpha_hat ** 0.5
        coeff2 = (1 - self.alpha_hat) / (1 - self.alpha) ** 0.5
        if self.num_steps > 1:
            sigmas = (((1.0 - self.alpha[:-1]) / (1.0 - self.alpha[1:]) * self.beta[1:]) ** 0.5).to(self.device)

        for i in range(n_samples):
            # generate noisy observation for unconditional model
            if self.is_unconditional: 
                noisy_obs = observed_data
                noisy_cond_history = []
                for t in range(self.num_steps):
                    noise = torch.randn_like(noisy_obs)
                    noisy_obs = (self.alpha_hat[t] ** 0.5) * noisy_obs + self.beta[t] ** 0.5 * noise
                    noisy_cond_history.append(noisy_obs * cond_mask)

            current_sample = torch.randn_like(observed_data)

            for t in range(self.num_steps - 1, -1, -1):
                if self.is_unconditional:
                    diff_input = cond_mask * noisy_cond_history[t] + (1.0 - cond_mask) * current_sample
                    diff_input = diff_input.unsqueeze(1)  # (B,1,K,L)
                else:
                    cond_obs = (cond_mask * observed_data).unsqueeze(1)
                    noisy_target = ((1 - cond_mask) * current_sample).unsqueeze(1)
                    diff_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)

                predicted = self.diffmodel(diff_input, side_info, timesteps[t])

                current_sample = coeff1[t] * (current_sample - coeff2[t] * predicted)

                if t > 0:
                    noise = torch.randn_like(current_sample)
                    current_sample += sigmas[t - 1] * noise

            imputed_samples[:, i] = current_sample.detach()

        return imputed_samples

    def forward(self, batch, is_train=1):
        (
            observed_data,
            observed_mask,
            cond_mask,
            observed_tp,
            gt_mask,
            #focus_weights
        ) = self.process_data(batch)
        
        
        if is_train == 0:
            cond_mask = gt_mask
        
        side_info = self.get_side_info(observed_tp, cond_mask)

        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid

        return loss_func(observed_data, cond_mask, observed_mask, side_info, is_train) #observed_data_noisy focus_weights

    def evaluate(self, batch, n_samples):
        (
            observed_data,
            observed_mask,
            _,
            observed_tp,
            gt_mask,
        ) = self.process_data(batch)

        with torch.no_grad():
            cond_mask = gt_mask
            target_mask = observed_mask - cond_mask

            side_info = self.get_side_info(observed_tp, cond_mask)

            samples = self.impute(observed_data, observed_mask, cond_mask, side_info, n_samples) #added observed mask

        return samples, observed_data, target_mask, observed_mask, observed_tp

    
class CSDI_spectra(CSDI_base_spectra):
    def __init__(self, config, device, data_config, target_dim=1):
        super().__init__(target_dim, config, data_config ,device)

    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        cond_mask = batch["cond_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()
        #focus_weights = batch["focus_weights"].to(self.device).float()
        
        return (
            observed_data,
            observed_mask,
            cond_mask,
            observed_tp,
            gt_mask,
            #focus_weights
        )
