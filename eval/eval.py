import torch
from tqdm import tqdm
from pathlib import Path

# Make predictions from test data
def run_inference(model, loader, num_samp):
    """Runs model.evaluate and collects all batches."""
    model.eval()

    all_samples, all_observed_data = [], []
    all_target_mask, all_observed_mask = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating model"):
            samples, observed_data, target_mask, observed_mask, timepoints = model.evaluate(batch, num_samp)

            all_samples.append(samples)
            all_observed_data.append(observed_data)
            all_target_mask.append(target_mask)
            all_observed_mask.append(observed_mask)

    # Concatenate
    all_samples = torch.cat(all_samples, dim=0)
    all_observed_data = torch.cat(all_observed_data, dim=0)
    all_target_mask = torch.cat(all_target_mask, dim=0)
    all_observed_mask = torch.cat(all_observed_mask, dim=0)

    return all_samples, all_observed_data, all_target_mask, all_observed_mask, timepoints

# Compute metrics 
def compute_metrics(preds, truths, mask):
    """Computes MSE and MAE over masked regions."""
    diff = (preds-truths) * mask # B x K x L 
    rmse_vect = (diff**2).sum(dim=-1) / mask.sum(dim=-1) # B x K
    rmse_vect = rmse_vect.squeeze(-1) # B
    
    
    rmse_mean = rmse_vect.mean().item()
    rmse_med = rmse_vect.quantile(0.5).item()
    rmse_p10 = rmse_vect.quantile(0.1).item()
    rmse_p90 = rmse_vect.quantile(0.9).item()
    mae = torch.abs((preds - truths) * mask).mean().item()
    
    return {"MAE": mae,'RMSE_MEAN': rmse_mean ,'RMSE_MED': rmse_med, 'RMSE_P10': rmse_p10, 'RMSE_P90': rmse_p90}

# Save predictions and metrics
def save_eval_outputs(save_path, preds, truths, mask, obs_mask, timepoints, metrics):
    results = {
        "all_samples": preds,
        "all_observed_data": truths,
        "all_target_mask": mask,
        "all_observed_mask": obs_mask,
        "all_timepoints": timepoints,
        "metrics": metrics,
    }
    
    torch.save(results, save_path / "eval.pt")

# Entire evaluation pipeline
def eval_model(model, loader, normaliser, num_samp, save_path: Path, save: bool = True):
    """
    Evaluates a trained model on a dataset using imputation.
    """
    samples, truths, target_mask, obs_mask, timepoints = run_inference(model, loader, num_samp)

    # Unnormalize
    samples = normaliser.unnormalise(samples)
    truths = normaliser.unnormalise(truths)

    # Take median of sampled reconstructions
    samples_med = samples.median(dim=1).values

    # Compute metrics
    metrics = compute_metrics(samples_med, truths, target_mask)

    # Save everything
    if save:
        save_eval_outputs(save_path, samples, truths, target_mask, obs_mask, timepoints, metrics)

    return samples, truths, target_mask, obs_mask, timepoints
