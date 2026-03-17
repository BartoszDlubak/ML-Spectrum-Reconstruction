from tqdm import tqdm
import torch
import random

from torch.utils.data import Subset
from .utils import save_checkpoint, setup_logger
from eval.eval import run_inference, compute_metrics
from .scheduler import scheduler

def downsample_dataloader(dataloader, fraction=0.1):
    dataset = dataloader.dataset
    total_len = len(dataset)
    subset_size = int(total_len * fraction)

    indices = random.sample(range(total_len), subset_size)
    subset = Subset(dataset, indices)

    return torch.utils.data.DataLoader(
        subset,
        batch_size=dataloader.batch_size,
        shuffle=False, 
        num_workers=dataloader.num_workers,
        pin_memory=dataloader.pin_memory,
    )

def train_one_epoch(model, train_loader, optimizer, epoch_no, logger):
    model.train()
    total_loss = 0

    with tqdm(train_loader, desc=f"Train Epoch {epoch_no}") as it:
        for batch_no, batch in enumerate(it, start=1):
            optimizer.zero_grad()
            loss = model(batch, is_train = 1)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            it.set_postfix({
                "train_loss": f"{total_loss / batch_no:.4f}",
                "epoch": epoch_no
            }, refresh=False)

    avg_loss = total_loss / len(train_loader)
    logger.info(f"Epoch {epoch_no} | Train Loss: {avg_loss:.5f}")
    return avg_loss


def validate_one_epoch(model, valid_loader, epoch_no, logger):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        with tqdm(valid_loader, desc=f"Valid Epoch {epoch_no}") as it:
            for batch_no, batch in enumerate(it, start=1):
                loss = model(batch, is_train=1)
                total_loss += loss.item()
                it.set_postfix({
                    "val_loss": f"{total_loss / batch_no:.4f}",
                    "epoch": epoch_no
                }, refresh=False)

    avg_loss = total_loss / len(valid_loader)
    logger.info(f"Epoch {epoch_no} | Val Loss: {avg_loss:.5f}")
    return avg_loss


def inference_error(model , valid_loader, epoch_no, logger):
    sub_dataloader = downsample_dataloader(valid_loader, fraction=0.1)
    samples, truths, target_mask, _, _ = run_inference(model, sub_dataloader, num_samp=1)
    samples_med, _ = samples.median(dim=1)

    # Compute metrics
    metrics = compute_metrics(samples_med, truths, target_mask)
    rmse_mean = metrics['RMSE_MEAN']
    rmse_med = metrics['RMSE_MED']
    rmse_p10 = metrics['RMSE_P10']
    rmse_p90 = metrics['RMSE_P90']
        
    logger.info(f"Epoch {epoch_no} | Impute RMSE: {rmse_mean:.5f} ; RMSE_P10: {rmse_p10:.5f}; RMSE_MED: {rmse_med:.5f} ; RMSE_P90: {rmse_p90:.5f}")
    return metrics


def train_model(
    model,
    config,
    train_loader,
    valid_loader,
    save_dir,
    valid_epoch_interval = 2,
    inference_interval = 5):
    
    best_val_loss = float("inf")
    epochs = config["train"]["epochs"]
    
    scheduler_func = scheduler
    logger = setup_logger(save_dir / 'training.log')
    
    # Preparing log file
    csv_path = save_dir / "loss_log.csv"
    with open(csv_path, "w") as f:
        f.write("epoch,train_loss,val_loss,inference_val_loss\n")
        
    # Setting optimizers:
    optimizer = torch.optim.Adam(model.parameters(), lr=config["train"]["lr"], weight_decay=1e-6)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler_func)

    for epoch_no in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, epoch_no, logger)
        lr_scheduler.step()

        val_loss = None
        if valid_loader is not None and (epoch_no + 1) % valid_epoch_interval == 0:
            val_loss = validate_one_epoch(model, valid_loader, epoch_no, logger)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # Saving and logging best model
                save_checkpoint(model, optimizer, config, save_dir, name="best.pt")
                print('\n')
                logger.info(f"Epoch {epoch_no}] New best model saved with val loss {val_loss:.5f}")

        if (epoch_no + 1) % inference_interval == 0:
            inference_loss = inference_error(model, valid_loader, epoch_no, logger)
        else:
            inference_loss = 'NA'
            
        # Saving current checkpoint
        save_checkpoint(model, optimizer, config, save_dir, name="last.pt")

        # Logging to CSV
        with open(csv_path, "a") as f:
            f.write(f"{epoch_no},{train_loss:.5f},{val_loss if val_loss else 'NA'},{inference_loss}\n")
