import os

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler


def _default_num_workers():
    return 0 if os.name == "nt" else 4


class EarlyStopping:
    def __init__(self, patience=10, delta=0.0, verbose=False):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} / {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


def train_model(
    device,
    writer,
    train_dataset,
    validate_dataset,
    model,
    epoch_num,
    batch_size,
    num_batch,
    lr,
    accumulation_steps=1,
    adaptlr=False,
    early_stopping=True,
    patience=25,
    sample_weights=None,
    warmup_epochs=10,
):
    """scMIGRA-v1 训练流程。"""

    small_dataset = len(train_dataset) < batch_size
    drop_last = not small_dataset
    num_workers = _default_num_workers()

    if sample_weights is not None:
        sample_weights = torch.tensor(sample_weights, dtype=torch.double)
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        train_data = DataLoader(train_dataset, batch_size, shuffle=False, sampler=sampler, drop_last=drop_last, num_workers=num_workers, pin_memory=True)
    else:
        train_data = DataLoader(train_dataset, batch_size, shuffle=True, drop_last=drop_last, num_workers=num_workers, pin_memory=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = None
    if adaptlr:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=epoch_num * max(1, num_batch))

    stopper = EarlyStopping(patience=patience, verbose=True) if early_stopping else None

    lambda_transport_max = float(model.lambda_transport)
    lambda_cycle_max = float(model.lambda_cycle)
    lambda_batch_ind_max = float(model.lambda_batch_ind)

    global_step = 0
    for epoch in range(epoch_num):
        model.train()
        stats = {
            "loss": 0.0,
            "recon_loss": 0.0,
            "kl_mod": 0.0,
            "kl_tech": 0.0,
            "program_loss": 0.0,
            "transport_loss": 0.0,
            "cycle_loss": 0.0,
            "batch_ind_loss": 0.0,
            "anchor_coverage": 0.0,
            "anchor_count": 0.0,
        }

        # 中文说明:
        # 第一版训练先让重构与 program state 稳住，再逐步拉起 transport / cycle / batch independence。
        if warmup_epochs > 0:
            t = min(1.0, max(0.0, epoch / max(1, warmup_epochs)))
            scale = t * t
        else:
            scale = 1.0
        model.lambda_transport = lambda_transport_max * scale
        model.lambda_cycle = lambda_cycle_max * scale
        model.lambda_batch_ind = lambda_batch_ind_max * scale

        optimizer.zero_grad(set_to_none=True)
        step_count = 0
        for step, (X, b, m, i, w) in enumerate(train_data):
            X, b, m, i, w = X.to(device), b.to(device), m.to(device), i.to(device), w.to(device)
            _, _, loss, loss_dict, _ = model(X, b, m, i, w, stage="train")
            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            if (step + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            step_count += 1

            if scheduler is not None:
                scheduler.step()

            stats["loss"] += loss.item()
            for key in stats:
                if key == "loss":
                    continue
                stats[key] += float(loss_dict[key])

            global_step += 1
            if writer is not None:
                writer.add_scalar("Loss/train", loss.item(), global_step)
                for key, value in loss_dict.items():
                    writer.add_scalar(f"{key}/train", value, global_step)
                writer.add_scalar("lambda_transport/train", model.lambda_transport, global_step)
                writer.add_scalar("lambda_cycle/train", model.lambda_cycle, global_step)
                writer.add_scalar("lambda_batch_ind/train", model.lambda_batch_ind, global_step)
                if scheduler is not None:
                    writer.add_scalar("lr/train", scheduler.get_last_lr()[0], global_step)

        if step_count > 0 and (step_count % accumulation_steps) != 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        denom = max(1, num_batch)
        print(
            "epoch {}: loss={:.4f}, recon={:.4f}, kl_mod={:.4f}, kl_tech={:.4f}, program={:.4f}, transport={:.4f}, cycle={:.4f}, batch_ind={:.4f}, anchor_cov={:.3f}".format(
                epoch + 1,
                stats["loss"] / denom,
                stats["recon_loss"] / denom,
                stats["kl_mod"] / denom,
                stats["kl_tech"] / denom,
                stats["program_loss"] / denom,
                stats["transport_loss"] / denom,
                stats["cycle_loss"] / denom,
                stats["batch_ind_loss"] / denom,
                stats["anchor_coverage"] / denom,
            )
        )

        if writer is not None:
            for key, value in stats.items():
                writer.add_scalar(f"{key}_epoch/train", value / denom, epoch + 1)

        if stopper is not None:
            validate_loss = validate_model(device, validate_dataset, model, batch_size)
            stopper(validate_loss)
            if stopper.early_stop:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    model.lambda_transport = lambda_transport_max
    model.lambda_cycle = lambda_cycle_max
    model.lambda_batch_ind = lambda_batch_ind_max


def validate_model(device, validate_dataset, model, batch_size):
    model.eval()
    validate_data = DataLoader(validate_dataset, batch_size, shuffle=False, drop_last=False, num_workers=_default_num_workers(), pin_memory=True)
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for X, b, m, i, w in validate_data:
            X, b, m, i, w = X.to(device), b.to(device), m.to(device), i.to(device), w.to(device)
            _, _, loss, _, _ = model(X, b, m, i, w, stage="train")
            total_loss += loss.item()
            n_batches += 1
    return total_loss / max(1, n_batches)


def inference_model(device, inference_dataset, model, batch_size):
    """推理阶段额外导出 program assignments 与 tech residual。"""

    model.eval()
    inference_data = DataLoader(inference_dataset, batch_size, shuffle=False, drop_last=False, num_workers=_default_num_workers(), pin_memory=True)

    z_bio_list, z_mod_list, z_tech_list, q_list = [], [], [], []
    total_loss = 0.0
    with torch.no_grad():
        for X, b, m, i, w in inference_data:
            X, b, m, i, w = X.to(device), b.to(device), m.to(device), i.to(device), w.to(device)
            z_bio, z_mod, loss, _, aux = model(X, b, m, i, w, stage="train")
            z_bio_list.append(z_bio.detach().cpu().numpy())
            z_mod_list.append(z_mod.detach().cpu().numpy())
            z_tech_list.append(aux["z_tech"].detach().cpu().numpy())
            q_list.append(aux["assignments"].detach().cpu().numpy())
            total_loss += loss.item()

    z_bio = np.concatenate(z_bio_list, axis=0)
    z_mod = np.concatenate(z_mod_list, axis=0)
    z_tech = np.concatenate(z_tech_list, axis=0)
    assignments = np.concatenate(q_list, axis=0)
    num_batch = np.ceil(len(inference_dataset) / batch_size)
    print(f"inference: loss={total_loss / max(1, num_batch):.4f}")
    return z_bio, z_mod, z_tech, assignments
