# %%
import abc
import json
import os
from pathlib import Path

import joblib
import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torchsort


try:
    from lighting.pytorch import seed_everything
except ImportError:
    print("lighting.pytorch is not installed, using custom seed_everything")
    from utils.seed import seed_everything

from numpy.typing import NDArray
from pydantic import BaseModel, model_validator
from schedulefree import RAdamScheduleFree
from scipy.stats import rankdata
from sklearn.preprocessing import RobustScaler
from timm.utils import ModelEmaV3
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from kaggle_evaluation import mitsui_inference_server
from models.itransformer.model.itransformer import ITransformer
from settings import DirectorySettings
from plot import plot_training_history


# %%
class Config(BaseModel):
    exp_name: str = "train"

    sliding_window_view_size: int = 100
    feature_encoders: list[dict] = [
        {"name": "RawEncoder", "params": {"log_transform": False}}
    ]

    use_fp16: bool = False
    max_grad_norm: float = 1.0
    batch_size: int = 32
    lr: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.999)
    num_epochs: int = 500
    use_ema: bool = True
    ema_use_warmup: bool = True
    ema_decay: float = 0.9999

    early_stopping_patience: int = 36
    early_stopping_min_delta: float = 0.0
    early_stopping_min_epochs: int= 0

    num_recent_steps: int = 0
    d_model: int = sliding_window_view_size
    no_embd: bool = False
    use_norm: bool = True

    regularization_strength: float = 1.0
    corr_std_weight: float = 0.5
    corr_mean_weight: float = 1
    point_loss_wight: float = 0

    online_training_frequency: int = 60
    online_training_batch_size: int = 30
    online_training_epochs: int = 2
    online_training_lr: float = 1e-5
    online_training_stack_additional: int = 0  # extra history to keep beyond each flow batch

    feature_prefix: str = 'f_'
    debug: bool = False

    seeds: list[int] = list(range(10))

    only_test: bool = True  # skip training
    full_train: bool = False  # train on all data, no validation

    # --- fixed parameters ---
    # test_start_date_id=1827
    # test_windows_size=134 (is_scored)
    test_start_date_id: int = 1827
    test_window_size: int = 90

    holdout_date_range: tuple[int, int] = (
        test_start_date_id - 1,
        test_start_date_id + test_window_size - 1,
    )  # (]: left exclusive, right inclusive

    @model_validator(mode="after")
    def after_init(self) -> "Config":
        if self.debug:
            self.num_epochs = 4
            self.batch_size = 2
            self.seeds = [0, 1]

        if self.only_test:
            self.full_train = False

        return self


class BaseEncoder:
    _fitted: bool
    feature_prefix: str

    @abc.abstractmethod
    def fit(self, df: pl.DataFrame) -> None:
        pass

    @abc.abstractmethod
    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        pass

    def fit_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        self.fit(df)
        return self.transform(df)


class RawEncoder(BaseEncoder):
    def __init__(
        self,
        feature_prefix: str = "f_",
        exclude_features: list[str] = ["date_id"],  # noqa
        log_transform: bool = False,
        suffix: str = "",
    ):
        self.feature_prefix = feature_prefix
        self.exclude_features = exclude_features
        self.log_transform = log_transform
        self._fitted = False
        self.suffix = suffix

    def fit(self, df: pl.DataFrame) -> None:
        self._fitted = True

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        expr = (
            pl.exclude(self.exclude_features)
            if not self.log_transform
            else pl.exclude(self.exclude_features).log().name.suffix("__log")
        )
        feature_df = df.select(expr.name.prefix(self.feature_prefix))
        feature_cols = [x for x in feature_df.columns if x.startswith(self.feature_prefix)]
        feature_df = feature_df.select(pl.col(feature_cols).name.suffix(self.suffix))
        return feature_df


class MAEncoder(BaseEncoder):
    """Moving average encoder"""

    def __init__(
            self,
            feature_prefix: str = "f_",
            windows: list[int] = [3, 5, 50, 200],  # noqa
            exclude_features: list[str] = ['date_id'],  # noqa
    ):
        self.feature_prefix = feature_prefix
        self.windows = windows
        self.exclude_features = exclude_features
        self._fitted = False

    def fit(self, df: pl.DataFrame) -> None:
        self._fitted = True

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        feature_df = df.select(self.exclude_features)
        for ws in self.windows:
            ma_df = df.select(
                pl.exclude(self.exclude_features)
                .rolling_mean(window_size=ws)
                .interpolate(method="linear")
                .fill_null(strategy="forward")
                .fill_null(strategy="backward")
                .name.suffix(f"__ma={ws}")
                .name.prefix(self.feature_prefix),
                pl.col(self.exclude_features),
            )
            feature_df = feature_df.join(ma_df, on=self.exclude_features, how="left")
        return df


class DiffEncoder(BaseEncoder):
    def __init__(
        self,
        feature_prefix: str = "f_",
        diffs: list[int] = [1],  # noqa
        exclude_features: list[str] = ["date_id"],  # noqa
        log_transform: bool = False,
    ):
        self.feature_prefix = feature_prefix
        self._fitted = False
        self.exclude_features = exclude_features
        self.log_transform = log_transform
        self.diffs = diffs

    def fit(self, df: pl.DataFrame) -> None:
        self._fitted = True

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        feature_df = pl.DataFrame()
        src_df = df.clone()
        suffix = "__diff="
        if self.log_transform:
            src_df = src_df.select(pl.col(self.exclude_features), pl.exclude(self.exclude_features).log1p())
            suffix = "__log_diff="

        for n in self.diffs:
            _df = src_df.select(
                pl.exclude(self.exclude_features)
                .diff(n=n)
                .name.suffix(f"{suffix}={n}")
                .name.prefix(self.feature_prefix)
            )
            feature_df = pl.concat([feature_df, _df], how="horizontal")
        return feature_df


class ShiftEncoder(BaseEncoder):
    def __init__(
        self,
        feature_prefix: str = "f_",
        shifts: list[int] = [1],  # noqa
        exclude_features: list[str] = ["date_id"],  # noqa
    ):
        self.feature_prefix = feature_prefix
        self._fitted = False
        self.exclude_features = exclude_features
        self.shifts = shifts

    def fit(self, df: pl.DataFrame) -> None:
        self._fitted = True

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        feature_df = df.select(self.exclude_features)
        for n in self.shifts:
            ma_df = df.select(
                pl.exclude(self.exclude_features)
                .shift(n=n)
                .name.suffix(f"__shift=={n}")
                .name.prefix(self.feature_prefix),
                pl.col(self.exclude_features),
            )
            feature_df = feature_df.join(ma_df, on=self.exclude_features, how="left")
        return feature_df

class TargetPairEncoder(BaseEncoder):
    def __init__(
            self,
            target_pairs_df: pl.DataFrame | None = None,
            feature_prefix: str = "f_",
            use_all_target_values: bool = False,
    ):
        self.target_pairs = (
            target_pairs_df.with_columns(pl.col("pair").str.split(" ")).drop("lag").to_dicts()
            if target_pairs_df is not None
            else []
        )
        self.feature_prefix = feature_prefix
        self._fitted = False
        self.use_all_target_values = use_all_target_values

    def fit(self, df: pl.DataFrame) -> None:
        self._fitted = True

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        if not self.target_pairs:
            return pl.DataFrame()

        exprs = []
        for d in self.target_pairs:
            target_name = d["target"]
            feature_source = d["pair"]
            if len(feature_source) == 3:
                expr = pl.col(feature_source[0]) - pl.col(feature_source[2])
                expr = expr.alias(f"{self.feature_prefix}{target_name}")
            else:
                if self.use_all_target_values:
                    expr = pl.col(feature_source[0])
                else:
                    continue

            exprs.append(expr)
        feature_df = df.select(*exprs)
        return feature_df


class Preprocessor:
    def __init__(self, config: Config, **kwargs):
        self.config = config

        # only transform encoders
        self.encoders = []
        for encoder_setting in self.config.feature_encoders:
            encoder_name = encoder_setting["name"]
            encoder_params = encoder_setting.get("params", {})
            if encoder_name == "RawEncoder":
                encoder = RawEncoder(
                    feature_prefix=self.config.feature_prefix,
                    exclude_features=["date_id"],
                    **encoder_params,
                )

            elif encoder_name == "MAEncoder":
                encoder = MAEncoder(
                    feature_prefix=self.config.feature_prefix,
                    exclude_features=["date_id"],
                    **encoder_params,
                )
            elif encoder_name == "TargetPairEncoder":
                encoder = TargetPairEncoder(
                    feature_prefix=self.config.feature_prefix,
                    target_pairs_df=kwargs.get("target_pairs_df"),
                    **encoder_params,
                )

            elif encoder_name == "DiffEncoder":
                encoder = DiffEncoder(
                    feature_prefix=self.config.feature_prefix,
                    exclude_features=["date_id"],
                    **encoder_params,
                )

            elif encoder_name == "ShiftEncoder":
                encoder = ShiftEncoder(
                    feature_prefix=self.config.feature_prefix,
                    exclude_features=["date_id"],
                    **encoder_params,
                )
            else:
                raise ValueError(f"Unknown encoder: {encoder_name}")
            self.encoders.append(encoder)

        self.scaler = RobustScaler()

    def fit(self, raw_df: pl.DataFrame) -> pl.DataFrame:
        output_df = raw_df.select(pl.col("date_id"))
        for encoder in self.encoders:
            feature_df = encoder.fit_transform(raw_df)
            output_df = pl.concat([output_df, feature_df], how="horizontal")

        feature_cols = [x for x in output_df.columns if x.startswith(self.config.feature_prefix)]
        self.scaler.fit(
            output_df.filter(pl.col("date_id") < self.config.test_start_date_id).select(feature_cols).to_numpy()
        )

    def transform(self, raw_df: pl.DataFrame) -> pl.DataFrame:
        base_df = raw_df.select(["date_id"])
        output_df = pl.DataFrame()
        for encoder in self.encoders:
            feature_df = encoder.transform(raw_df)
            output_df = pl.concat([output_df, feature_df], how="horizontal")

        feature_cols = [x for x in output_df.columns if x.startswith(self.config.feature_prefix)]
        output_df = pl.DataFrame(self.scaler.transform(output_df.select(feature_cols).to_numpy()), schema=feature_cols)
        output_df - pl.concat([base_df, output_df], how="horizontal")
        return output_df

    def fit_transform(self, raw_df: pl.DataFrame) -> pl.DataFrame:
        self.fit(raw_df)
        return self.transform(raw_df)

    def save(self, dirpath: Path) -> None:
        dirpath.mkdir(parents=True, exist_ok=True)

        # save scaler
        joblib.dump(self.scaler, dirpath / "scaler.pkl")

    def load(self, dirpath: Path) -> None:
        # load scaler
        self.scaler = joblib.load(dirpath / "scaler.pkl")


def create_sliding_window_x(
    X: NDArray[np.floating],  # (N, F)
    seq_length: int,
    y: NDArray[np.floating] | None = None,  # (N, L)
) -> tuple[NDArray[np.floating], NDArray[np.floating]] | NDArray[np.floating]:
    """X: sliding window, y: scaler target"""
    X_seq = np.lib.stride_tricks.sliding_window_view(X, seq_length, axis=0)
    # (n_samples, F, seq_length) -> (n_samples, seq_length, F)
    X_seq = X_seq.transpose(0, 2, 1)

    if y is None:
        return X_seq  # ((N-seq_length), seq_length, F)

    y_seq = y[seq_length - 1 :]
    return X_seq, y_seq  # ((N-seq_length), seq_length, F), ((N-seq_length), L)


class TrainDataset(Dataset):
    def __init__(self, X_seq: NDArray[np.floating], y: NDArray[np.floating]):
        self.X_seq = X_seq  # (n_samples, seq_length, num_input_channels)
        self.y = y  # (n_samples, num_target_channels)

    def __len__(self):
        return len(self.X_seq)

    def __getitem__(self, idx):
        # observed_mask
        past_values = torch.tensor(self.X_seq[idx], dtype=torch.float32)  # (sequence_length, num_input_channels)
        past_observed_mask = ~torch.isnan(past_values)  # (sequence_length, num_input_channels)
        past_values = torch.nan_to_num(past_values, nan=0)  # (sequence_length, num_input_channels)

        target_values = torch.tensor(self.y[idx], dtype=torch.float32)  # (num_target_channels,)
        target_mask = ~torch.isnan(target_values)  # (num_target_channels,)
        target_values = torch.nan_to_num(target_values, nan=0)  # (num_target_channels,)
        return {
            "past_values": past_values,  # ( sequence_length, num_input_channels)
            "past_observed_mask": past_observed_mask,  # ( sequence_length, num_input_channels)
            "target_values": target_values,  # (num_target_channels,)
            "target_mask": target_mask,  # (num_target_channels,)
        }


class TestDataset(Dataset):
    def __init__(self, X_seq: NDArray[np.floating]):
        self.X_seq = X_seq  # (n_samples, seq_length, num_input_channels)

    def __len__(self):
        return len(self.X_seq)

    def __getitem__(self, idx):
        # observed_mask
        past_values = torch.tensor(self.X_seq[idx], dtype=torch.float32)  # (sequence_length, num_input_channels)
        past_observed_mask = ~torch.isnan(past_values)  # (sequence_length, num_input_channels)
        past_values = torch.nan_to_num(past_values, nan=0)  # (sequence_length, num_input_channels)

        return {
            "past_values": past_values,  # ( sequence_length, num_input_channels)
            "past_observed_mask": past_observed_mask,  # ( sequence_length, num_input_channels)
        }


class CustomModel(nn.Module):
    def __init__(
            self,
            config: Config,
            num_input_channels: int = 557,
            num_output_channels: int = 424,
    ):
        super().__init__()

        self.config = config

        self.model = ITransformer(
            seq_len=self.config.sliding_window_view_size,
            d_model=getattr(self.config, "d_model", 128),
            n_heads=getattr(self.config, "num_attention_heads", 4),
            e_layers=getattr(self.config, "num_hidden_layers", 3),
            d_ff=getattr(self.config, "ffn_dim", 512),
            dropout=getattr(self.config, "dropout", 0.1),
            activation="relu",
            no_embd=getattr(self.config, "no_embd", False),
            use_norm=getattr(self.config, "use_norm", True),
        )

        self.flatten = nn.Flatten(start_dim=1)
        self.num_recent_steps = getattr(self.config, "num_recent_steps", 3)

        # ITransformer output [B, N, E]
        # flattened to num_input_channels * d_model
        self.transformer_dim = num_input_channels * self.model.d_model

        if self.num_recent_steps > 0:
            recent_input_dim = num_input_channels * self.num_recent_steps
            recent_feature_dim = getattr(self.config, "recent_feature_dim", 512)

            self.recent_feature_extractor = nn.Sequential(
                nn.Linear(recent_input_dim, recent_feature_dim),
                nn.ReLU(),
                nn.Dropout(getattr(self.config, "dropout", 0.0)),
                nn.Linear(recent_feature_dim, recent_feature_dim // 2),
                nn.ReLU(),
                nn.Dropout(getattr(self.config, "dropout", 0.0)),
                nn.Linear(recent_feature_dim // 2, recent_feature_dim),
            )

            self.head_dim = self.transformer_dim + recent_feature_dim
        else:
            self.recent_feature_extractor = None
            self.head_dim = self.transformer_dim

        self.regression_head = nn.Linear(self.head_dim, num_output_channels)

    def forward(self, batch: dict) -> torch.Tensor:
        hidden_states = self.model(batch["past_values"], x_mark_enc=None)  # [B, N, E]

        transformer_features = self.flatten(hidden_states)  # (B, num_channels * d_model)

        if self.num_recent_steps > 0:
            recent_features = batch["past_values"][:, -self.num_recent_steps:, :]
            recent_features = self.flatten(recent_features)  # (B, num_recent_steps * num_channels)
            recent_features = self.recent_feature_extractor(recent_features)
            combined_features = torch.cat([transformer_features, recent_features], dim=1)
        else:
            combined_features = transformer_features

        x = self.regression_head(combined_features)
        return x


def load_best_model(model: nn.Module, model_path: Path, device: torch.device) -> nn.Module:
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    return model


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def _to_device(batch: dict, device: torch.device) -> dict:
    item_candidates = ["past_values", "past_observed_mask", "target_values"]
    for k in item_candidates:
        if k in batch:
            batch[k] = batch[k].to(device)
    return batch


class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(self.mse(outputs, targets))


class CustomLoss(nn.Module):
    """
    Vectorized version of RankCorrelationSharpeLoss for better performance.
    Process multiple samples in parallel where possible.
    """

    def __init__(
            self,
            regularization_strength: float = 1.0,
            eps: float = 1e-8,
            point_loss_weight: float = 0.0,
            corr_std_weight: float = 0.0,
            corr_mean_weight: float = 1.0,
    ):
        super().__init__()
        self.regularization_strength = regularization_strength
        self.eps = eps
        self.point_loss_weight = point_loss_weight
        self.point_loss = RMSELoss()
        self.corr_std_weight = corr_std_weight
        self.corr_mean_weight = corr_mean_weight

    def forward(
            self,
            outputs: torch.Tensor,
            target_values: torch.Tensor,
            target_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Compute loss as negative Sharpe ratio of rank correlations.
        Vectorized implementation for better performance.

        Args:
            outputs: Model predictions [B, L]
            target_values: Target values [B, L]
            target_mask: Valid mask [B, L]

        Returns:
            Loss scalar (negative Sharpe ratio)
        """
        if target_mask is None:
            target_mask = torch.ones_like(outputs, dtype=torch.bool)

        device = outputs.device

        # Replace invalid values with 0 (will be ignored in correlation)
        masked_outputs = outputs.clone()
        masked_targets = target_values.clone()
        masked_outputs[~target_mask] = 0
        masked_targets[~target_mask] = 0

        # Count valid points per sample
        valid_counts = target_mask.sum(dim=1)
        valid_samples = valid_counts >= 2

        if not valid_samples.any():
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Get valid batch indices
        valid_batch_outputs = masked_outputs[valid_samples].to(torch.float32)
        valid_batch_targets = masked_targets[valid_samples].to(torch.float32)
        valid_batch_mask = target_mask[valid_samples]

        # Compute ranks for all valid samples at once
        pred_ranks = torchsort.soft_rank(valid_batch_outputs, regularization_strength=self.regularization_strength)
        target_ranks = torchsort.soft_rank(valid_batch_targets, regularization_strength=self.regularization_strength)

        # Zero out ranks for invalid positions
        pred_ranks[~valid_batch_mask] = 0
        target_ranks[~valid_batch_mask] = 0

        # Compute correlations for each sample
        rank_correlations = []
        for i in range(valid_batch_outputs.shape[0]):
            mask = valid_batch_mask[i]
            if mask.sum() < 2:
                continue

            pred = pred_ranks[i][mask]
            target = target_ranks[i][mask]

            # Check variance
            if torch.std(valid_batch_outputs[i][mask], unbiased=False) < self.eps:
                continue
            if torch.std(valid_batch_targets[i][mask], unbiased=False) < self.eps:
                continue

            # Normalize and compute correlation
            pred = pred - pred.mean()
            pred = pred / (pred.norm() + self.eps)
            target = target - target.mean()
            target = target / (target.norm() + self.eps)

            corr = (pred * target).sum()
            rank_correlations.append(corr)

        if len(rank_correlations) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Stack correlations
        rank_corr_tensor = torch.stack(rank_correlations)

        mean_corr = rank_corr_tensor.mean()
        std_corr = (
            rank_corr_tensor.std(unbiased=False)
            if len(rank_corr_tensor) > 1
            else torch.tensor(0.0, device=device) + self.eps
        )

        valid_outputs = outputs[target_mask]
        valid_targets = target_values[target_mask]
        point_loss = self.point_loss(valid_outputs, valid_targets)

        # Changed from optimizing sharpe ratio to minimizing stddev
        loss = (
            (-mean_corr) * self.corr_mean_weight
            + (std_corr * self.corr_std_weight)
            + (point_loss * self.point_loss_weight)
        )
        return loss


def loss_fn(
        config: Config,
        outputs: torch.Tensor,
        target_values: torch.Tensor,
        target_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    loss_module = CustomLoss(
        regularization_strength=getattr(config, "regularization_strength", 1.0),
        eps=getattr(config, "eps", 1e-8),
        point_loss_weight=getattr(config, "point_loss_weight", 1.0),
        corr_std_weight=getattr(config, "corr_std_weight", 0.0),
        corr_mean_weight=getattr(config, "corr_mean_weight", 1.0),
    )

    return loss_module(outputs, target_values, target_mask)


def train_fn(
    config: Config,
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    model_ema: ModelEmaV3 | None = None,
) -> float:
    model.train()
    optimizer.train()
    losses = AverageMeter()

    scaler = GradScaler() if config.use_fp16 and torch.cuda.is_available() else None
    use_fp16 = config.use_fp16 and torch.cuda.is_available()
    max_grad_norm = getattr(config, "max_grad_norm", 1.0)

    pbar = tqdm(dataloader, desc="batch")
    for batch in pbar:
        batch = _to_device(batch, device)
        batch_size = batch["target_values"].size(0)

        with autocast(enabled=use_fp16, device_type=device.type):
            outputs = model(batch)
            loss = loss_fn(
                config=config,
                outputs=outputs,
                target_values=batch["target_values"],
                target_mask=batch["target_mask"],
            )

        losses.update(loss.item(), batch_size)
        optimizer.zero_grad()

        if use_fp16 and scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()

        # Update EMA model after optimizer step
        if model_ema is not None:
            model_ema.update(model)

        pbar.set_postfix(
            {
                "loss": f"{loss.item():.4f}",
                "avg_loss": f"{losses.avg:.4f}",
                "grad_norm": f"{grad_norm:.4f}",
            }
        )
    pbar.close()

    return losses.avg


def valid_fn(
    config: Config,
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> dict:
    model.eval()
    predictions = []
    losses = AverageMeter()

    pbar = tqdm(dataloader, desc="batch", leave=False)
    for batch in pbar:
        batch = _to_device(batch, device)
        batch_size = batch["target_values"].size(0)

        with torch.inference_mode():
            outputs = model(batch)
            loss = loss_fn(
                config=config,
                outputs=outputs,
                target_values=batch["target_values"],
                target_mask=batch["target_mask"],
            )
            losses.update(loss.item(), batch_size)
            predictions.append(outputs.cpu().numpy())

        pbar.set_postfix({"loss": f"{loss.item():.4f}", "avg_loss": f"{losses.avg:.4f}"})

    predictions = np.vstack(predictions)
    targets = [batch["target_values"].cpu().numpy() for batch in dataloader]
    target_masks = [batch["target_mask"].cpu().numpy() for batch in dataloader]

    outputs = {
        "predictions": predictions,
        "targets": np.vstack(targets),
        "target_masks": np.vstack(target_masks),
        "loss": losses.avg,
    }
    return outputs


def inference_fn(
        config: Config,
        models: list[nn.Module],
        dataloader: DataLoader,
        device: torch.device,
        verbose: bool = True,
) -> np.ndarray:
    for model in models:
        model.eval()

    all_predictions = []

    progressbar = tqdm(dataloader, desc="Inference", leave=True) if verbose else dataloader
    for batch in progressbar:
        batch = _to_device(batch, device)

        batch_predictions = [] # Predictions for each model for this batch
        with torch.inference_mode():
            for model in models:
                outputs = model(batch)
                batch_predictions.append(outputs.cpu().numpy())

        batch_avg = np.mean(batch_predictions, axis=0)  # (batch_size, num_output_channels)
        all_predictions.append(batch_avg)

    predictions = np.vstack(all_predictions)
    return predictions


def rank_normalize(predictions: np.ndarray) -> np.ndarray:
    batch_size, num_channels = predictions.shape
    normalized = np.zeros_like(predictions)

    for i in range(batch_size):
        ranks = predictions[i].argsort().argsort()
        normalized[i] = ranks / (num_channels - 1)

    return normalized


def score_fn(
    predictions: np.ndarray,
    targets: np.ndarray,
    targets_mask: np.ndarray,
) -> dict[str, float]:
    assert predictions.shape == targets.shape == targets_mask.shape, "All input arrays must have the same shape"

    N, L = predictions.shape
    daily_rank_corrs = []

    for i in range(N):
        valid_mask = targets_mask[i]

        if not np.any(valid_mask):
            raise ValueError(f"No valid values founbd for sample {i}")

        valid_predictions = predictions[i][valid_mask]
        valid_targets = targets[i][valid_mask]

        if np.std(valid_predictions, ddof=0) == 0 or np.std(valid_targets, ddof=0) == 0:
            raise ZeroDivisionError(f"Zero variance found for sample {i}, unable to compute rank correlation.")

        pred_ranks = rankdata(valid_predictions, method="average")
        target_ranks = rankdata(valid_targets, method="average")

        correlation = np.corrcoef(pred_ranks, target_ranks)[0, 1]
        daily_rank_corrs.append(correlation)

    daily_rank_corrs = np.array(daily_rank_corrs)

    std_dev = np.std(daily_rank_corrs, ddof=0)
    if std_dev == 0:
        raise ZeroDivisionError("Zero standard deviation, unable to compute Sharpe ratio.")

    mean_corr = np.mean(daily_rank_corrs)
    sharpe_ratio =mean_corr / std_dev
    return {"score": float(sharpe_ratio), "mean_corr": float(mean_corr), "std_corr": float(std_dev)}


def evaluate_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    targets_mask: np.ndarray,
    target_cols: list = None,
) -> dict:
    _, L = predictions.shape

    if target_cols is None:
        target_cols = [f"target_{i}" for i in range(L)]

    scores = score_fn(predictions, targets, targets_mask)

    return scores


def train_loop(
    config: Config,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader | None,
    best_model_path: Path,
    device: torch.device,
) -> pl.DataFrame:
    model = CustomModel(
        config=config,
        num_input_channels=train_dataloader.dataset.X_seq.shape[2],
        num_output_channels=train_dataloader.dataset.y.shape[1],
    ).to(device)

    optimizer = RAdamScheduleFree(
        model.parameters(),
        lr=config.lr,
        betas=config.betas,
    )

    model_ema = (
        ModelEmaV3(
            model,
            use_warmup=getattr(config, "ema_use_warmup", True),
            decay=getattr(config, "ema_decay", 0.999),
            warmup_power=getattr(config, "ema_warmup_power", 1.0),
            warmup_gamma=getattr(config, "ema_warmup_gamma", 1.0),
            exclude_buffers=True,
        )
        if getattr(config, "use_ema", False)
        else None
    )

    patience = max(0, getattr(config, "early_stopping_patience", 0))
    min_delta = getattr(config, "early_stopping_min_delta", 0.0)
    min_epochs = max(0, getattr(config, "early_stopping_min_epochs", 0))

    best_score = -float("inf")
    best_epoch = 0
    epochs_without_improve = 0
    stopped_epoch: int | None = None
    history = []

    print(f"\nTraining on {device} | {config.num_epochs} epochs | LR: {config.lr}")
    print("=" * 80)

    for epoch in range(config.num_epochs):
        print(f"\n[Epoch {epoch + 1}/{config.num_epochs}]")

        # Training
        train_loss = train_fn(
            config=config,
            model=model,
            model_ema=model_ema,
            dataloader=train_dataloader,
            optimizer=optimizer,
            device=device,
        )

        # Validation
        if val_dataloader is not None:
            # Choose which model to validate based on use_ema
            eval_model = model_ema.module if model_ema is not None else model

            val_output = valid_fn(
                config=config,
                model=eval_model,
                dataloader=val_dataloader,
                device=device,
            )
            val_loss = val_output["loss"]
            val_scores = score_fn(
                predictions=val_output["predictions"],
                targets=val_output["targets"],
                targets_mask=val_output["target_masks"],
            )
            val_score = val_scores["score"]

            improvement = val_score - best_score
            is_best = improvement > min_delta
            old_best = best_score

            if is_best:
                best_score = val_score
                best_epoch = epoch + 1
                epochs_without_improve = 0
                # Save the model that was actually evaluated
                if model_ema is not None:
                    torch.save(model_ema.module.state_dict(), best_model_path)
                else:
                    torch.save(model.state_dict(), best_model_path)
            else:
                epochs_without_improve += 1

            # Print results
            model_type = " (EMA)" if model_ema is not None else ""
            if is_best:
                delta_for_print = 0.0 if old_best == -float("inf") else val_score - old_best
                print(
                    f"train loss {train_loss:.4f} | val loss {val_loss:.4f} | "
                    f"val score{model_type} {val_score:.4f} | "
                    f"BEST! {delta_for_print:.4f}"
                )
            else:
                print(
                    f"train loss {train_loss:.4f} | val loss {val_loss:.4f} | val score{model_type} {val_score:.4f}"
                )

            print(f"   val scores: {val_scores}")

            history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_score": val_score,
                    "val_corr_mean": val_scores["mean_corr"],
                    "val_corr_std": val_scores["std_corr"],
                    "is_best": is_best,
                    "best_score": best_score,
                    "no_improve_count": epochs_without_improve,
                    "model_type": "ema" if model_ema is not None else "normal",
                }
            )

            # Early stopping check
            if patience > 0 and (epoch + 1) >= max(1, min_epochs) and epochs_without_improve >= patience:
                stopped_epoch = epoch + 1
                print(f"Early stopping triggered (no improvement in {patience} epochs).")
                break
        else:
            # No validation, just save the model
            print(f"train loss {train_loss:.4f}")
            if model_ema is not None:
                torch.save(model_ema.module.state_dict(), best_model_path)
            else:
                torch.save(model.state_dict(), best_model_path)

            history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": None,
                    "val_score": None,
                    "val_corr_mean": None,
                    "val_corr_std": None,
                    "is_best": False,
                    "best_score": None,
                    "no_improve_count": None,
                    "model_type": "ema" if model_ema is not None else "normal",
                }
            )

    print("=" * 80)

    if val_dataloader is not None:
        model_type_str = " (EMA)" if model_ema is not None else ""
        if best_score == -float("inf"):
            print(f"\nNo validation improvements recorded{model_type_str}.\n")
        else:
            print(f"\nBest Score{model_type_str}: {best_score:.6f} (epoch {best_epoch})\n")
        if stopped_epoch is not None:
            print(f"Early stopping at epoch {stopped_epoch} (patience {patience}).\n")
        history_df = pl.DataFrame(history)
        print(history_df.sort("val_score", descending=True).head(5))
    else:
        print("\nTraining completed (no validation)\n")
        history_df = pl.DataFrame(history)
        print(history_df)

    return history_df


class InferenceEnv:
    def __init__(
        self,
        config: Config,
        preprocessor: Preprocessor,
        models: list[nn.Module],
        device: torch.device,
        raw_train_df: pl.DataFrame,
        target_cols: list[str],
        feature_cols: list[str],
    ):
        self.preprocessor = preprocessor
        self.config = config
        self.models = models
        self.device = device

        # latest_df -> test
        self.latest_df = raw_train_df.filter(pl.col("date_id") <= 1826)
        self.dtype_map = {col: dtype for col, dtype in zip(self.latest_df.columns, self.latest_df.dtypes, strict=True)}
        self.feature_cols = feature_cols
        self.target_cols = target_cols
        self.seq_length = self.config.sliding_window_view_size

        self.online_training_frequency = getattr(self.config, "online_training_frequency", 0)
        self.online_training_batch_size = getattr(
            self.config,
            "online_training_batch_size",
            self.config.batch_size,
        )
        self.online_training_epochs = getattr(self.config, "online_training_epochs", 1)
        self.online_training_lr = getattr(self.config, "online_training_lr", self.config.lr)
        self.online_training_stack_additional = max(0, getattr(self.config, "online_training_stack_additional", 0))
        self.online_training_stack_width = (
            self.online_training_frequency + self.online_training_stack_additional
            if self.online_training_frequency > 0
            else 0
        )

        self.pred_list = []
        self.lag_dfs: list[pl.DataFrame] = []
        self.label_count = 0
        self.label_history_df = None

    def _append_label_batch(self, batch: pl.DataFrame) -> None:
        if batch.is_empty():
            return
        drop_cols = [col for col in ("date_id",) if col in batch.columns]
        self.lag_dfs.append(batch.drop(drop_cols) if drop_cols else batch)

    def _get_label_df(self) -> pl.DataFrame:
        if not self.lag_dfs:
            return pl.DataFrame()

        label_df = (
            pl.concat(self.lag_dfs, how="diagonal")
            .group_by("label_date_id")
            .mean()
            .sort("label_date_id")
            .rename({"label_date_id": "date_id"})
        )

        missing_targets = [col for col in self.target_cols if col not in label_df.columns]
        if missing_targets:
            label_df = label_df.with_columns([pl.lit(None).cast(pl.Float32).alias(col) for col in missing_targets])

        return label_df.select(["date_id", *self.target_cols])

    def _update_label_history(self, new_label_df: pl.DataFrame) -> pl.DataFrame:
        if new_label_df.is_empty():
            return self.label_history_df if isinstance(self.label_history_df, pl.DataFrame) else pl.DataFrame()

        if isinstance(self.label_history_df, pl.DataFrame):
            history_df = pl.concat(
                [self.label_history_df, new_label_df],
                how="vertical_relaxed",
            )
            history_df = history_df.unique(subset=["date_id"], keep="last")
        else:
            history_df = new_label_df

        history_df = history_df.sort("date_id")

        if self.online_training_stack_width > 0 and history_df.height > self.online_training_stack_width:
            history_df = history_df.tail(self.online_training_stack_width)

        self.label_history_df = history_df
        return history_df

    def _reset_lag(self) -> None:
        self.lag_dfs = []
        self.label_count = 0

    def _run_online_training(self) -> bool:
        new_label_df = self._get_label_df()
        if new_label_df.is_empty():
            return False

        stack_label_df = self._update_label_history(new_label_df)
        if stack_label_df.is_empty():
            return False

        feature_df = self.preprocessor.transform(self.latest_df)
        feature_np = feature_df.select(pl.col(self.feature_cols)).to_numpy().astype(np.float32)
        if feature_np.shape[0] < self.seq_length:
            return False

        X_seq_all = create_sliding_window_x(X=feature_np, seq_length=self.seq_length)
        date_ids = feature_df["date_id"].to_numpy()
        seq_date_ids = date_ids[self.seq_length - 1 :]

        label_dates = stack_label_df["date_id"].to_numpy()
        target_values_np = stack_label_df.select(self.target_cols).to_numpy().astype(np.float32)
        label_map = {int(date): values for date, values in zip(label_dates, target_values_np, strict=True)}

        mask = np.isin(seq_date_ids, label_dates)
        if not np.any(mask):
            return False

        X_seq_new = X_seq_all[mask].astype(np.float32)
        y_seq_new = np.stack([label_map[int(date)] for date in seq_date_ids[mask]], axis=0)

        if X_seq_new.size == 0:
            return False

        dataset = TrainDataset(X_seq=X_seq_new, y=y_seq_new)
        dataloader = DataLoader(
            dataset,
            batch_size=min(self.online_training_batch_size, len(dataset)),
            shuffle=True,
        )

        print(
            f"[Online training] samples={len(dataset)} epochs={self.online_training_epochs} "
            f"batch_size={self.online_training_batch_size} stack_width={stack_label_df.height}"
        )

        for model_idx, model in enumerate(self.models):
            optimizer = RAdamScheduleFree(
                model.parameters(),
                lr=self.online_training_lr,
                betas=self.config.betas,
            )
            for epoch in range(self.online_training_epochs):
                loss = train_fn(
                    config=self.config,
                    model=model,
                    dataloader=dataloader,
                    optimizer=optimizer,
                    device=self.device,
                    model_ema=None,
                )
                print(f"  model={model_idx} epoch={epoch + 1}/{self.online_training_epochs} loss={loss:.4f}")
            model.eval()
        return True

    def predict(
        self,
        test: pl.DataFrame,
        label_lags_1_batch: pl.DataFrame,
        label_lags_2_batch: pl.DataFrame,
        label_lags_3_batch: pl.DataFrame,
        label_lags_4_batch: pl.DataFrame,
    ) -> pl.DataFrame:
        assert (test["date_id"].item() - 1) == self.latest_df["date_id"].max(), "Test data_id should be continuous."

        if self.online_training_frequency > 0:
            self._append_label_batch(label_lags_1_batch)
            self._append_label_batch(label_lags_2_batch)
            self._append_label_batch(label_lags_3_batch)
            if not label_lags_4_batch.is_empty():
                self._append_label_batch(label_lags_4_batch)
                unique_dates = label_lags_4_batch["label_date_id"].n_unique()
                self.label_count += int(unique_dates)
            if self.label_count >= self.online_training_frequency:
                trained = self._run_online_training()
                if trained:
                    self._reset_lag()

        # add target test record to latest_df
        self.latest_df = (
            pl.concat(
                [
                    self.latest_df,
                    test.drop("is_scored"),
                ],
                how="diagonal_relaxed",
            )
            .sort("date_id")
            .cast(self.dtype_map)
        )

        # preprocess
        latest_feature_df = self.preprocessor.transform(self.latest_df)

        latest_X_seq = create_sliding_window_x(
            X=latest_feature_df.select(pl.col(self.feature_cols)).to_numpy(),
            seq_length=self.seq_length,
        )[[-1]]  # get the latest sequence only

        # Dataset
        latest_dataset = TestDataset(X_seq=latest_X_seq)
        latest_dataloader = DataLoader(latest_dataset, batch_size=1)

        # inference
        test_preds = inference_fn(
            config=self.config,
            models=self.models,
            dataloader=latest_dataloader,
            device=self.device,
            verbose=False,
        )

        test_pred_df = pl.DataFrame(test_preds, schema=self.target_cols)
        self.pred_list.append(test_pred_df.with_columns(pl.lit(test["date_id"].item()).alias("date_id")))

        return test_pred_df

if __name__ == "__main__":
    import rootutils

    rootutils.setup_root(".", cwd=True)

    # %%
    # Settings
    config = Config()
    settings = DirectorySettings(exp_name=config.exp_name)
    settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # save config
    with open(settings.OUTPUT_DIR / "config.json", "w") as f:
        json.dump(config.model_dump(), f, indent=4)

    # %%
    # Data Load
    raw_train_df = pl.read_csv(settings.COMP_DATASET_DIR / "train.csv")
    raw_test_df = pl.read_csv(settings.COMP_DATASET_DIR / "test.csv")
    train_labels_df = pl.read_csv(settings.COMP_DATASET_DIR / "train_labels.csv")
    target_pairs_df = pl.read_csv(settings.COMP_DATASET_DIR / "target_pairs.csv")

    preprocessor = Preprocessor(config=config, target_pairs_df=target_pairs_df)
    train_feature_df = preprocessor.fit_transform(raw_train_df)
    preprocessor.save(dirpath=settings.OUTPUT_DIR)

    # %%
    # Preprocess
    feature_cols = [x for x in train_feature_df.columns if x.startswith(config.feature_prefix)]
    target_cols = [x for x in train_labels_df.columns if x.startswith("target_")]

    if config.debug:
        feature_cols = feature_cols[:10]

    # save feature_cols and target_cols
    joblib.dump(feature_cols, settings.OUTPUT_DIR / "feature_cols.pkl")
    joblib.dump(target_cols, settings.OUTPUT_DIR / "target_cols.pkl")

    Xt_seq, yt = create_sliding_window_x(
        X=train_feature_df.select(pl.col("date_id")).to_numpy(),
        y=train_labels_df.select(pl.col("date_id")).to_numpy(),
        seq_length=config.sliding_window_view_size,
    )
    X_seq, y = create_sliding_window_x(
        X=train_feature_df.select(pl.col(feature_cols)).to_numpy(),
        y=train_labels_df.select(pl.col(target_cols)).to_numpy(),
        seq_length=config.sliding_window_view_size,
    )

    # %%
    # [Optional] Full Training
    if config.full_train:
        train_dataset = TrainDataset(X_seq=X_seq, y=y)
        train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        best_model_paths = []
        for seed in config.seeds:
            print(f"Seed: {seed}")
            out_dir = settings.OUTPUT_DIR / "full_training" / f"seed_{seed}"
            out_dir.mkdir(parents=True, exist_ok=True)
            seed_everything(seed)

            best_model_path = out_dir / "best_model.pth"
            history_df = train_loop(
                config=config,
                train_dataloader=train_dataloader,
                val_dataloader=None,
                best_model_path=best_model_path,
                device=device,
            )
            history_df.write_csv(out_dir / "history_full.csv")
            plot_training_history(
                history_df=history_df,
                filepath=out_dir / "training_history_full.png",
                figsize=(10, 6),
            )
            best_model_paths.append(best_model_path)

        print("Full training completed.")

        # Inference Server Demo
        preprocessor = Preprocessor(config=config, target_pairs_df=target_pairs_df)
        preprocessor.load(settings.OUTPUT_DIR)

        best_models = [
            load_best_model(
                model=CustomModel(
                    config=config,
                    num_input_channels=X_seq.shape[2],
                    num_output_channels=y.shape[1],
                ),
                model_path=settings.OUTPUT_DIR / "full_training" / f"seed_{seed}" / "best_model.pth",
                device=device,
            )
            for seed in config.seeds
        ]
        inference_env = InferenceEnv(
            config=config,
            preprocessor=preprocessor,
            models=best_models,
            device=device,
            raw_train_df=raw_train_df,
            target_cols=target_cols,
            feature_cols=feature_cols,
        )

        inference_server = mitsui_inference_server.MitsuiInferenceServer(inference_env.predict)
        if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
            inference_server.serve()
        else:
            inference_server.run_local_gateway((settings.COMP_DATASET_DIR,))

        test_pred_df = pl.concat(inference_env.pred_list).sort("date_id")
        print(test_pred_df)
        print("✔Inference server demo completed.")
        exit(0)

        # %%
        # Train/Validation Split
        val_mask = ((yt > config.holdout_date_range[0]) & (yt <= config.holdout_date_range[1])).reshape(-1)
        tr_mask = (yt <= config.holdout_date_range[0]).reshape(-1)  # train only on data before the holdout range
        tr_x, tr_y, tr_t = X_seq[tr_mask], y[tr_mask], yt[tr_mask].reshape(-1)
        val_x, val_y, val_t = X_seq[val_mask], y[val_mask], yt[val_mask].reshape(-1)
        print(f"tr_t: {tr_t.min()} - {tr_t.max()} | {len(tr_t)} samples")
        print(f"val_t: {val_t.min()} - {val_t.max()} | {len(val_t)} samples")

        # Dataset
        train_dataset = TrainDataset(X_seq=tr_x, y=tr_y)
        val_dataset = TrainDataset(X_seq=val_x, y=val_y)
        train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
        val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size * 2, shuffle=False, num_workers=4)

        # %%
        # Training
        if not config.only_test:
            best_model_paths = []
            for seed in config.seeds:
                print(f"Seed: {seed}")
                seed_everything(seed)
                out_dir = settings.OUTPUT_DIR / f"seed_{seed}"
                out_dir.mkdir(parents=True, exist_ok=True)

                best_model_path = out_dir / "best_model.pth"
                history_df = train_loop(
                    config=config,
                    train_dataloader=train_dataloader,
                    val_dataloader=val_dataloader,
                    best_model_path=best_model_path,
                    device=device,
                )
                history_df.write_csv(out_dir / "history.csv")
                plot_training_history(
                    history_df=history_df,
                    filepath=out_dir / "training_history.png",
                    figsize=(10, 6),
                )
                best_model_paths.append(best_model_path)

        # %%
        # Validation
        best_models = [
            load_best_model(
                model=CustomModel(
                    config=config,
                    num_input_channels=X_seq.shape[2],
                    num_output_channels=y.shape[1],
                ),
                model_path=settings.OUTPUT_DIR / f"seed_{seed}" / "best_model.pth",
                device=device,
            )
            for seed in config.seeds
        ]

        val_preds = inference_fn(
            config=config,
            models=best_models,
            dataloader=val_dataloader,
            device=device,
        )
        val_pred_df = pl.DataFrame(
            val_preds,
            schema=target_cols,
        ).with_columns(pl.Series(val_t).alias("date_id").cast(pl.Int32))

        # score
        out_dir = settings.OUTPUT_DIR / "val_predictions"
        out_dir.mkdir(parents=True, exist_ok=True)

        val_metrics = evaluate_metrics(
            predictions=val_preds,
            targets=val_y,
            targets_mask=~np.isnan(val_y),
            target_cols=target_cols,
        )
        print(f"\nFinal Validation Score: {val_metrics['score']:.6f}\n")
        with open(out_dir / "val_metrics.json", "w") as f:
            json.dump(val_metrics, f, indent=4)

        # %%
        # Inference Server Demo
        preprocessor = Preprocessor(config=config)
        preprocessor.load(settings.OUTPUT_DIR)
        inference_env = InferenceEnv(
            config=config,
            preprocessor=preprocessor,
            models=best_models,
            device=device,
            raw_train_df=raw_train_df,
            target_cols=target_cols,
            feature_cols=feature_cols,
        )

        inference_server = mitsui_inference_server.MitsuiInferenceServer(inference_env.predict)
        if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
            inference_server.serve()
        else:
            inference_server.run_local_gateway((settings.COMP_DATASET_DIR,))

        test_pred_df = (
            pl.concat(inference_env.pred_list)
            .join(raw_test_df.select(pl.col(["date_id", "is_scored"])), on="date_id")
            .sort("date_id")
        )
        print(test_pred_df)

        test_score_df = test_pred_df.filter(pl.col("is_scored") == 1)
        test_label_df = train_labels_df.join(test_score_df, on="date_id", how="inner")
        print(test_score_df)

        score = score_fn(
            predictions=test_score_df.select(pl.col(target_cols)).to_numpy(),
            targets=test_label_df.select(pl.col(target_cols)).to_numpy(),
            targets_mask=~np.isnan(test_label_df.select(pl.col(target_cols)).to_numpy()),
        )
        print(f"\nTest Score on public labels: {score['score']:.6f}\n")

    # %%
