from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from tqdm import tqdm

def plot_timeseries_predictions(  # noqa: C901
    val_pred_df: pl.DataFrame,
    train_labels_df: pl.DataFrame,
    target_cols_to_plot: list[str] | None = None,
    target_pairs_df: pl.DataFrame | None = None,
    n_context: int = 50,
    figsize_per_plot: tuple[float, float] = (4, 3),
    max_cols: int = 8,
    alpha_pred: float = 0.8,
    alpha_gt: float = 0.8,
    filepath: Path | None = None,
) -> None:
    val_date_ids = val_pred_df["date_id"].to_list()
    val_start = min(val_date_ids)
    val_end = max(val_date_ids)

    all_date_ids = train_labels_df["date_id"].to_list()

    # Determine the plot range(n_context points before and after the validation period)
    plot_start_idx = max(0, all_date_ids.index(val_start) - n_context)
    plot_end_idx = min(len(all_date_ids), all_date_ids.index(val_end) + n_context + 1)
    plot_date_ids = all_date_ids[plot_start_idx:plot_end_idx]

    plot_labels_df = train_labels_df.filter(pl.col("date_id").is_in(plot_date_ids))

    tr_target_df = plot_labels_df.filter(pl.col("date_id") < val_start)
    val_target_df = plot_labels_df.filter(pl.col("date_id").is_in(val_date_ids))

    all_target_cols = [col for col in val_pred_df.columns if col.startswith("target_")]

    if target_cols_to_plot is None:
        cols_to_plot = all_target_cols

    else:
        cols_to_plot = []
        for col in target_cols_to_plot:
            if col in all_target_cols:
                cols_to_plot.append(col)

    n_plots = len(cols_to_plot)
    n_cols = min(n_plots, max_cols)
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig_width = figsize_per_plot[0] * n_cols
    fig_height = figsize_per_plot[1] * n_rows

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))

    if n_plots == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.flatten()
    elif n_cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    # target_pairs_df exist, convert meta info into a dict
    target_meta = {}
    if target_pairs_df is not None:
        for row in target_pairs_df.iter_rows(named=True):
            target_meta[row["target"]] = {"lag": row["lag"], "pair": row["pair"]}

    val_pred_pd = val_pred_df.to_pandas()
    val_target_pd = val_target_df.to_pandas()
    tr_target_pd = tr_target_df.to_pandas()

    post_val_df = plot_labels_df.filter(pl.col("date_id") > val_end)
    post_val_pd = post_val_df.to_pandas()

    for idx, col in tqdm(enumerate(cols_to_plot), total=n_plots, desc="Plotting"):
        ax = axes[idx]

        # Training data (GT)
        if len(tr_target_pd) > 0:
            ax.plot(
                tr_target_pd["date_id"],
                tr_target_pd[col],
                label="Training GT",
                color="blue",
                alpha=alpha_gt,
                linewidth=1,
            )

        # Validation GT
        if len(val_target_pd) > 0:
            ax.plot(
                val_target_pd["date_id"],
                val_target_pd[col],
                label="Validation GT",
                color="green",
                alpha=alpha_gt,
                linewidth=1.5,
            )

        # Validation Prediction
        ax.plot(
            val_pred_pd["date_id"],
            val_pred_pd[col],
            label="Prediction",
            color="red",
            alpha=alpha_pred,
            linewidth=1.5,
            linestyle="--",
        )

        # Post-validation GT
        if len(post_val_pd) > 0:
            ax.plot(
                post_val_pd["date_id"],
                post_val_pd[col],
                label="Post-val GT",
                color="purple",
                alpha=alpha_gt * 0.7,
                linewidth=1,
            )

        ax.axvline(x=val_start, color="gray", linestyle=":", alpha=0.5, label="Val start")
        ax.axvline(x=val_end, color="gray", linestyle=":", alpha=0.5)

        ax.axvspan(val_start, val_end, alpha=0.1, color="yellow")

        if col in target_meta:
            meta = target_meta[col]
            title = f"{col} | lag={meta['lag']} | {meta['pair']}"
            if len(title) > 80:
                pair_short = meta["pair"][:60] + "..."
                title = f"{col} | lag={meta['lag']} | {pair_short}"
        else:
            title = col

        ax.set_title(title, fontsize=9)
        ax.set_xlabel("date_id", fontsize=8)
        ax.set_ylabel("value", fontsize=8)
        ax.tick_params(axis="both", labelsize=7)
        ax.grid(True, alpha=0.3)

        ax.set_xlim(plot_date_ids[0], plot_date_ids[-1])

        if idx == 0:
            ax.legend(fontsize=8, loc="best")

    for idx in range(n_plots, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    title = f"Time Series Predictions ({n_plots} labels, context={n_context})"
    plt.suptitle(title, y=1.02, fontsize=12)

    if filepath:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(filepath, dpi=100, bbox_inches="tight")
    plt.show()

def plot_training_history(
    history_df: pl.DataFrame,
    filepath: str | Path | None = None,
    figsize: tuple = (12, 10),
) -> None:
    sns.set_style("whitegrid")
    df = history_df.to_pandas()

    fig, axes = plt.subplots(2, 1, figsize=figsize, height_ratios=[1, 1])

    # ========== Upper subplot: Loss and Score ==========
    ax1 = axes[0]

    color_train = "tab:blue"
    color_val = "tab:orange"
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    line1 = ax1.plot(
        df["epoch"],
        df["train_loss"],
        color=color_train,
        label="Train Loss",
        linewidth=2,
        marker="o",
        markersize=4,
        alpha=0.7,
    )
    line2 = ax1.plot(
        df["epoch"],
        df["val_loss"],
        color=color_val,
        label="Val Loss",
        linewidth=2,
        marker="s",
        markersize=4,
        alpha=0.7,
    )
    ax1.tick_params(axis="y")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    color_score = "tab:green"
    ax2.set_ylabel("Val Score", fontsize=12, color=color_score)
    line3 = ax2.plot(
        df["epoch"],
        df["val_score"],
        color=color_score,
        label="Val Score",
        linewidth=2,
        marker="^",
        markersize=5,
    )
    ax2.tick_params(axis="y", labelcolor=color_score)

    best_epochs = df[df["is_best"]]
    if len(best_epochs) > 0:
        ax2.scatter(
            best_epochs["epoch"],
            best_epochs["val_score"],
            color="red",
            s=100,
            zorder=5,
            marker="*",
            label="Best",
        )

    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]  # noqa
    if len(best_epochs) > 0:
        best_marker = plt.Line2D(
            [0],
            [0],
            marker="*",
            color="w",
            markerfacecolor="red",
            markersize=10,
            label="Best",
        )
        lines.append(best_marker)
        labels.append("Best")
    ax1.legend(lines, labels, loc="upper right", framealpha=0.9)
    ax1.set_title("Training History", fontsize=14, fontweight="bold", pad=10)

    # ========== Lower subplot: Correlation Mean and Std ==========
    ax3 = axes[1]

    color_mean = "tab:purple"
    color_std = "tab:brown"

    # Plot correlation mean on primary y-axis
    ax3.set_xlabel("Epoch", fontsize=12)
    ax3.set_ylabel("Correlation Mean", fontsize=12, color=color_mean)
    line4 = ax3.plot(
        df["epoch"],
        df["val_corr_mean"],
        color=color_mean,
        label="Val Corr Mean",
        linewidth=2,
        marker="o",
        markersize=4,
        alpha=0.7,
    )
    ax3.tick_params(axis="y", labelcolor=color_mean)
    ax3.grid(True, alpha=0.3)

    # Plot correlation std on secondary y-axis
    ax4 = ax3.twinx()
    ax4.set_ylabel("Correlation Std", fontsize=12, color=color_std)
    line5 = ax4.plot(
        df["epoch"],
        df["val_corr_std"],
        color=color_std,
        label="Val Corr Std",
        linewidth=2,
        marker="d",
        markersize=4,
        alpha=0.7,
    )
    ax4.tick_params(axis="y", labelcolor=color_std)

    # Highlight best epochs in correlation plot
    if len(best_epochs) > 0:
        ax3.scatter(
            best_epochs["epoch"],
            best_epochs["val_corr_mean"],
            color="red",
            s=100,
            zorder=5,
            marker="*",
            alpha=0.6,
        )

    # Add legend for correlation subplot
    lines_corr = line4 + line5
    labels_corr = [l.get_label() for l in lines_corr]  # noqa
    if len(best_epochs) > 0:
        best_marker_corr = plt.Line2D(
            [0],
            [0],
            marker="*",
            color="w",
            markerfacecolor="red",
            markersize=10,
            label="Best Epoch",
        )
        lines_corr.append(best_marker_corr)
        labels_corr.append("Best Epoch")
    ax3.legend(lines_corr, labels_corr, loc="upper right", framealpha=0.9)
    ax3.set_title("Validation Correlations", fontsize=14, fontweight="bold", pad=10)

    plt.suptitle("Model Training Metrics", fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()

    if filepath:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(filepath, dpi=100, bbox_inches="tight")

    plt.show()
