# %%
import os

import joblib
import polars as pl
import torch

from train import Config, CustomModel, InferenceEnv, Preprocessor, load_best_model
from kaggle_evaluation import mitsui_inference_server
from settings import DirectorySettings

config = Config()
settings = DirectorySettings(exp_name=config.exp_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
raw_train_df = pl.read_csv(settings.COMP_DATASET_DIR / "train.csv")

preprocessor = Preprocessor(config=config)
preprocessor.load(settings.ARTIFACT_EXP_DIR)

feature_cols = joblib.load(settings.ARTIFACT_EXP_DIR / "feature_cols.pkl")
target_cols = joblib.load(settings.ARTIFACT_EXP_DIR / "target_cols.pkl")

best_models = [
    load_best_model(
        model=CustomModel(
            config=config,
            num_input_channels=len(feature_cols),
            num_output_channels=len(target_cols),
        ),
        model_path=settings.ARTIFACT_EXP_DIR / f"seed_{seed}" / "best_model.pth"
        if not config.full_train
        else settings.ARTIFACT_EXP_DIR / "full_training" / f"seed_{seed}" / "best_model.pth",
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
    inference_server.server()
else:
    inference_server.run_local_gateway((settings.COMP_DATASET_DIR,))
    df = pl.read_parquet("submission.parquet")
    print(df)

print("Inference completed")
