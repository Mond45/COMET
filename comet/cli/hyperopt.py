from copy import deepcopy
from functools import partial
import warnings

import optuna
import wandb
from comet.cli.train import initialize_model, initialize_trainer, read_arguments
from pytorch_lightning import seed_everything


def optuna_objective(trial: optuna.trial.Trial, cfg):
    cfg = deepcopy(cfg)

    if cfg.regression_metric is not None:
        model_cfg = cfg.regression_metric.init_args
    elif cfg.referenceless_regression_metric is not None:
        model_cfg = cfg.referenceless_regression_metric.init_args
    elif cfg.ranking_metric is not None:
        model_cfg = cfg.ranking_metric.init_args
    elif cfg.unified_metric is not None:
        model_cfg = cfg.unified_metric.init_args
    else:
        raise Exception("Model configurations missing!")

    model_cfg.learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True)

    trainer = initialize_trainer(cfg)
    model = initialize_model(cfg)

    trainer.fit(model)

    wandb.finish()

    return trainer.early_stopping_callback.best_score.item()


def hyperopt():
    parser = read_arguments()
    cfg = parser.parse_args()

    seed_everything(cfg.seed_everything)

    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message=".*Consider increasing the value of the `num_workers` argument` .*",
    )

    study = optuna.create_study(direction="maximize")
    study.optimize(partial(optuna_objective, cfg=cfg), n_trials=10)
    print(study.best_params)


if __name__ == "__main__":
    hyperopt()
