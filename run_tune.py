import optuna


def objective(trial):
    x = trial.suggest_float("x", -10, 10)
    return (x - 2) ** 2


if __name__ == "__main__":
    study_name = "study1"
    storage = f"sqlite:///{study_name}.db"
    study = optuna.create_study(
        study_name=study_name, storage=storage, load_if_exists=True
    )

    study.optimize(objective, n_trials=400)
    df = study.trials_dataframe()
    df.to_csv(f"{study_name}.csv", index=False)
