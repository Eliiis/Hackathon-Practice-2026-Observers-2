configfile: "config.yaml"

from tailgate import stages


rule all:
    input:
        "submissions/submission.csv",


rule validate:
    input:
        "artifacts/out_of_fold.npy",


rule report:
    input:
        "reports/analysis.md",


rule parse:
    input:
        config["data"]["train"],
        config["data"]["test"],
    output:
        "data/processed/train.parquet",
        "data/processed/test.parquet",
    run:
        stages.parse()


rule features:
    input:
        "data/processed/train.parquet",
        "data/processed/test.parquet",
    output:
        "data/processed/train_features.parquet",
        "data/processed/test_features.parquet",
    run:
        stages.build_features()


rule folds:
    input:
        "data/processed/train_features.parquet",
    output:
        "artifacts/folds.json",
    run:
        stages.folds()


rule shift:
    input:
        "data/processed/train_features.parquet",
        "data/processed/test_features.parquet",
    output:
        "artifacts/shift.json",
    run:
        stages.estimate_shift()


rule cross_validate:
    input:
        "data/processed/train_features.parquet",
        "artifacts/folds.json",
    output:
        "artifacts/out_of_fold.npy",
    run:
        stages.validate()


rule analysis:
    input:
        "data/processed/train_features.parquet",
        "artifacts/folds.json",
        "artifacts/out_of_fold.npy",
    output:
        "reports/analysis.md",
    run:
        stages.analyze()


rule submit:
    input:
        "data/processed/train_features.parquet",
        "data/processed/test_features.parquet",
    output:
        "submissions/submission.csv",
        "artifacts/test_prediction.npy",
    run:
        stages.submit()
