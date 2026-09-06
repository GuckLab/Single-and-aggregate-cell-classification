import logging
import sys

logging.basicConfig(level=logging.INFO, stream=sys.stdout)

import argparse
import pathlib
import time


from dotenv import load_dotenv
import torch
from torch.utils.data import DataLoader

from dcml.utils.logger import retrieve_logger
from dcml.training import Trainer
from dcml.evaluation.helpers import map_abbr_2_ml_score_label, map_weights_abbr_2_labels
from dcml.utils.data import create_datasets, create_data_sampler
import yaml
from src.deepclassifier.evaluate_utils import evaluate_models, evaluate_models_on_gmm
import mlflow
from dcml.training.metrics import EvaluationMetrics
import random
import os
import numpy as np

load_dotenv()

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--param_file', type=str)
    parser.add_argument('--data_path_gmm_eval_out', type=str, default="", help='Output folder path where rtdc files with'
                                                                           ' predictions on GMM data will be saved')
    parser.add_argument('--data_path_gmm_eval_in', type=str, default="", help='Input folder with GMM rtdc measurements')
    parser.add_argument('--mlflow_gmm_folder_name', type=str, default="evaluation_on_gmm_measurements",
                        help='folder name within mlflow where gmm evaluation results will be stored')
    parser.add_argument('--mlflow', action='store_true', default=True, help='True for model to be saved in mlflow')
    parser.add_argument('--seed', type=int, default=0, help='sets seed to make results reproducable')
    parser.add_argument('--rm_pred', action='store_true', default=False, help='removes folder with predictions after evaluation')

    args = parser.parse_args()

    # if args.seed:
    #     set_seed(args.seed) # do not use seed, random run

    data_absolute_path = pathlib.Path(args.data_path).resolve()
    params = yaml.safe_load(open(args.param_file))

    # use seed from the command line if given, if not use from the configuration file, if not - no seed is used - random run
    if args.seed:
        set_seed(args.seed)
        params["training"]['seed'] = args.seed
    else:
        seed = params["training"].get('seed', None)
        if seed is not None:
            set_seed(seed)

    dataset_params = params["create_dataset"]
    trainer_params = params["create_trainer"]
    training_params = params["training"]
    device = 'cpu'  # default device value
    num_workers = 0  # default number of workers value

    hdf5_absolute_paths = [data_absolute_path / path
                           for path in dataset_params["hdf5_paths"]["train"]]

    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        num_workers = 8
        print("GPU is available.", flush=True)
        print(f"CUDA version: "
              f"{torch.version.cuda}",
              flush=True)
        print(f"GPU device name: "
              f"{torch.cuda.get_device_name(device)}",
              flush=True)
    else:
        print("GPU is not available. Using CPU by default.",
              flush=True)

    # TODO design a function that check all constraints on yaml configuration file, including the one below
    par_check_train = dataset_params.get('unknowns_target_grouping_train_val', None)
    if par_check_train:
        par_check_test = dataset_params.get('unknowns_target_grouping_test')
        for key in par_check_train:
            assert set(par_check_test[key]) <= set(par_check_train[key]), "unknowns_target_grouping_test must be subset of unknowns_target_grouping_train_val"

    MTL_mode = trainer_params['architecture']['type'] == 'multitask'
    print(f"Create dataset: train and val {time.ctime()}", flush=True)
    augmentation = dataset_params["augmentation"]
    train_p99 = dataset_params.get("p99", None)
    train_mean = dataset_params.get("mean", None)
    train_std = dataset_params.get("std", None)
    dataset_train, dataset_val = create_datasets(
        hdf5_paths=hdf5_absolute_paths,
        required_data=dataset_params["required_data"],
        target_grouping=dataset_params["target_grouping"],
        augmentation=augmentation,
        crop_size=dataset_params["crop_size"],
        correct_background=dataset_params["correct_background"],
        train_size=dataset_params["train_size"],
        mean=train_mean,
        std=train_std,
        p99_compute=train_p99 is None,
        ml_score_to_int=dataset_params.get('ml_score_to_int', None),
        MTL=MTL_mode,
        unknowns_target_grouping=dataset_params.get('unknowns_target_grouping_train_val', None)
    )

    assert (len(dataset_train)) > 0, (
        "the is no valid data (rtdc files with the labels defined in the configuration file) "
        "at given input path")
    assert (len(dataset_val)) > 0, (
        "the is no valid data (rtdc files with the labels defined in the configuration file) "
        "at given input path")

    print(f"the number of examples in train set is {len(dataset_train)}", flush=True)
    print(f"the number of examples in validation set is {len(dataset_val)}", flush=True)

    # add mean and std to logged params file
    if train_mean is None:
        params["create_dataset"]["mean"] = round(dataset_train.mean, 4)

    if train_std is None:
        params["create_dataset"]["std"] = round(dataset_train.std, 4)

    if train_p99 is None:
        params["create_dataset"]["p99"] = round(dataset_train.p99, 4)

    print(f"Create dataloader {time.ctime()}", flush=True)
    batch_size = trainer_params["batch_size"]
    dataset_train_sampler = None
    dataset_train_shuffle = True

    if "sampler" in dataset_params:

        sampler_weights = dataset_train.calculate_sample_weights()
        num_samples = dataset_params["sampler"].get("num_samples", len(dataset_train))
        sampler_type = dataset_params["sampler"].get("type", '')
        dataset_train_sampler = create_data_sampler(sampler_type, sampler_weights, num_samples)
        if dataset_train_sampler:
            print(f"Sampler created {time.ctime()}", flush=True)
            dataset_train_shuffle = False # TODO put here None instead False according to the help for Dataloader

    print(f"Create dataloader {time.ctime()}", flush=True)
    dataloader_train = DataLoader(dataset_train,
                                  batch_size=batch_size,
                                  shuffle=dataset_train_shuffle,
                                  sampler=dataset_train_sampler,
                                  num_workers=num_workers)
    dataloader_val = DataLoader(dataset_val,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=num_workers)

    print(f"the number of batches in train loader is {len(dataloader_train)}", flush=True)
    print(f"the number of batches in validation loader is {len(dataloader_val)}", flush=True)
    print(f"bach size is {batch_size}")

    # retrieve our_logger
    class_label_dict = trainer_params["class_label_dict"]
    bio_relevance_weights = params["create_trainer"].get("bio_relevance_weights", None)
    f1_score_weights = params["create_trainer"].get("f_beta_score_weights", None)

    print(f"Retrieve Logger {time.ctime()}", flush=True)
    logger = retrieve_logger(logger_params=trainer_params.get('logger', {}),
                             class_label_dict=class_label_dict,
                             log_dir=None)

    print(f"Create trainer {time.ctime()}", flush=True)

    mapping = map_abbr_2_ml_score_label(class_label_dict, trainer_params["ml_score_features"])
    bio_relevance_weights_mapped = map_weights_abbr_2_labels(bio_relevance_weights, mapping)
    f1_score_weights_mapped = map_weights_abbr_2_labels(f1_score_weights, mapping)

    metrics_val = EvaluationMetrics(target_names=class_label_dict, bio_weights=bio_relevance_weights_mapped,
                                    beta_weights=f1_score_weights_mapped,
                                    used_metrics=trainer_params["performance_metrics"])
    metric_names = metrics_val.requested_metric_names
    metrics_train = EvaluationMetrics(target_names=class_label_dict, bio_weights=bio_relevance_weights_mapped,
                                      beta_weights=f1_score_weights_mapped,
                                      used_metrics=trainer_params["performance_metrics"])

    trainer = Trainer(session_params=params,
                      dataloader_train=dataloader_train,
                      dataloader_val=dataloader_val,
                      device=device,
                      verbose=True,
                      logger=logger,
                      )

    # log parameters
    logger.log_params(params=params, filename="configuration.yaml")

    print(f"START TIME: {time.ctime()}", flush=True)

    train_step_metrics = {}
    for epoch in range(training_params["num_epochs"]):
        print(f"Epoch: {epoch}, {time.ctime()}", flush=True)

        # # # # # # # # TODO remove - debugging
        # trainer.evaluate(stage='val', last_epoch=True, eval_metrics=metrics_val)

        av_loss = trainer.train_step(epoch, every_n_batches=training_params.get("every_n_batches", 100))

        mlflow.log_metric(key="TRAIN - Average batch loss", value=av_loss, step=epoch)

        print("model evaluation on validation set...", flush=True)

        last_epoch_flag = (epoch == training_params["num_epochs"] - 1)
        trainer.evaluate(stage='val', last_epoch=last_epoch_flag, eval_metrics=metrics_val)

        for metric_name in metric_names:
            current_metric_value = getattr(metrics_val, metric_name)
            # print(f"{metric_name}: {current_metric_value}")

            if current_metric_value == getattr(metrics_val, "best_" + metric_name):
                trainer.save_model(f"best_model_{metric_name}")
                train_step_metrics[f"best_model_{metric_name}"] = '{:.2f}'.format(current_metric_value)
                train_step_metrics[f"best_model_{metric_name}_epoch"] = epoch

                if last_epoch_flag:
                    train_step_metrics[f"last_model_{metric_name}"] = '{:.2f}'.format(current_metric_value)

        if last_epoch_flag:
            # save final model
            trainer.save_model("last_model")
            train_step_metrics["last_model_epoch"] = epoch

        if (epoch % training_params["eval_interval"] == 0) or last_epoch_flag:
            print("Evaluation: Validation Dataset", flush=True)
            metrics_val.print_metrics()
            # trainer.print_metrics(scores_val)

            print("Evaluation: Training Dataset", flush=True)
            trainer.evaluate(stage='train', eval_metrics=metrics_train)
            metrics_train.print_metrics()
            # trainer.print_metrics(scores_train)

    # log the best performances on validation set and corresponding epoch for all requested metrics
    logger.log_params(params=train_step_metrics, filename="model_epoch_val_performance.yaml")

    # save run-uuid for following evaluation
    # TODO do I need this?
    run_id = logger.active_run.info.run_uuid
    with open('run_uuid.txt', 'w') as file:
        file.write(run_id)

    logger.close()

    print(f"END TRAINING TIME: {time.ctime()}", flush=True)

    # model evaluation
    print("starting evaluation on test dataset")

    model_names = []
    for metric_name in metric_names:
        model_names.append(f"best_model_{metric_name}")
    # model_names.append("last_model")
    evaluate_models(run_id, args.data_path, model_names=model_names, batch_size=batch_size)

    if args.data_path_gmm_eval_in:

        # model evaluation on GMM data
        print("starting evaluation on GMM test dataset")

        evaluate_models_on_gmm(run_id, args.data_path_gmm_eval_in, args.data_path_gmm_eval_out, model_names,
                               remove_predictions=args.rm_pred, mlflow_folder_name=args.mlflow_gmm_folder_name)

    print(f"END EVALUATION TIME: {time.ctime()}", flush=True)


if __name__ == "__main__":
    main()

