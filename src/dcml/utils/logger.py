from dcml.logger import MLflowLogger  # TensorboardLogger


def retrieve_logger(logger_params, class_label_dict, log_dir=None):
    """Select logger using logger params."""
    if logger_params.get('name', '') == 'mlflow':
        logger = MLflowLogger(class_label_dict=class_label_dict,
                              log_dir=log_dir,
                              experiment_name=logger_params.get(
                                  'experiment_name'),
                              run_name=logger_params.get(
                                  'run_name'),
                              experiment_description=logger_params.get('experiment_description')
                              )
    else:
        raise Exception("tensorflow logger is deprecated")
        # logger = TensorboardLogger(log_dir=log_dir,
        #                            class_label_dict=class_label_dict,
        #                            ml_score_features=ml_score_features
        #                            )
    return logger
