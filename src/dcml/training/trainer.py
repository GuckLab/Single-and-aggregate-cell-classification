import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .criterion import CELoss_with_costs, MultiTaskCrossEntropy
from ..utils.models import build_model
from ..predict import predict
from ._utils import print_batch, param_dict_to_list  # SaveImage, print_metrics
from ..logger.base_logger import BaseLogger
from ..training.metrics import EvaluationMetrics
import numpy as np


class Trainer:
    """Responsible for the training process of a Model in Pytorch

    This class deals with the training process of a Neural Network. It includes
    all necessary functionality for training and storing of model and metrics.
    """

    def __init__(self,
                 session_params: dict,
                 dataloader_train: DataLoader,
                 dataloader_val: DataLoader,
                 device: str,
                 verbose: bool,
                 logger: BaseLogger,) -> None:
        """Initializes the Trainer Class

        Parameters
        ----------
        session_params:
            dictionary containing relevant information about the training, such
            as experiment-name, architecture, optimizer etc.
        dataloader_train: torch.utils.data.DataLoader
            Contains the training data
        dataloader_val: torch.utils.data.DataLoader
            Contains the validation data
        verbose: bool
            indicates if training information should be printed out.
        logger: BaseLogger
            Logger-instance used for logging evaluation output
        MTL: True if Multi Task Learning
        """
        self.session_params = session_params
        self.params = session_params["create_trainer"]
        self.MTL = self.params['architecture']['type'] == 'multitask'

        self.verbose = verbose
        self.dataloader = {}
        self.dataloader["train"] = dataloader_train
        self.dataloader["val"] = dataloader_val
        self.logger = logger
        self.lr = self.params["lr"]
        self.class_label_dict = self.params["class_label_dict"]
        # self.ml_score_features = self.params["ml_score_features"] # never used
        self.num_classes = len(self.class_label_dict)
        self.evaluate_artifacts = []
        self.device = device
        self.initialize()

    def initialize(self) -> None:
        """Prepares Trainer by initializing datasets and dataloader etc."""
        self.create_model()

        self.create_optimizer()
        self.create_scheduler()
        self.create_criterion()
        self.epoch = 0

    def create_model(self) -> None:
        """Load Model based on parameters set in params file"""
        self.model = build_model(self.params).to(self.device)

    def create_optimizer(self) -> None:
        """Defines Optimizer based on parameters set in params-filename

        Needs `self.model` to be defined via `self.create_model()`.
        """
        # Get optimizer class-constructor set in params file
        optim_class = getattr(torch.optim, self.params["optimizer"])

        # instantiate optimizer
        self.optimizer = optim_class(self.model.parameters(), lr=self.lr)

    def create_scheduler(self) -> None:
        """Defines Scheduler based on parameters set in params-filename

        Needs `self.model` to be defined via `self.create_model()`.
        """
        # Get scheduler params in params file
        params = self.params["scheduler"]

        # Get scheduler class-constructor set in params file
        scheduler_class = getattr(torch.optim.lr_scheduler,
                                  params["type"])

        # instantiate scheduler
        self.scheduler = scheduler_class(optimizer=self.optimizer,
                                         step_size=params["step_size"],
                                         gamma=params["gamma"])

    def create_criterion(self) -> None:
        """Define Loss function

        Currently it is hard coded to CE-Loss with class weights defined in
        params-file.
        """
        criterion_params = self.params["criterion"]

        if criterion_params["type"] == "MultiTaskCrossEntropy":
            assert self.MTL == True, "model arcithecture for MultiTaskCrossEntropy must be multitask"
            inv_costs = criterion_params.get("inv_costs", None)
            num_heads = self.params["MTL_classes"]
            task_weights = criterion_params.get("task_weights", None)
            mtl_mc_weights = criterion_params.get("MTL_MC_weights", None)
            mc_class_weights = criterion_params.get("MC_class_weights", None)
            pos_weights = criterion_params.get("pos_weights", None) # dictionary
            if not pos_weights:
                print("calculation of the positive weights for binary classifiers", flush=True)
                pos_weights = self.find_pos_weights(num_heads) # outputs dictionary
                print(f"positive weights:{pos_weights}")
                self.session_params["create_trainer"]["criterion"]['pos_weights'] = pos_weights # configuration file will be updated

            pos_weights = param_dict_to_list(pos_weights, self.class_label_dict) # outputs list


            if pos_weights is not None:
                assert num_heads == len(pos_weights), "the number of pos_weights must ne equal to the number num_heads"

            if task_weights is not None:
                assert num_heads == len(task_weights), ("the number of MTL classes must be equal to the number of "
                                                             "task weights in the configuration file")

            if task_weights is not None and mc_class_weights is not None:
                assert len(task_weights) + len(mc_class_weights) == len(self.class_label_dict), ("the number tasks "
                                                            "weights plus the number of MC classes weights must be equal"
                                                            "to the number of labels in the configuration file")

            if isinstance(task_weights, dict):
                print("not yet implemented") ## TODO implement this if needed
                raise
            else:
                task_weights_tensor = torch.tensor(task_weights)

            if isinstance(mtl_mc_weights, dict):
                print("not yet implemented")  ## TODO implement this if needed
                raise
            else:
                mtl_mc_weights_tensor = torch.tensor(mtl_mc_weights)

            if isinstance(mc_class_weights, dict):
                print("not yet implemented")  ## TODO implement this if needed
                raise
            else:
                mc_class_weights_tensor = torch.tensor(mc_class_weights)

            task_weights_tensor = task_weights_tensor.to(self.device).type(torch.float32)  # noqa: E501
            mtl_mc_weights_tensor = mtl_mc_weights_tensor.to(self.device).type(torch.float32)
            mc_class_weights_tensor = mc_class_weights_tensor.to(self.device).type(torch.float32)
            self.criterion = MultiTaskCrossEntropy(num_heads=num_heads, weights_MTL=task_weights_tensor, weights_MC=mc_class_weights_tensor,
                                                   weights_MTL_MC=mtl_mc_weights_tensor, pos_weights=pos_weights, inv_costs=inv_costs, device=self.device)
            self.criterion = self.criterion.to(self.device)


        elif criterion_params["type"] == "CELoss_with_costs":
            class_weights = self.params["class_weights"]
            inv_costs = criterion_params.get("inv_costs", None)
            # costs = criterion_params.get("costs", None)
            # confidence = criterion_params.get("confidence", None)
            if isinstance(class_weights, dict):
                class_weights_tensor = torch.tensor([class_weights[k] for k in range(self.num_classes)])
            else:
                class_weights_tensor = torch.tensor(class_weights)
            class_weights_tensor = class_weights_tensor.to(self.device).type(torch.float32)  # noqa: E501
            self.criterion = CELoss_with_costs(self.num_classes, inv_costs=inv_costs, weights=class_weights_tensor,
                                               device=self.device)
            self.criterion = self.criterion.to(self.device)

        elif criterion_params["type"] == "cross_entropy":
            class_weights = self.params["class_weights"]
            if isinstance(class_weights, dict):
                class_weights_tensor = torch.tensor([class_weights[k] for k in range(self.num_classes)])
            else:
                class_weights_tensor = torch.tensor(class_weights)
            class_weights_tensor = class_weights_tensor.to(self.device).type(torch.float32)  # noqa: E501
            self.criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
            self.criterion = self.criterion.to(self.device)

        # elif "focal" in criterion_params["type"].lower():
        #     Commented out during cleanup: focal criterion variants are not used in current train/evaluate flow.

    def get_model_output(self, model_input):
        # this function gets model output and format it
        output = self.model(model_input)
        architecture_type = self.params["architecture"]["type"]
        if architecture_type == 'inception_v3':
            output = output.logits
        return output

    def train_step(self, epoch: int, every_n_batches: int = 100) -> float:
        """Runs Training for one epoch

        Uses the dataloader of the train dataset with weighted sampler.
        If verbose is set to True, it will print out the loss value
        every 100th batch.

        Outputs mean value of the loss over all batches

        """
        self.model.train()
        self.epoch = epoch

        # # debugging
        # image_processor = SaveImage(path_to_save="/mnt/ZPE_cluster_results/igor_tests/augmentations_check",
        #                             file_prefix='noinv_noblur_02noise', ext='png', str_labels=None)

        av_loss = 0
        for idx, batch in enumerate(self.dataloader['train']):

            # # # TODO remove debugging:
            # if idx > 1:
            #     break

            # Define Input
            model_input = batch["image"].to(self.device)
            # TODO - do I need long() for multiclass? For multitask I do not
            target = batch["target"]
            if not self.MTL:
                target = target.long()

            target = target.to(self.device)

            # # TODO remove this debugging
            # if torch.isnan(target).any():
            #     print("has nan")
            #     raise


            # # debugging
            # image_processor(model_input, target)

            self.optimizer.zero_grad()
            # Get Output
            with torch.set_grad_enabled(True):
                output = self.get_model_output(model_input)
                # _, prediction = torch.max(output, 1)

                # Backpropagation Step
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
            if self.verbose:
                if idx % every_n_batches == 0:
                    print_batch(epoch=self.epoch,
                                batch_idx=idx,
                                loss_value=loss.item(),
                                )

            av_loss += loss.item()

        self.scheduler.step()
        av_loss = av_loss / len(self.dataloader['train'])

        return av_loss

    def evaluate(self, stage: str, eval_metrics: EvaluationMetrics, last_epoch=False) -> None:
        """Evaluates current model

        Applies the model to dataset of given dataloader, computes
        'Precision', 'Recall' and 'F1-Score' and other requested metrics.

        Parameters
        ----------
        stage:
            This defines the used Pytorch Dataloader. Can be one of
            ['train', 'val'].

        eval_metrics: the instance of a class that defines which metrics need to be used for evaluation
        """

        print(f"running prediction on {stage}, {time.ctime()}", flush=True)
        # print(f"the number of batches in {stage} dataloader is {len(self.dataloader[stage])}", flush=True)
        targets, prediction_scores, predictions, loss_values = predict(
            model=self.model,
            dataloader=self.dataloader[stage],
            criterion=self.criterion,
            device=self.device,
            class_label_dict=self.class_label_dict
            #free_dependent=self.session_params["create_dataset"]["free_dependent"]
        )

        # compute requested metrics
        eval_metrics(targets, predictions, prediction_scores)

        # scores = metrics.precision_recall_fscore_support(targets,
        #                                                  predictions,
        #                                                  average=None)
        #
        # # convert scores from tuple to list
        # scores = [list(row) for row in scores]
        #
        # precision_scores = scores[0]
        # recall_scores = scores[1]
        #
        # ###
        # # Compute F_beta_Score
        # ###
        # print(f"Compute F_beta_score {time.ctime()}", flush=True)
        # f_beta_scores = []
        # for idx, class_label in self.class_label_dict.items():
        #     delimiter = (self.logger.f1_score_weights[class_label]**2
        #                  * precision_scores[idx]) + recall_scores[idx]
        #     if delimiter != 0:
        #         f_beta_score = (
        #                 (1 + self.logger.f1_score_weights[class_label]**2)
        #                 * precision_scores[idx]
        #                 * recall_scores[idx]
        #                 / delimiter
        #         )
        #     else:
        #         f_beta_score = 0
        #     f_beta_scores.append(f_beta_score)
        # scores.append(f_beta_scores)
        #
        # ###
        # # Compute bio_avg_f_beta_score
        # ###
        # print(f"Compute bio avg f_beta_score {time.ctime()}", flush=True)
        # bio_f_beta_scores = []
        # for idx, class_label in self.class_label_dict.items():
        #     bio_f_beta_score = (
        #             f_beta_scores[idx]
        #             * self.logger.bio_relevance_weights[class_label]
        #     )
        #     bio_f_beta_scores.append(bio_f_beta_score)
        # scores.append(bio_f_beta_scores)
        # # print(f"Log with targets {time.ctime()}", flush=True)
        # # self.logger.log_with_target(predictions=predictions,
        # #                             prediction_scores=prediction_scores,
        # #                             targets=targets,
        # #                             loss_values=loss_values,
        # #                             scores=scores,
        # #                             stage=stage,
        # #                             epoch=self.epoch)
        #
        # #scores.append(recall_scores) # used for computation of balanced accuracy

        # logging loss values and requested metrics
        print(f"Log scalar {time.ctime()}", flush=True)
        self.logger.log_scalar(loss_values=loss_values,
                               metrics=eval_metrics,
                               stage=stage,
                               epoch=self.epoch,
                               individual_classes=self.params.get("learning_curve_show", []))

        # ###
        # # Compute balanced accuracy - remove when you see that averaging recall scores is the same
        # ###
        # print(f"Compute balanced accuracy {time.ctime()}", flush=True)
        # balanced_acc = metrics.balanced_accuracy_score(targets, predictions)
        # mlflow.log_metric(key=f"{stage.upper()} - Balanced Accuracy", value=balanced_acc, step=self.epoch)

        # logging confusion matrix, performance table with requested metrics (report), false detections (images)
        if last_epoch:
            print(f"Log artifacts {time.ctime()}", flush=True)
            self.logger.log_artifacts(metrics=eval_metrics, epoch=self.epoch)

            # # TODO bring it back (does not work for MTL now)
            # print(f"Log images {time.ctime()}", flush=True)
            # dataset = self.dataloader[stage].dataset
            # self.logger.log_images(torch_dataset=dataset,
            #                        predictions=predictions,
            #                        prediction_scores=prediction_scores,
            #                        targets=targets
            #                        )

        # return scores, targets, predictions, prediction_scores, loss_values
        return

    def save_model(self, filename: str) -> None:
        """Saves model and relevant parts of training process"""
        transform = self.dataloader["val"].dataset.transform
        self.logger.save_model(model=self.model,
                               model_name=filename,
                               transform=transform,
                               params=self.session_params,
                               epoch=self.epoch)

    # def print_metrics(self, scores: List) -> None:
    #     """Pretty-Prints out Precision, Recall and F1-Score for all classes"""
    #     print_metrics(scores, self.class_label_dict)

    # def close(self):
    #     self.logger.close()

    def find_pos_weights(self, n_MTLclasses: int) -> dict:
        """

        Parameters
        ----------
        n_MTLclasses: the number of MTL classes (first classes defined in "class_label_dict" in configuration file )

        Returns
        -------
        pos_weights - dictionary with keys as names of the MTL classes and values as positive weights for training binary classifiers
        """

        cls_label_counts_pos = {}
        cls_label_counts_neg = {}

        def update_cls_labels(cls_label_counts, targets):

            for tar in targets:
                current_label = self.class_label_dict[tar]
                if current_label in cls_label_counts:
                    cls_label_counts[current_label] += 1
                else:
                    cls_label_counts[current_label] = 1

            return cls_label_counts

        for batch_obj in self.dataloader["train"]:

            # TODO check how it works for batch size == 1

            batch_target = batch_obj['target']
            for target in batch_target:


                if torch.is_tensor(target):
                    target = target.numpy()

                target = np.squeeze(target)

                try:
                    _ = len(target)
                except TypeError:
                    target = [target.item(), ]

                # in case of MTL, targets are vectors instead of numbers
                #if len(target) > 1:
                assert len(target) > 1, "MTL case only"
                    # MTL

                targets = np.flatnonzero(target[:n_MTLclasses] == 1)
                targets_neg = np.flatnonzero(target[:n_MTLclasses] == 0)

                cls_label_counts_pos = update_cls_labels(cls_label_counts_pos, targets)
                cls_label_counts_neg = update_cls_labels(cls_label_counts_neg, targets_neg)

        assert set(cls_label_counts_pos.keys()) == set(cls_label_counts_neg.keys()), "sets must be equal"
        pos_weights = {}
        for cls in cls_label_counts_pos.keys():
            pos_weights[cls] = cls_label_counts_neg[cls]/cls_label_counts_pos[cls]

        return pos_weights
