import time
from collections import OrderedDict

from omegaconf import DictConfig

import torch
from sklearn.svm import LinearSVC

from model import SimpleCNN, MLP, test
from torch.utils.data import Dataset, DataLoader

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, precision_score, recall_score, f1_score, confusion_matrix

from dataset import get_data_numpy
import numpy as np
import xgboost as xgb
from flwr.common.logger import log
from logging import INFO
from sklearn.metrics import accuracy_score


def get_on_fit_config(config: DictConfig, model_name: str):
    """Return function that prepares config to send to clients."""

    def fit_config_fn_scnn(server_round: int):
        # This function will be executed by the strategy in its
        # `configure_fit()` method.

        # Here we are returning the same config on each round but
        # here you might use the `server_round` input argument to
        # adapt over time these settings so clients. For example, you
        # might want clients to use a different learning rate at later
        # stages in the FL process (e.g. smaller lr after N rounds)

        return {
            "lr": config.lr,
            "momentum": config.momentum,
            "local_epochs": config.local_epochs,
            "batch_size": config.batch_size,
            "server_round": server_round,
            "is_malicious": False # Will be modified in strategy
        }

    def fit_config_fn_lgr(server_round: int):
        # This function will be executed by the strategy in its
        # `configure_fit()` method.

        # Here we are returning the same config on each round but
        # here you might use the `server_round` input argument to
        # adapt over time these settings so clients. For example, you
        # might want clients to use a different learning rate at later
        # stages in the FL process (e.g. smaller lr after N rounds)

        return {
            "penalty": config.penalty,
            "warm_start": config.warm_start,
            "local_epochs": config.local_epochs,
            "server_round": server_round,
            "is_malicious": False # Will be modified in strategy
        }

    def fit_config_fn_lsvc(server_round: int):
        # This function will be executed by the strategy in its
        # `configure_fit()` method.

        # Here we are returning the same config on each round but
        # here you might use the `server_round` input argument to
        # adapt over time these settings so clients. For example, you
        # might want clients to use a different learning rate at later
        # stages in the FL process (e.g. smaller lr after N rounds)

        return {
            "penalty": config.penalty,
            # "warm_start": config.warm_start,
            "local_epochs": config.local_epochs,
            "C": config.C,
            "server_round": server_round,
            "is_malicious": False # Will be modified in strategy
        }

    def fit_config_fn_mlp(server_round: int):
        # This function will be executed by the strategy in its
        # `configure_fit()` method.

        # Here we are returning the same config on each round but
        # here you might use the `server_round` input argument to
        # adapt over time these settings so clients. For example, you
        # might want clients to use a different learning rate at later
        # stages in the FL process (e.g. smaller lr after N rounds)

        return {
            "lr": config.lr,
            "local_epochs": config.local_epochs,
            "batch_size": config.batch_size,
            "server_round": server_round,
            "is_malicious": False # Will be modified in strategy
        }

    def fit_config_fn_xgb(server_round: int):
        # This function will be executed by the strategy in its
        # `configure_fit()` method.

        # Here we are returning the same config on each round but
        # here you might use the `server_round` input argument to
        # adapt over time these settings so clients. For example, you
        # might want clients to use a different learning rate at later
        # stages in the FL process (e.g. smaller lr after N rounds)
        on_fit_config = {
            "local_epochs": config.local_epochs,
            # "early_stopping": config.early_stopping,
            "eta": config.eta,
            "max_depth": config.max_depth,
            "subsample": config.subsample,
            "colsample_bytree": config.colsample_bytree,
            "objective": config.objective,
            "eval_metric": config.eval_metric,
            "alpha": config.alpha,
            "lambda": config.Lambda,
            "tree_method": config.tree_method,
            "device": config.device,
            "server_round": server_round,
            "is_malicious": False # Will be modified in strategy
        }
        # Random Forest Params
        if model_name == "RF":
            on_fit_config["num_parallel_tree"] = config.num_parallel_tree

        return on_fit_config

    if model_name == "SCNN":
        return fit_config_fn_scnn
    elif model_name == "LGR":
        return fit_config_fn_lgr
    elif model_name == "MLP":
        return fit_config_fn_mlp
    elif model_name == "LSVC":
        return fit_config_fn_lsvc
    elif model_name == "XGB" or model_name == "RF":
        return fit_config_fn_xgb
    else:
        return None


def get_evaluate_fn(num_classes: int, testset: Dataset, model_name: str):
    """Define function for global evaluation on the server."""

    def evaluate_fn_xgboost(server_round: int, parameters, config):
        # For round 1 skip evaluation
        if server_round == 0:
            return 0, {}
        else:
            params = {
                "num_class": 10,
                "eta": 0.08,
                "max_depth": 6,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "objective": "multi:softmax",
                "eval_metric": "mlogloss",
                "alpha": 8,
                "lambda": 2,
                "tree_method": "hist",
                "device": "cuda"
            }
            # Random Forest Params
            if model_name == "RF":
                params["num_parallel_tree"] = 100

            bst = xgb.Booster(params=params)
            para_b = None
            for para in parameters.tensors:
                para_b = bytearray(para)

            # Load global model
            bst.load_model(para_b)
            # Run evaluation
            X_test, y_test = get_data_numpy(DataLoader(testset))
            test_dmatrix = xgb.DMatrix(X_test, label=y_test)

            eval_results = bst.eval_set(
                evals=[(test_dmatrix, "valid")],
                iteration=bst.num_boosted_rounds() - 1,
            )

            mlogloss = float(eval_results.split("\t")[1].split(":")[1])

            # Making predictions
            y_pred = bst.predict(test_dmatrix)
            y_pred_classes = [round(value) for value in y_pred]

            # Calculating additional metrics
            accuracy = accuracy_score(y_test, y_pred_classes)
            precision = precision_score(y_test, y_pred_classes, average='weighted')
            recall = recall_score(y_test, y_pred_classes, average='weighted')
            f1 = f1_score(y_test, y_pred_classes, average='weighted')
            conf_matrix = confusion_matrix(y_test, y_pred_classes)

            return mlogloss, {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1,
                     "confusion_matrix": conf_matrix}

    def evaluate_fn_scnn(server_round: int, parameters, config):
        # This function is called by the strategy's `evaluate()` method
        # and receives as input arguments the current round number and the
        # parameters of the global model.
        # this function takes these parameters and evaluates the global model
        # on a evaluation / test dataset.

        model = SimpleCNN(num_classes)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

        # Here we evaluate the global model on the test set. Recall that in more
        # realistic settings you'd only do this at the end of your FL experiment
        # you can use the `server_round` input argument to determine if this is the
        # last round. If it's not, then preferably use a global validation set.

        # We leave the test set intact (i.e. we don't partition it)
        # This test set will be left on the server side and we'll be used to evaluate the
        # performance of the global model after each round.
        # Please note that a more realistic setting would instead use a validation set on the server for
        # this purpose and only use the testset after the final round.
        # Also, in some settings (specially outside simulation) it might not be feasible to construct a validation
        # set on the server side, therefore evaluating the global model can only be done by the clients. (see the comment
        # in main.py above the strategy definition for more details on this)
        testloader = DataLoader(testset, batch_size=128)
        loss, accuracy, precision, recall, f1, conf_matrix = test(model, testloader, device)

        # Report the loss and any other metric (inside a dictionary). In this case
        # we report the global test accuracy.
        return loss, {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1,
                      "confusion_matrix": conf_matrix}

    def evaluate_fn_mlp(server_round: int, parameters, config):
        # This function is called by the strategy's `evaluate()` method
        # and receives as input arguments the current round number and the
        # parameters of the global model.
        # this function takes these parameters and evaluates the global model
        # on a evaluation / test dataset.

        model = MLP(num_classes)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

        # Here we evaluate the global model on the test set. Recall that in more
        # realistic settings you'd only do this at the end of your FL experiment
        # you can use the `server_round` input argument to determine if this is the
        # last round. If it's not, then preferably use a global validation set.

        # We leave the test set intact (i.e. we don't partition it)
        # This test set will be left on the server side and we'll be used to evaluate the
        # performance of the global model after each round.
        # Please note that a more realistic setting would instead use a validation set on the server for
        # this purpose and only use the testset after the final round.
        # Also, in some settings (specially outside simulation) it might not be feasible to construct a validation
        # set on the server side, therefore evaluating the global model can only be done by the clients. (see the comment
        # in main.py above the strategy definition for more details on this)
        testloader = DataLoader(testset, batch_size=128)
        loss, accuracy, precision, recall, f1, conf_matrix = test(model, testloader, device)

        # Report the loss and any other metric (inside a dictionary). In this case
        # we report the global test accuracy.
        return loss, {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1,
                      "confusion_matrix": conf_matrix}

    def evaluate_fn_lgr(server_round: int, parameters, config):
        # This function is called by the strategy's `evaluate()` method
        # and receives as input arguments the current round number and the
        # parameters of the global model.
        # this function takes these parameters and evaluates the global model
        # on a evaluation / test dataset.

        model = LogisticRegression()

        model.classes_ = np.array([i for i in range(num_classes)])
        model.coef_ = parameters[0]
        if model.fit_intercept:
            model.intercept_ = parameters[1]

        # Here we evaluate the global model on the test set. Recall that in more
        # realistic settings you'd only do this at the end of your FL experiment
        # you can use the `server_round` input argument to determine if this is the
        # last round. If it's not, then preferably use a global validation set.

        # We leave the test set intact (i.e. we don't partition it)
        # This test set will be left on the server side and we'll be used to evaluate the
        # performance of the global model after each round.
        # Please note that a more realistic setting would instead use a validation set on the server for
        # this purpose and only use the testset after the final round.
        # Also, in some settings (specially outside simulation) it might not be feasible to construct a validation
        # set on the server side, therefore evaluating the global model can only be done by the clients. (see the comment
        # in main.py above the strategy definition for more details on this)
        testloader = DataLoader(testset)
        X_test, y_test = get_data_numpy(testloader)

        y_pred_prob = model.predict_proba(X_test)
        y_pred = model.predict(X_test)

        # Loss
        loss = log_loss(y_test, y_pred_prob)
        # Accuracy 
        accuracy = model.score(X_test, y_test)
        # Precision, Recall, F1_score
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        # Confusion Matrix
        conf_matrix = confusion_matrix(y_test, y_pred, labels=list(range(10)))

        # Report the loss and any other metric (inside a dictionary). In this case
        # we report the global test accuracy.
        return loss, {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1,
                      "confusion_matrix": conf_matrix}

    def evaluate_fn_lscv(server_round: int, parameters, config):
        # This function is called by the strategy's `evaluate()` method
        # and receives as input arguments the current round number and the
        # parameters of the global model.
        # this function takes these parameters and evaluates the global model
        # on a evaluation / test dataset.

        model = LinearSVC(dual=False)

        model.classes_ = np.array([i for i in range(num_classes)])
        model.coef_ = parameters[0]
        if model.fit_intercept:
            model.intercept_ = parameters[1]

        # Here we evaluate the global model on the test set. Recall that in more
        # realistic settings you'd only do this at the end of your FL experiment
        # you can use the `server_round` input argument to determine if this is the
        # last round. If it's not, then preferably use a global validation set.

        # We leave the test set intact (i.e. we don't partition it)
        # This test set will be left on the server side and we'll be used to evaluate the
        # performance of the global model after each round.
        # Please note that a more realistic setting would instead use a validation set on the server for
        # this purpose and only use the testset after the final round.
        # Also, in some settings (specially outside simulation) it might not be feasible to construct a validation
        # set on the server side, therefore evaluating the global model can only be done by the clients. (see the comment
        # in main.py above the strategy definition for more details on this)
        testloader = DataLoader(testset)
        X_test, y_test = get_data_numpy(testloader)


        descision_scores = model.decision_function(X_test)
        y_pred_prob = 1 / (1 + np.exp(-descision_scores))

        y_pred = model.predict(X_test)

        # Loss
        loss = log_loss(y_test, y_pred_prob)
        # Accuracy
        accuracy = model.score(X_test, y_test)
        # Precision, Recall, F1_score
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        # Confusion Matrix
        conf_matrix = confusion_matrix(y_test, y_pred, labels=list(range(10)))

        # Report the loss and any other metric (inside a dictionary). In this case
        # we report the global test accuracy.
        return loss, {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1,
                      "confusion_matrix": conf_matrix}

    if model_name == "SCNN":
        return evaluate_fn_scnn
    elif model_name == "LGR":
        return evaluate_fn_lgr
    elif model_name == "MLP":
        return evaluate_fn_mlp
    elif model_name == "LSVC":
        return evaluate_fn_lscv
    elif model_name == "XGB" or model_name == "RF":
        return evaluate_fn_xgboost
    else:
        return None
