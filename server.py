from collections import OrderedDict


from omegaconf import DictConfig

import torch

from model import SimpleCNN, MLP, test
from torch.utils.data import Dataset, DataLoader

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, precision_score, recall_score, f1_score, confusion_matrix

from dataset import get_data_numpy
import numpy as np



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
            "is_malicious": False
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
            "is_malicious": False
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
            "is_malicious": False
        }

    if model_name == "SCNN": 
        return fit_config_fn_scnn
    elif model_name == "LGR": 
        return fit_config_fn_lgr
    elif model_name == "MLP":
        return fit_config_fn_mlp
    else:
        return None


def get_evaluate_fn(num_classes: int, testset: Dataset, model_name: str):
    """Define function for global evaluation on the server."""

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
        return loss, {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, "confusion_matrix": conf_matrix}
    
    
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
        return loss, {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, "confusion_matrix": conf_matrix}
    
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
        return loss, {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, "confusion_matrix": conf_matrix}

    if model_name == "SCNN": 
        return evaluate_fn_scnn
    elif model_name == "LGR": 
        return evaluate_fn_lgr
    elif model_name == "MLP":
        return evaluate_fn_mlp
    else: 
        return None
