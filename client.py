from collections import OrderedDict
from typing import Dict, Tuple
from flwr.common import NDArrays, Scalar

import torch
from torch.utils.data import Dataset, DataLoader
import flwr as fl

from model import SimpleCNN, MLP, train, test
from typing import List
from attacks import label_flipping_attack, targeted_label_flipping_attack, gan_attack, partial_dataset_for_GAN_attack
from dataset import get_data_numpy

from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
from sklearn.metrics import log_loss, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import warnings

import time


def generate_client_fn(traindataset_list: List[Dataset], valdataset_list: List[Dataset], num_classes: int, model: str):
    """Return a function that can be used by the VirtualClientEngine.

    to spawn a FlowerClient with client id `cid`.
    """

    def client_fn_scnn(cid: str):
        # This function will be called internally by the VirtualClientEngine
        # Each time the cid-th client is told to participate in the FL
        # simulation (whether it is for doing fit() or evaluate())

        # Returns a normal FLowerClient that will use the cid-th train/val
        # dataloaders as it's local data.
        return FlowerClientSCNN(
            traindataset=traindataset_list[int(cid)],
            valdataset=valdataset_list[int(cid)],
            num_classes=num_classes,
        )

    def client_fn_lgr(cid: str):
        # This function will be called internally by the VirtualClientEngine
        # Each time the cid-th client is told to participate in the FL
        # simulation (whether it is for doing fit() or evaluate())

        # Returns a normal FLowerClient that will use the cid-th train/val
        # dataloaders as it's local data.
        return FlowerClientLGR(
            traindataset=traindataset_list[int(cid)],
            valdataset=valdataset_list[int(cid)],
            num_classes=num_classes,
            num_features=28 * 28  # TODO configurable?
        )

    def client_fn_mlp(cid: str):
        # This function will be called internally by the VirtualClientEngine
        # Each time the cid-th client is told to participate in the FL
        # simulation (whether it is for doing fit() or evaluate())

        # Returns a normal FLowerClient that will use the cid-th train/val
        # dataloaders as it's local data.
        return FlowerClientMLP(
            traindataset=traindataset_list[int(cid)],
            valdataset=valdataset_list[int(cid)],
            num_classes=num_classes
        )

    # Control logic for other models
    # return the function to spawn client
    if model == "SCNN":
        return client_fn_scnn
    elif model == "LGR":
        return client_fn_lgr
    elif model == "MLP":
        return client_fn_mlp
    else:
        return None


class FlowerClientSCNN(fl.client.NumPyClient):
    """Define a Flower Client."""

    def __init__(self, traindataset: Dataset, valdataset: Dataset, num_classes: int) -> None:
        super().__init__()

        # the dataloaders that point to the data associated to this client
        self.trainDataset = traindataset
        self.valdataset = valdataset

        # a model that is randomly initialised at first
        self.model = SimpleCNN(num_classes)
        self.num_classes = num_classes

        # figure out if this client has access to GPU support or not
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

    def set_parameters(self, parameters):
        """Receive parameters and apply them to the local model."""
        params_dict = zip(self.model.state_dict().keys(), parameters)

        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})

        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config: Dict[str, Scalar]):
        """Extract model parameters and return them as a list of numpy arrays."""

        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        """Train model received by the server (parameters) using the data.

        that belongs to this client. Then, send it back to the server.
        """
        # Poison the dataset if the client is malicious
        self.trainDataset = applyAttacks(self.trainDataset, config)

        # copy parameters sent by the server into client's local model
        self.set_parameters(parameters)

        # fetch elements in the config sent by the server. Note that having a config
        # sent by the server each time a client needs to participate is a simple but
        # powerful mechanism to adjust these hyperparameters during the FL process. For
        # example, maybe you want clients to reduce their LR after a number of FL rounds.
        # or you want clients to do more local epochs at later stages in the simulation
        # you can control these by customising what you pass to `on_fit_config_fn` when
        # defining your strategy.
        lr = config["lr"]
        momentum = config["momentum"]
        epochs = config["local_epochs"]

        # a very standard looking optimiser
        optim = torch.optim.SGD(self.model.parameters(),
                                lr=lr, momentum=momentum)

        # do local training. This function is identical to what you might
        # have used before in non-FL projects. For more advance FL implementation
        # you might want to tweak it but overall, from a client perspective the "local
        # training" can be seen as a form of "centralised training" given a pre-trained
        # model (i.e. the model received from the server)
        trainloader = DataLoader(
            self.trainDataset, batch_size=config["batch_size"], shuffle=True, num_workers=2)
        train(self.model, trainloader, optim, epochs, self.device)

        # Flower clients need to return three arguments: the updated model, the number
        # of examples in the client (although this depends a bit on your choice of aggregation
        # strategy), and a dictionary of metrics (here you can add any additional data, but these
        # are ideally small data structures)
        return self.get_parameters({}), len(trainloader), {}

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        self.set_parameters(parameters)

        valloader = DataLoader(
            self.valdataset, batch_size=config["batch_size"], shuffle=True, num_workers=2)
        loss, accuracy, precision, recall, f1, conf_matrix = test(self.model, valloader, self.device)

        return float(loss), len(valloader), {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1,
                                             "confusion_matrix": conf_matrix}


class FlowerClientLGR(fl.client.NumPyClient):
    '''Define a Flower Client'''

    def __init__(self, traindataset: Dataset, valdataset: Dataset, num_classes: int, num_features) -> None:
        super().__init__()

        # the dataloaders that point to the data associated to this client
        self.traindataset = traindataset
        self.valdataset = valdataset

        # a model that is randomly initialised at first
        self.model = LogisticRegression()
        self.num_classes = num_classes
        self.num_features = num_features

        self.model.classes_ = np.array([i for i in range(num_classes)])
        self.model.coef_ = np.zeros((num_classes, self.num_features))
        if self.model.fit_intercept:
            self.model.intercept_ = np.zeros((num_classes,))

    def set_parameters(self, parameters):
        """Receive parameters and apply them to the local model."""
        """Sets the parameters of a sklean LogisticRegression model."""
        self.model.coef_ = parameters[0]
        if self.model.fit_intercept:
            self.model.intercept_ = parameters[1]

    def get_parameters(self, config: Dict[str, Scalar]):
        """Extract model parameters and return them as a list of numpy arrays."""
        if self.model.fit_intercept:
            params = [self.model.coef_, self.model.intercept_]
        else:
            params = [self.model.coef_, ]
        return params

    def fit(self, parameters, config):
        """Train model received by the server (parameters) using the data.

        that belongs to this client. Then, send it back to the server.
        """
        # Poison the dataset if the client is malicious
        self.traindataset = applyAttacks(self.traindataset, config, model="LGR")

        # copy parameters sent by the server into client's local model
        self.set_parameters(parameters)

        # fetch elements in the config sent by the server. Note that having a config
        # sent by the server each time a client needs to participate is a simple but
        # powerful mechanism to adjust these hyperparameters during the FL process. For
        # example, maybe you want clients to reduce their LR after a number of FL rounds.
        # or you want clients to do more local epochs at later stages in the simulation
        # you can control these by customising what you pass to `on_fit_config_fn` when
        # defining your strategy.
        penalty = config["penalty"]
        warm_start = config["warm_start"]
        epochs = config["local_epochs"]

        self.model.penalty = penalty
        self.model.warm_start = warm_start
        self.model.max_iter = epochs

        # do local training. This function is identical to what you might
        # have used before in non-FL projects. For more advance FL implementation
        # you might want to tweak it but overall, from a client perspective the "local
        # training" can be seen as a form of "centralised training" given a pre-trained
        # model (i.e. the model received from the server)
        trainloader = DataLoader(self.traindataset)
        # Convert to numpy data 
        X_train, y_train = get_data_numpy(trainloader)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(X_train, y_train)
            # print(f"Training finished for round {config['server_round']}")

        # Flower clients need to return three arguments: the updated model, the number
        # of examples in the client (although this depends a bit on your choice of aggregation
        # strategy), and a dictionary of metrics (here you can add any additional data, but these
        # are ideally small data structures)
        return self.get_parameters({}), len(X_train), {}

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        self.set_parameters(parameters)

        valloader = DataLoader(self.valdataset)
        X_test, y_test = get_data_numpy(valloader)

        y_pred_prob = self.model.predict_proba(X_test)
        y_pred = self.model.predict(X_test)

        loss = log_loss(y_test, y_pred_prob)
        accuracy = self.model.score(X_test, y_test)

        # Precision, Recall, F1_score
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        # Confusion Matrix
        conf_matrix = confusion_matrix(y_test, y_pred, labels=list(range(10)))

        return float(loss), len(X_test), {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1,
                                          "confusion_matrix": conf_matrix}


class FlowerClientMLP(fl.client.NumPyClient):
    """Define a Flower Client."""

    def __init__(self, traindataset: Dataset, valdataset: Dataset, num_classes: int) -> None:
        super().__init__()

        # the dataloaders that point to the data associated to this client
        self.traindataset = traindataset
        self.valdataset = valdataset

        # a model that is randomly initialised at first
        self.model = MLP(num_classes)
        self.num_classes = num_classes

        # figure out if this client has access to GPU support or not
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

    def set_parameters(self, parameters):
        """Receive parameters and apply them to the local model."""
        params_dict = zip(self.model.state_dict().keys(), parameters)

        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})

        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config: Dict[str, Scalar]):
        """Extract model parameters and return them as a list of numpy arrays."""

        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        """Train model received by the server (parameters) using the data.

        that belongs to this client. Then, send it back to the server.
        """
        # Poison the dataset if the client is malicious
        self.traindataset = applyAttacks(trainset=self.traindataset, config=config)

        # copy parameters sent by the server into client's local model
        self.set_parameters(parameters)

        # fetch elements in the config sent by the server. Note that having a config
        # sent by the server each time a client needs to participate is a simple but
        # powerful mechanism to adjust these hyperparameters during the FL process. For
        # example, maybe you want clients to reduce their LR after a number of FL rounds.
        # or you want clients to do more local epochs at later stages in the simulation
        # you can control these by customising what you pass to `on_fit_config_fn` when
        # defining your strategy.
        lr = config["lr"]
        epochs = config["local_epochs"]

        # a very standard looking optimiser
        optim = torch.optim.Adam(self.model.parameters(), lr=lr)

        # do local training. This function is identical to what you might
        # have used before in non-FL projects. For more advance FL implementation
        # you might want to tweak it but overall, from a client perspective the "local
        # training" can be seen as a form of "centralised training" given a pre-trained
        # model (i.e. the model received from the server)
        trainloader = DataLoader(
            self.traindataset, batch_size=config["batch_size"], shuffle=True, num_workers=2)
        train(self.model, trainloader, optim, epochs, self.device)

        # Flower clients need to return three arguments: the updated model, the number
        # of examples in the client (although this depends a bit on your choice of aggregation
        # strategy), and a dictionary of metrics (here you can add any additional data, but these
        # are ideally small data structures)
        return self.get_parameters({}), len(trainloader), {}

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        self.set_parameters(parameters)

        valloader = DataLoader(
            self.valdataset, batch_size=config["batch_size"], shuffle=True, num_workers=2)
        loss, accuracy, precision, recall, f1, conf_matrix = test(self.model, valloader, self.device)

        return float(loss), len(valloader), {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1,
                                             "confusion_matrix": conf_matrix}


def applyAttacks(trainset: Dataset, config, model: str = None) -> Dataset:
    # NOTE: this attack ratio is different, This is for number of samples to attack.
    ## The one in the config file is to select number of malicious clients

    if config["attack_type"] == "TLF":
        if config["is_malicious"]:
            print("----------------------------------Dataset Attacked------------------------------")
            return targeted_label_flipping_attack(trainset=trainset, attack_ratio=1.0)
    elif config["attack_type"] == "GAN":
        if config["is_malicious"]:
            print("----------------------------------Dataset Attacked------------------------------")
            return gan_attack(trainset=trainset)  # Change this if the program crashes
        # LGR model needs samples for all labels
        if model != "LGR":
            return partial_dataset_for_GAN_attack(trainset=trainset)
    else:
        if config["is_malicious"]:
            print("----------------------------------Dataset Attacked------------------------------")
            return label_flipping_attack(dataset=trainset, num_classes=10, attack_ratio=1.0)

    return trainset
