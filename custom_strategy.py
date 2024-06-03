import flwr as fl
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedXgbBagging, FedAvg
from flwr.server.strategy.fedxgb_bagging import aggregate as xgb_aggregate
from flwr.server.strategy.aggregate import aggregate, aggregate_inplace, parameters_to_ndarrays
from typing import Callable, Dict, List, Optional, Tuple, Any, Union, cast
from flwr.common.logger import log
from logging import WARNING
import pickle
from joblib import load
from flwr.common import (
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
)
from omegaconf import DictConfig
from torch.utils.data import Dataset
from server import get_on_fit_config, get_evaluate_fn
import pandas as pd
from sklearn.preprocessing import QuantileTransformer
from sklearn.svm import OneClassSVM


def make_malicious_clients(max_attack_ratio: float, client_instructions: list) -> list:
    num_malicious_clients = round(len(client_instructions) * max_attack_ratio)
    for i in range(num_malicious_clients):
        _, fit_ins = client_instructions[i]
        fit_ins.config["is_malicious"] = True

    return client_instructions


def prepare_malicious_clients(attack_round: str, num_rounds: int, server_round: int, attack_type: str,
                              max_attack_ratio: float, client_instructions: list, defence=False) -> List[Tuple[ClientProxy, FitIns]]:
    # TODO Convert this to first 30%, mid 30% and last 30% - The server rounds may increase
    if attack_round == "FULL":
        client_instructions = make_malicious_clients(max_attack_ratio, client_instructions)
    elif attack_round == "MID":
        if (num_rounds / 2 - 1 <= server_round <= num_rounds / 2 + 1):
            client_instructions = make_malicious_clients(max_attack_ratio, client_instructions)
    elif attack_round == "END":
        if (num_rounds - 2 <= server_round <= num_rounds):
            client_instructions = make_malicious_clients(max_attack_ratio, client_instructions)
    # Attack type is set in all clients (This is just for easier programming flow)
    for i in range(len(client_instructions)):
        _, fit_ins = client_instructions[i]
        fit_ins.config["attack_type"] = attack_type
        fit_ins.config["defence"] = defence
    return client_instructions


class MaliciousClientFedAvg(FedAvg):
    def __init__(
            self,
            *,
            fraction_fit: float = 1.0,
            fraction_evaluate: float = 1.0,
            min_fit_clients: int = 2,
            min_evaluate_clients: int = 2,
            min_available_clients: int = 2,
            evaluate_fn: Optional[
                Callable[
                    [int, NDArrays, Dict[str, Scalar]],
                    Optional[Tuple[float, Dict[str, Scalar]]],
                ]
            ] = None,
            on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
            on_evaluate_config_fn: Optional[Callable[[
                int], Dict[str, Scalar]]] = None,
            accept_failures: bool = True,
            initial_parameters: Optional[Parameters] = None,
            fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
            evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
            # This is the only variable added in the custom class
            max_attack_ratio: float = 0.3,
            attack_round="FULL",
            attack_type="LF",
            defence=False,
            num_rounds
    ) -> None:
        """Federated Averaging strategy.

        Implementation based on https://arxiv.org/abs/1602.05629

        Parameters
        ----------
        fraction_fit : float, optional
            Fraction of clients used during training. In case `min_fit_clients`
            is larger than `fraction_fit * available_clients`, `min_fit_clients`
            will still be sampled. Defaults to 1.0.
        fraction_evaluate : float, optional
            Fraction of clients used during validation. In case `min_evaluate_clients`
            is larger than `fraction_evaluate * available_clients`,
            `min_evaluate_clients` will still be sampled. Defaults to 1.0.
        min_fit_clients : int, optional
            Minimum number of clients used during training. Defaults to 2.
        min_evaluate_clients : int, optional
            Minimum number of clients used during validation. Defaults to 2.
        min_available_clients : int, optional
            Minimum number of total clients in the system. Defaults to 2.
        evaluate_fn : Optional[Callable[[int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]]]
            Optional function used for validation. Defaults to None.
        on_fit_config_fn : Callable[[int], Dict[str, Scalar]], optional
            Function used to configure training. Defaults to None.
        on_evaluate_config_fn : Callable[[int], Dict[str, Scalar]], optional
            Function used to configure validation. Defaults to None.
        accept_failures : bool, optional
            Whether or not accept rounds containing failures. Defaults to True.
        initial_parameters : Parameters, optional
            Initial global model parameters.
        fit_metrics_aggregation_fn : Optional[MetricsAggregationFn]
            Metrics aggregation function, optional.
        evaluate_metrics_aggregation_fn : Optional[MetricsAggregationFn]
            Metrics aggregation function, optional.
        """
        super().__init__(fraction_fit=fraction_fit,
                         fraction_evaluate=fraction_evaluate,
                         min_fit_clients=min_fit_clients,
                         min_evaluate_clients=min_evaluate_clients,
                         min_available_clients=min_available_clients,
                         evaluate_fn=evaluate_fn,
                         on_fit_config_fn=on_fit_config_fn,
                         on_evaluate_config_fn=on_evaluate_config_fn,
                         accept_failures=accept_failures,
                         initial_parameters=initial_parameters,
                         fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
                         evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn)
        self.max_attack_ratio = max_attack_ratio
        self.attack_round = attack_round
        self.attack_type = attack_type
        self.num_rounds = num_rounds
        self.defence = defence
        self.outlier_model = None

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager: ClientManager) -> List[
        Tuple[ClientProxy, FitIns]]:
        # return client_instructions
        """Configure the next round of training."""

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        client_instructions = []
        for client in clients:
            config = {}
            if self.on_fit_config_fn is not None:
                # Custom fit config function provided
                config = self.on_fit_config_fn(server_round)

            fit_ins = FitIns(parameters, config)
            client_instructions.append((client, fit_ins))

        return prepare_malicious_clients(attack_round=self.attack_round,
                                         num_rounds=self.num_rounds,
                                         server_round=server_round,
                                         attack_type=self.attack_type,
                                         max_attack_ratio=self.max_attack_ratio,
                                         client_instructions=client_instructions,
                                         defence=self.defence)

        # if self.attack_round == "FULL":
        #     client_instructions = self.make_malicious_clients(
        #         client_instructions)
        # elif self.attack_round == "MID":
        #     if (self.num_rounds / 2 - 1 <= server_round <= self.num_rounds / 2 + 1):
        #         client_instructions = self.make_malicious_clients(
        #             client_instructions)
        # elif self.attack_round == "END":
        #     if (self.num_rounds - 2 <= server_round <= self.num_rounds):
        #         client_instructions = self.make_malicious_clients(
        #             client_instructions)
        # # Attack type is set in all clients (This is just for easier programming flow)
        # for i in range(len(client_instructions)):
        #     _, fit_ins = client_instructions[i]
        #     fit_ins.config["attack_type"] = self.attack_type
        # return client_instructions

    # def make_malicious_clients(self, client_instructions: list) -> list:
    #     num_malicious_clients = int(len(client_instructions) * self.max_attack_ratio)
    #     for i in range(num_malicious_clients):
    #         _, fit_ins = client_instructions[i]
    #         fit_ins.config["is_malicious"] = True
    #
    #     return client_instructions
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""

        # _, fitRes_0 = results[0]
        # metrics = fitRes_0.metrics
        # output_file_name ="outlier_analysis/" + metrics["attack_type"] + "_" + str(server_round) + ".pkl"
        # outlier_data = [fitRes for _, fitRes in results]
        # with open(str(output_file_name), "wb") as h:
        #     pickle.dump(outlier_data, h, protocol=pickle.HIGHEST_PROTOCOL)
        if self.defence:
            o_b = len(results)
            results = remove_outliers(results=results)
            o_a = len(results)
            print("Number of Honest Clients=", o_a, "Number of Outliers=", o_b - o_a)

        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        if self.inplace:
            # Does in-place weighted average of results
            aggregated_ndarrays = aggregate_inplace(results)
        else:
            # Convert results
            weights_results = [
                (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
                for _, fit_res in results
            ]
            aggregated_ndarrays = aggregate(weights_results)

        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated

class MCFedXGBBagging(FedXgbBagging):
    """Configurable FedXgbBagging strategy implementation."""

    # pylint: disable=too-many-arguments,too-many-instance-attributes, line-too-long
    def __init__(
        self,
        evaluate_function: Optional[
            Callable[
                [int, Parameters, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]],
            ]
        ] = None,
        max_attack_ratio: float = 0.3,
        attack_round="FULL",
        attack_type="LF",
        num_rounds: int = 10,
        defence=False,
        **kwargs: Any,
    ):
        # self.evaluate_function = evaluate_function
        self.global_model: Optional[bytes] = None
        super().__init__(evaluate_function= evaluate_function, **kwargs)

        self.max_attack_ratio = max_attack_ratio
        self.attack_round = attack_round
        self.attack_type = attack_type
        self.num_rounds = num_rounds
        self.defence = defence
        self.outlier_model = None


    def configure_fit(self, server_round: int, parameters: Parameters, client_manager: ClientManager) -> List[
        Tuple[ClientProxy, FitIns]]:
        # return client_instructions
        """Configure the next round of training."""

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        client_instructions = []
        for client in clients:
            config = {}
            if self.on_fit_config_fn is not None:
                # Custom fit config function provided
                config = self.on_fit_config_fn(server_round)

            fit_ins = FitIns(parameters, config)
            client_instructions.append((client, fit_ins))

        return prepare_malicious_clients(attack_round=self.attack_round,
                                         num_rounds=self.num_rounds,
                                         server_round=server_round,
                                         attack_type=self.attack_type,
                                         max_attack_ratio=self.max_attack_ratio,
                                         client_instructions=client_instructions,
                                         defence=self.defence)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        # _, fitRes_0 = results[0]
        # metrics = fitRes_0.metrics
        # output_file_name ="outlier_analysis/" + metrics["attack_type"] + "_" + str(server_round) + ".pkl"
        # outlier_data = [fitRes for _, fitRes in results]
        # with open(str(output_file_name), "wb") as h:
        #     pickle.dump(outlier_data, h, protocol=pickle.HIGHEST_PROTOCOL)
        if self.defence:
            o_b = len(results)
            results = remove_outliers(results=results)
            o_a = len(results)
            print("Number of Honest Clients=", o_a, "Number of Outliers=", o_b-o_a)

        """Aggregate fit results using bagging."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Aggregate all the client trees
        global_model = self.global_model
        for _, fit_res in results:
            update = fit_res.parameters.tensors
            for bst in update:

                global_model = xgb_aggregate(global_model, bst)

        self.global_model = global_model

        return (
            Parameters(tensor_type="", tensors=[cast(bytes, global_model)]),
            {},
        )


def get_custom_strategy(model: str, cfg: DictConfig, testdataset: Dataset):
    if model == 'XGB' or model == "RF":
        if cfg.train_method == "bagging":
            # Bagging training
            strategy = MCFedXGBBagging(
                evaluate_function=get_evaluate_fn(cfg.num_classes, testdataset, cfg.model),
                fraction_fit=0.0,
                min_fit_clients=cfg.num_clients_per_round_fit,
                min_available_clients=cfg.num_clients,  # total clients in the simulation
                min_evaluate_clients=cfg.num_clients_per_round_eval,
                fraction_evaluate=0.0,
                on_evaluate_config_fn=get_on_fit_config(cfg.config_fit, cfg.model),
                on_fit_config_fn=get_on_fit_config(cfg.config_fit, cfg.model),
            # evaluate_metrics_aggregation_fn TODO?
                max_attack_ratio=cfg.max_attack_ratio,
                attack_round=cfg.attack_round,
                attack_type=cfg.attack_type,
                num_rounds=cfg.num_rounds,
                defence=cfg.defence
            )
            return strategy
        else:
            #Cyclic training
            return None

    strategy = MaliciousClientFedAvg(  # fl.server.strategy.FedAvg(
        fraction_fit=0.0,
        # in simulation, since all clients are available at all times, we can just use `min_fit_clients` to control exactly how many clients we want to involve during fit
        # number of clients to sample for fit()
        min_fit_clients=cfg.num_clients_per_round_fit,
        # similar to fraction_fit, we don't need to use this argument.
        fraction_evaluate=0.0,
        # number of clients to sample for evaluate()
        min_evaluate_clients=cfg.num_clients_per_round_eval,
        min_available_clients=cfg.num_clients,  # total clients in the simulation
        on_fit_config_fn=get_on_fit_config(
            cfg.config_fit, cfg.model
        ),  # a function to execute to obtain the configuration to send to the clients during fit()
        evaluate_fn=get_evaluate_fn(cfg.num_classes, testdataset, cfg.model),
        max_attack_ratio=cfg.max_attack_ratio,
        attack_round=cfg.attack_round,
        attack_type=cfg.attack_type,
        num_rounds=cfg.num_rounds,
        defence=cfg.defence
    )  # a function to run on the server side to evaluate the global model.
    return strategy



def remove_outliers(results:List[Tuple[ClientProxy, FitRes]]) -> List[Tuple[ClientProxy, FitRes]]:
    metrics = [get_metrics(fitres.metrics) for (_, fitres) in results]
    df = pd.DataFrame(metrics)

    features = ["loss","precision","recall","f1"]
    # features.extend([str(i) for i in range(0, 10)])

    qt = QuantileTransformer(output_distribution='uniform', random_state=0)
    df[features] = qt.fit_transform(df[features])
    X_test = df[features].values
    y_pred = OneClassSVM(nu=0.25, gamma='auto').fit_predict(X_test) # you should set nu value only if you know how many attackers might be present

    results_filtered = [item for item, pred in zip(results, y_pred) if pred != -1]
    return results_filtered


def get_class_wise_f1(metrics):
    conf_matrix = metrics["confusion_matrix"]
    num_classes = conf_matrix.shape[0]
    f1_scores = {}
    # Calculate precision, recall, and F1 score for each class
    for i in range(num_classes):
        tp = conf_matrix[i, i]  # True positive: diagonal element
        fp = conf_matrix[:, i].sum() - tp  # False positive: sum of column i minus tp
        fn = conf_matrix[i, :].sum() - tp  # False negative: sum of row i minus tp

        # Calculate precision and recall
        if tp + fp > 0:
            precision = tp / (tp + fp)
        else:
            precision = 0  # Avoid division by zero

        if tp + fn > 0:
            recall = tp / (tp + fn)
        else:
            recall = 0  # Avoid division by zero

        # Calculate F1 score
        if precision + recall > 0:
            f1_scores[str(i)] = 2 * (precision * recall) / (precision + recall)
        else:
            f1_scores[str(i)] = 0  # Avoid division by zero to prevent NaN values
    return f1_scores


def get_metrics(metrics) -> dict:
    result = dict()
    result["loss"] = metrics["loss"]

    result["f1"] = metrics["f1"]
    result["precision"] = metrics["precision"]
    result["recall"] = metrics["recall"]
    # result.update(get_class_wise_f1(metrics))
    # result["attack_type"] = metrics["attack_type"]
    # result["is_malicious"] = metrics["is_malicious"]
    return result
