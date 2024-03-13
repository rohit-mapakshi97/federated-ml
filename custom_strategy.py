import flwr as fl
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedXgbBagging, FedAvg
from typing import Callable, Dict, List, Optional, Tuple, Any
from flwr.common import (
    FitIns,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
)
from omegaconf import DictConfig
from torch.utils.data import Dataset
from server import get_on_fit_config, get_evaluate_fn


def make_malicious_clients(max_attack_ratio: float, client_instructions: list) -> list:
    num_malicious_clients = int(len(client_instructions) * max_attack_ratio)
    for i in range(num_malicious_clients):
        _, fit_ins = client_instructions[i]
        fit_ins.config["is_malicious"] = True

    return client_instructions


def prepare_malicious_clients(attack_round: str, num_rounds: int, server_round: int, attack_type: str,
                              max_attack_ratio: float, client_instructions: list) -> List[Tuple[ClientProxy, FitIns]]:
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
                                         client_instructions=client_instructions)

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
        **kwargs: Any,
    ):
        # self.evaluate_function = evaluate_function
        self.global_model: Optional[bytes] = None
        super().__init__(evaluate_function= evaluate_function, **kwargs)

        self.max_attack_ratio = max_attack_ratio
        self.attack_round = attack_round
        self.attack_type = attack_type
        self.num_rounds = num_rounds

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
                                         client_instructions=client_instructions)


def get_custom_strategy(model: str, cfg: DictConfig, testdataset: Dataset):
    if model == 'XGB':
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
                num_rounds=cfg.num_rounds
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
        num_rounds=cfg.num_rounds
    )  # a function to run on the server side to evaluate the global model.
    return strategy
