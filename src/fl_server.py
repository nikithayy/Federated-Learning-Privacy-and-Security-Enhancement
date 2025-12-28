import os
import csv
from datetime import datetime, timezone
from typing import List, Tuple, Any
from pathlib import Path
import numpy as np
import flwr as fl

# -------------------------------
# Default directories
# -------------------------------
RESULTS_DIR_DEFAULT = Path(__file__).resolve().parent.parent / "results"
NO_ATTACK_SUBDIR = "no_attack"
ATTACK_SUBDIR = "attack"


# -------------------------------
# Custom FedAvg strategy
# -------------------------------
class CustomFedAvg(fl.server.strategy.FedAvg):
    """
    Custom Flower FedAvg strategy with per-client and per-round metrics logging.
    Handles creation and writing of CSV logs for each training round.
    """
    def __init__(self, *args, results_dir: str = RESULTS_DIR_DEFAULT, **kwargs):
        """
        Initializes the custom FedAvg strategy and prepares CSV files for metrics logging.
        """
        super().__init__(*args, **kwargs)
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)

        # CSV file paths
        self.per_client_csv = os.path.join(self.results_dir, "per_client_metrics.csv")
        self.round_csv = os.path.join(self.results_dir, "round_metrics.csv")

        # Create CSV headers if files don't exist
        if not os.path.exists(self.per_client_csv):
            with open(self.per_client_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "round", "client_id", "num_examples", "loss", "accuracy"])

        if not os.path.exists(self.round_csv):
            with open(self.round_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "round", "avg_accuracy", "avg_loss", "total_examples"])

    # ---------------------------
    # Helpers to write CSVs
    # ---------------------------
    def _append_per_client(self, rows: List[List[Any]]):
        """
        Appends per-client metrics to CSV.
        """
        with open(self.per_client_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(rows)

    def _append_round(self, row: List[Any]):
        """
        Appends round-level aggregated metrics to CSV.
        """
        with open(self.round_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)

    # ---------------------------
    # Weighted averaging of client weights
    # ---------------------------
    def weighted_average(self, weights_results: List[Tuple[List[np.ndarray], int]]) -> List[np.ndarray]:
        """
        Performs weighted averaging of model weights (FedAvg).
        """
        if not weights_results:
            return []

        counts = [c for _, c in weights_results]
        total = float(sum(counts))

        weighted = None
        for ndarrays, c in weights_results:
            if weighted is None:
                weighted = [np.array(layer) * c for layer in ndarrays]
            else:
                for i, layer in enumerate(ndarrays):
                    weighted[i] += np.array(layer) * c

        avg = [w / total for w in weighted]
        return avg

    # ---------------------------
    # Aggregate Fit Results
    # ---------------------------
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ):
        """
        Aggregates training results from clients and logs metrics.
        """
        if not results:
            print(f"[Server] No fit results for round {server_round}")
            return None, {}

        timestamp = datetime.now(timezone.utc).isoformat()
        per_client_rows = []
        metrics_list = []
        weights_results = []
        total_examples = 0

        for client_proxy, fit_res in results:
            num_examples = fit_res.num_examples
            total_examples += num_examples
            metrics = fit_res.metrics or {}
            loss = metrics.get("loss")
            accuracy = metrics.get("accuracy")

            try:
                client_name = metrics.get("client_name")
                if not client_name:
                    client_name = getattr(client_proxy, "cid", None)
                if client_name is None:
                    client_name = getattr(client_proxy.transport, "_identity", None)
                if client_name is None:
                    client_name = str(client_proxy)
            except Exception:
                client_name = str(client_proxy)

            per_client_rows.append([timestamp, server_round, client_name, int(num_examples),
                                    float(loss) if loss is not None else None,
                                    float(accuracy) if accuracy is not None else None])
            metrics_list.append({"loss": loss, "accuracy": accuracy, "num_examples": num_examples})

            # Convert parameters to ndarrays
            ndarrays = fl.common.parameters_to_ndarrays(fit_res.parameters)
            weights_results.append((ndarrays, num_examples))

            print(f"[Server] Round {server_round} fit from client {client_name}: "
                  f"examples={num_examples}, loss={loss}, acc={accuracy}")

        # Save per-client metrics
        self._append_per_client(per_client_rows)

        # Compute averages
        losses = [m["loss"] for m in metrics_list if m["loss"] is not None]
        accs = [m["accuracy"] for m in metrics_list if m["accuracy"] is not None]
        avg_loss = float(np.mean(losses)) if losses else None
        avg_acc = float(np.mean(accs)) if accs else None

        # Save round metrics
        self._append_round([timestamp, server_round, avg_acc, avg_loss, total_examples])
        print(f"[Server] Round {server_round} aggregated: total_examples={total_examples}, "
              f"avg_acc={avg_acc}, avg_loss={avg_loss}")

        # Aggregate weights (FedAvg style)
        aggregated_weights = self.weighted_average(weights_results)
        aggregated_parameters = fl.common.ndarrays_to_parameters(aggregated_weights)

        return aggregated_parameters, {}

    # ---------------------------
    # Aggregate Evaluation Results
    # ---------------------------
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]],
        failures: List[BaseException],
    ):
        """
        Aggregates evaluation results from clients.
        """
        if not results:
            print(f"[Server] No evaluation results for round {server_round}")
            return None, {}

        timestamp = datetime.now(timezone.utc).isoformat()
        eval_losses = []
        eval_accs = []
        total_examples = 0

        for client_proxy, eval_res in results:
            num_examples = eval_res.num_examples
            total_examples += num_examples
            metrics = eval_res.metrics or {}
            loss = metrics.get("loss")
            accuracy = metrics.get("accuracy")

            # Prefer client_name from metrics
            try:
                client_name = metrics.get("client_name")
                if not client_name:
                    client_name = getattr(client_proxy, "cid", None)
                if client_name is None:
                    client_name = getattr(client_proxy.transport, "_identity", None)
                if client_name is None:
                    client_name = str(client_proxy)
            except Exception:
                client_name = str(client_proxy)

            if loss is not None:
                eval_losses.append(loss)
            if accuracy is not None:
                eval_accs.append(accuracy)

        avg_loss = float(np.mean(eval_losses)) if eval_losses else None
        avg_acc = float(np.mean(eval_accs)) if eval_accs else None

        return None, {}


# -------------------------------
# Start Flower Server (Normal FL)
# -------------------------------
def start_server(
    min_available_clients: int = 10,
    results_dir: str = os.path.join(RESULTS_DIR_DEFAULT, NO_ATTACK_SUBDIR),
    num_rounds: int = 3,
):
    """
    Starts the federated learning server with a custom strategy.
    """
    print(f"[Server] Starting Flower server (results -> {results_dir})")

    strategy = CustomFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=min_available_clients,
        min_evaluate_clients=min_available_clients,
        min_available_clients=min_available_clients,
        results_dir=results_dir
    )

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
    )

    print("[Server] Flower server finished.")


# -------------------------------
# Start Flower Server (PID-Controlled)
# -------------------------------
def start_server_with_pid(
    min_available_clients: int = 10,
    results_dir: str = os.path.join(RESULTS_DIR_DEFAULT, ATTACK_SUBDIR),
    num_rounds: int = 3,
    Kp: float = 1.0,
    Ki: float = 0.05,
    Kd: float = 0.5,
    pid_threshold: float = 0.7,
):
    """
    Starts a Flower server using the FedAvgPID strategy for attack scenarios (Part 3).
    Includes PID-based client filtering based on behavior over training rounds.
    """
    print(f"[Server PID] Starting Flower server with PID filtering (results -> {results_dir})")

    # Import locally to prevent circular dependency issues
    from server_pid import FedAvgPID

    strategy = FedAvgPID(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=min_available_clients,
        min_evaluate_clients=min_available_clients,
        min_available_clients=min_available_clients,
        Kp=Kp, Ki=Ki, Kd=Kd, pid_threshold=pid_threshold,
        results_dir=results_dir,
        removal_log_dir=results_dir,
    )

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
    )

    print("[Server PID] Flower server finished.")
