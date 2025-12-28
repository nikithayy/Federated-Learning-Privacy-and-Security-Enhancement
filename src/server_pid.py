import os
import csv
from datetime import datetime, timezone
from typing import List, Tuple, Any
import numpy as np
import flwr as fl
from fl_server import CustomFedAvg


class FedAvgPID(fl.server.strategy.FedAvg):
    """FedAvg strategy with PID-based anomaly detection and CustomFedAvg logging."""

    def __init__(
        self,
        *args,
        Kp: float = 1.0,
        Ki: float = 0.05,
        Kd: float = 0.5,
        pid_threshold: float = 0.5,
        removal_log_dir: str = "./",
        results_dir: str = None,
        **kwargs,
    ):
        """Initialize PID parameters, logging, and removal tracking."""
        if results_dir is None:
            results_dir = removal_log_dir

        # Use CustomFedAvg for logging metrics (per-client and round)
        self.logger_strategy = CustomFedAvg(results_dir=results_dir, *args, **kwargs)

        # PID hyperparameters
        self.Kp = float(Kp)
        self.Ki = float(Ki)
        self.Kd = float(Kd)
        self.pid_threshold = float(pid_threshold)

        # State for PID calculation
        self.prev_dist = {}
        self.integral = {}
        self.removed_clients = set()

        # CSV for removed clients
        self.removal_csv = os.path.join(results_dir, "removed_clients.csv")
        os.makedirs(results_dir, exist_ok=True)
        if not os.path.exists(self.removal_csv):
            with open(self.removal_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "round", "client_name", "pid_score",
                                 "norm_distance", "accuracy", "reason"])

        super().__init__(*args, **kwargs)

    # -------------------------------
    # Delegated methods (logging & init)
    # -------------------------------
    def initialize_parameters(self, client_manager):
        """Initialize model parameters using CustomFedAvg."""
        return self.logger_strategy.initialize_parameters(client_manager)

    def evaluate(self, server_round, parameters):
        """Evaluate global model using CustomFedAvg logic."""
        return self.logger_strategy.evaluate(server_round, parameters)

    def configure_fit(self, server_round, parameters, client_manager):
        """Configure client selection and filter out removed clients."""
        config = self.logger_strategy.configure_fit(server_round, parameters, client_manager)
        # Filter out clients already removed
        if isinstance(config, tuple) and len(config) == 2:
            clients, fit_config = config
            filtered_clients = [c for c in clients if getattr(c, "cid", str(c)) not in self.removed_clients]
            return filtered_clients, fit_config
        return config

    # -------------------------------
    # Utility
    # -------------------------------
    def _l2_distance(self, w1: List[np.ndarray], w2: List[np.ndarray]) -> float:
        """Compute L2 distance between two weight sets."""
        return float(np.sqrt(sum(np.sum((a - b) ** 2) for a, b in zip(w1, w2))))

    def _write_removed(self, rows: List[List[Any]]):
        """Append removed client data to CSV log."""
        with open(self.removal_csv, "a", newline="") as f:
            csv.writer(f).writerows(rows)

    # -------------------------------
    # Aggregate fit with PID
    # -------------------------------
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures,
    ):
        """Aggregate client updates with PID-based filtering."""
        if not results:
            return self.logger_strategy.aggregate_fit(server_round, results, failures)

        timestamp = datetime.now(timezone.utc).isoformat()
        weights_with_counts, client_meta, original_pairs = [], [], []

        # Collect weights and metrics from clients
        for client_proxy, fit_res in results:
            metrics = fit_res.metrics or {}
            name = metrics.get("client_name") or getattr(client_proxy, "cid", None) or str(client_proxy)
            if name in self.removed_clients:
                continue  # Ignore previously removed clients

            nds = fl.common.parameters_to_ndarrays(fit_res.parameters)
            num_examples = int(fit_res.num_examples)
            acc = metrics.get("accuracy")
            loss = metrics.get("loss")

            weights_with_counts.append((nds, num_examples))
            client_meta.append((name, num_examples, acc, loss))
            original_pairs.append((client_proxy, fit_res))

        if not weights_with_counts:
            return self.logger_strategy.aggregate_fit(server_round, results, failures)

        # Compute global centroid
        total = sum(n for _, n in weights_with_counts)
        centroid = [
            sum(w[i] * n for w, n in weights_with_counts) / total
            for i in range(len(weights_with_counts[0][0]))
        ]

        # Normalized L2 distances
        distances = [self._l2_distance(nds, centroid) for nds, _ in weights_with_counts]
        mean_dist = np.mean(distances) if np.mean(distances) > 0 else 1.0
        norm_dists = [d / mean_dist for d in distances]

        # Mean accuracy for reference
        accs = [a for (_, _, a, _) in client_meta if a is not None]
        mean_acc = float(np.mean(accs)) if accs else 0.0

        kept_indices, removed_rows = [], []

        # Apply PID + accuracy filter only for poisoned clients
        for idx, ((nds, _), (name, _, acc, _)) in enumerate(zip(weights_with_counts, client_meta)):
            d = norm_dists[idx]
            self.integral[name] = self.integral.get(name, 0.0) + d
            P = self.Kp * d
            I = self.Ki * self.integral[name]
            D = self.Kd * (d - self.prev_dist.get(name, 0.0))
            pid_score = P + I + D
            self.prev_dist[name] = d

            remove_by_pid = pid_score > self.pid_threshold
            remove_by_acc = acc is not None and acc < mean_acc * 0.8

            # Only remove if both PID and low accuracy trigger (i.e., poisoned client)
            if remove_by_pid and remove_by_acc:
                removed_rows.append([timestamp, server_round, name, pid_score, d, acc, "PID+LowAcc"])
                self.removed_clients.add(name)
            else:
                kept_indices.append(idx)

        if removed_rows:
            self._write_removed(removed_rows)

        kept_results = [original_pairs[i] for i in kept_indices] or original_pairs
        return self.logger_strategy.aggregate_fit(server_round, kept_results, failures)

    # -------------------------------
    # Aggregate evaluate
    # -------------------------------
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]],
        failures: List[BaseException],
    ):
        """Aggregate evaluation results using CustomFedAvg."""
        return self.logger_strategy.aggregate_evaluate(server_round, results, failures)
