import multiprocessing
import time
import os
import random
import socket
from pathlib import Path
import shutil
from typing import List
import torch
torch.set_num_threads(1)

from fl_server import start_server, start_server_with_pid, RESULTS_DIR_DEFAULT, NO_ATTACK_SUBDIR, ATTACK_SUBDIR
from fl_client import start_client_process

# Paths
BASE_DIR = Path("Dataset_Derma/data_preprocessed")
CLEAN_DIR = BASE_DIR / "dermanist_clean"
POISON_DIR = BASE_DIR / "dermanist_poison_flip"
MODEL_PATH = "Part 3/src/initial_resnet18_pretrained.pth"

ROUNDS = 3
BATCH_SIZE = 10  # Number of clients to run at a time
CLIENT_TIMEOUT = 2000  # seconds


# ---------------- Server utils ----------------
def wait_for_server(address="localhost", port=8080, timeout=30):
    """
    Wait for the server to start and accept connections.
    Retries every 0.5 seconds until the timeout is reached.
    """
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.create_connection((address, port), timeout=2):
                print("[Runner] Server accepting connections")
                return True
        except OSError:
            time.sleep(0.5)
    raise TimeoutError(f"Server not available at {address}:{port}")


def start_server_process(num_rounds: int, results_dir_subpath: str, min_available_clients: int, use_pid: bool = False, pid_params: dict = None):
    """
    Start the server process to run the federated learning rounds.
    If use_pid is True, start a PID-controlled server version.
    Creates the result directory and waits for the server to initialize.
    """
    results_dir = os.path.join(RESULTS_DIR_DEFAULT, results_dir_subpath)
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)
    os.makedirs(results_dir, exist_ok=True)

    if pid_params is None:
        pid_params = {}

    if use_pid:
        target = start_server_with_pid
        kwargs = {
            "num_rounds": num_rounds,
            "results_dir": results_dir,
            "min_available_clients": min_available_clients,
        }
        kwargs.update({k: pid_params[k] for k in ("Kp", "Ki", "Kd", "pid_threshold") if k in pid_params})
    else:
        target = start_server
        kwargs = {
            "num_rounds": num_rounds,
            "results_dir": results_dir,
            "min_available_clients": min_available_clients,
        }

    proc = multiprocessing.Process(target=target, kwargs=kwargs)
    proc.start()
    print("[Runner] Waiting for server to initialize...")
    wait_for_server()
    return proc


# ---------------- Client utils ----------------
def _extract_client_id_from_path(p: Path) -> int:
    """
    Extract client ID from the folder name (e.g., 'Client_1' → 1).
    """
    stem = p.stem
    if "_" not in stem:
        raise ValueError(f"Unexpected client folder name '{stem}'. Expected format 'Client_<id>'.")
    return int(stem.split("_")[-1])


def run_client_batch(batch: List[Path], poisoned: bool):
    """
    Run a batch of clients (clean or poisoned) in parallel processes.
    Each client runs as an independent process and is terminated if hung.
    """
    procs = []
    for p in batch:
        cid = _extract_client_id_from_path(p)
        proc = multiprocessing.Process(
            target=start_client_process,
            args=(cid, poisoned, str(BASE_DIR), MODEL_PATH)
        )
        proc.start()
        procs.append(proc)
        time.sleep(0.5)  # Stagger client starts slightly

    # Wait for batch completion
    for proc in procs:
        proc.join(timeout=CLIENT_TIMEOUT)
        if proc.is_alive():
            print(f"[Runner] Terminating hung client PID={proc.pid}")
            proc.terminate()
            proc.join(timeout=5)


def run_phase(client_folders: List[Path], poisoned: bool, results_subdir: str, use_pid: bool = False, pid_params: dict = None):
    """
    Run one complete federated learning phase (clean or attack).
    Launches the server, executes all clients in batches, and saves results.
    """
    if not client_folders:
        raise FileNotFoundError(f"No Client_* folders found in {client_folders}")

    server_proc = start_server_process(
        num_rounds=ROUNDS,
        results_dir_subpath=results_subdir,
        min_available_clients=BATCH_SIZE,
        use_pid=use_pid,
        pid_params=pid_params,
    )

    try:
        run_client_batch(client_folders, poisoned)
        time.sleep(2)
    finally:
        if server_proc.is_alive():
            server_proc.terminate()
            server_proc.join(timeout=5)
        print(f"[Runner] Phase complete. Results in {os.path.join(RESULTS_DIR_DEFAULT, results_subdir)}")


def run_experiment():
    """
    Main function to run the full experiment.
    Selects specific clients for poisoning, shuffles the rest,
    and runs both clean and attack (PID-controlled) phases.
    """
    random.seed(42)
    all_client_folders = sorted([p for p in CLEAN_DIR.glob("Client_*")])

    if not all_client_folders:
        raise FileNotFoundError(f"No Client_* folders found in {CLEAN_DIR}")

    selected_client_ids = {1, 2, 12, 15}
    selected_clients = [p for p in all_client_folders if _extract_client_id_from_path(p) in selected_client_ids]
    remaining_clients = [p for p in all_client_folders if _extract_client_id_from_path(p) not in selected_client_ids]

    random.shuffle(remaining_clients)
    all_client_folders = selected_clients + remaining_clients
    clean_batch = all_client_folders[:BATCH_SIZE]
    poisoned_batch = all_client_folders[:BATCH_SIZE]

    # Phase 1: Clean clients (no PID control)
    run_phase(clean_batch, poisoned=False, results_subdir=NO_ATTACK_SUBDIR, use_pid=False)

    # Phase 2: Poisoned clients (PID-controlled)
    pid_cfg = {"Kp": 1.0, "Ki": 0.05, "Kd": 0.5, "pid_threshold": 0.5}
    run_phase(poisoned_batch, poisoned=True, results_subdir=ATTACK_SUBDIR, use_pid=True, pid_params=pid_cfg)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    run_experiment()
