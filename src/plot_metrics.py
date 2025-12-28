import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Directories
RESULTS_DIR = "Part 3/results"
NO_ATTACK_DIR = os.path.join(RESULTS_DIR, "no_attack")
ATTACK_DIR = os.path.join(RESULTS_DIR, "attack")
NO_ATTACK_PLOTS = os.path.join(NO_ATTACK_DIR, "plots")
ATTACK_PLOTS = os.path.join(ATTACK_DIR, "plots")

# Make sure subfolders exist
os.makedirs(NO_ATTACK_PLOTS, exist_ok=True)
os.makedirs(ATTACK_PLOTS, exist_ok=True)

FLIP_MAP = {1: 0.1, 12: 0.25, 15: 0.5, 2: 0.75}


def ensure_client_col(df):
    """Return correct column for client name/id."""
    if "client_id" in df.columns:
        return "client_id"
    elif "client_name" in df.columns:
        return "client_name"
    raise ValueError("Neither client_id nor client_name found in CSV columns.")


# -------------------------------------------------------
# Generic plotting functions
# -------------------------------------------------------
def plot_per_client_history(df, metric, phase, ylabel, save_dir):
    """Plot history of a metric per client (line per client)."""
    client_col = ensure_client_col(df)

    plt.figure(figsize=(10, 6))
    for name, group in df.groupby(client_col):
        plt.plot(group["round"], group[metric], label=name, alpha=0.7)
    plt.title(f"History of {ylabel} per Client over Rounds ({phase.replace('_', ' ').title()})")
    plt.xlabel("Round")
    plt.ylabel(ylabel)
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), ncol=2)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{metric}_per_client.png"))
    plt.close()


def plot_average_history(df_round, metric, phase, ylabel, save_dir):
    """Plot average metric (loss or accuracy) over rounds."""
    plt.figure(figsize=(8, 5))
    plt.plot(df_round["round"], df_round[metric], marker="o")
    plt.title(f"Average {ylabel} over Rounds ({phase.replace('_', ' ').title()})")
    plt.xlabel("Round")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"avg_{metric}.png"))
    plt.close()


def plot_attack_severity_relation(per_client_csv, save_dir):
    """Plot final loss and accuracy vs flip rate (attack severity)."""
    df = pd.read_csv(per_client_csv)
    client_col = ensure_client_col(df)
    last_round = df[df["round"] == df["round"].max()]

    # Map flip rates to client names
    severity_data = []
    for _, row in last_round.iterrows():
        cname = str(row[client_col])
        # Extract numeric part (Client_1 → 1)
        try:
            cid = int(''.join([c for c in cname if c.isdigit()]))
        except ValueError:
            cid = None
        flip_rate = FLIP_MAP.get(cid, 0.0)
        severity_data.append({
            "client": cname,
            "flip_rate": flip_rate,
            "accuracy": row.get("accuracy"),
            "loss": row.get("loss"),
        })

    df_severity = pd.DataFrame(severity_data)

    # Accuracy vs flip rate
    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=df_severity, x="flip_rate", y="accuracy", s=80)
    plt.title("Final Accuracy vs Flip Rate (Attack Severity)")
    plt.xlabel("Flipping Rate")
    plt.ylabel("Final Accuracy")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "attack_accuracy_vs_fliprate.png"))
    plt.close()

    # Loss vs flip rate
    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=df_severity, x="flip_rate", y="loss", s=80)
    plt.title("Final Loss vs Flip Rate (Attack Severity)")
    plt.xlabel("Flipping Rate")
    plt.ylabel("Final Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "attack_loss_vs_fliprate.png"))
    plt.close()


def plot_removed_clients(removal_csv_path, save_dir):
    """Plot removal dynamics: number of removed clients per round and which clients were removed."""
    if not os.path.exists(removal_csv_path):
        print(f"[Plotter] No removal CSV at {removal_csv_path} (skipping)")
        return

    df = pd.read_csv(removal_csv_path)
    if df.empty:
        print("[Plotter] removal CSV is empty (skipping)")
        return

    # Ensure required columns
    for col in ("round", "client_name", "pid_score"):
        if col not in df.columns:
            print(f"[Plotter] removal CSV missing column '{col}' (skip)")
            return

    # Number removed per round
    removed_count = df.groupby("round").size().reset_index(name="num_removed")

    plt.figure(figsize=(8, 4))
    plt.bar(removed_count["round"], removed_count["num_removed"])
    plt.xlabel("Round")
    plt.ylabel("Number of removed clients")
    plt.title("Removed Clients per Round (PID)")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "removed_clients_count_per_round.png"))
    plt.close()

    # Which clients removed by round (scatter)
    df["round"] = df["round"].astype(int)
    df["client_short"] = df["client_name"].astype(str)

    plt.figure(figsize=(10, 5))
    for _, row in df.iterrows():
        plt.scatter(row["round"], row["client_short"], s=80)
    plt.xlabel("Round")
    plt.ylabel("Client")
    plt.title("Clients Removed by PID per Round")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "removed_clients_by_round.png"))
    plt.close()


# -------------------------------------------------------
# MAIN DRIVER
# -------------------------------------------------------
def main():
    print("[Plotter] Generating graphs for both phases...")

    # Paths
    clean_clients = "Part 3/results/no_attack/per_client_metrics.csv"
    clean_round = "Part 3/results/no_attack/round_metrics.csv"
    attack_clients = "Part 3/results/attack/per_client_metrics.csv"
    attack_round = "Part 3/results/attack/round_metrics.csv"

    # Validate presence
    if not all(os.path.exists(p) for p in [clean_clients, clean_round, attack_clients, attack_round]):
        print(" Missing result CSV files. Run the experiment first.")
        return

    # ---------------------------
    # NO ATTACK (4 plots)
    # ---------------------------
    print("[Plotter] Processing No-Attack results...")
    df_clean_clients = pd.read_csv(clean_clients)
    df_clean_round = pd.read_csv(clean_round)
    plot_per_client_history(df_clean_clients, "loss", "no_attack", "Loss", NO_ATTACK_PLOTS)
    plot_average_history(df_clean_round, "avg_loss", "no_attack", "Loss", NO_ATTACK_PLOTS)
    plot_per_client_history(df_clean_clients, "accuracy", "no_attack", "Accuracy", NO_ATTACK_PLOTS)
    plot_average_history(df_clean_round, "avg_accuracy", "no_attack", "Accuracy", NO_ATTACK_PLOTS)

    # ---------------------------
    # ATTACK (6 plots)
    # ---------------------------
    print("[Plotter] Processing Attack results...")
    df_attack_clients = pd.read_csv(attack_clients)
    df_attack_round = pd.read_csv(attack_round)
    plot_per_client_history(df_attack_clients, "loss", "attack", "Loss", ATTACK_PLOTS)
    plot_average_history(df_attack_round, "avg_loss", "attack", "Loss", ATTACK_PLOTS)
    plot_per_client_history(df_attack_clients, "accuracy", "attack", "Accuracy", ATTACK_PLOTS)
    plot_average_history(df_attack_round, "avg_accuracy", "attack", "Accuracy", ATTACK_PLOTS)
    plot_attack_severity_relation(attack_clients, ATTACK_PLOTS)

    print(f"[Plotter] Plots saved to:\n  - {NO_ATTACK_PLOTS}\n  - {ATTACK_PLOTS}")

    # After plotting attack plots
    removal_csv = os.path.join(ATTACK_DIR, "removed_clients.csv")
    plot_removed_clients(removal_csv, ATTACK_PLOTS)


if __name__ == "__main__":
    main()
