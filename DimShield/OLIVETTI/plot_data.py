from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Where the CSVs are and where to save plots (relative to this script)
RESULT_DIR = Path(__file__).resolve().parent / "Result"

# CSV filenames for all attacks
FILES = {
    "FGSM": "DimShield_Result_FGSM.csv",
    "BIM":  "DimShield_Result_BIM.csv",
    "CW":   "DimShield_Result_CW.csv",
    "JSMA": "DimShield_Result_JSMA.csv",
}

# Column names in the provided CSVs
COL_EPS = "Epsilon"
COL_ITERS = "Iterations"
COL_ACC_WO = "w/o DimShield (Acc.)"
COL_ACC_W  = "w DimShield (Acc.)"
COL_GAMMA = "Gamma"

def plot_attack(attack_name: str, csv_name: str) -> Path:
    """Load one CSV and save a plot for the given attack."""
    csv_path = RESULT_DIR / csv_name
    if not csv_path.exists():
        print(f"Warning: Could not find {csv_path}, skipping.")
        return None

    df = pd.read_csv(csv_path)

    # Choose x-axis column based on attack type
    if attack_name == "CW":
        x_col = COL_ITERS
        xlabel = "Iterations"
        title = f"{attack_name} — Accuracy vs Iterations"
        out_png = f"{attack_name}_accuracy_vs_iterations.png"
    elif attack_name == "JSMA":
        x_col = COL_GAMMA
        xlabel = "Gamma"
        title = f"{attack_name} — Accuracy vs Gamma"
        out_png = f"{attack_name}_accuracy_vs_gamma.png"
    else:
        x_col = COL_EPS
        xlabel = "Epsilon"
        title = f"{attack_name} — Accuracy vs Epsilon"
        out_png = f"{attack_name}_accuracy_vs_epsilon.png"

    # Basic sanity checks
    for c in [x_col, COL_ACC_WO, COL_ACC_W]:
        if c not in df.columns:
            print(f"Warning: Expected column '{c}' not found in {csv_path.name}. Skipping.")
            return None

    # Sort by x-axis and coerce to numeric
    df = df.copy()
    df[x_col] = pd.to_numeric(df[x_col], errors="coerce").fillna(0.0)
    df[COL_ACC_WO] = pd.to_numeric(df[COL_ACC_WO], errors="coerce")
    df[COL_ACC_W]  = pd.to_numeric(df[COL_ACC_W],  errors="coerce")
    df = df.sort_values(x_col)

    x = df[x_col].values
    y_wo = df[COL_ACC_WO].values
    y_w  = df[COL_ACC_W].values

    plt.figure()
    plt.plot(x, y_wo, marker="o", label="w/o DimShield")
    plt.plot(x, y_w,  marker="s", label="w DimShield")
    plt.xlabel(xlabel)
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()

    out_path = RESULT_DIR / out_png
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"Saved: {out_path}")
    return out_path

def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    for attack, fname in FILES.items():
        plot_attack(attack, fname)

if __name__ == "__main__":
    main()