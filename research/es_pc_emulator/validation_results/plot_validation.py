"""Reproduce the held-out accuracy figure from the committed measurements."""
import json
from pathlib import Path

import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
result = json.loads((root / "training_20261027.json").read_text())["runs"]["warm_channel"]
arm = result["accuracy_report"]["selected_feature_set"]
rows = result["arms"][arm]["per_snapshot"]
z = [row["redshift"] for row in rows]
fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)
for name, color, label in (("es", "#a54b32", "ES"), ("nn", "#246b98", "NN")):
    axes[0].plot(z, [row[name]["Q_volume"] for row in rows], "o-", color=color, label=label)
    axes[1].plot(z, [100 * row[name]["MAE"] for row in rows], "o-", color=color, label=label)
axes[0].plot(z, [row["es"]["Q_volume"] - row["es"]["delta_Q_volume"] for row in rows],
             "k--", label="RT teacher")
for ax in axes[:2]:
    ax.set_xlabel("Redshift")
    ax.invert_xaxis()
    ax.grid(alpha=0.2)
axes[0].set_ylabel("Volume-averaged ionized fraction")
axes[1].set_ylabel("Field MAE (percentage points)")
axes[0].legend(frameon=False)
last = min(rows, key=lambda row: row["redshift"])
spec = last["spectra"]
axes[2].plot(spec["k_cMpc_inv"], spec["ratio_es"], "o-", color="#a54b32", label="ES / RT")
axes[2].plot(spec["k_cMpc_inv"], spec["ratio_nn"], "o-", color="#246b98", label="NN / RT")
axes[2].axhline(1, color="black", linestyle="--", linewidth=1)
axes[2].set_xlabel(r"$k$ (cMpc$^{-1}$)")
axes[2].set_ylabel("21-cm power ratio at z=7.9")
axes[2].legend(frameon=False)
axes[2].grid(alpha=0.2)
fig.suptitle("Independent seed 333; model selected using seed 222; 50 cMpc / 16 cubed")
fig.savefig(root / "held_out_accuracy.png", dpi=160)
