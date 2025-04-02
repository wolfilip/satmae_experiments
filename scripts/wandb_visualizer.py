import wandb
import matplotlib.pyplot as plt
import pandas as pd


# Initialize a W&B API client
api = wandb.Api()

# List all runs in your project

username = "uni-lj-dl4rs"

# Replace with your W&B project name and run ID
project_name = "crossmodal-ssl"
run_id = "nrk383h0"

# Construct the run path
run_path = f"{username}/{project_name}/{run_id}"

# Fetch the run
run = api.run(run_path)

history = []
# Fetch the history data from the run
for row in run.scan_history():
    history.append(row)

df = pd.DataFrame(history)

aerial_reconstruction_loss_epoch = df["train/aerial_reconstruction_loss_epoch"].dropna()
contrastive_loss_epoch = df["train/contrastive_loss_epoch"].dropna()
s2_reconstruction_loss_epoch = df["train/s2_reconstruction_loss_epoch"].dropna()
loss_epoch = df["train/loss_epoch"].dropna()

plt.figure(figsize=(10, 5))
plt.plot(aerial_reconstruction_loss_epoch)
plt.xlabel("Epoch")
plt.ylabel("Aerial Reconstruction Train Loss")

plt.figure(figsize=(10, 5))
plt.plot(contrastive_loss_epoch)
plt.xlabel("Epoch")
plt.ylabel("Contrastive Train Loss")

plt.figure(figsize=(10, 5))
plt.plot(s2_reconstruction_loss_epoch)
plt.xlabel("Epoch")
plt.ylabel("S2 Reconstruction Train Loss")

plt.figure(figsize=(10, 5))
plt.plot(loss_epoch)
plt.xlabel("Epoch")
plt.ylabel("Train Loss")

plt.show()
