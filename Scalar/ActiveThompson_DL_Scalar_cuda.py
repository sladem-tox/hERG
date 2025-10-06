# Get data and libraries
import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from rich.progress import track
from rich.console import Console
console = Console()

parser = argparse.ArgumentParser(description="Load training and test data for active learning model.")
parser.add_argument('trainfile', type=str, help='Path to the training CSV data file')
parser.add_argument('testfile', type=str, help='Path to the test CSV data file')
parser.add_argument('--q_num', type=int, default=300, help='Number of active learning queries (default: 300)')
args = parser.parse_args()


TrainFile = args.trainfile
TestFile = args.testfile
q_num = args.q_num

Test_file_name = TestFile

filename_base = os.path.splitext(TrainFile)[0]
results_dir = os.path.join(os.getcwd(), "results") #Creates string for path.

# Create a subdirectory for this experiment, based on q_num
run_dir = os.path.join(results_dir, f"ActiveLearning_scalar_{q_num}_queries")
os.makedirs(run_dir, exist_ok=True)

# Variables take names from filenames above
df = pd.read_csv(TrainFile)

# Extract the base name of the training file (without path or .csv)
base_name = os.path.splitext(os.path.basename(TrainFile))[0]

# Define all result file paths inside the run_dir
FigName = os.path.join(run_dir, f"{base_name}_TS_AL_scalar_{q_num}_iterations.png")
Train_performance_file = os.path.join(run_dir, f"{base_name}_TS_AL_Scalar_{q_num}_Train_Performance.txt")
Model_name = os.path.join(run_dir, f"{base_name}_TS_AL_Scalar_{q_num}_model.pt")
Pred_filename = os.path.join(run_dir, f"{base_name}_TS_AL_Scalar_{q_num}_Predictions.csv")
Holdout_filename = os.path.join(run_dir, f"{base_name}_TS_AL_Scalar_{q_num}_Holdout_Performance.txt")
print(f"Results will be saved in directory: {run_dir}")

# Don't need to change below this line
X = df.drop(columns=['COMPOUND_ID', 'target'])
y = df['target']
print("Initial dimensions of the dataset, X is:", X.shape,"and y is:", y.shape)

#Test file saved for the holdout evaluation at the end
#Train file used for active learning

# 1. First split off the test set (20%)
X_train_pool, X_test, y_train_pool, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 2. From remaining 80%, take small initial training set (2000 samples)
X_train, X_pool, y_train, y_pool = train_test_split(
    X_train_pool, y_train_pool, train_size=2000, random_state=42
)

# Convert pandas DataFrames/Series to NumPy arrays for PyTorch compatibility
X_train = X_train.values
y_train = y_train.values
X_pool = X_pool.values
y_pool = y_pool.values
X_test = X_test.values
y_test = y_test.values


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# Define tensor conversion helper
def to_tensor(x): return torch.tensor(x, dtype=torch.float32)

# Convert training data to PyTorch tensors
X_train_tensor = to_tensor(X_train).to(device)
y_train_tensor = to_tensor(y_train).unsqueeze(1).to(device)

# ----- Define neural network -----
class DropoutNet(nn.Module):
    def __init__(self, input_dim=2048, p_dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.out = nn.Linear(256, 1)
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), negative_slope=0.01)  # default is 0.01
        x = self.dropout(x)
        x = F.leaky_relu(self.fc2(x), negative_slope=0.01)
        x = self.dropout(x)
        return self.out(x)

print("Initializing model...")
model = DropoutNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# Training loop

# ----- Training function -----
def train_model(model, X_train, y_train, epochs=100):
    print(f"Training model for {epochs} epochs on {len(X_train)} samples...")
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        output = model(X_train)
        loss = loss_fn(output, y_train)
        loss.backward()
        optimizer.step()

# ----- MC Dropout inference -----
def mc_dropout_predict(model, X, n_samples=20):
    print(f"Performing MC Dropout inference with {n_samples} samples...")
    model.train()  # important: keep dropout active
    X_tensor = to_tensor(X).to(device)
    preds = [model(X_tensor).detach().cpu().numpy() for _ in range(n_samples)]
    return np.array(preds)

# ----- Thompson sampling acquisition -----
# n_samples is usually kept same as the number of MC samples (above)
def thompson_sampling_query(model, X_pool, n_samples=20):
    print("Running Thompson Sampling query selection...")
    preds = mc_dropout_predict(model, X_pool, n_samples=n_samples)
    stds = preds.std(axis=0).squeeze()  # select one model sample
    idx = np.argmax(stds)        # point with highest uncertainty
    console.print(f"\nSelected index {idx} from pool with predicted std {stds[idx]:.3f}", style="bold blue")
    return idx, preds

# ----- Active Learning Loop -----
print("Starting Active Learning loop...")
mse_scores, mae_scores, r2_scores = [], [], []

for i in track(range(q_num), description="Active Learning Iterations"):  # number of active learning iterations
    print(f"\n--- Active Learning Iteration {i + 1} ---")
    train_model(model, X_train_tensor, y_train_tensor, epochs=50)
    
    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        x_test_tensor = to_tensor(X_test).to(device)           # Move input to device
        y_pred_tensor = model(x_test_tensor)                  # Output is on device (GPU)
        y_pred = y_pred_tensor.detach().cpu().numpy().ravel() # Move to CPU and convert to numpy
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mse_scores.append(mse)
        mae_scores.append(mae)
        r2_scores.append(r2)
        console.print(f"\nMSE on test set: {mse:.3f}", style="bold green")
        console.print(f"MAE on test set: {mae:.3f}", style="bold green")
        console.print(f"R² on test set: {r2:.3f}", style="bold green")

    idx, _ = thompson_sampling_query(model, X_pool, n_samples=20)

    new_x = X_pool[idx].reshape(1, -1)
    new_y = y_pool[idx].reshape(1)

    # Add selected instance to training set
    print("Updating training set with selected sample...")
    X_train = np.vstack([X_train, new_x])
    y_train = np.append(y_train, new_y)
    X_train_tensor = to_tensor(X_train).to(device)
    y_train_tensor = to_tensor(y_train).unsqueeze(1).to(device)

    # Remove selected instance from pool
    print("Removing selected sample from pool...")
    X_pool = np.delete(X_pool, idx, axis=0)
    y_pool = np.delete(y_pool, idx)

    print(f"Training set size: {len(X_train)} | Pool size: {len(X_pool)}")

# ----- Final training evaluation -----

X_test_tensor = to_tensor(X_test).to(device)
model.eval()
with torch.no_grad():
    y_pred = model(X_test_tensor).detach().cpu().numpy().ravel()
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Final training results: MSE: {mse:.3f} | MAE: {mae:.3f} | R²: {r2:.3f}")
# Save metrics to file
    with open(Train_performance_file, 'w') as f:
        f.write(f'Final MSE after TS active learning: {mse:.3f}\n')
        f.write(f'Final MAE after TS active learning: {mae:.3f}\n')
        f.write(f'Final R² after TS active learning: {r2:.3f}\n')

# ----- Plotting training results -----
plt.figure(figsize=(8,5))
plt.plot(range(1, len(r2_scores)+1), r2_scores, marker='o')
plt.title(f"R² on Test Set Over {q_num} Active Learning Iterations")
plt.xlabel("Active Learning Iteration")
plt.ylabel("R² Score")
plt.grid(True)
plt.tight_layout()
plt.savefig(FigName, dpi=300)
plt.close()


# Save the model
torch.save(model.state_dict(), Model_name)

# ----- Evaluate on held-out test set -----
print("\nTraining complete. Evaluating final model on held out test set...")
# Load the model and evaluate on the test set
model = DropoutNet(input_dim=X_train.shape[1])
model.load_state_dict(torch.load(Model_name))
model.to(device)
model.eval()

X_test_holdout = pd.read_csv(Test_file_name)
compound_ids = X_test_holdout['COMPOUND_ID']
y_test_holdout = X_test_holdout['target']
X_test_holdout = X_test_holdout.drop(columns=['COMPOUND_ID', 'target'])

# Evaluate the loaded model on the test set
X_test_holdout = X_test_holdout.values
X_test_holdout_tensor = to_tensor(X_test_holdout).to(device)
with torch.no_grad():
    y_pred_holdout = model(X_test_holdout_tensor).detach().cpu().numpy().ravel()


predictions_df = pd.DataFrame({
    'COMPOUND_ID': compound_ids,
    'True_Label': y_test_holdout,
    'Predicted_pIC50': y_pred_holdout
})
predictions_df.to_csv(Pred_filename, index=False)

mse = mean_squared_error(y_test_holdout, y_pred_holdout)
mae = mean_absolute_error(y_test_holdout, y_pred_holdout)
r2 = r2_score(y_test_holdout, y_pred_holdout)

# Save metrics to file
with open(Holdout_filename, 'w') as f:
    f.write(f'Final test MSE after active learning: {mse:.3f}\n')
    f.write(f'Final test MAE after active learning: {mae:.3f}\n')
    f.write(f'Final test R² after active learning: {r2:.3f}\n')
print(f"Holdout set performance saved to {Holdout_filename}")   
