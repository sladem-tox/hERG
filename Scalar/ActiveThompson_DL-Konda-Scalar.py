import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split


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



df = pd.read_csv(TrainFile)
# Outputs take names from filenames above
FigName = TrainFile.split(".csv")[0] + f"_TS_AL_scalar_{q_num}_iterations.png"
Train_performance_file = filename_base + f"_TS_AL_Scalar_{q_num}_Train_Performance.txt"
Model_name = filename_base + f"_TS_AL_Scalar_{q_num}_model.pt"
Pred_filename = filename_base + f"TS_AL_Scalar_{q_num}_Predictions.csv"
Holdout_filename = filename_base + f"TS_AL_Scalar_{q_num}_Holdout_Performance.txt"


#q_num = 3 #args.queries
# ----- Load and prepare data -----
print(f"Loading data from {TrainFile}...")

#df = pd.read_csv("Konda_regression.csv", dtype={"COMPOUND_ID": str})


# Don't need to change below this line, but based on 2048 bit-fingerprint input except to match columns!
X = df.drop(columns=['COMPOUND_ID', 'target'])
y = df['target']
print("Initial dimensions of the dataset, X is:", X.shape,"and y is:", y.shape)

# 1. First split off the test set (20%)
X_train_pool, X_test, y_train_pool, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 2. From remaining 80%, take small initial training set (2000 samples)
X_train, X_pool, y_train, y_pool = train_test_split(
    X_train_pool, y_train_pool, train_size=2000, random_state=42
)


# Convert pandas DataFrames/Series to NumPy arrays
X_train = X_train.values
y_train = y_train.values
X_pool = X_pool.values
y_pool = y_pool.values
X_test = X_test.values
y_test = y_test.values

# Tensor conversion helper
def to_tensor(x): return torch.tensor(x, dtype=torch.float32)

# Convert training data to PyTorch tensors
X_train_tensor = to_tensor(X_train)
y_train_tensor = to_tensor(y_train).view(-1, 1)  # Ensure shape [N, 1]

# ----- Define neural network -----
class DropoutNet(nn.Module):
    def __init__(self, input_dim=2048, p_dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.out = nn.Linear(256, 1)
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), negative_slope=0.01)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc2(x), negative_slope=0.01)
        x = self.dropout(x)
        return self.out(x)  # No sigmoid for regression

print("Initializing model...")
model = DropoutNet()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()  # Regression loss

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
    model.train()  # Keep dropout active
    preds = [model(to_tensor(X)).detach().numpy().ravel() for _ in range(n_samples)]
    return np.array(preds)

# ----- Thompson sampling acquisition -----
def thompson_sampling_query(model, X_pool, n_samples=20):
    print("Running Thompson Sampling query selection...")
    preds = mc_dropout_predict(model, X_pool, n_samples=n_samples)
    stds = preds.std(axis=0)  # Use prediction uncertainty
    idx = np.argmax(stds)     # Most uncertain prediction
    print(f"Selected index {idx} from pool with predicted std {stds[idx]:.3f}")
    return idx, preds

# ----- Active Learning Loop -----
print("Starting Active Learning loop...")
mse_scores, mae_scores, r2_scores = [], [], []

for i in range(q_num):  # number of active learning iterations
    print(f"\n--- Active Learning Iteration {i + 1} ---")
    train_model(model, X_train_tensor, y_train_tensor, epochs=50)

    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        y_pred = model(to_tensor(X_test)).numpy().ravel()
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mse_scores.append(mse)
        mae_scores.append(mae)
        r2_scores.append(r2)
        print(f"MSE: {mse:.3f} | MAE: {mae:.3f} | R²: {r2:.3f}")

    idx, _ = thompson_sampling_query(model, X_pool, n_samples=20)

    new_x = X_pool[idx].reshape(1, -1)
    new_y = y_pool[idx].reshape(1)

    # Update training set
    print("Updating training set with selected sample...")
    X_train = np.vstack([X_train, new_x])
    y_train = np.append(y_train, new_y)
    X_train_tensor = to_tensor(X_train)
    y_train_tensor = to_tensor(y_train).view(-1, 1)

    # Remove selected instance from pool
    print("Removing selected sample from pool...")
    X_pool = np.delete(X_pool, idx, axis=0)
    y_pool = np.delete(y_pool, idx)

    print(f"Training set size: {len(X_train)} | Pool size: {len(X_pool)}")

# ----- Final evaluation -----
print("\nTraining complete. Evaluating final model on held out test set...")
model.eval()
with torch.no_grad():
    y_pred = model(to_tensor(X_test)).numpy().ravel()
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Final MSE: {mse:.3f} | MAE: {mae:.3f} | R²: {r2:.3f}")
# ----- Plotting results -----
    plt.figure(figsize=(8,5))
    plt.plot(range(1, len(r2_scores)+1), r2_scores, marker='o')
    plt.title("R-Square on hERG pIC50 Predictions Over TS AL Iterations")
    plt.xlabel("Active Learning Iteration")
    plt.ylabel(r"$R^2$")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(FigName, dpi=300)
    plt.close()


# Save the model
torch.save(model, Model_name)

# Load the model and evaluate on the test set
model = torch.load(Model_name)
model.eval()

X_test_holdout = pd.read_csv(Test_file_name)
compound_ids = X_test_holdout['COMPOUND_ID']
y_test_holdout = X_test_holdout['target'].values.astype(float)
X_test_holdout = X_test_holdout.drop(columns=['COMPOUND_ID', 'target'])

# Evaluate the loaded model on the test set
X_test_holdout = X_test_holdout.values
X_test_holdout_tensor = to_tensor(X_test_holdout)
with torch.no_grad():
    y_pred_holdout = model(X_test_holdout_tensor).numpy().ravel()


predictions_df = pd.DataFrame({
    'COMPOUND_ID': compound_ids,
    'True_Label': y_test_holdout,
    'Predicted_Label': y_pred_holdout
})
predictions_df.to_csv(Pred_filename, index=False)


# Compute regression metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mse = mean_squared_error(y_test_holdout, y_pred_holdout)
mae = mean_absolute_error(y_test_holdout, y_pred_holdout)
r2 = r2_score(y_test_holdout, y_pred_holdout)

# Save metrics to file
with open(Holdout_filename, 'w') as f:
    f.write(f'Final MSE on holdout set: {mse:.3f}\n')
    f.write(f'Final MAE on holdout set: {mae:.3f}\n')
    f.write(f'Final R² score on holdout set: {r2:.3f}\n')
print(f"Holdout set performance saved to {Holdout_filename}")