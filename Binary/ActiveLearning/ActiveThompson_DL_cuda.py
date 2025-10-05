# Get data and libraries
import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

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
run_dir = os.path.join(results_dir, f"ActiveLearning_bin_{q_num}_queries")
os.makedirs(run_dir, exist_ok=True)

# Variables take names from filenames above
df = pd.read_csv(TrainFile)

FigName = os.path.join(run_dir, filename_base + f"_F1_over_{q_num}_queries.png")
Train_performance_file = os.path.join(run_dir, filename_base + f'_Train_Performance_{q_num}_queries.txt')
Model_name = os.path.join(run_dir, filename_base + f'_Thompson_AL_model_{q_num}_queries.pt')
Pred_filename = os.path.join(run_dir, filename_base + f'_Predictions_{q_num}_queries.csv')
Holdout_filename = os.path.join(run_dir, filename_base + f'_Holdout_Performance_{q_num}_queries.txt')

# Don't need to change below this line
X = df.drop(columns=['COMPOUND_ID', 'target'])
y = df['target']
print("Initial dimensions of the dataset, X is:", X.shape,"and y is:", y.shape)

# 1. First split off the test set (20%)
X_train_pool, X_test, y_train_pool, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 2. From remaining 80%, take small initial training set (2000 samples)
X_train, X_pool, y_train, y_pool = train_test_split(
    X_train_pool, y_train_pool, train_size=2000, stratify=y_train_pool, random_state=42
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
        return torch.sigmoid(self.out(x))

print("Initializing model...")
model = DropoutNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.BCELoss()

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
    sample = preds[np.random.choice(n_samples)]  # select one model sample
    idx = np.argmin(np.abs(sample - 0.5))        # point with highest uncertainty
    console.print(f"\nSelected index {idx} from pool with predicted probability {sample[idx].item():.3f}", style="bold blue")
    return idx, sample

# ----- Active Learning Loop -----
print("Starting Active Learning loop...")
f1_scores = []
for i in track(range(q_num), description="Active Learning Iterations"):  # number of active learning iterations
    print(f"\n--- Active Learning Iteration {i + 1} ---")
    train_model(model, X_train_tensor, y_train_tensor, epochs=50)
    
    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        x_test_tensor = to_tensor(X_test).to(device)           # Move input to device
        y_pred_tensor = model(x_test_tensor)                  # Output is on device (GPU)
        y_pred = y_pred_tensor.detach().cpu().numpy().ravel() # Move to CPU and convert to numpy
        y_pred_binary = (y_pred > 0.5).astype(int)
        f1 = f1_score(y_test, y_pred_binary)
        f1_scores.append(f1)
        console.print(f"\nF1 Score on test set: {f1:.3f}", style="bold green")

        
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

# ----- Final evaluation -----
print("\nTraining complete. Evaluating final model on held out test set...")
X_test_tensor = to_tensor(X_test).to(device)
model.eval()
with torch.no_grad():
    y_pred = model(X_test_tensor).detach().cpu().numpy().ravel()
    y_pred_binary = (y_pred > 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred_binary)
    f1 = f1_score(y_test, y_pred_binary, zero_division=0)
    precision = precision_score(y_test, y_pred_binary, zero_division=0)
    recall = recall_score(y_test, y_pred_binary, zero_division=0)
    print(f"\nTest Accuracy after Active Learning: {acc:.3f}")
    print(f"Test F1 Score: {f1:.3f}")
    print(f"Test Precision: {precision:.3f}")
    print(f"Test Recall: {recall:.3f}")

    with open(Train_performance_file, 'w') as f:
        f.write(f'Final accuracy after TS active learning: {acc:.3f}\n')
        f.write(f'Test F1 Score: {f1:.3f}\n')
        f.write(f'Test Precision: {precision:.3f}\n')
        f.write(f'Test Recall: {recall:.3f}\n')

    plt.figure(figsize=(8,5))
    plt.plot(range(1, len(f1_scores)+1), f1_scores, marker='o')
    plt.title(f"F1 Score on Test Set Over {q_num} Active Learning Iterations")
    plt.xlabel("Active Learning Iteration")
    plt.ylabel("F1 Score")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(FigName, dpi=300)
    plt.close()


# Save the model
torch.save(model.state_dict(), Model_name)

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

y_pred_binary = (y_pred_holdout > 0.5).astype(int)

predictions_df = pd.DataFrame({
    'COMPOUND_ID': compound_ids,
    'True_Label': y_test_holdout,
    'Predicted_Label': y_pred_binary
})
predictions_df.to_csv(Pred_filename, index=False)


test_accuracy = accuracy_score(y_test_holdout, y_pred_binary)
f1 = f1_score(y_test_holdout, y_pred_binary, zero_division=0)
test_precision = precision_score(y_test_holdout, y_pred_binary, zero_division=0)
test_recall = recall_score(y_test_holdout, y_pred_binary, zero_division=0)

with open(Holdout_filename, 'w') as f:
    f.write(f'Final test accuracy after active learning: {test_accuracy:.3f}\n')
    f.write(f'Test f1 score: {f1:.3f}\n')
    f.write(f'Test Precision: {test_precision:.3f}\n')
    f.write(f'Test Recall: {test_recall:.3f}\n')