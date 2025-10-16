import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import TensorDataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- model parameterized ----
class DropoutNet(nn.Module):
    def __init__(self, input_dim=2048, hidden1=512, hidden2=256, p_dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.out = nn.Linear(hidden2, 1)
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = self.dropout(x)
        x = F.leaky_relu(self.fc2(x))
        x = self.dropout(x)
        return torch.sigmoid(self.out(x))

# ---- Objective function ----
def objective(trial):
    hidden1 = trial.suggest_int("hidden1", 128, 1024, step=128)
    hidden2 = trial.suggest_int("hidden2", 64, 512, step=64)
    p_dropout = trial.suggest_float("p_dropout", 0.1, 0.6)
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    epochs = trial.suggest_int("epochs", 30, 150, step=20)

    model = DropoutNet(input_dim=X.shape[1], hidden1=hidden1, hidden2=hidden2, p_dropout=p_dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCELoss()

    # Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y)
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(device)

    # Training loop
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(X_train_t)
        loss = loss_fn(out, y_train_t)
        loss.backward()
        optimizer.step()

    # Validation performance
    model.eval()
    with torch.no_grad():
        preds = model(X_val_t).cpu().numpy().ravel()
    y_pred_bin = (preds > 0.5).astype(int)

    # Balanced accuracy or F1
    score = f1_score(y_val, y_pred_bin)

    return score  # Optuna maximizes by default

#Run study
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

print("Best params:", study.best_params)
print("Best F1:", study.best_value)

# Save for later use in Active Learning
import json, os
os.makedirs("results", exist_ok=True)
with open("results/best_params.json", "w") as f:
    json.dump(study.best_params, f)
