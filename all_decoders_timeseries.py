from scipy.io import loadmat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
""" preprocessing """
# Load data 
fam_time_bhs = loadmat(folder+'all_mice_familiar.mat')
nov_time_bhs = loadmat(folder+'all_mice_novel.mat')
# Extract relevant data from dictionary 
fam_time_lhs = fam_time_bhs['VEP_LHS_all'] #unlabelled timeseries - familiar stimuli responses, left hemisphere
nov_time_lhs = nov_time_bhs['VEP_LHS_nov_all']
# Label mouse ID
test_mouse = np.arange(1, 34, 1, dtype=int) #array of mice IDs
test_mouse_idx = np.repeat(test_mouse, 1000) #repeat each element in test_mouse for n times, n = obs per mouse

df_fam = pd.DataFrame(fam_time_lhs)
df_fam['target'] = [0]*int(df_fam.shape[0])
df_fam['mice'] = test_mouse_idx
df_nov = pd.DataFrame(nov_time_lhs)
df_nov['target'] = [1]*int(df_fam.shape[0])
df_nov['mice'] = test_mouse_idx

df = pd.concat([df_fam, df_nov])
df = df.sample(frac=1)

# Initialize GroupKFold
n_splits = 5
group_kfold = GroupKFold(n_splits=n_splits)

# Prepare to store results
folds = []

# Split the data into training, validation, and test sets
for fold, (train_val_idx, test_idx) in enumerate(group_kfold.split(df, df['target'], groups=df['mice'])):
    train_val_df = df.iloc[train_val_idx]
    test_df = df.iloc[test_idx]

    # Further split train_val_df into training and validation sets
    train_idx, val_idx = next(GroupKFold(n_splits=5).split(
        train_val_df, train_val_df['target'], groups=train_val_df['mice']))
    train_df = train_val_df.iloc[train_idx]
    val_df = train_val_df.iloc[val_idx]

    # Scale the data
    scaler = StandardScaler()
    X_train = train_df.drop(columns=['mice', 'target'])
    X_val = val_df.drop(columns=['mice', 'target'])
    X_test = test_df.drop(columns=['mice', 'target'])

    X_train_scaled = scaler.fit_transform(X_train)  # only fit to the training set
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    train_df.loc[:, X_train.columns] = X_train_scaled
    val_df.loc[:, X_val.columns] = X_val_scaled
    test_df.loc[:, X_test.columns] = X_test_scaled

    folds.append({
        'fold': fold,
        'train': train_df,
        'validation': val_df,
        'test': test_df
    })

# Display an example fold
example_fold = folds[0]
print("Fold 0:")
print("Train set:", example_fold['train'].shape)
print("Validation set:", example_fold['validation'].shape)
print("Test set:", example_fold['test'].shape)

# -----------------------------------------------------------
# Check label distribution for each fold
# -----------------------------------------------------------
for fold_data in folds:
    fold = fold_data['fold']
    train_df = fold_data['train']
    val_df = fold_data['validation']
    test_df = fold_data['test']

    print(f"\n=== Fold {fold} ===")

    # Print counts and proportions for Train
    train_counts = train_df['target'].value_counts()
    train_ratio = train_df['target'].value_counts(normalize=True)
    print("Train label distribution:")
    print(train_counts)
    print(train_ratio)

    # Print counts and proportions for Validation
    val_counts = val_df['target'].value_counts()
    val_ratio = val_df['target'].value_counts(normalize=True)
    print("Validation label distribution:")
    print(val_counts)
    print(val_ratio)

    # Print counts and proportions for Test
    test_counts = test_df['target'].value_counts()
    test_ratio = test_df['target'].value_counts(normalize=True)
    print("Test label distribution:")
    print(test_counts)
    print(test_ratio)

"""Function to plot losses over epochs"""

def plot_metrics(fold, model_name, train_losses, val_losses, val_accuracies):
    epochs = range(1, len(train_losses) + 1)
    # Plot training and validation loss on the same figure
    plt.figure()
    plt.plot(epochs, train_losses, label='Train Loss', color='blue')
    plt.plot(epochs, val_losses, label='Validation Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{model_name} - Fold {fold} Loss')
    plt.legend()
    #plt.savefig(f'{model_name}_fold_{fold}_losses.png')
    plt.show()  # display the loss figure

    # Plot validation accuracy on a separate figure
    plt.figure()
    plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, label='Validation Accuracy', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'{model_name} - Fold {fold} Validation Accuracy')
    plt.legend()
    #plt.savefig(f'{model_name}_fold_{fold}_accuracy.png')
    plt.show()  # display the accuracy figure

"""RF"""

cont = 0
# Train and evaluate non-neural models
for fold_data in folds:
    cont += 1
    print(cont)
    train_df = fold_data['train']
    val_df = fold_data['validation']
    test_df = fold_data['test']

    # Prepare data
    X_train, y_train = train_df.drop(columns=['mice', 'target']), train_df['target']
    X_val, y_val = val_df.drop(columns=['mice', 'target']), val_df['target']
    X_test, y_test = test_df.drop(columns=['mice', 'target']), test_df['target']

    # Dummy classifier
    dummy_clf = DummyClassifier(strategy="most_frequent")
    dummy_clf.fit(X_train, y_train)
    dummy_y_pred = dummy_clf.predict(X_test)
    dummy_accuracy = accuracy_score(y_test, dummy_y_pred)
    dummy_cm = confusion_matrix(y_test, dummy_y_pred)
    dummy_specificity = dummy_cm[0, 0] / (dummy_cm[0, 0] + dummy_cm[0, 1]) if (dummy_cm[0, 0] + dummy_cm[0, 1]) > 0 else 0

    # Hyperparameter tuning for Random Forest using validation set
    rf_clf = RandomForestClassifier()
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }
    grid_search = GridSearchCV(rf_clf, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Best model from grid search
    best_rf_clf = grid_search.best_estimator_

    # Evaluate the best model on the validation set
    val_y_pred = best_rf_clf.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_y_pred)
    print(f"Fold {fold_data['fold']} Validation Accuracy: {val_accuracy}")

    # Evaluate the best model on the test set
    rf_y_pred = best_rf_clf.predict(X_test)
    rf_y_proba = best_rf_clf.predict_proba(X_test)[:, 1]
    rf_accuracy = accuracy_score(y_test, rf_y_pred)
    rf_cm = confusion_matrix(y_test, rf_y_pred)
    rf_auc = roc_auc_score(y_test, rf_y_proba)
    rf_specificity = rf_cm[0, 0] / (rf_cm[0, 0] + rf_cm[0, 1]) if (rf_cm[0, 0] + rf_cm[0, 1]) > 0 else 0
    rf_classification_report = classification_report(y_test, rf_y_pred, output_dict=True)

    # Show results
    print(f"Fold {fold_data['fold']} Results:")
    print(f"Dummy Classifier Accuracy: {dummy_accuracy}")
    print(f"Dummy Classifier Specificity: {dummy_specificity}")
    print(f"Dummy Classifier Confusion Matrix:\n{dummy_cm}")
    print(f"Random Forest Classifier Accuracy: {rf_accuracy}")
    print(f"Random Forest Classifier AUC: {rf_auc}")
    print(f"Random Forest Classifier Specificity: {rf_specificity}")
    print(f"Random Forest Classifier Confusion Matrix:\n{rf_cm}")
    print(f"Random Forest Classifier Detailed Report:\n{rf_classification_report}")
    print("\n")

"""MLP"""

# Define the MLP model
class MLPClassifier(nn.Module):
    def __init__(self, input_size):
        super(MLPClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 30),
            nn.ReLU(),
            nn.Linear(30, 15),
            nn.ReLU(),
            nn.Linear(15, 1)  # Single logit output for BCEWithLogitsLoss
        )

    def forward(self, x):
        # Return raw logits (no Sigmoid, no Softmax),
        # because BCEWithLogitsLoss handles the Sigmoid internally.
        return self.model(x)


# Training loop for MLP
num_epochs = 200
for fold_data in folds:
    fold = fold_data['fold']
    train_df = fold_data['train']
    val_df = fold_data['validation']
    test_df = fold_data['test']

    # Prepare data
    X_train, y_train = train_df.drop(columns=['mice', 'target']), train_df['target']
    X_val, y_val = val_df.drop(columns=['mice', 'target']), val_df['target']
    X_test, y_test = test_df.drop(columns=['mice', 'target']), test_df['target']

    # Convert to PyTorch tensors
    train_dataset = TensorDataset(
        torch.tensor(X_train.values, dtype=torch.float32),
        torch.tensor(y_train.values, dtype=torch.long)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val.values, dtype=torch.float32),
        torch.tensor(y_val.values, dtype=torch.long)
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test.values, dtype=torch.float32),
        torch.tensor(y_test.values, dtype=torch.long)
    )

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Initialize the model, loss, optimizer
    input_size = X_train.shape[1]
    model = MLPClassifier(input_size).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Lists to track metrics across epochs (for plotting)
    mlp_train_losses = []
    mlp_val_losses = []
    mlp_val_accuracies = []

    # Variables to store the final epoch's metrics
    final_train_loss = None
    final_val_loss = None
    final_val_acc = None
    final_val_auc = None

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            # For BCEWithLogitsLoss, y_batch needs to be float in [0,1]
            y_batch = y_batch.float().to(device)

            optimizer.zero_grad()
            y_pred = model(X_batch)                     # shape: (batch_size, 1) logits
            loss = criterion(y_pred, y_batch.unsqueeze(1))
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()

        epoch_train_loss /= len(train_loader)

        # Validation
        model.eval()
        epoch_val_loss = 0.0
        epoch_val_acc = 0.0
        val_y_true = []
        val_y_scores = []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.float().to(device)
                y_pred = model(X_batch)                   # (batch_size, 1) logits

                loss = criterion(y_pred, y_batch.unsqueeze(1))
                epoch_val_loss += loss.item()

                # Convert logits -> probabilities [0..1]
                y_probs = torch.sigmoid(y_pred)           # shape: (batch_size, 1)
                y_preds = (y_probs >= 0.5).long().squeeze()  # threshold at 0.5

                # Accuracy: compare predicted class vs. true class
                epoch_val_acc += (y_preds == y_batch.long()).float().mean().item()

                # For AUC, store the probability for class=1
                val_y_true.extend(y_batch.cpu().numpy())                # shape: [batch_size]
                val_y_scores.extend(y_probs.squeeze().cpu().numpy())    # shape: [batch_size]

        epoch_val_loss /= len(val_loader)
        epoch_val_acc /= len(val_loader)

        # Compute AUC with true labels vs. predicted probabilities
        if len(np.unique(val_y_true)) == 2:
            epoch_val_auc = roc_auc_score(val_y_true, val_y_scores)
        else:
            epoch_val_auc = 0.5

        # Append to arrays for plotting
        mlp_train_losses.append(epoch_train_loss)
        mlp_val_losses.append(epoch_val_loss)
        mlp_val_accuracies.append(epoch_val_acc)

        # If it's the last epoch, store final metrics
        if epoch == num_epochs - 1:
            final_train_loss = epoch_train_loss
            final_val_loss = epoch_val_loss
            final_val_acc = epoch_val_acc
            final_val_auc = epoch_val_auc

        # Optional: print each epoch's progress
        #print(f"Fold {fold}, Epoch {epoch+1}/{num_epochs} => "
        #      f"Train Loss: {epoch_train_loss:.4f}, "
        #      f"Val Loss: {epoch_val_loss:.4f}, "
        #      f"Val Acc: {epoch_val_acc:.4f}, "
        #      f"Val AUC: {epoch_val_auc:.4f}")

    # After training: plot the metrics for the entire training run
    plot_metrics(fold, "MLP", mlp_train_losses, mlp_val_losses, mlp_val_accuracies)

    # Print only the final epoch's results (optional)
    #print(f"==== MLP Fold {fold} Final Metrics ====")
    #print(f"Final Epoch Train Loss: {final_train_loss:.4f}")
    #print(f"Final Epoch Validation Loss: {final_val_loss:.4f}")
    #print(f"Final Epoch Validation Accuracy: {final_val_acc:.4f}")
    #print(f"Final Epoch Validation AUC: {final_val_auc:.4f}")

    # Test evaluation
    model.eval()
    test_accuracy = 0.0
    test_probs_list = []
    test_y_true, test_y_pred_all = [], []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            y_pred = model(X_batch)                # shape: (batch_size, 1) logits
            y_probs = torch.sigmoid(y_pred)        # shape: (batch_size, 1)
            y_preds = (y_probs >= 0.5).long().squeeze()

            # Accuracy
            test_accuracy += (y_preds == y_batch).float().mean().item()

            # Store labels & predictions for confusion matrix
            test_y_true.extend(y_batch.cpu().numpy())
            test_y_pred_all.extend(y_preds.cpu().numpy())

            # Also store probabilities for computing AUC
            test_probs_list.extend(y_probs.squeeze().cpu().numpy())

    test_accuracy /= len(test_loader)
    test_cm = confusion_matrix(test_y_true, test_y_pred_all)
    # For AUC, pass the probabilities (not the predicted classes)
    test_auc = roc_auc_score(test_y_true, test_probs_list)

    #print(f"MLP Fold {fold} Test Accuracy: {test_accuracy:.4f}")
    #print(f"MLP Fold {fold} Test AUC: {test_auc:.4f}")
    #print(f"MLP Fold {fold} Test Confusion Matrix:\n{test_cm}")
    print("\n")

"""LSTM"""

# Define the LSTM model
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.4)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h_0, c_0))
        out = self.fc(out[:, -1, :])
        return out

hidden_size = 32
num_layers = 3
num_classes = 2

for fold_data in folds:
    fold = fold_data['fold']
    train_df = fold_data['train']
    val_df = fold_data['validation']
    test_df = fold_data['test']

    # Prepare data
    X_train, y_train = train_df.drop(columns=['mice', 'target']), train_df['target']
    X_val, y_val = val_df.drop(columns=['mice', 'target']), val_df['target']
    X_test, y_test = test_df.drop(columns=['mice', 'target']), test_df['target']

    # Convert to PyTorch tensors (LSTM expects input as [batch, seq_len, feature_dim])
    train_dataset = TensorDataset(torch.tensor(X_train.values, dtype=torch.float32).unsqueeze(1), torch.tensor(y_train.values, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(X_val.values, dtype=torch.float32).unsqueeze(1), torch.tensor(y_val.values, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(X_test.values, dtype=torch.float32).unsqueeze(1), torch.tensor(y_test.values, dtype=torch.long))

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    input_size = X_train.shape[1]
    model = LSTMClassifier(input_size, hidden_size, num_layers, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    lstm_train_losses = []
    lstm_val_losses = []
    lstm_val_accuracies = []
    lstm_val_aucs = []

    # Training loop
    for epoch in range(50):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(X_batch).squeeze()
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validation loop
        model.eval()
        val_loss = 0
        val_accuracy = 0
        val_y_true = []
        val_y_score = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred = model(X_batch).squeeze()
                loss = criterion(y_pred, y_batch)
                val_loss += loss.item()
                val_accuracy += (y_pred.argmax(dim=1) == y_batch).float().mean().item()

                # For AUC calculation
                val_y_true.extend(y_batch.cpu().numpy())
                # Extract probability for the positive class (index = 1)
                probs_pos_class = nn.Softmax(dim=1)(y_pred)[:, 1].cpu().numpy()
                val_y_score.extend(probs_pos_class)

        val_loss /= len(val_loader)
        val_accuracy /= len(val_loader)

        # Compute AUC
        if len(np.unique(val_y_true)) == 2:
            val_auc = roc_auc_score(val_y_true, val_y_score)
        else:
            val_auc = 0.5

        lstm_train_losses.append(train_loss)
        lstm_val_losses.append(val_loss)
        lstm_val_accuracies.append(val_accuracy)
        lstm_val_aucs.append(val_auc)

        print(f"LSTM Fold {fold} Epoch {epoch+1}: Train Loss = {train_loss:.4f}, "
              f"Val Loss = {val_loss:.4f}, Val Accuracy = {val_accuracy:.4f}, Val AUC = {val_auc:.4f}")

    # After training, plot metrics
    plot_metrics(fold, "LSTM", lstm_train_losses, lstm_val_losses, lstm_val_accuracies)

    # Save average validation accuracy and AUC across epochs
    avg_val_acc = np.mean(lstm_val_accuracies)
    avg_val_auc = np.mean(lstm_val_aucs)
    print(f"LSTM Fold {fold} Average Validation Accuracy Across Epochs: {avg_val_acc:.4f}")
    print(f"LSTM Fold {fold} Average Validation AUC Across Epochs: {avg_val_auc:.4f}")

    # Test evaluation
    model.eval()
    test_accuracy = 0
    y_true, y_pred_all = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch).squeeze()
            test_accuracy += (y_pred.argmax(dim=1) == y_batch).float().mean().item()
            y_true.extend(y_batch.cpu().numpy())
            y_pred_all.extend(y_pred.argmax(dim=1).cpu().numpy())
    test_accuracy /= len(test_loader)
    test_cm = confusion_matrix(y_true, y_pred_all)
    test_auc = roc_auc_score(y_true, y_pred_all)

    print(f"LSTM Fold {fold} Test Accuracy: {test_accuracy:.4f}")
    print(f"LSTM Fold {fold} Test AUC: {test_auc:.4f}")
    print(f"LSTM Fold {fold} Test Confusion Matrix:\n{test_cm}")
    print("\n")

"""RNN"""

class RNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNNClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, dropout=0.4, nonlinearity='tanh')
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h_0)
        out = self.fc(out[:, -1, :])
        return out

hidden_size = 32
num_layers = 3
num_classes = 2

for fold_data in folds:
    fold = fold_data['fold']
    train_df = fold_data['train']
    val_df = fold_data['validation']
    test_df = fold_data['test']

    # Prepare data
    X_train, y_train = train_df.drop(columns=['mice', 'target']), train_df['target']
    X_val, y_val = val_df.drop(columns=['mice', 'target']), val_df['target']
    X_test, y_test = test_df.drop(columns=['mice', 'target']), test_df['target']

    # Convert to PyTorch tensors (RNN expects [batch, seq_len, feature_dim])
    train_dataset = TensorDataset(torch.tensor(X_train.values, dtype=torch.float32).unsqueeze(1), torch.tensor(y_train.values, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(X_val.values, dtype=torch.float32).unsqueeze(1), torch.tensor(y_val.values, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(X_test.values, dtype=torch.float32).unsqueeze(1), torch.tensor(y_test.values, dtype=torch.long))

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    input_size = X_train.shape[1]
    model = RNNClassifier(input_size, hidden_size, num_layers, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    rnn_train_losses = []
    rnn_val_losses = []
    rnn_val_accuracies = []
    rnn_val_aucs = []

    # Training loop
    for epoch in range(50):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(X_batch).squeeze()
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validation loop
        model.eval()
        val_loss = 0
        val_accuracy = 0
        val_y_true = []
        val_y_score = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred = model(X_batch).squeeze()
                loss = criterion(y_pred, y_batch)
                val_loss += loss.item()
                val_accuracy += (y_pred.argmax(dim=1) == y_batch).float().mean().item()

                # For AUC calculation
                val_y_true.extend(y_batch.cpu().numpy())
                # Probability for the positive class (index = 1)
                probs_pos_class = nn.Softmax(dim=1)(y_pred)[:, 1].cpu().numpy()
                val_y_score.extend(probs_pos_class)

        val_loss /= len(val_loader)
        val_accuracy /= len(val_loader)

        # Compute AUC
        if len(np.unique(val_y_true)) == 2:
            val_auc = roc_auc_score(val_y_true, val_y_score)
        else:
            val_auc = 0.5

        rnn_train_losses.append(train_loss)
        rnn_val_losses.append(val_loss)
        rnn_val_accuracies.append(val_accuracy)
        rnn_val_aucs.append(val_auc)

        print(f"RNN Fold {fold} Epoch {epoch+1}: Train Loss = {train_loss:.4f}, "
              f"Val Loss = {val_loss:.4f}, Val Accuracy = {val_accuracy:.4f}, Val AUC = {val_auc:.4f}")

    # After training, plot metrics
    plot_metrics(fold, "RNN", rnn_train_losses, rnn_val_losses, rnn_val_accuracies)

    # Save average validation accuracy and AUC across epochs
    avg_val_acc = np.mean(rnn_val_accuracies)
    avg_val_auc = np.mean(rnn_val_aucs)
    print(f"RNN Fold {fold} Average Validation Accuracy Across Epochs: {avg_val_acc:.4f}")
    print(f"RNN Fold {fold} Average Validation AUC Across Epochs: {avg_val_auc:.4f}")

    # Test evaluation
    model.eval()
    test_accuracy = 0
    y_true, y_pred_all = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch).squeeze()
            test_accuracy += (y_pred.argmax(dim=1) == y_batch).float().mean().item()
            y_true.extend(y_batch.cpu().numpy())
            y_pred_all.extend(y_pred.argmax(dim=1).cpu().numpy())
    test_accuracy /= len(test_loader)
    test_cm = confusion_matrix(y_true, y_pred_all)
    test_auc = roc_auc_score(y_true, y_pred_all)

    print(f"RNN Fold {fold} Test Accuracy: {test_accuracy:.4f}")
    print(f"RNN Fold {fold} Test AUC: {test_auc:.4f}")
    print(f"RNN Fold {fold} Test Confusion Matrix:\n{test_cm}")
    print("\n")