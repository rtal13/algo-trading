import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from math import sqrt
import matplotlib.pyplot as plt

def plot_epoch_results(epoch, folder_name, 
                       scaled_targets, scaled_predictions, 
                       non_scaled_targets, non_scaled_predictions,
                       residuals, rmse, mae, mape):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    # Top Left: Non-scaled real vs predicted
    axs[0, 0].plot(non_scaled_targets, label='Real (Non-Scaled)')
    axs[0, 0].plot(non_scaled_predictions, label='Predicted (Non-Scaled)')
    axs[0, 0].set_title('Non-Scaled Real vs Predicted')
    axs[0, 0].set_xlabel('Time Steps')
    axs[0, 0].set_ylabel('Close Price')
    axs[0, 0].legend()

    metrics_text = f"Epoch: {epoch}\nRMSE: {rmse:.4f}\nMAE: {mae:.4f}\nMAPE: {mape:.2f}%"
    axs[0, 0].text(0.05, 0.95, metrics_text, transform=axs[0, 0].transAxes,
                   fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Top Right: Scaled real vs predicted
    axs[0, 1].plot(scaled_targets, label='Real (Scaled)')
    axs[0, 1].plot(scaled_predictions, label='Predicted (Scaled)')
    axs[0, 1].set_title('Scaled Real vs Predicted')
    axs[0, 1].set_xlabel('Time Steps')
    axs[0, 1].set_ylabel('Scaled Value')
    axs[0, 1].legend()

    # Bottom Left: Residuals
    axs[1, 0].plot(residuals, label='Residuals (Non-Scaled)')
    axs[1, 0].set_title('Residuals Over Time')
    axs[1, 0].set_xlabel('Time Steps')
    axs[1, 0].set_ylabel('Difference')
    axs[1, 0].legend()

    # Bottom Right: Histogram of residuals
    axs[1, 1].hist(residuals, bins=30, alpha=0.7, color='g')
    axs[1, 1].set_title('Residuals Distribution')
    axs[1, 1].set_xlabel('Residual')
    axs[1, 1].set_ylabel('Frequency')

    plt.tight_layout()
    plot_path = os.path.join(folder_name, f'Epoch{epoch}.png')
    plt.savefig(plot_path)
    plt.close(fig)
    print(f"Plot saved: {plot_path}")


def train_model(
    model, 
    criterion, 
    optimizer, 
    X_train_seq, 
    y_train_seq,
    X_test_seq,
    y_test_seq,
    epochs=50, 
    batch_size=32, 
    patience=5, 
    gradient_clip_value=1.0,
    folder_name='.',
    scheduler_patience=3,
    target_scaler=None
):
    """
    Training loop that uses a separate validation set (X_test_seq, y_test_seq) for early stopping.
    Computes metrics and saves plots based on validation performance each epoch.
    """

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        

    train_dataset = TensorDataset(X_train_seq, y_train_seq)
    val_dataset = TensorDataset(X_test_seq, y_test_seq)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=scheduler_patience, factor=0.5)

    best_val_loss = np.inf
    epochs_no_improve = 0
    best_model_path = os.path.join(folder_name, 'best_model.pt')

    for epoch in range(1, epochs+1):
        # Training
        model.train()
        train_loss = 0.0
        with tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", unit="batch") as t:
            for batch_x, batch_y in t:
                optimizer.zero_grad()
                predictions = model(batch_x)
                loss = criterion(predictions, batch_y)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_value)
                optimizer.step()
                train_loss += loss.item() * batch_x.size(0)
                t.set_postfix(train_loss=loss.item())

        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        predictions_list = []
        targets_list = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                pred = model(batch_x)
                loss = criterion(pred, batch_y)
                val_loss += loss.item() * batch_x.size(0)
                predictions_list.append(pred.cpu().numpy())
                targets_list.append(batch_y.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        predictions_array = np.concatenate(predictions_list, axis=0)  # (val_samples, 1)
        targets_array = np.concatenate(targets_list, axis=0)          # (val_samples, 1)

        print(f"Epoch [{epoch}/{epochs}] - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        scheduler.step(val_loss)

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"Validation loss improved. Model saved: {best_model_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                # We'll plot for the last epoch and then break
                # If you prefer not to plot on the last epoch after triggering, just break here
                pass

        # Compute metrics (based on validation set)
        if target_scaler is not None:
            non_scaled_predictions = target_scaler.inverse_transform(predictions_array)
            non_scaled_targets = target_scaler.inverse_transform(targets_array)
        else:
            non_scaled_predictions = predictions_array
            non_scaled_targets = targets_array

        scaled_predictions = predictions_array
        scaled_targets = targets_array

        residuals = non_scaled_targets - non_scaled_predictions
        mae = np.mean(np.abs(residuals))
        rmse = sqrt(np.mean(residuals**2))
        valid_mape_indices = non_scaled_targets != 0
        if np.any(valid_mape_indices):
            mape = np.mean(np.abs((non_scaled_targets[valid_mape_indices] - non_scaled_predictions[valid_mape_indices]) / 
                                  non_scaled_targets[valid_mape_indices])) * 100
        else:
            mape = np.nan

        # Plot results for validation data
        plot_epoch_results(epoch, folder_name, 
                           scaled_targets, scaled_predictions, 
                           non_scaled_targets, non_scaled_predictions,
                           residuals, rmse, mae, mape)

        if epochs_no_improve >= patience:
            break

    model.load_state_dict(torch.load(best_model_path))
    print(f"Training complete. Best validation loss: {best_val_loss:.4f}")
    return model
