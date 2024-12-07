import torch
import matplotlib.pyplot as plt
import os

def evaluate_model(model, criterion, X_test_tensor, y_test_tensor, target_scaler, folder_name, epoch_number=1):
    """
    Evaluate the model on the test dataset.
    """
    model.eval()

    # Ensure X_test_tensor has correct shape: (N_test, seq_len, num_features)
    # If there's only one sample and it's missing batch dimension:
    if X_test_tensor.ndimension() == 2:
        # This means shape is (seq_len, features), reshape to (1, seq_len, features)
        X_test_tensor = X_test_tensor.unsqueeze(0)
    elif X_test_tensor.ndimension() == 1:
        # If it's completely flat, this is likely incorrect data.
        raise ValueError("X_test_tensor is 1D. It should be at least 2D or 3D.")

    with torch.no_grad():
        predictions = model(X_test_tensor)
        test_loss = criterion(predictions, y_test_tensor).item()

    predictions = predictions.cpu().numpy()
    y_test = y_test_tensor.cpu().numpy()

    # If we have a target scaler, inverse transform
    if target_scaler is not None:
        predictions = target_scaler.inverse_transform(predictions)
        y_test = target_scaler.inverse_transform(y_test)

    # Plot predicted vs real close
    plt.figure(figsize=(10, 6))
    plt.plot(y_test, label='Real Close')
    plt.plot(predictions, label='Predicted Close')
    plt.title(f'Epoch {epoch_number}: Predictions vs Real Close')
    plt.xlabel('Time Steps')
    plt.ylabel('Close Price')
    plt.legend()

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    file_name = f'Epoch{epoch_number}.png'
    file_path = os.path.join(folder_name, file_name)
    plt.savefig(file_path)
    plt.close()

    print(f"Test Loss at Epoch {epoch_number}: {test_loss}")
    print(f"Plot saved: {file_path}")

    return test_loss
