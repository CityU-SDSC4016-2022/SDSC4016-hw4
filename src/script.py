import torch

from data import get_test_dataloader, get_train_dataloader
from model import Classifier
from test import testing
from train import training
from utils import save_results

# Install conformer from pip
# !pip install conformer


def main():
    """Main function."""

    # Arguments
    data_dir = "./data/Dataset/"        # Data directory, default is "./data/Dataset/"
    save_path = "./model.ckpt"          # Model path, default is "./model.ckpt"
    output_path = "./output.csv"        # Output file path, default is "./output.csv"
    batch_size = 128                    # Batch size, default is 32
    n_workers = 8                       # Number of workers, default is 8
    warmup_steps = 1000                 # Warmup steps, default is 1000
    valid_steps = 2000                  # Validating steps, default is 2000
    save_steps = valid_steps * 5        # Saving steps, default is Validating steps * 5
    total_steps = save_steps * 25       # Total steps, default is Saving steps * 7
    learning_rate = 1e-3                # Learning rate, default is 1e-3
    d_model = 512                       # Dimension of model, default is 40
    dim_head = 64                       # Dimension of head, default is 64
    head = 8                            # Head size, default is 8
    ff_mult = 4                         # Multi-head feedforward, default is 4
    conv_exp = 2                        # Conv layer expansion, default is 2
    conv_k_size = 32                    # Conv layer kernel size, default is 31
    dropout = 0.1                       # Dropout rate, default is 0

    # Load Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info]: Use {device} now!")

    # Training Part
    train_loader, valid_loader, speaker_num = get_train_dataloader(data_dir, batch_size, n_workers)
    print("[Info]: Finish loading data!", flush=True)

    model = Classifier(d_model, speaker_num, dim_head, head, ff_mult, conv_exp, conv_k_size, dropout).to(device)
    print("[Info]: Finish creating model!", flush=True)

    steps = [warmup_steps, valid_steps, save_steps, total_steps]
    training(train_loader, valid_loader, model, device, learning_rate, steps, save_path)
    print("[Info]: Finish training model!", flush=True)

    # Testing Part
    test_loader, speak2id = get_test_dataloader(data_dir, 1, n_workers)
    print("[Info]: Finish loading data!", flush=True)

    speaker_num = len(speak2id)
    model = Classifier(d_model, speaker_num, dim_head, head, ff_mult, conv_exp, conv_k_size, dropout).to(device)
    model.load_state_dict(torch.load(save_path))
    model.eval()
    print("[Info]: Finish loading model!", flush=True)

    result = testing(test_loader, speak2id, model, device)
    print("[Info]: Finish testing dataset!", flush=True)

    # Save Result as CSV
    save_results(result, output_path)


if __name__ == "__main__":
    main()
