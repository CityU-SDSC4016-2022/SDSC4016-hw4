import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import Classifier


def testing(test_loader: DataLoader, id2speaker, model: Classifier, device: torch.device):
    """Test on test set."""
    results = [["Id", "Category"]]
    for feat_paths, mels in tqdm(test_loader):
        with torch.no_grad():
            mels = mels.to(device)
            outs = model(mels)
            preds = outs.argmax(1).cpu().numpy()
            for feat_path, pred in zip(feat_paths, preds):
                results.append([feat_path, id2speaker[str(pred)]])
    return results
