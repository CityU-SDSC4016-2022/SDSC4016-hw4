import json
import os
import random
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, random_split


class SpeakerDataset(Dataset):
    def __init__(self, data_dir: str, segment_len: int = 128):
        self.data_dir = data_dir
        self.segment_len = segment_len

        # Load the mapping from speaker name to their corresponding id.
        mapping_path = Path(data_dir) / "mapping.json"
        mapping = json.load(mapping_path.open())
        self.speaker2id = mapping["speaker2id"]

        # Load metadata of training data.
        metadata_path = Path(data_dir) / "metadata.json"
        metadata = json.load(metadata_path.open())
        self.speakers = metadata["speakers"]

        # Get the total number of speaker.
        self.speaker_num = len(self.speakers.keys())

        self.data = []
        for speaker in self.speakers.keys():
            for utterances in self.speakers[speaker]:
                self.data.append([utterances["feature_path"], self.speaker2id[speaker]])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        feat_path, speaker = self.data[index]
        # Load preprocessed mel-spectrogram.
        mel = torch.load(os.path.join(self.data_dir, feat_path))

        # Segmemt mel-spectrogram into "segment_len" frames.
        if len(mel) > self.segment_len:
            # Randomly get the starting point of the segment.
            start = random.randint(0, len(mel) - self.segment_len)
            # Get a segment with "segment_len" frames.
            mel = torch.FloatTensor(mel[start: start + self.segment_len])
        else:
            mel = torch.FloatTensor(mel)
        # Turn the speaker id into long for computing loss later.
        speaker = torch.FloatTensor([speaker]).long()
        return mel, speaker

    def get_speaker_number(self):
        return self.speaker_num


class InferenceDataset(Dataset):
    def __init__(self, data_dir: str):
        # Load metadata of training data.
        testdata_path = Path(data_dir) / "testdata.json"
        metadata = json.load(testdata_path.open())
        self.data_dir = data_dir
        self.data = metadata["utterances"]

        # Load the mapping from id to their corresponding speaker name.
        mapping_path = Path(data_dir) / "mapping.json"
        mapping = json.load(mapping_path.open())
        self.id2speaker = mapping["id2speaker"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        utterance = self.data[index]
        feat_path = utterance["feature_path"]
        mel = torch.load(os.path.join(self.data_dir, feat_path))
        return feat_path, mel

    def get_speaker2id(self):
        return self.id2speaker


def collate_batch(batch: list):
    # Process features within a batch.
    """Collate a batch of data."""
    mel, speaker = zip(*batch)
    # Because we train the model batch by batch, we need to pad the features in the same batch to make their lengths the same.
    mel = pad_sequence(mel, batch_first=True, padding_value=-20)    # pad log 10^(-20) which is very small value.
    # mel: (batch size, length, 40)
    return mel, torch.FloatTensor(speaker).long()


def inf_collate_batch(batch):
    """Collate a batch of data."""
    feat_paths, mels = zip(*batch)
    return feat_paths, torch.stack(mels)


def get_train_dataloader(data_dir: str, batch_size: int, n_workers: int):
    """Generate dataloader"""
    dataset = SpeakerDataset(data_dir)
    speaker_num = dataset.get_speaker_number()
    # Split dataset into training dataset and validation dataset with ratio 9:1
    trainlen = int(0.9 * len(dataset))
    lengths = [trainlen, len(dataset) - trainlen]
    trainset, validset = random_split(dataset, lengths)

    train_loader = DataLoader(trainset, batch_size, num_workers=n_workers, drop_last=True, pin_memory=True, collate_fn=collate_batch, shuffle=True,)
    valid_loader = DataLoader(validset, batch_size, num_workers=n_workers, drop_last=True, pin_memory=True, collate_fn=collate_batch)

    return train_loader, valid_loader, speaker_num


def get_test_dataloader(data_dir: str, batch_size: int, n_workers: int):
    """Generate dataloader"""
    dataset = InferenceDataset(data_dir)
    id2speaker = dataset.get_speaker2id()
    dataloader = DataLoader(dataset, batch_size, num_workers=n_workers, drop_last=False, collate_fn=inf_collate_batch, shuffle=False)
    return dataloader, id2speaker
