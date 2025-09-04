import random

import torchaudio
import datasets
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from tokenizers.processors import TemplateProcessing
from pathlib import Path
import sounddevice as sd


# Custom collate function for DataLoader
# Takes a list of dictionaries (batch) and pads audio and input_ids
# to ensure uniform tensor shapes within the batch.
def collate_fn(batch):
    # Get max audio length in the current batch
    max_audio_len = max([item["audio"].shape[0] for item in batch])

    # Check if 'input_ids' are present in the batch items
    max_ids_len = 0
    has_input_ids = "input_ids" in batch[0]
    if has_input_ids:
        # Get max input_ids length if present
        max_ids_len = max([len(item["input_ids"]) for item in batch])

    # Pad audio sequences to the max length with zeros
    audio_tensor = torch.stack(
        [
            F.pad(item["audio"], (0, max_audio_len - item["audio"].shape[0]))
            for item in batch
        ]
    )

    # Initialize the output dictionary with padded audio and original text
    output_dict = {
        "audio": audio_tensor,
        "text": [item["text"] for item in batch],
    }

    # Handle input_ids padding if present
    if has_input_ids:
        # Pad input_ids sequences to the max length with the padding value (0)
        input_ids = torch.stack(
            [
                F.pad(
                    torch.tensor(item["input_ids"]),
                    (0, max_ids_len - len(item["input_ids"])),
                    value=0,  # Use 0 for padding token ID
                )
                for item in batch
            ]
        )
        output_dict["input_ids"] = input_ids

    # Return the processed batch dictionary
    return output_dict


# Function to create or load a BPE tokenizer
def get_tokenizer(save_path="tokenizer.json"):
    # Initialize a BPE (Byte-Pair Encoding) model
    tokenizer = Tokenizer(models.BPE())
    # Add a special blank token '□' often used in CTC loss
    tokenizer.add_special_tokens(["□"])
    # Add uppercase letters, space, and apostrophe as basic tokens
    tokenizer.add_tokens(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ '"))

    # Set up pre-tokenization and decoding using ByteLevel (handles raw bytes)
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    tokenizer.decoder = decoders.ByteLevel()
    # Assign the ID of the blank token for easy access
    tokenizer.blank_token = tokenizer.token_to_id("□")
    # Save the tokenizer configuration to the specified path
    tokenizer.save(save_path)
    # Return the configured tokenizer instance
    return tokenizer


# Custom Dataset class for Common Voice, integrating the tokenizer
class CommonVoiceDataset(Dataset):
    def __init__(
        self,
        common_voice_dataset,  # The dataset object from `datasets` library
        num_examples=None,  # Optional limit on the number of examples to use
        tokenizer=None,  # The tokenizer instance to use for encoding text
    ):
        self.dataset = common_voice_dataset
        # Determine the number of examples: either the limit or the full dataset size
        self.num_examples = (
            min(num_examples, len(common_voice_dataset))
            if num_examples is not None
            else len(common_voice_dataset)
        )
        self.tokenizer = tokenizer

    def __len__(self):
        # Return the effective size of the dataset
        return self.num_examples

    def __getitem__(self, idx):
        # Retrieve a single item from the original dataset by index
        item = self.dataset[idx]
        # Convert the audio numpy array to a PyTorch float tensor
        waveform = torch.from_numpy(item["audio"]["array"]).float()
        # Get the transcription and convert it to uppercase
        text = item["transcription"].upper()  # Assuming Common Voice provides lowercase

        # If a tokenizer is provided, encode the text
        if self.tokenizer:
            encoded = self.tokenizer.encode(text)
            # Return audio, original text, and tokenized input_ids
            return {"audio": waveform, "text": text, "input_ids": encoded.ids}

        # If no tokenizer, return only audio and original text
        return {"audio": waveform, "text": text}


# Function to load the dataset, create the tokenizer, and return a DataLoader
def get_dataset(
    batch_size=32,  # Batch size for the DataLoader
    num_examples=None,  # Optional limit on the number of examples
    num_workers=4,  # Number of worker processes for data loading
):
    # Load the specified Common Voice dataset split from Hugging Face datasets
    # (using a local or cached version if available)
    dataset = datasets.load_dataset(
        "m-aliabbas/idrak_timit_subsample1",  # Dataset identifier
        split="train",  # Specify the 'train' split
    )
    # Get the tokenizer (creates or loads it)
    tokenizer = get_tokenizer()

    dataset = CommonVoiceDataset(
        dataset,
        tokenizer=tokenizer,  # Pass the tokenizer for text encoding
        num_examples=num_examples,  # Pass the example limit
    )

    # Create a PyTorch DataLoader for batching and parallel loading
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # Keep shuffle=False for reproducibility or specific needs
        collate_fn=collate_fn,  # Use the custom collate function for padding
        num_workers=num_workers,  # Use multiple workers for faster loading
    )
    # Return the configured DataLoader
    return dataloader


# Main execution block for testing or demonstration purposes
if __name__ == "__main__":
    dataloader = get_dataset(batch_size=32)
    for batch in dataloader:
        audio = batch["audio"]
        input_ids = batch["input_ids"]
        print("Audio batch shape:", audio.shape)
        print("Input IDs batch shape:", input_ids.shape)

        # breakpoint() # Useful for debugging the batch content
        break