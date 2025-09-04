# Speech2Text-Transformer
Speech2Text-Transformer is a PyTorch-based Speech-to-Text system built on transformer architectures developed as part of an MS project at `IIIT-Hyderabad`. It integrates custom dataset handling, byte-pair encoding (BPE) tokenization, and vector quantization (VQ) loss. The project supports the Common Voice / TIMIT dataset and includes an extendable training loop for experimentation.


# Speech2TextTransformer

A PyTorch-based **Speech-to-Text Transformer** that integrates:
- Custom audio preprocessing and batching.
- Byte-Pair Encoding (BPE) tokenizer.
- Transformer-based transcription model with **Vector Quantization (VQ) loss**.
- Flexible training loop with gradient clipping, logging, and checkpointing.

---

## Installation

```bash
# Create environment
conda create -n torch_env python=3.12
conda activate torch_env

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```text
├── train.py            # Main training loop
├── dataset.py          # Dataset and DataLoader utilities
├── downsampling.py        
├── rvq.py     
├── transcribe.py     
├── train.py      
├── self_attention.py   # Transformer-based speech model
├── requirements.txt    # Dependencies
└── README.md           # Project description
```


## Configuration

The following parameters are currently hardcoded in `train.py`:

- **Training Loop**
  - `num_epochs = 100`
  - `BATCH_SIZE = 64`
  - `LEARNING_RATE = 0.005`
  - `num_examples = None`
  - `num_batch_repeats = 10000`
  - `starting_steps = 0`
  - `model_id = "test1"`

- **VQ Loss**
  - `vq_initial_loss_weight = 10`
  - `vq_warmup_steps = 1000`
  - `vq_final_loss_weight = 0.5`

- **Model Architecture**
  - `num_codebooks = 3`
  - `codebook_size = 48`
  - `embedding_dim = 48`
  - `num_transformer_layers = 3`
  - `strides = [6, 6, 6]`
  - `initial_mean_pooling_kernel_size = 4`
  - `max_seq_length = 400`
  - `vocab_size` → dynamically set from tokenizer

- **Miscellaneous**
  - Gradient clipping: `max_norm = 10.0`
  - Logging every 20 steps
  - Evaluation & checkpoint saving every 50 steps
  - First 4 examples displayed during evaluation

---

## Training

Run training with:

```bash
python train.py
```

> [!IMPORTANT]
> - You can modify hyperparameters directly in train.py.


## Example Output

During training:
- **Loss values** are logged every 20 steps.
- **Intermediate transcriptions** are displayed every 50 steps.
- **Checkpoints** are automatically saved.

---

## Future Improvements
- Add CTC loss for alignment-free training.
- Support multilingual datasets (Common Voice, LibriSpeech).
- Implement beam search decoding for better transcription quality.
- Add config file support instead of hardcoded parameters.

---

## 🙌 Contributing
Feel free to open issues or PRs if you’d like to extend this repo with new datasets, models, or features.

---
