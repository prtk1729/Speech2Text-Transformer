import os

# Set environment variables for tokenizer parallelism and PyTorch MPS fallback
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Import TensorBoard for logging
from torch.utils.tensorboard import SummaryWriter


import torch

# Enable anomaly detection for debugging gradients
torch.autograd.set_detect_anomaly(True)
# Import custom modules for dataset, tokenizer, and the model
from dataset import get_dataset, get_tokenizer
from transcribe_model import TranscribeModel

# Import PyTorch neural network module
from torch import nn

# --- Hyperparameters for VQ (Vector Quantization) Loss ---
# Initial weight for the VQ loss component
vq_initial_loss_weight = 10
# Number of steps over which to linearly decrease the VQ loss weight
vq_warmup_steps = 1000
# Final weight for the VQ loss component after warmup
vq_final_loss_weight = 0.5

# --- Training Configuration ---
# Total number of training epochs
num_epochs = 100
# Starting step count (useful for resuming training)
starting_steps = 0
# Number of examples to use from the dataset (can be smaller for faster iteration)
num_examples = None
# Identifier for the current training run/model version
model_id = "test1"
# Number of times to repeat each batch within an epoch (for overfitting a single batch)
num_batch_repeats = 1

starting_steps = 0
# Number of samples per batch
BATCH_SIZE = 64
# Learning rate for the Adam optimizer
LEARNING_RATE = 0.005


# --- CTC Loss Function ---
def run_loss_function(log_probs, target, blank_token):
    """
    Calculates the Connectionist Temporal Classification (CTC) loss.
    Args:
        log_probs: Log probabilities output by the model (Batch, Time, VocabSize).
        target: Target token sequences (Batch, TargetLength).
        blank_token: ID of the blank token used in CTC.
    Returns:
        torch.Tensor: The calculated CTC loss.
    """
    # Instantiate the CTC Loss function
    loss_function = nn.CTCLoss(blank=blank_token)
    # Calculate input lengths (all sequences in the batch have the same time dimension length)
    input_lengths = tuple(log_probs.shape[1] for _ in range(log_probs.shape[0]))
    # Calculate target lengths (sum of non-blank tokens for each sequence)
    target_lengths = (target != blank_token).sum(dim=1)
    target_lengths = tuple(t.item() for t in target_lengths)
    # Permute log_probs to (Time, Batch, VocabSize) as required by nn.CTCLoss
    input_seq_first = log_probs.permute(1, 0, 2)
    # Calculate and return the CTC loss
    loss = loss_function(input_seq_first, target, input_lengths, target_lengths)
    return loss


# --- Main Training Function ---
def main():
    """
    Sets up and runs the main training loop.
    """
    # --- Setup TensorBoard ---
    log_dir = f"runs/speech2text_training/{model_id}"
    # Remove previous logs if they exist
    if os.path.exists(log_dir):
        import shutil

        shutil.rmtree(log_dir)
    # Create a SummaryWriter instance for logging
    writer = SummaryWriter(log_dir)

    # --- Tokenizer ---
    # Load the tokenizer
    tokenizer = get_tokenizer()
    # Get the ID for the blank token
    blank_token = tokenizer.token_to_id("□")

    # --- Device Selection ---
    # Select GPU (CUDA or MPS) if available, otherwise fallback to CPU
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    # --- Model Loading / Initialization ---
    model_path = f"models/{model_id}/model_latest.pth"
    # Check if a saved model checkpoint exists
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        # Load the existing model
        model = TranscribeModel.load(model_path).to(device)
    else:
        print("Initializing a new model")
        # Initialize a new TranscribeModel instance
        model = TranscribeModel(
            num_codebooks=3,
            codebook_size=48,
            embedding_dim=48,
            num_transformer_layers=3,
            vocab_size=len(tokenizer.get_vocab()),
            strides=[6, 6, 6],
            initial_mean_pooling_kernel_size=4,
            max_seq_length=400,
        ).to(device)

    # Calculate and print the number of trainable parameters
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_trainable_params}")

    # --- Optimizer ---
    # Initialize the Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- Dataloader ---
    # Get the data loader for the training dataset
    dataloader = get_dataset(
        batch_size=BATCH_SIZE,
        num_examples=num_examples,
        num_workers=1,  # Number of worker processes for data loading
    )

    # --- Training Loop Initialization ---
    # Lists to store losses for averaging
    ctc_losses = []
    vq_losses = []
    # Total number of batches per epoch
    num_batches = len(dataloader)
    # Initialize step counter
    steps = starting_steps

    # --- Epoch Loop ---
    for i in range(num_epochs):
        print(f"--- Starting Epoch {i+1}/{num_epochs} ---")
        # --- Batch Loop ---
        for idx, batch in enumerate(dataloader):
            # --- Batch Repetition Loop ---
            # Repeat the current batch multiple times
            for repeat_batch in range(num_batch_repeats):
                # Extract audio, target tokens, and text from the batch
                audio = batch["audio"]
                target = batch["input_ids"]
                text = batch["text"]

                # --- Data Preprocessing ---
                # Pad audio if the target sequence is longer (ensure model input dimensions are sufficient)
                if target.shape[1] > audio.shape[1]:
                    print(
                        "Padding audio, target is longer than audio. Audio Shape: ",
                        audio.shape,
                        "Target Shape: ",
                        target.shape,
                    )
                    # Pad the time dimension of the audio tensor
                    audio = torch.nn.functional.pad(
                        audio, (0, 0, 0, target.shape[1] - audio.shape[1])
                    )
                    print("After padding: ", audio.shape)

                # Move data to the selected device
                audio = audio.to(device)
                target = target.to(device)

                # --- Forward Pass & Loss Calculation ---
                # Reset gradients
                optimizer.zero_grad()
                # Perform the forward pass through the model
                output, vq_loss = model(audio)
                # Calculate the CTC loss
                ctc_loss = run_loss_function(output, target, blank_token)

                # Calculate the dynamic VQ loss weight using a linear warmup/cooldown schedule
                vq_loss_weight = max(
                    vq_final_loss_weight,
                    vq_initial_loss_weight
                    - (vq_initial_loss_weight - vq_final_loss_weight)
                    * (steps / vq_warmup_steps),
                )

                # Combine CTC and VQ loss (if VQ is used)
                if vq_loss is None:
                    loss = ctc_loss
                else:
                    loss = ctc_loss + vq_loss_weight * vq_loss

                # Skip step if loss becomes infinite (potential numerical instability)
                if torch.isinf(loss):
                    print("Loss is inf, skipping step", audio.shape, target.shape)
                    continue

                # --- Backward Pass & Optimization ---
                # Compute gradients
                loss.backward()
                # Clip gradients to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=10.0
                )  # Increased max_norm
                # Update model parameters
                optimizer.step()

                # --- Logging & Tracking ---
                # Append current batch losses
                ctc_losses.append(ctc_loss.item())
                if vq_loss is not None:  # Only append if vq_loss is calculated
                    vq_losses.append(vq_loss.item())
                # Increment step counter
                steps += 1

                # Log average losses to console and TensorBoard periodically
                if steps % 20 == 0:
                    avg_ctc_loss = (
                        sum(ctc_losses) / len(ctc_losses) if ctc_losses else 0
                    )
                    avg_vq_loss = sum(vq_losses) / len(vq_losses) if vq_losses else 0
                    # Recalculate average total loss based on current weight
                    avg_loss = avg_ctc_loss + vq_loss_weight * avg_vq_loss
                    print(
                        f"Epoch: {i+1}, Step: {steps}, Batch: {idx + 1}/{num_batches}, "
                        f"ctc_loss: {avg_ctc_loss:.3f}, vq_loss: {avg_vq_loss:.3f}, "
                        f"total_loss: {avg_loss:.3f}, vq_weight: {vq_loss_weight:.3f}"
                    )
                    # Reset loss lists for the next averaging window
                    ctc_losses = []
                    vq_losses = []
                    # Write losses to TensorBoard
                    writer.add_scalar("Loss/CTC", avg_ctc_loss, steps)
                    writer.add_scalar("Loss/VQ", avg_vq_loss, steps)
                    writer.add_scalar("Loss/Total", avg_loss, steps)
                    writer.add_scalar("Params/VQ_Weight", vq_loss_weight, steps)

                # Periodically evaluate and display transcription examples
                if steps % 50 == 0:
                    print(f"Saving model checkpoint at step {steps}")
                    os.makedirs(f"models/{model_id}", exist_ok=True)
                    # model.save(f"models/{model_id}/model_{steps}.pth")
                    model.save(f"models/{model_id}/model_latest.pth")

                    # --- Transcription Example Display ---
                    # Ensure evaluation is done without gradient calculation
                    with torch.no_grad():
                        # Run model inference on the current batch's audio
                        outs, _ = model(
                            audio
                        )  # Re-run inference or use 'output' from training? Using 'output' might be slightly stale. Re-running.
                        # Get the most likely token IDs from the output probabilities
                        tokens = torch.argmax(outs, dim=2)

                        # Move tokens to CPU and convert to numpy for decoding
                        tokens = tokens.detach().cpu().numpy()
                        # Decode the token sequences back into text
                        tokens = tokenizer.decode_batch(tokens)

                        # Use Rich library to display examples in a formatted table
                        from rich.table import Table
                        from rich.console import Console

                        table = Table(title=f"Transcription Examples (Step {steps})")
                        table.add_column("Example #", justify="right", style="cyan")
                        table.add_column("Model Output", style="green")
                        table.add_column("Ground Truth", style="yellow")

                        # Display the first 4 examples from the batch
                        for example_idx in range(
                            min(4, BATCH_SIZE)
                        ):  # Ensure we don't exceed batch size
                            table.add_row(
                                str(example_idx),
                                tokens[example_idx],
                                text[example_idx],  # Ground truth text from the batch
                            )

                        console = Console()
                        console.print(table)  # Print the table to the console


# --- Script Entry Point ---
if __name__ == "__main__":
    # Call the main function when the script is executed directly
    main()