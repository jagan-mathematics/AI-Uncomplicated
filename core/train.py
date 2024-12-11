"""Training implementation"""
import torch
from spacy.cli.train import train

from core.configurations.base import BaseConfiguration
from core.dataloaders.dataloader import dataloader_v1, load_tokenizer
from core.models.model import LLM


def loss_batch_calculation(
        input_batch, target_batch, model, device
):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)['logits']
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = loss_batch_calculation(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def train_model(
        model: LLM,
        train_loader,
        val_loader,
        optimizer,
        device,
        config: BaseConfiguration
):
    train_losses, val_losses = [], []
    global_steps = -1

    for epoch in range(config.num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = loss_batch_calculation(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            global_steps += 1

            # Need to implement evaluation also
            if global_steps % config.eval_frequency == 0:
                train_loss, eval_loss = evaluate_model(
                    model, train_loader, val_loader, device, config.eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(eval_loss)
                print(f"Ep {epoch + 1} (Step {global_steps:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {eval_loss:.3f}")

    return train_losses, val_losses


def main(config: BaseConfiguration):
    """Main function for training"""
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    file_path = "txt path here"

    with open(file_path, "r", encoding="utf-8") as file:
        text_data = file.read()


    # Initialize the model
    model = LLM(config)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    # Setup Dataloaders
    train_ratio = 0.90
    split_idx = int(train_ratio * len(text_data))

    train_loader = dataloader_v1(
        text_data[:split_idx],
        config
    )

    val_loader = dataloader_v1(
        text_data[split_idx:],
        config
    )

    # Train model
    tokenizer = load_tokenizer(config.tokenizer_path)

    train_loss, val_loss = train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        config
    )
    return train_loss, val_loss, model
