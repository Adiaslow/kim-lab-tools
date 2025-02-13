"""Base trainer implementation."""

import wandb
import torch
from PIL import Image
from pathlib import Path
from typing import Optional, List, Any
from torch.utils.data import DataLoader
from torch.nn import Module
from torch.optim import Optimizer


class BaseTrainer:
    """Base trainer for deep learning models."""

    def __init__(
        self,
        model: Module,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        criterion: Module,
        optimizer: Optimizer,
        device: str = "cuda",
        project_name: str = "default_project",
    ):
        """Initialize trainer.

        Args:
            model: Model to train.
            train_loader: Training data loader.
            valid_loader: Validation data loader.
            criterion: Loss function.
            optimizer: Optimizer.
            device: Device to use.
            project_name: WandB project name.
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.epoch = 0
        self.best_loss = float("inf")

        # Initialize wandb
        wandb.init(project=project_name)
        wandb.watch(self.model)

    def train_one_epoch(self) -> float:
        """Train model for one epoch.

        Returns:
            Average training loss.
        """
        self.model.train()
        train_loss = 0.0

        for batch, (samples, labels) in enumerate(self.train_loader):
            print(
                f"Train | Epoch: {self.epoch}, "
                f"Batch: {batch}/{len(self.train_loader)}",
                end="\r",
            )
            samples = samples.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(samples)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item() * samples.size(0)

        avg_loss = train_loss / len(self.train_loader.dataset)
        print(f"\nTrain Loss: {avg_loss}")
        self.epoch += 1

        return avg_loss

    def validate(self) -> float:
        """Validate model.

        Returns:
            Average validation loss.
        """
        self.model.eval()
        valid_loss = 0.0

        with torch.no_grad():
            for batch, (samples, labels) in enumerate(self.valid_loader):
                print(f"Valid | Batch: {batch}/{len(self.valid_loader)}", end="\r")
                samples = samples.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(samples)
                loss = self.criterion(outputs, labels)
                valid_loss += loss.item() * samples.size(0)

        avg_loss = valid_loss / len(self.valid_loader.dataset)
        print(f"\nValid Loss: {avg_loss}")
        return avg_loss

    def should_continue_training(
        self,
        validation_losses: List[float],
        patience: int = 5,
        min_delta: float = 0.01,
    ) -> bool:
        """Check if training should continue.

        Args:
            validation_losses: History of validation losses.
            patience: Epochs to wait for improvement.
            min_delta: Minimum improvement required.

        Returns:
            Whether to continue training.
        """
        if len(validation_losses) < patience:
            return True

        latest_losses = validation_losses[-patience:]
        return not all(
            abs(latest_losses[i] - latest_losses[i + 1]) < min_delta
            for i in range(len(latest_losses) - 1)
        )

    def run(
        self,
        epochs: int,
        checkpoint_dir: Optional[Path] = None,
        patience: int = 20,
        min_delta: float = 1e-8,
    ) -> None:
        """Run training loop.

        Args:
            epochs: Number of epochs to train.
            checkpoint_dir: Directory to save checkpoints.
            patience: Early stopping patience.
            min_delta: Minimum improvement for early stopping.
        """
        prior_loss = []
        checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else Path.cwd()

        try:
            for _ in range(epochs):
                train_loss = self.train_one_epoch()
                valid_loss = self.validate()
                prior_loss.append(valid_loss)

                # Save best model
                if valid_loss < self.best_loss:
                    self.best_loss = valid_loss
                    self.save_checkpoint(checkpoint_dir / "best_model.pt")

                # Early stopping check
                if not self.should_continue_training(
                    prior_loss, patience=patience, min_delta=min_delta
                ):
                    print("Early stopping triggered")
                    break

                # Log metrics
                wandb.log(
                    {
                        "train_loss": train_loss,
                        "valid_loss": valid_loss,
                    }
                )

        except KeyboardInterrupt:
            print("\nTraining interrupted")

        finally:
            wandb.finish()

    def show_reconstruction(self, dataloader: DataLoader) -> None:
        """Show sample reconstruction.

        Args:
            dataloader: Data loader for samples.
        """
        self.model.eval()
        with torch.no_grad():
            for samples, _ in dataloader:
                samples = samples.to(self.device)
                outputs = self.model(samples)

                # Convert to images
                sample = Image.fromarray(
                    (samples[0][0].cpu().numpy() * 255).astype("uint8")
                ).convert("RGB")
                output = Image.fromarray(
                    (outputs[0][0].cpu().numpy() * 255).astype("uint8")
                ).convert("RGB")

                sample.show()
                output.show()
                break

    def save_checkpoint(self, path: Path) -> None:
        """Save model checkpoint.

        Args:
            path: Path to save checkpoint.
        """
        torch.save(
            {
                "epoch": self.epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_loss": self.best_loss,
            },
            path,
        )

    def load_checkpoint(self, path: Path) -> None:
        """Load model checkpoint.

        Args:
            path: Path to checkpoint file.
        """
        checkpoint = torch.load(path)
        self.epoch = checkpoint["epoch"]
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.best_loss = checkpoint["best_loss"]
