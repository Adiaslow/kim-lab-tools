"""Domain-Adversarial Neural Network trainer implementation."""

from typing import Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..models.discriminator import DomainDiscriminator


class DANNTrainer:
    """Trainer for Domain-Adversarial Neural Networks."""

    def __init__(
        self,
        unet: nn.Module,
        num_classes: int,
        device: torch.device,
        unet_lr: float = 0.01,
        disc_lr: float = 0.1,
        domain_weight: float = 1.0,
    ):
        """Initialize the trainer.

        Args:
            unet: UNet model with GRL.
            num_classes: Number of segmentation classes.
            device: Device to train on.
            unet_lr: Learning rate for UNet.
            disc_lr: Learning rate for discriminator.
            domain_weight: Weight for domain loss.
        """
        self.unet = unet.to(device)
        self.domain_classifier = DomainDiscriminator(128).to(device)
        self.device = device
        self.num_classes = num_classes
        self.domain_weight = domain_weight

        # Loss functions
        self.segmentation_loss = nn.CrossEntropyLoss()
        self.domain_loss = nn.BCELoss()

        # Optimizers
        self.unet_optimizer = optim.SGD(unet.parameters(), lr=unet_lr, momentum=0.9)
        self.domain_optimizer = optim.SGD(
            self.domain_classifier.parameters(), lr=disc_lr, momentum=0.9
        )

    def train_epoch(
        self,
        source_dataloader: DataLoader,
        target_dataloader: DataLoader,
        alpha: float = 2.0,
    ) -> tuple[float, float]:
        """Train for one epoch.

        Args:
            source_dataloader: DataLoader for source domain.
            target_dataloader: DataLoader for target domain.
            alpha: GRL alpha parameter.

        Returns:
            Tuple of (domain loss, segmentation loss).
        """
        total_domain_loss = 0.0
        total_seg_loss = 0.0
        num_source = len(source_dataloader.dataset)
        num_total = num_source + len(target_dataloader.dataset)

        pbar = tqdm(source_dataloader, desc="Training")
        for i, (source_images, source_labels) in enumerate(pbar):
            # Move to device
            source_images = source_images.to(self.device)
            source_labels = self._prepare_labels(source_labels)

            # Get target batch
            target_images = next(iter(target_dataloader)).to(self.device)

            # Forward passes
            source_predictions = self.unet(source_images, alpha=alpha)
            seg_loss = self.segmentation_loss(source_predictions, source_labels)

            # Domain adaptation
            source_features = self.unet.get_features(source_images)
            target_features = self.unet.get_features(target_images)

            domain_loss = self._compute_domain_loss(
                source_features[1], target_features[1], source_images.size(0)
            )

            # Update networks
            total_loss = seg_loss + self.domain_weight * domain_loss
            self._update_networks(total_loss, domain_loss)

            # Track losses
            total_domain_loss += domain_loss.item() * source_images.size(0)
            total_seg_loss += seg_loss.item() * source_images.size(0)

            # Update progress bar
            pbar.set_postfix(
                {
                    "Domain Loss": f"{total_domain_loss/num_total:.4f}",
                    "Seg Loss": f"{total_seg_loss/num_source:.4f}",
                }
            )

        return total_domain_loss / num_total, total_seg_loss / num_source

    def train(
        self,
        source_dataloader: DataLoader,
        target_dataloader: DataLoader,
        num_epochs: int,
        save_path: Optional[Path] = None,
    ) -> None:
        """Train the network.

        Args:
            source_dataloader: DataLoader for source domain.
            target_dataloader: DataLoader for target domain.
            num_epochs: Number of epochs to train.
            save_path: Optional path to save models to.
        """
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            domain_loss, seg_loss = self.train_epoch(
                source_dataloader, target_dataloader
            )

            if save_path:
                torch.save(self.unet.state_dict(), save_path / "adapted_unet.pth")
                torch.save(
                    self.domain_classifier.state_dict(),
                    save_path / "domain_classifier.pth",
                )

    def _prepare_labels(self, labels: torch.Tensor) -> torch.Tensor:
        """Prepare labels for loss computation.

        Args:
            labels: Input labels tensor.

        Returns:
            Processed labels tensor.
        """
        return (
            nn.functional.one_hot(labels.long(), num_classes=self.num_classes)
            .permute(0, 4, 1, 2, 3)
            .reshape([labels.shape[0], self.num_classes, 256, 256])
            .float()
            .to(self.device)
        )

    def _compute_domain_loss(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """Compute domain adaptation loss.

        Args:
            source_features: Features from source domain.
            target_features: Features from target domain.
            batch_size: Batch size.

        Returns:
            Domain loss tensor.
        """
        source_domain_labels = torch.ones(batch_size, 1).to(self.device)
        target_domain_labels = torch.zeros(batch_size, 1).to(self.device)

        source_domain_preds = self.domain_classifier(source_features.detach())
        target_domain_preds = self.domain_classifier(target_features.detach())

        return self.domain_loss(
            source_domain_preds, source_domain_labels
        ) + self.domain_loss(target_domain_preds, target_domain_labels)

    def _update_networks(
        self, total_loss: torch.Tensor, domain_loss: torch.Tensor
    ) -> None:
        """Update network parameters.

        Args:
            total_loss: Combined loss for UNet.
            domain_loss: Domain loss for discriminator.
        """
        # Update UNet
        self.unet_optimizer.zero_grad()
        total_loss.backward(retain_graph=True)
        self.unet_optimizer.step()

        # Update Domain Classifier
        self.domain_optimizer.zero_grad()
        domain_loss.backward()
        self.domain_optimizer.step()
