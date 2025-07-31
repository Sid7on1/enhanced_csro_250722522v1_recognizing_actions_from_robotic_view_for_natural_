import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Union


class LossFunctions:
    """
    Custom loss functions for enhanced action recognition.

    This class encapsulates various loss functions required for training the
    action recognition model. It aims to provide a flexible and extensible
    interface for different loss scenarios.
    """

    @staticmethod
    def _validate_inputs(inputs: List[torch.Tensor]) -> None:
        """
        Validate the inputs to the loss function.

        Args:
            inputs (List[torch.Tensor]): A list of tensors containing the inputs.

        Raises:
            ValueError: If the input tensors are not valid.
        """
        if not isinstance(inputs, list) or not all(isinstance(t, torch.Tensor) for t in inputs):
            raise ValueError("Inputs must be a list of tensors.")

        if len(inputs) != 6:
            raise ValueError("Expected 6 tensors, received different number of inputs.")

        for tensor in inputs:
            if tensor.dim() != 4:
                raise ValueError("Input tensors should have 4 dimensions.")

    @staticmethod
    def _apply_mask(predictions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Apply the binary mask to the predictions.

        Args:
            predictions (torch.Tensor): Predicted values for action recognition.
            mask (torch.Tensor): Binary mask to apply to predictions.

        Returns:
            torch.Tensor: Masked predictions.
        """
        if predictions.shape != mask.shape:
            raise ValueError("Predictions and mask shapes don't match.")

        return predictions * mask

    @staticmethod
    def _compute_flow(prev_features: torch.Tensor, curr_features: torch.Tensor) -> torch.Tensor:
        """
        Compute optical flow between two feature maps.

        Args:
            prev_features (torch.Tensor): Previous feature map.
            curr_features (torch.Tensor): Current feature map.

        Returns:
            torch.Tensor: Optical flow between the feature maps.
        """
        if prev_features.shape != curr_features.shape:
            raise ValueError("Previous and current features shapes don't match.")

        flow_x = curr_features - prev_features
        flow_x = flow_x.abs()
        flow_y = torch.zeros_like(flow_x)
        flow = torch.stack((flow_x, flow_y), dim=1)
        return flow

    @staticmethod
    def _velocity_threshold(flow: torch.Tensor, threshold: float) -> torch.Tensor:
        """
        Apply velocity thresholding on the optical flow.

        Args:
            flow (torch.Tensor): Optical flow values.
            threshold (float): Velocity threshold.

        Returns:
            torch.Tensor: Thresholded optical flow.
        """
        if flow.shape[-1] != 2:
            raise ValueError("Expected 2 channels in flow tensor, received more or less.")

        flow_mag = torch.sqrt(flow[:, 0] ** 2 + flow[:, 1] ** 2)
        return (flow_mag > threshold).float() * flow

    @staticmethod
    def _compute_accuracy(predictions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute accuracy metric for action recognition.

        Args:
            predictions (torch.Tensor): Predicted values.
            labels (torch.Tensor): Ground truth labels.

        Returns:
            torch.Tensor: Accuracy metric.
        """
        acc = (predictions.argmax(dim=1) == labels).float()
        return acc.mean()

    def __init__(self, threshold: float = 0.5) -> None:
        """
        Initialize the loss functions with threshold for velocity.

        Args:
            threshold (float): Velocity threshold value.
        """
        self.threshold = threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        prev_features: torch.Tensor,
        curr_features: torch.Tensor,
        mask: torch.Tensor,
        flow_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the custom loss function for action recognition.

        This function encapsulates the entire loss computation process, including
        applying masks, computing optical flow, thresholding, and combining
        various losses.

        Args:
            predictions (torch.Tensor): Current frame predictions.
            labels (torch.Tensor): Ground truth labels.
            prev_features (torch.Tensor): Previous frame features.
            curr_features (torch.Tensor): Current frame features.
            mask (torch.Tensor): Binary mask for predictions.
            flow_weights (torch.Tensor): Weights for flow loss.

        Returns:
            torch.Tensor: Combined loss value.
        """
        self._validate_inputs([predictions, labels, prev_features, curr_features, mask, flow_weights])

        # Apply mask to predictions
        masked_predictions = self._apply_mask(predictions, mask)

        # Compute optical flow
        flow = self._compute_flow(prev_features, curr_features)

        # Threshold optical flow
        thresholded_flow = self._velocity_threshold(flow, self.threshold)

        # Compute flow loss
        flow_loss = torch.mean(flow_weights * (thresholded_flow * flow).sum(dim=(1, 2, 3)))

        # Compute accuracy
        acc = self._compute_accuracy(masked_predictions, labels)

        # Combine losses
        loss = flow_loss + (1 - acc)

        return loss


class LossFunctionsConfig:
    """Configuration parameters for LossFunctions."""

    def __init__(self, threshold: float = 0.5) -> None:
        """
        Initialize the configuration.

        Args:
            threshold (float): Velocity threshold value.
        """
        self.threshold = threshold


def create_loss_function(config: LossFunctionsConfig) -> LossFunctions:
    """Factory function to create LossFunctions instance.

    Args:
        config (LossFunctionsConfig): Configuration for the loss functions.

    Returns:
        LossFunctions: Initialized loss functions instance.
    """
    return LossFunctions(config.threshold)