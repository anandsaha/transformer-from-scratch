import torch
import torch.nn as nn
from typing import Tuple

# A small value added to the denominator to prevent division by zero.
# This is a common practice in normalization layers.
EPSILON = 1e-5

class MyBatchNorm1d(nn.Module):
    """
    A manual implementation of Batch Normalization for 1D inputs.

    This class mimics the behavior of torch.nn.BatchNorm1d.
    It highlights the key difference between training and evaluation modes.
    """
    def __init__(self, num_features: int, momentum: float = 0.1):
        """
        Initializes the BatchNorm layer.

        Args:
            num_features (int): The number of features in the input tensor.
            momentum (float): The momentum for the running mean and running variance.
                              This controls how much the new batch statistics influence the
                              running statistics.
        """
        super().__init__()
        self.num_features = num_features
        self.momentum = momentum

        # Gamma (γ) and Beta (β) are trainable parameters.
        # Gamma scales the normalized output, Beta shifts it.
        # They are initialized to 1s and 0s respectively.
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

        # running_mean and running_var are not trainable parameters.
        # They are buffers that hold the running statistics.
        # We register them as buffers so they are saved and loaded with the model's state_dict.
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        
        # Set running_var to 1 initially to avoid division by zero
        self.running_var.fill_(1)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass for BatchNorm.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, num_features).

        Returns:
            torch.Tensor: The normalized output tensor.
        """
        if self.training:
            # === TRAINING MODE ===
            # We calculate the mean and variance for the current mini-batch.
            # The dimension to normalize over is the batch dimension (dim=0).
            mean = x.mean(dim=0)
            var = x.var(dim=0, unbiased=False)

            # Update the running mean and variance using a momentum-based average.
            # This is how the model "remembers" statistics from previous batches.
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
            
            # Use the current batch's mean and variance for normalization.
            normalized_x = (x - mean) / torch.sqrt(var + EPSILON)
        else:
            # === EVALUATION MODE ===
            # During inference, we don't have a mini-batch to calculate statistics from.
            # We use the accumulated running mean and running variance instead.
            normalized_x = (x - self.running_mean) / torch.sqrt(self.running_var + EPSILON)
        
        # Apply the trainable scale (gamma) and shift (beta) parameters.
        output = self.gamma * normalized_x + self.beta
        return output

class MyLayerNorm(nn.Module):
    """
    A manual implementation of Layer Normalization.

    This class mimics the behavior of torch.nn.LayerNorm.
    It computes statistics per-example, making it independent of batch size.
    """
    def __init__(self, normalized_shape: Tuple[int, ...]):
        """
        Initializes the LayerNorm layer.

        Args:
            normalized_shape (tuple): The shape of the input to normalize over.
                                      For a 2D tensor (batch, features), this would be `(num_features,)`.
        """
        super().__init__()
        self.normalized_shape = normalized_shape

        # Gamma (γ) and Beta (β) are trainable parameters.
        # Unlike BatchNorm, they are applied to the entire feature dimension.
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass for LayerNorm.

        Args:
            x (torch.Tensor): The input tensor. The last dimension should match
                              `self.normalized_shape`.

        Returns:
            torch.Tensor: The normalized output tensor.
        """
        # === TRAINING & EVALUATION MODE ===
        # LayerNorm's logic is the same in both modes.
        # We calculate the mean and variance along the last dimension(s).
        # We need to keep the dimensions for broadcasting later.
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)

        # Normalize the input using the computed mean and variance.
        normalized_x = (x - mean) / torch.sqrt(var + EPSILON)

        # Apply the trainable scale (gamma) and shift (beta) parameters.
        output = self.gamma * normalized_x + self.beta
        return output

if __name__ == '__main__':
    # === DEMONSTRATION OF MyBatchNorm1d ===
    print("--- Demonstrating MyBatchNorm1d ---")
    
    num_features = 5
    batch_norm_layer = MyBatchNorm1d(num_features)
    
    # Let's see the initial running stats
    print(f"Initial running mean: {batch_norm_layer.running_mean.detach().numpy()}")
    print(f"Initial running variance: {batch_norm_layer.running_var.detach().numpy()}")
    
    # Generate some fake data for a mini-batch
    batch_size = 4
    input_data = torch.randn(batch_size, num_features)
    
    # Put the model in training mode.
    # The statistics will be computed from this batch and the running stats will be updated.
    batch_norm_layer.train()
    print("\nBatch norm in TRAINING mode:")
    output_bn_train = batch_norm_layer(input_data)
    print(f"Batch Norm Output (Training):\n{output_bn_train.detach()}")
    
    # Let's check the updated running stats after one training step
    print(f"\nUpdated running mean after training batch:\n{batch_norm_layer.running_mean.detach().numpy()}")
    print(f"Updated running variance after training batch:\n{batch_norm_layer.running_var.detach().numpy()}")
    
    # Now, put the model in evaluation mode.
    # It will use the running stats, not the new batch's stats.
    batch_norm_layer.eval()
    new_input_data = torch.randn(batch_size, num_features)
    print("\nBatch norm in EVALUATION mode:")
    print("Note: The output is based on the running mean/var, not the new input data's mean/var.")
    output_bn_eval = batch_norm_layer(new_input_data)
    print(f"Batch Norm Output (Evaluation):\n{output_bn_eval.detach()}")

    # === DEMONSTRATION OF MyLayerNorm ===
    print("\n\n--- Demonstrating MyLayerNorm ---")
    
    num_features_ln = 10
    layer_norm_layer = MyLayerNorm((num_features_ln,))
    
    # Generate some fake data for a mini-batch
    batch_size_ln = 4
    input_data_ln = torch.randn(batch_size_ln, num_features_ln)
    
    # LayerNorm's forward pass is the same in both training and evaluation modes.
    # Let's show it in training mode first.
    layer_norm_layer.train()
    print("Layer norm in TRAINING mode:")
    output_ln_train = layer_norm_layer(input_data_ln)
    print(f"Layer Norm Output (Training):\n{output_ln_train.detach()}")
    
    # Now in evaluation mode. The output should be identical for the same input.
    layer_norm_layer.eval()
    print("\nLayer norm in EVALUATION mode:")
    output_ln_eval = layer_norm_layer(input_data_ln)
    print(f"Layer Norm Output (Evaluation):\n{output_ln_eval.detach()}")
    
    # Let's compare the outputs from the same input data
    is_identical = torch.allclose(output_ln_train, output_ln_eval)
    print(f"\nAre training and evaluation outputs identical for the same input? {is_identical}")
    print("Yes, because LayerNorm does not use running averages..")