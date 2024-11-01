import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange
from typing import Type, Tuple, Optional

class SALTLinear(nn.Linear):
    """
    A linear layer that combines truncated SVD decomposition with LoRA-style adaptation.
    Only keeps top r singular values and vectors, then adds LoRA adaptation.
    """
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        rank: int,  # truncation rank for SVD
        r_lora: int = 8,  # LoRA rank
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        seed: int = 42
    ) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        torch.manual_seed(seed)
        # Keep only top r components
        # We freeze all parameters
        self.weight.requires_grad = False
        self.done_svd = False
        # I don't know why we are doing this step I am just sticking to the implementations of SVDLinear XD
        self.U, self.S, self.Vt = torch.linalg.svd(self.weight, full_matrices=False)

        max_possible_rank = min(self.U.shape[1], self.S.shape[0], self.Vt.shape[0])
        print("\nThe max possible rank is", max_possible_rank)

        
        # Initialize LoRA matrices to match truncated dimensions it should be [S_r , L_r] \times [L_r , S_r] where S_r is the turncated rank
        self.X = nn.Parameter(
            torch.randn(max_possible_rank, r_lora, generator=torch.Generator().manual_seed(seed)) * 0.01
        )
        self.Y = nn.Parameter(
            torch.randn(r_lora, max_possible_rank, generator=torch.Generator().manual_seed(seed + 1)) * 0.01
        )
        
        self.reset_parameters()

    def perform_svd(self) -> None:
        """Performs truncated SVD decomposition on the weight matrix."""
        # Update truncated components
        self.U, self.S, self.Vt = torch.linalg.svd(self.weight, full_matrices=False)
        self.done_svd = True

    def get_modified_singular_values(self) -> torch.Tensor:
        """
        Computes modified singular values using LoRA adaptation.
        Returns:
            Modified singular values tensor
        """
        return torch.diag(self.S) + self.X @ self.Y

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with LoRA-modified truncated singular values.
        
        Args:
            input: Input tensor
            
        Returns:
            Tuple containing:
                - Output tensor after linear transformation
                - Regularization loss
        """
        if not self.done_svd:
            self.perform_svd()

        # Compute modified singular values with LoRA
        s_new = F.relu(self.get_modified_singular_values().to(input.device))
        
        # Reconstruct weight matrix using truncated components
        print(self.U.shape)
        print(s_new.shape)
        print(self.Vt.shape)
        weight_updated = self.U @ s_new @ self.Vt
        
        # Compute regularization loss
        reg_loss = torch.norm(self.X @ self.Y)

        return F.linear(input, weight_updated, self.bias), reg_loss

class SALTConv2d(nn.Conv2d):
    """
    A 2D convolutional layer that combines truncated SVD decomposition with LoRA-style adaptation.
    The weight matrix is reshaped before applying truncated SVD and LoRA modifications.
    """
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int,
        kernel_size: int,
        rank: int,  # truncation rank for SVD
        r_lora: int = 8,  # LoRA rank
        seed: int = 42,
        **kwargs
    ):
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
        torch.manual_seed(seed)
        
        # Reshape and perform initial truncated SVD
        weight_reshaped = rearrange(self.weight, 'co cin h w -> co (cin h w)')
        U, S, Vt = torch.linalg.svd(weight_reshaped, full_matrices=False)
        
        # Keep only top r components
        self.U = nn.Parameter(U[:, :rank], requires_grad=False)
        self.S = nn.Parameter(S[:rank], requires_grad=False)
        self.Vt = nn.Parameter(Vt[:rank, :], requires_grad=False)
        
        self.weight.requires_grad = False
        self.done_svd = False
        
        # Initialize LoRA matrices to match truncated dimensions
        self.X = nn.Parameter(
            torch.randn(rank, r_lora, generator=torch.Generator().manual_seed(seed)) * 0.01
        )
        self.Y = nn.Parameter(
            torch.randn(r_lora, rank, generator=torch.Generator().manual_seed(seed + 1)) * 0.01
        )
        
        self.reset_parameters()

    def perform_svd(self) -> None:
        """Performs truncated SVD decomposition on the reshaped weight matrix."""
        weight_reshaped = rearrange(self.weight, 'co cin h w -> co (cin h w)')
        U, S, Vt = torch.linalg.svd(weight_reshaped, full_matrices=False)
        rank = self.X.size(0)  # Get rank from X matrix dimensions
        
        # Update truncated components
        self.U.data = U[:, :rank]
        self.S.data = S[:rank]
        self.Vt.data = Vt[:rank, :]
        self.done_svd = True

    def get_modified_singular_values(self) -> torch.Tensor:
        """
        Computes modified singular values using LoRA adaptation.
        Returns:
            Modified singular values tensor
        """
        return torch.diag(self.S) + self.X @ self.Y

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with LoRA-modified truncated singular values.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple containing:
                - Output tensor after convolution
                - Regularization loss
        """
        if not self.done_svd:
            self.perform_svd()

        # Compute modified singular values with LoRA
        s_new = F.relu(self.get_modified_singular_values().to(x.device))
        
        # # Reconstruct weight matrix using truncated components
        print(self.U.shape)
        print(s_new.shape)
        print(self.Vt.shape)
        weight_updated = self.U @ s_new @ self.Vt
        
        # Reshape weight back to conv2d format
        weight_updated = rearrange(
            weight_updated, 
            'co (cin h w) -> co cin h w', 
            cin=self.weight.size(1), 
            h=self.weight.size(2), 
            w=self.weight.size(3)
        )
        
        # Compute regularization loss
        reg_loss = torch.norm(self.X @ self.Y)

        return F.conv2d(
            x, weight_updated, self.bias, 
            self.stride, self.padding, 
            self.dilation, self.groups
        ), reg_loss