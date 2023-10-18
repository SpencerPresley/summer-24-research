import torch
from torch import nn
from zeta.nn.attention.attend import Attend
from bitnet.bitlinear import BitLinear


# Feedforward network
def FeedForward(dim, dropout=0.0):
    """
    Feedforward network for Transformer with BitLinear layers instead.

    Args:
        dim: int, dimension of the input.
        dropout: float, dropout rate.

    Returns:
        nn.Sequential, feedforward network.

    Usage:
        >>> x = torch.randn(10, 512)
        >>> ff = FeedForward(512)
        >>> y = ff(x)
        >>> print(y)

    """
    return nn.Sequential(
        nn.LayerNorm(dim),
        BitLinear(dim),
        nn.GELU(),
        nn.Dropout(dropout),
        BitLinear(dim),
        nn.Dropout(dropout),
    )


class BitNetTransformer(nn.Module):
    """
    Transformer with BitLinear layers instead.

    Args:
        dim: int, dimension of the input.
        depth: int, number of layers.
        heads: int, number of heads.
        dim_head: int, dimension of each head.
        dropout: float, dropout rate.

    Returns:
        tensor, output of the transformer.

    Usage:
        >>> x = torch.randn(10, 512)
        >>> layer = BitNetTransformer(512, 8, 8, 64)
        >>> y = layer(x)
        >>> print(y)

    """

    def __init__(
        self,
        dim,
        depth,
        heads,
        dim_head,
        dropout=0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attend(dropout=dropout, heads=heads, flash=False),
                        FeedForward(dim, dropout),
                    ]
                )
            )
        self.norm = nn.LayerNorm(dim)
        
        # Bi Linear Layer
        self.bitlinear = BitLinear(dim)

    def forward(
        self,
        x,
        mask=None,
        # attn_mask = None
    ):
        """
        Forward pass of the transformer.

        """
        for attn, ff in self.layers:
            #apply by linear to x to create q, k, v
            q = self.bitlinear(x)
            k = self.bitlinear(x)
            v = self.bitlinear(x)

            out, intermediates = attn(q, k, v, mask=mask)

            x = out + x

            x = self.bitlinear(x)

            x = ff(x) + x

        return self.norm(x)