from dataclasses import dataclass


@dataclass
class LLaDAConfig:
    """1B Parameter Model"""

    vocab_size: int = 126_464

    """The embedding size is set to the next multiple
    of 128 that's greater than vocab_size to improve throughput"""
    # set in __post_init__
    embedding_size: int = None

    """embedding dimension / hidden size"""
    d_embed: int = 2048

    """dropout probability"""
    dropout: float = 0.1

    """maximum sequence length"""
    max_sequence_length: int = 1024

    """number of attention heads"""
    n_heads: int = 32

    """number of layers"""
    n_layers: int = 22

    """mlp hidden size"""
    # set in __post_init__
    mlp_hidden_size: int = None

    """rope"""
    # TODO rope true

    """pad token id"""
    pad_token_id: int = 126081

    """eos token id"""
    eos_token_id: int = 126081

    """mask token id"""
    mask_token_id: int = 126336

    """precision"""
    # TODO precision

    def __post_init__(self):
        if self.embedding_size is None:
            self.embedding_size = self.vocab_size + (128 - self.vocab_size % 128)

        if self.mlp_hidden_size is None:
            self.mlp_hidden_size = 4 * self.d_embed


test_config = LLaDAConfig(
    n_layers=12,
    vocab_size=1000,
    d_embed=128,
    n_heads=4,
)

gpt2_config = LLaDAConfig(
    n_layers=12,
    vocab_size=50257,
    d_embed=768,
    n_heads=12,
)
