import jax
import optax
import equinox as eqx
import jax.numpy as jnp
from dataclasses import dataclass
@dataclass
class LLaDAConfig:
    vocab_size: int = 50257
    
    """The embedding size is set to the next multiple
    of 128 that's greater than vocab_size to improve throughput"""
    embedding_size: int = vocab_size + (128 - vocab_size % 128)
    
    """embedding dimension"""
    d_embed: int = 768
    
    """dropout probability"""
    dropout: float = 0.1
    
    """maximum sequence length"""
    max_sequence_length: int = 1024


config = LLaDAConfig()

class Transformer(eqx.Module):
    config: LLaDAConfig
    # word token embedding
    wte: eqx.nn.Embedding
    # position embedding
    wpe: eqx.nn.Embedding
    drop: eqx.nn.Dropout
    

    def __init__(self, config: LLaDAConfig, key: jax.random.PRNGKey):
        ekey, pkey, dkey = jax.random.split(key, 3)
        self.config = config
        self.wte = eqx.nn.Embedding(config.embedding_size, config.d_embed, key=ekey)
        self.wpe = eqx.nn.Embedding(config.max_sequence_length, config.d_embed, key=pkey)
        self.drop = eqx.nn.Dropout(config.dropout)
        # TODO past key values

    def __call__(self, input_ids: jnp.ndarray, key: jax.random.PRNGKey):
        pos = jnp.arange(0, input_ids.shape[0], dtype=jnp.int64)
        x = jax.vmap(self.wte)(input_ids)
        pos_emb = jax.vmap(self.wpe)(pos)
        x = self.drop(x + pos_emb, key=key)

        return x


class LLaDAModel(eqx.Module):
    def __init__(self, config: LLaDAConfig, key: jax.random.PRNGKey):
        
        pass

        # self.transformer

    def __call__(self, input_ids: jnp.ndarray):
        pass



transformer = Transformer(config, jax.random.PRNGKey(0))

print(transformer(jnp.zeros(1024, dtype=jnp.int32), jax.random.PRNGKey(0)).shape)