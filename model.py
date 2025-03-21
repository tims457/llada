import jax
import optax
import equinox as eqx
import jax.numpy as jnp
from dataclasses import dataclass
import chex

@dataclass
class LLaDAConfig:
    vocab_size: int = 50257
    
    """The embedding size is set to the next multiple
    of 128 that's greater than vocab_size to improve throughput"""
    embedding_size: int = vocab_size + (128 - vocab_size % 128)
    
    """embedding dimension / hidden size"""
    d_embed: int = 768
    
    """dropout probability"""
    dropout: float = 0.1
    
    """maximum sequence length"""
    max_sequence_length: int = 1024

    """number of attention heads"""
    n_heads: int = 12

    """number of layers"""
    n_layers: int = 12

config = LLaDAConfig()



class MLP(eqx.Module):
    c_fc: eqx.nn.Linear
    swiglu: eqx.nn.SwiGLU
    c_proj: eqx.nn.Linear
    dropout: eqx.nn.Dropout
    

class LLaDABlock(eqx.Module):
    attn_norm: eqx.nn.LayerNorm
    attn: eqx.nn.MultiheadAttention
    ff_norm: eqx.nn.LayerNorm
    ff: eqx.nn.Linear

    def __init__(self, config: LLaDAConfig, key: jax.random.PRNGKey):
        ckey, akey, fkey = jax.random.split(key, 3)
        self.attn_norm = eqx.nn.LayerNorm(config.d_embed)
        self.attn = eqx.nn.MultiheadAttention(config.n_heads, config.d_embed, key=akey)
        self.ff_norm = eqx.nn.LayerNorm(config.d_embed)
        self.ff = eqx.nn.Linear(config.d_embed, config.d_embed, key=fkey)

    def __call__(self, x: jnp.ndarray, key: jax.random.PRNGKey):
        # TODO add RoPE
        x = jax.vmap(self.attn_norm)(x)
        x = self.attn(x, x, x, key=key)
        x = jax.vmap(self.ff_norm)(x)
        x = self.ff(x)
        return x


class Transformer(eqx.Module):
    config: LLaDAConfig
    # word token embedding
    wte: eqx.nn.Embedding
    # position embedding
    wpe: eqx.nn.Embedding
    drop: eqx.nn.Dropout
    blocks: list[LLaDABlock]

    def __init__(self, config: LLaDAConfig, key: jax.random.PRNGKey):
        ekey, pkey, dkey, bkey = jax.random.split(key, 4)
        self.config = config
        self.wte = eqx.nn.Embedding(config.embedding_size, config.d_embed, key=ekey)
        self.wpe = eqx.nn.Embedding(config.max_sequence_length, config.d_embed, key=pkey)
        self.drop = eqx.nn.Dropout(config.dropout)
        self.blocks = [LLaDABlock(config, bkey) for _ in range(config.n_layers)]
        # TODO past key values

    def __call__(self, input_ids: jnp.ndarray, key: jax.random.PRNGKey):
        t, = input_ids.shape
        chex.assert_shape(input_ids, (t,))

        pos = jnp.arange(0, input_ids.shape[0], dtype=jnp.int64)
        token_emb = jax.vmap(self.wte)(input_ids)
        pos_emb = jax.vmap(self.wpe)(pos)
        x = self.drop(token_emb + pos_emb, key=key)

        # TODO attention mask
        for block in self.blocks:
            x = block(x, key=key)

        return x


class LLaDAModel(eqx.Module):
    def __init__(self, config: LLaDAConfig, key: jax.random.PRNGKey):
        tkey, lmhkey = jax.random.split(key, 2)
    

        self.transformer = Transformer(config, key)

    def __call__(self, input_ids: jnp.ndarray):
        x = self.transformer(input_ids, key=jax.random.PRNGKey(0))



transformer = Transformer(config, jax.random.PRNGKey(0))

print(transformer(jnp.zeros(1024, dtype=jnp.int32), jax.random.PRNGKey(0)).shape)