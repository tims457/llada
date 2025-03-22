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

    """mlp hidden size"""
    mlp_hidden_size: int = 4 * d_embed


config = LLaDAConfig()


class SwiGLU(eqx.Module):
    """
    Implementation of the SwiGLU activation function in the paper by Noam Shazeer at Google

    References:
        GLU Variants Improve Transformer paper  : https://arxiv.org/abs/2002.05202
        Aziz et al. Paper Summaries             : https://azizbelaweid.substack.com/p/what-is-swiglu-how-to-implement-it
    """

    W: jax.Array
    V: jax.Array
    b: jax.Array
    c: jax.Array

    def __init__(self, dim_in, dim_out, key):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.W = jax.random.normal(k1, (dim_in, dim_out))
        self.V = jax.random.normal(k2, (dim_in, dim_out))
        self.b = jax.random.normal(k3, (dim_out,))
        self.c = jax.random.normal(k4, (dim_out,))

    def __call__(self, x):
        return jax.nn.swish(jnp.dot(x, self.W) + self.b) * (jnp.dot(x, self.V) + self.c)


class LLaDABlock(eqx.Module):
    attn_norm: eqx.nn.LayerNorm
    attn: eqx.nn.MultiheadAttention
    ff_norm: eqx.nn.LayerNorm
    ff: eqx.nn.Linear
    fused_dims: tuple[int, int, int]
    act: eqx.Module
    att_proj: eqx.nn.Linear
    ff_proj: eqx.nn.Linear
    ff_out: eqx.nn.Linear
    drop: eqx.nn.Dropout

    def __init__(self, config: LLaDAConfig, key: jax.random.PRNGKey):
        ckey, akey, fkey, dkey = jax.random.split(key, 4)
        self.attn_norm = eqx.nn.LayerNorm(config.d_embed)

        self.ff_norm = eqx.nn.LayerNorm(config.d_embed)
        self.ff = eqx.nn.Linear(config.d_embed, config.d_embed, key=fkey)
        self.drop = eqx.nn.Dropout(config.dropout)

        self.attn = eqx.nn.MultiheadAttention(config.n_heads, config.d_embed, key=akey)
        # x -> (q, k, v)
        head_dim = config.d_embed // config.n_heads
        self.fused_dims = (
            config.d_embed,
            config.n_heads * head_dim,
            config.n_heads * head_dim,
        )
        hidden_dim = config.mlp_hidden_size
        # attention input projection
        self.att_proj = eqx.nn.Linear(config.d_embed, sum(self.fused_dims), key=akey)
        # feed forward input projection
        self.ff_proj = eqx.nn.Linear(config.d_embed, hidden_dim, key=fkey)
        # feed forward output projection
        self.ff_out = eqx.nn.Linear(hidden_dim, config.d_embed, key=fkey)
        self.act = SwiGLU(hidden_dim, hidden_dim, key=fkey)

    def __call__(self, x: jnp.ndarray, key: jax.random.PRNGKey):
        eqx.error_if(
            x,
            x.shape != (config.max_sequence_length, config.d_embed),
            f"Expected shape (max_sequence_length, d_embed), got {x.shape}",
        )
        attn_key, drop1_key, drop2_key = jax.random.split(key, 3)
        # TODO add RoPE
        x_norm = jax.vmap(self.attn_norm)(x)
        # ?? cache
        # ?? attention bias
        # ?? layer past
        q, k, v = jnp.split(jax.vmap(self.att_proj)(x_norm), 3, axis=1)
        attn_out = self.attn(q, k, v, key=attn_key)
        # dropout and skip connection
        # shape: (B, max_sequence_length, d_embed)
        x = x + self.drop(attn_out, key=drop1_key)
        # save skip connection
        og_x = x
        x_norm = jax.vmap(self.ff_norm)(x)
        x_proj = jax.vmap(self.ff_proj)(x_norm)
        x_act = jax.vmap(self.act)(x_proj)
        x_out = jax.vmap(self.ff_out)(x_act)
        x = og_x + self.drop(x_out, key=drop2_key)
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
        self.wpe = eqx.nn.Embedding(
            config.max_sequence_length, config.d_embed, key=pkey
        )
        self.drop = eqx.nn.Dropout(config.dropout)
        self.blocks = [LLaDABlock(config, bkey) for _ in range(config.n_layers)]
        # TODO past key values

    def __call__(self, input_ids: jnp.ndarray, key: jax.random.PRNGKey):
        (t,) = input_ids.shape
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
    transformer: Transformer

    def __init__(self, config: LLaDAConfig, key: jax.random.PRNGKey):
        tkey, lmhkey = jax.random.split(key, 2)
        self.transformer = Transformer(config, tkey)

    def __call__(self, input_ids: jnp.ndarray, key: jax.random.PRNGKey = None):
        if key is None:
            key = jax.random.PRNGKey(0)
        return self.transformer(input_ids, key=key)


transformer = Transformer(config, jax.random.PRNGKey(0))

print(transformer(jnp.zeros(1024, dtype=jnp.int32), jax.random.PRNGKey(0)).shape)
# check with vmap
print(
    jax.vmap(transformer, in_axes=(0, None))(
        jnp.zeros((10, 1024), dtype=jnp.int32), jax.random.PRNGKey(0)
    ).shape
)

# import time
# # Add a jitted call to transformer with a test input
# @jax.jit
# def jitted_transformer_call(input_ids, key):
#     return transformer(input_ids, key)

# # Test the jitted function
# test_input = jnp.zeros(1024, dtype=jnp.int32)
# test_key = jax.random.PRNGKey(42)
# jitted_output = jitted_transformer_call(test_input, test_key)
# jitted_start_time = time.time()
# jitted_output = jitted_transformer_call(test_input, test_key)
# jitted_end_time = time.time()
# print(f"Jitted transformer call output shape: {jitted_output.shape}")
# print(f"Jitted call execution time: {(jitted_end_time - jitted_start_time) * 1000:.2f} ms")

# # Compare with non-jitted call
# non_jitted_start_time = time.time()
# non_jitted_output = transformer(test_input, test_key)
# non_jitted_end_time = time.time()
# print(f"Non-jitted call execution time: {(non_jitted_end_time - non_jitted_start_time) * 1000:.2f} ms")
# print(f"Speedup from jitting: {(non_jitted_end_time - non_jitted_start_time) / ((jitted_end_time - jitted_start_time) + 1e-10):.2f}x")
