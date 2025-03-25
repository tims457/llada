import jax
import optax
import equinox as eqx
import jax.numpy as jnp
from config import test_config, gpt2_config, LLaDAConfig
from model import LLaDAModel

config = test_config

print(config)


# it looks like they just assign a token to the masked position
# and don't use the attention mask for anything, instead computing the loss
# on just the masked tokens
def forward_process(input_ids, key, eps=1e-3):
    key1, key2 = jax.random.split(key, 2)
    b, l = input_ids.shape
    p_mask = jax.random.uniform(key1, (b,), minval=eps, maxval=1.0)
    p_mask = p_mask[:, None].repeat(l, 1)

    masked_indices = jax.random.uniform(key2, (b, l)) < p_mask
    noisy_batch = jnp.where(masked_indices, config.mask_token_id, input_ids)

    return noisy_batch, masked_indices, p_mask


# input_ids = jax.random.randint(jax.random.PRNGKey(0), (2, 1024), 0, 10)

# noisy_batch, masked_indices, p_mask = forward_process(input_ids, jax.random.PRNGKey(0))
# print(noisy_batch)
# print(masked_indices)
# print(p_mask)

model = LLaDAModel(config, jax.random.PRNGKey(0))
# logits = jax.vmap(model)(noisy_batch)
# print(logits.shape)


# # compute loss
# loss = (
#     optax.softmax_cross_entropy(
#         logits[masked_indices],
#         jax.nn.one_hot(input_ids[masked_indices], config.embedding_size),
#     )
#     / p_mask[masked_indices]
# )
# print(loss)
# print(loss.shape)
# loss = loss.sum() / (input_ids.shape[0] * input_ids.shape[1])
# print(loss)
