@jit
def f_eduardo(identity_for_n: int, array_of_H):
    n = identity_for_n.shape[0]
    return jax.lax.dynamic_slice(array_of_H, (0, 0, 0), (n, 2, 2))


f_eduardo(jnp.ones(6), array_of_h)
