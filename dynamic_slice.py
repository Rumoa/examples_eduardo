@jit
def f_eduardo(identity_for_n, array_of_H):
    n = identity_for_n.shape[0]
    return jax.lax.dynamic_slice(array_of_H, (0, 0, 0), (n, 2, 2))


f_eduardo(jnp.ones(6), array_of_h)


jax.vmap((
    lambda i, arr: jax.lax.cond(
         i < 1.5,
        lambda i, arr: arr,
        lambda i, arr: 0 * arr,
        *[i, arr]
    )
), in_axes=(0, 0))(times, A)