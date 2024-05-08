import jax.numpy as jnp
from jax import lax
from flax import linen as nn

class ChebyshevKANLayer(nn.Module):
    in_dim: int  # Number of input features
    out_dim: int  # Number of output features
    num_polynomials: int  # Number of Chebyshev polynomials
    add_bias: bool = True  # Whether to add a bias term

    def setup(self):
        # Coefficients for Chebyshev polynomials, shape: (in_dim, num_polynomials, out_dim)
        self.cheb_coeffs = self.param('cheb_coeffs', 
                                      nn.initializers.normal(stddev=(1.0 / jnp.sqrt(self.in_dim * self.num_polynomials))), 
                                      (self.in_dim, self.num_polynomials, self.out_dim))

        if self.add_bias:
            self.bias = self.param('bias', nn.initializers.zeros, (self.out_dim,))
        else:
            self.bias = 0

    def chebyshev_polynomials(self, x, n):
        # Computes Chebyshev polynomials up to degree n-1 for each element in x
        def chebyshev_scan(carry, _):
            T_nm2, T_nm1 = carry
            T_n = 2 * x * T_nm1 - T_nm2
            return (T_nm1, T_n), T_n

        # Initialize first two terms
        T0, T1 = jnp.ones_like(x), x

        # Recurse to generate polynomials
        _, T_rest = lax.scan(chebyshev_scan, init=(T0, T1), xs=None, length=n-2)  
        
        return jnp.concatenate([T0[..., None], T1[..., None], jnp.moveaxis(T_rest, 0, -1)], axis=-1)

    def __call__(self, x):
        # Calculate Chebyshev polynomials
        x = nn.tanh(x) # Ensure input is in (-1, 1)
        T = self.chebyshev_polynomials(x, self.num_polynomials)  # (batch_size, in_dim, num_polynomials)
        
        # Apply coefficients
        output = jnp.einsum("bin, ino -> bo", T, self.cheb_coeffs)  # (batch_size, out_dim)
        output += self.bias
        return output