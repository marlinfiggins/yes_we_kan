import jax.numpy as jnp
from flax import linen as nn

class FourierKANLayer(nn.Module):
    in_dim: int  # Number of input features
    out_dim: int  # Number of output features
    num_harmonics: int  # Number of harmonics in the Fourier series
    add_bias: bool = True  # Whether to add a bias term

    def setup(self):
        # Fourier coefficients for cosine and sine, shape: (in_dim, num_harmonics, 2, out_dim)
        self.fourier_coeffs = self.param('fourier_coeffs', 
                                    nn.initializers.normal(stddev=(1.0 / jnp.sqrt(self.in_dim * self.num_harmonics))), 
                                    (self.in_dim, self.num_harmonics, 2, self.out_dim, ))

        # Optional bias, shape: (out_dim,)
        if self.add_bias:
            self.bias = self.param('bias', nn.initializers.zeros, (self.out_dim,))
        else:
            self.bias = 0

        # Harmonics for cosine and sine, shape: (self.num_harmonics,)
        self.k = jnp.arange(1, self.num_harmonics + 1)

    def __call__(self, x):
        # Compute cosine and sine transformations for each harmonic
        kx =  x[:, :, None] * self.k  # (batch_size, in_dim, num_harmonics)
        trig = jnp.stack([jnp.cos(kx), jnp.sin(kx)], axis=-1)  # (batch_size, in_dim, num_harmonics, 2)
        
        # Apply Fourier coefficients   
        output = jnp.einsum("bigf, igfo -> bo", trig, self.fourier_coeffs) # (batch_size, out_dim)
        output += self.bias
        return output