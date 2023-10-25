import jax
import jax.numpy as jnp
#from jax.ops import index, index_add, index_update

def exactKdVTwoSol(x, t):
    '''
    Same setup as in https://doi.org/10.1016/0021-9991(84)90004-4

    Analytical and numerical aspects of certain nonlinear evolution equations. III. Numerical, Korteweg-de Vries equation
    Thiab R Taha, Mark I Ablowitz
    Journal of Computational Physics, Volume 55, Issue 2, August 1984, Pages 231-253
    '''

    k = jnp.asarray([1., jnp.sqrt(5.)])
    eta = jnp.asarray([0., 10.73])
    t = jnp.asarray(t)

    etaMat1 = k[0]*x.reshape((-1,1)) - k[0]**3*t.reshape((1,-1)) + eta[0]
    etaMat2 = k[1]*x.reshape((-1,1)) - k[1]**3*t.reshape((1,-1)) + eta[1]
    c = ((k[0] - k[1])/(k[0] + k[1]))**2

    f = 1. + jnp.exp(etaMat1) + jnp.exp(etaMat2) + jnp.multiply(jnp.exp(etaMat1), jnp.exp(etaMat2)*c)
    df = k[0]*jnp.exp(etaMat1) + k[1]*jnp.exp(etaMat2) + c*(k[0] + k[1])*jnp.multiply(jnp.exp(etaMat1), jnp.exp(etaMat2))
    ddf = k[0]**2*jnp.exp(etaMat1) + k[1]**2*jnp.exp(etaMat2) + c*(k[0] + k[1])**2*jnp.multiply(jnp.exp(etaMat1), jnp.exp(etaMat2))

    y = 2*jnp.divide(jnp.multiply(f, ddf) - df**2, f**2);

    y = jnp.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0) # avoid numerical errors far outside of [-1, 2]

    return y
