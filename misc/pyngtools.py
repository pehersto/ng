import jax.numpy as jnp
import math
import jax

def mvnpdfFull(X, meanAndCovList = None, unnormalize = False):
    dim = meanAndCovList[0].shape[0]
    # if we get a list of means and cov then iterate over them and return a 2d array of evals
    if(len(meanAndCovList[0].shape) > 1):
        return jax.vmap(lambda i: mvnpdfFull(X, meanAndCovList = (meanAndCovList[0][:, i].reshape((dim,)), meanAndCovList[1][:, :, i].reshape((dim, dim))), unnormalize = unnormalize), in_axes = (0,), out_axes = 1)(jnp.arange(meanAndCovList[0].shape[1]))
    else:
        if(unnormalize):
            cons = 1
        else:
            cons = 1./jnp.sqrt((2*jnp.pi)**meanAndCovList[0].size*jnp.linalg.det(meanAndCovList[1]))
        inner = jnp.sum(jnp.multiply(X.T - meanAndCovList[0].reshape((-1,1)), jnp.linalg.solve(meanAndCovList[1], X.T - meanAndCovList[0].reshape((-1,1)))), axis = 0).reshape((-1,))
        return cons*jnp.exp(-0.5*inner)

def mvnpdf(X, mean = None, covdiag = None):
    cons = 1./jnp.sqrt((2*jnp.pi)**mean.size*jnp.prod(covdiag))
    inner = jnp.sum(jnp.multiply(X.T - mean.reshape((-1,1)), jnp.multiply((1./covdiag).reshape((-1,1)), X.T - mean.reshape((-1,1)))), axis = 0).reshape((-1,))
    return cons*jnp.exp(-0.5*inner)

def samplemvn(key, Nx, meanAndCov, facScale = 1):
    key, subkey = jax.random.split(key)
    dim = meanAndCov[0].shape[0]
    return jax.random.multivariate_normal(subkey, meanAndCov[0].reshape((dim,)), facScale*meanAndCov[1].reshape((dim, dim)), shape=(Nx,)), key

def sampleGaussianMixture(key, Nx, meanList, covdiagList):
    N = len(meanList)
    d = meanList[0].size
    NStep = int(math.ceil(Nx/N))
    key, subkey = jax.random.split(key)
    X = meanList[0].reshape((1,d)) + jnp.multiply(jax.random.normal(subkey, shape=(NStep, d)), covdiagList[0].reshape((1,d)))
    for i in jnp.arange(1, N):
        key, subkey = jax.random.split(key)
        X = jnp.vstack((X, meanList[i].reshape((1,d)) + jnp.multiply(jax.random.normal(subkey, shape=(NStep, d)), covdiagList[i].reshape((1,d)))))
    X = X[:Nx, :]
    return X, key

def hathh(X, mean, hleft, hright):
    return jnp.minimum(1, jnp.maximum(0.0, 1 + (X - mean)/hleft)) + jnp.minimum(1, jnp.maximum(0, 1 - (X - mean)/hright)) - 1
