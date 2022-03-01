import jax.numpy as jnp
from jax import jit, vmap
from scipy import optimize
from scipy import integrate
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def fpsolverddd(x):
    '''Computes y_i = \sum_{j = 1}^m (x_i - x_j) for i = 1, \dots, m'''
    return jnp.sum(- jnp.atleast_2d(x) + jnp.atleast_2d(x).T , axis = 1).reshape((-1,))

def fpesolverUT(y0, t, nu, alpha, beta, cov0 = []):
    dim = y0.shape[0]
    if(cov0 == []):
        cov0 = jnp.zeros((dim, dim))

    t = jnp.atleast_1d(t)
    if(t.size == 1 and t[0] == 0):
        return y0.reshape((dim, 1)), cov0.reshape((dim, dim, 1))

    # add mean to get second moment
    mom2nd = cov0 + jnp.dot(y0.reshape((-1, 1)), y0.reshape((1, -1)))

    # print(cov0, jnp.dot(y0.reshape((-1, 1)), y0.reshape((1, -1))), mom2nd)
    # quit()

    y0 = jnp.hstack((y0, mom2nd.reshape((-1,))))

    @jit
    def fpesolverRhsMean(t, y):
        # print('rhs:', t, '/', t_eval[-1])
        yMean = y[:dim]
        yMom2nd = y[dim:].reshape((dim, dim))

        # Eric's
        yMom2nd = (-2 - 2*alpha)*yMom2nd + ((nu(t)*yMean).reshape((-1, 1)) + (nu(t)*yMean).reshape((1, -1))) + alpha/dim*(jnp.sum(yMom2nd, axis = 0).reshape((1, -1)) + jnp.sum(yMom2nd, axis = 0).reshape((-1, 1))) + 2./beta*jnp.eye(dim)

        yMean = -(1 + alpha)*yMean + nu(t) + alpha/dim*jnp.sum(yMean)

        return jnp.hstack((yMean, yMom2nd.reshape((-1,))))

    sol = integrate.solve_ivp(fpesolverRhsMean, [0, t[-1]], y0, t_eval = t, method = 'RK45', rtol = 1e-08, atol = 1e-12)
    yMean = jnp.asarray(sol.y[:dim, :].reshape((dim, len(t))))
    yCov = sol.y[dim:, :].reshape((dim, dim, len(t)))
    for i in range(len(t)):
        yCov[:, :, i] = yCov[:, :, i] - jnp.dot(yMean[:, i].reshape((-1, 1)), yMean[:, i].reshape((1, -1)))
    yCov = jnp.asarray(yCov)
    return yMean, yCov
