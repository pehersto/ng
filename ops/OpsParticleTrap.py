import jax
from jax import grad, jit, vmap, jvp, value_and_grad
import jax.numpy as jnp
from functools import partial

class OpsParticleTrap:
    def __init__(self, prob, dnn, scheme, modeName):
        self.prob = prob # problem
        self.dnn = dnn
        self.modeName = modeName

        # check if dnn satisifies boundary conditions
        if(self.prob.OmegaPeriodic == 1 and self.dnn.unitIsPeriodic == 1):
            raise Exception('Periodic boundary conditions are not supported')
        else:
            print("Enforcing boundary conditions via penalty")
            self.bc = self.prob.bc

        if(scheme == "BwdE"):
            self.objJ = self.objJBwdE
            self.objJgrad = self.objJBwdEgrad
        else:
            raise Exception("not implemented")


    def objJBwdE(self, ZInit, x, alphaZ, t, deltat, cHat):
        return particleTrapObjJBwdE(self.dnn.ufunDimScalar, self.dnn.ufun, self.dnn.ufunScalarDXDXX, self.bc, self.prob.dim, self.prob.alpha, self.prob.beta, self.prob.nu, t, x, alphaZ, deltat, cHat)

    def objJBwdEgrad(self, ZInit, x, alphaZ, t, deltat, cHat):
        return particleTrapObjJBwdEgrad(self.dnn.ufunDimScalar, self.dnn.ufun, self.dnn.ufunScalarDXDXX, self.bc, self.prob.dim, self.prob.alpha, self.prob.beta, self.prob.nu, t, x, alphaZ, deltat, cHat)


@partial(jit, static_argnums=(0,1,2,3,4,5,6,7,))
def particleTrapObjJBwdE(ufunDimScalar, ufun, ufunScalarDXDXX, bc, dim, alpha, beta, nu, t, x, alphaZ, deltat, cHat):
    print('compile bwd')
    azBar = alphaZ + deltat*cHat

    u, dt = jvp(lambda az: ufun(x, az), (azBar,), (cHat,))

    dx, dxx = vmap(ufunScalarDXDXX, in_axes = (0, None), out_axes = 0)(x, azBar)
    dx = jnp.multiply(x - nu(t) + alpha*(x - 1.0/dim*jnp.sum(x, axis = 1).reshape((-1, 1))), dx)
    dx = dim*(1 + alpha/dim*(dim - 1))*u + jnp.sum(dx, axis = 1)


    J = 1./x.shape[0]*jnp.sum(jnp.square(dt - dx - 1./beta*dxx)) # + bc(lambda x: ufunScalar(x, azBar))
    return J

@partial(jit, static_argnums=(0,1,2,3,4,5,6,7,))
def particleTrapObjJBwdEgrad(ufunDimScalar, ufun, ufunScalarDXDXX, bc, dim, alpha, beta, nu, t, x, alphaZ, deltat, cHat):
    print('compile grad bwde')
    return grad(particleTrapObjJBwdE, argnums=12)(ufunDimScalar, ufun, ufunScalarDXDXX, bc, dim, alpha, beta, nu, t, x, alphaZ, deltat, cHat)
