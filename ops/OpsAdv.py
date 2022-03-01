import jax
from jax import grad, jit, vmap, jvp, value_and_grad
import jax.numpy as jnp
from functools import partial

class OpsAdv:
    def __init__(self, prob, dnn, scheme, modeName):
        self.prob = prob # problem
        self.dnn = dnn
        self.modeName = modeName

        # check if dnn satisifies boundary conditions
        if(self.prob.OmegaPeriodic == 1 and self.dnn.unitIsPeriodic == 1):
            print("No boundary penality because units satisfy boundary conditions")
            self.bc = lambda xdfun: 0.
        else:
            print("Enforcing boundary conditions via penalty")
            self.bc = self.prob.bc

        if(scheme == "RK23" or scheme == "RK45"):
            pass
        else:
            raise Exception("not implemented")

    def rhsJ(self, ZInit, x, alphaZ, t):
        return AdvRHS(self.dnn.ufunScalar, self.dnn.ufun, self.bc, self.prob.nu, self.prob.mu, x, alphaZ, t)

@partial(jit, static_argnums=(0,1,2,3,4,))
def AdvRHS(ufunScalar, ufun, bc, nuFun, muFun, x, alphaZ, t):
    print('compile rhs')

    dx = jax.vmap(grad(ufunScalar, argnums = 0), in_axes = (0, None), out_axes = 0)(x, alphaZ)
    Jac = jax.jacfwd(lambda az: ufun(x, az))(alphaZ)
    K = jnp.dot(Jac.T, Jac)
    f = jnp.dot(Jac.T, jnp.sum(jnp.multiply(nuFun(x, t), dx), axis = 1))

    J = jnp.linalg.lstsq(K, -f)[0]

    return J
