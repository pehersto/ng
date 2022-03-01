import jax
import jax.numpy as jnp
from functools import partial
from timeit import default_timer as timer

def adam(gradFun, sampleFun, h, nrIter, x, key, printETA = 0, returnBest = 0):
    '''Adam solver:
        Inputs:
            gradFun         ... gradient
            sampleFun       ... data sampling
            h               ... learning rate
            nrIter          ... number of iterations
            x               ... starting value
            outputFun       ... called whenever grad norm is printed
            key             ... key for random (jax)
        Outputs:
            x               ... approximation of optimal point
            key             ... key for random (jax)
    '''

    m = jnp.zeros_like(x)
    v = jnp.zeros_like(x)

    startT = timer()
    for i in range(nrIter):
        dat, key = sampleFun(x, key)
        sgCurr = gradFun(dat, x)

        x, m, v = adamupdate(x, h, m, v, sgCurr, i)

        if(((i % min(1000, jnp.ceil(nrIter/10.))) == 0) or (i == nrIter-1)):
            ng = jnp.linalg.norm(sgCurr);

            if(returnBest and (i == 0 or ngLast > ng)):
                ngLast = ng
                xLast = x
                flagUpdate = '*'
            else:
                flagUpdate = ' '

            if(printETA == 0):
                print("   " + flagUpdate, i, ": grad norm ", ng, ", time per step ", (timer()-startT)/(i+1));
            else:
                oneStepTime = (timer()-startT)/(i+1)
                oneStepETA = int((nrIter-(i+1))*oneStepTime)
                oneStepTimeDays = int(oneStepETA//86400)
                oneStepTimeHours = int((oneStepETA - oneStepTimeDays*86400)//3600)
                oneStepTimeMinutes = int((oneStepETA - oneStepTimeDays*86400 - oneStepTimeHours*3600)//60)
                oneStepTimeSeconds = int(oneStepETA - oneStepTimeDays*86400 - oneStepTimeHours*3600 - oneStepTimeMinutes*60)
                print("   " + flagUpdate, i, ": grad norm ", ng, ", time per step ", (timer()-startT)/(i+1), '(ETA: {days}-{hours}:{minutes}:{seconds})'.format(days=oneStepTimeDays, hours=oneStepTimeHours, minutes=oneStepTimeMinutes, seconds=oneStepTimeSeconds));

    if(returnBest):
        x = xLast
    return x, key


@jax.jit
def adamupdate(x, h, m, v, sgCurr, i):
    print("compile adamupdate")
    epsilon = 1e-08
    beta1 = 0.9
    beta2 = 0.999

    m = beta1*m + (1 - beta1)*sgCurr;
    v = beta2*v + (1 - beta2)*jnp.square(sgCurr);

    mhat = jnp.divide(m, (1 - beta1**(i + 1)))
    vhat = jnp.divide(v, (1 - beta2**(i + 1)))
    x = x - h * jnp.divide(mhat, jnp.sqrt(vhat) + epsilon)

    return x, m, v
