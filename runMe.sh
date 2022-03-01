cmd='python -u'

# KdV example with time-step size 1e-03
$cmd "testpyNG.py NG KdVTwoSol RBFp RK45 adamSGD zero uni 10 1 1000 1000 -1 1e-03 -1 uni 1e+05 1e+05 1e-01 fitSGD-l 5 -f 1"

# Advection example with time-varying coefficient and time-step size 1e-02
$cmd "testpyNG.py NG AdvTimeCoeff5 RBF RK45 adamSGD zero gauss1 50 1 1000 1000 -1 1e-02 -1 gauss 1e+5 1e+4 5e-02 fitSGD-l 5 -f 1"

# Particle in harmonic trap in 8 dimensions with adaptive sampling
$cmd "testpyNG.py NG Particle8 RBFNorm BwdE adamSGDMin reuse gauss2 30 1 1000 1000 1.0 1e-03 5000 uni -1 -1 -1 exact -1 -f 1"
