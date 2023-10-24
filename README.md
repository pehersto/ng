# Neural Galerkin with Active Learning for High-Dimensional Evolution Equations

This code implements the Neural Galerkin scheme with Active Learning for high-dimensional evolution equations as described in:

* Bruna, Peherstorfer, Vanden-Eijnden [Neural Galerkin Scheme with Active Learning for High-Dimensional Evolution Equations](tbd).

The numerical examples can be run as follows:
- Create a Python environment with

```
$ bash setuppy.sh
$ source venv/bin/activate
```

- The main file is `testpyNG.py`

```
$ python -u testpyNG.py --help
usage: testpyNG.py [-h] [-i {0,1}] [-p {0,1}] [-w {0,1}] [-o {0,1}] [-f {0,1}] [-r POSTFIX]
                   {NG,F2} probName {RBF,tanh,RBFp,tanhp,RBFNorm,hat} {FwdE,BwdE,RK45,RK23} {adamSGD,adamSGDMin}
                   {reuse,zero,randn} {uni,equi,gauss,gauss1,gauss2,gauss3,gauss4,gauss5} N nrLayers batchSize
                   miniBatchSize lr deltat nrIter {uni,equi,gauss} initBatchSize initNrIter initLr zInitMode
                   nrReplicates

positional arguments:
  {NG,F2}               mode
  probName              problem name
  {RBF,tanh,RBFp,tanhp,RBFNorm,hat}
                        unit name
  {FwdE,BwdE,RK45,RK23}
                        name of time integration scheme
  {adamSGD,adamSGDMin}  name of solver
  {reuse,zero,randn}    initializing weights at each time step
  {uni,equi,gauss,gauss1,gauss2,gauss3,gauss4,gauss5}
                        how to sample data
  N                     number of nodes
  nrLayers              number of hidden layers
  batchSize             batch size
  miniBatchSize         mini batch size
  lr                    learning rate
  deltat                time-step size
  nrIter                number of SGD iterations
  {uni,equi,gauss}      sampling for fitting initial condition
  initBatchSize         batch size of fitting initial condition
  initNrIter            number of SGD iterations when fitting initial condition
  initLr                learning rate for fitting initial condition
  zInitMode             mode of fitting initial condition
  nrReplicates          number of replicates

optional arguments:
  -h, --help            show this help message and exit
  -i {0,1}, --plotInit {0,1}
                        yes/no plot initial condition
  -p {0,1}, --plotFinal {0,1}
                        yes/no plot prediction
  -w {0,1}, --writeVideo {0,1}
                        yes/no write video
  -o {0,1}, --onlyInit {0,1}
                        yes/no only compute initial condition and truth and then exit
  -f {0,1}, --writeToFile {0,1}
                        yes/no write results to npz file
  -r POSTFIX, --postfix POSTFIX
                        add this to the filenames
```

- Examples of parameters for the experiments in the paper are in `runMe.sh'
