The scripts directory contains the Generative Adversarial Network Python code.

scripts/main.py contains code to create the computation graph for the Generative Adversarial Network and execute the training loop.  

Executing "python main.py --help" will display configurable options/hyper-parameters.  Without any arguments, "python main.py" will begin training a GAN with default options enabled.

The project has the following dependencies on external libraries:

    numpy v1.14.0
    matplotlib v2.0.2
    tensorflow v1.5.0
    imageio v2.3.0

The conda directory contains Anaconda environment scripts compatible with Python2 and Python3 to help with dependency download and environment configuration.
