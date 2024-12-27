# DEMs

> **Solving for diffusion-induced stresses using the energy method**

We provide code for the TensorFlow and Pytorch deep learning frameworks respectively.
> In TensorFlow(_version_ **tf2.11**)
> * We provide a total of two examples, a hollow cylinder `ThickCylinder_Diffusion_DEM.py` and a hollow sphere `Spherical_Diffusion_DEM.py`. 
> * We also give the models we have trained, and the related usage and notes are shown in the comments in the code. For example, the neural network structure used in our trained model is `2-32-32-1`, so the neural network structure set in the code should also be this structure
> * This code is referenced from [Samaniego, E., et al., An energy approach to the solution of partial differential equations in computational mechanics via machine learning. Concepts, implementation and applications. computer Methods in Applied Mechanics and Engineering, 2020. 362: p. 112790](https://doi.org/10.1016/j.cma.2019.112790).

> In Pytorch(_version_ **torch1.11**)
> * The two examples we provide are `Cylinder_DEM.py` and `Spherical_DEM.py` respectively.
> * Numerical solutions for COMSOL are also provided and can of course be used to test in TensorFlow.

> `pip install -r requirements.txt`Build the environment to run this code. 
