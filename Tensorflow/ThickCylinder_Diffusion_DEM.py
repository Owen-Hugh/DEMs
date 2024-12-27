import tensorflow as tf
import numpy as np
import time
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tfp_loss import tfp_function_factory
from Geom_Creat import Quadrilateral

tf.random.set_seed(42)
data_type = "float64"
model_save_path_diffusion = "model/ThickCylinder_Diffusion_DEM/diffusion.h5"
model_save_path_stress = "model/ThickCylinder_Diffusion_DEM/stress.h5"
cycle_number = 0 # cycle count
number_of_iterations = 10  # Setting the number of cycles
cumulative_epochs_diffusion = []  # Recorded loss
cumulative_epochs_stress = []

'''
    Setting values for parameters in hollow cylinders
'''
omiga = 3.497e-6
nu = 0.3
D = 7.08e-15
E = 1e10
R1 = 8e-7
j = 1e-3
R = 8.3145
T = 300
R2 = 4e-7
alfa = j*R1*omiga/D#0.06
theta=omiga*E/(R*T*3.0*0.4)#223.6

def geometric_shapes(xmin = 0.5, xmax = 1.0, tmin = 0.0, tmax = 0.5,
                     numPtsU = 80, numPtsV = 80,
                     numElemU = 20, numElemV = 20, numGauss = 4,
                     boundary_weight = 1):
    '''
        Constructing geometric shapes
        Description: Spatial coordinates 0.5-1.0, temporal coordinates 0.0-1.0, 
                     and create spatio-temporal training points.
        Input: xmin,xmax,tmin,tmax        -- The boundary point of the spatio-temporal coordinates of the model,
                                             note that xmin must not be 0 here
               numPtsU,numPtsU            -- Number of training points for the model's boundary and initial conditions
               numElemU,numElemV,numGauss -- Number of internal spatio-temporal coordinate points of the model and Gaussian parameterization
               boundary_weight            -- Weighting of boundary and initial conditions
        Returns: Xint, Yint, Wint,
                 Xbnd_l, Xbnd_r, Wbnd, Ybnd_l, Ybnd_r,
                 Xinit, Winit, Yinit      -- Spatio-temporal coordinate points of the model and their weights
                 domainGeom -- ...
    '''
    domainCorners = np.array([[xmin,tmin], [xmin,tmax], [xmax,tmin], [xmax,tmax]])
    domainGeom = Quadrilateral(domainCorners)

    xPhys, tPhys, Wint = domainGeom.getQuadIntPts(numElemU, numElemV, numGauss)

    Xint = np.concatenate((xPhys,tPhys),axis=1).astype(data_type)
    Wint = Wint.astype(data_type)
    Yint = np.zeros_like(xPhys).astype(data_type)

    # Setting of the left border adjustment
    xPhysBnd_l, tPhysBnd_l, _, _ = domainGeom.getUnifEdgePts(numPtsU, numPtsV, [0,0,0,1])
    Xbnd_l = np.concatenate((xPhysBnd_l, tPhysBnd_l), axis=1).astype(data_type)
    Ybnd_l = np.zeros_like(Xbnd_l).astype(data_type)
    Wbnd = boundary_weight*np.ones_like(Ybnd_l).astype(data_type) # Weighting of borders

    # Setting of the right border adjustment
    xPhysBnd_r, tPhysBnd_r, _, _ = domainGeom.getUnifEdgePts(numPtsU, numPtsV, [0,1,0,0])
    Xbnd_r = np.concatenate((xPhysBnd_r, tPhysBnd_r), axis=1).astype(data_type)
    Ybnd_r = np.ones_like(Xbnd_r).astype(data_type)
    # Here the same weights are used as for the left border
    # Wbnd = boundary_weight*np.ones_like(Ybnd).astype(data_type) # Weighting of borders

    # Setting of initial conditions
    xPhysInit, tPhysInit, _, _ = domainGeom.getUnifEdgePts(numPtsU, numPtsV, [1,0,0,0])
    Xinit = np.concatenate((xPhysInit, tPhysInit), axis=1).astype(data_type)
    Yinit = np.zeros_like(Xinit).astype(data_type)
    Winit = boundary_weight*np.ones_like(Yinit).astype(data_type) # Weighting of initial conditions

    # Displays the set training points
    plt.scatter(Xint[:,0], Xint[:,1], s=0.5)
    plt.scatter(Xbnd_l[:,0], Xbnd_l[:,1], s=1, c='red')
    plt.scatter(Xbnd_r[:,0], Xbnd_r[:,1], s=1, c='black')
    plt.scatter(Xinit[:,0], Xinit[:,1], s=1, c='yellow')
    plt.title("Initial/Boundary Matching and Internal Integration Points")
    plt.savefig('fig/distribution_chart.svg', format='svg', dpi=300)
    plt.show()
    return Xint, Yint, Wint, Xbnd_l, Xbnd_r, Wbnd, Ybnd_l, Ybnd_r, Xinit, Winit, Yinit, domainGeom


class Diffusion_DEM(tf.keras.Model):
    '''
        Classes for modeling computational diffusion
        DESCRIPTION: This class contains the computation of the PDE loss, the computation of the boundary condition loss and
                     the computation of the initial condition loss for the diffusion model of a hollow cylinder.
    '''
    def __init__(self, layers, train_op, num_epoch, print_epoch):
        super(Diffusion_DEM, self).__init__()
        self.model_layers = layers
        self.train_op = train_op
        self.num_epoch = num_epoch
        self.print_epoch = print_epoch
        self.adam_loss_hist = []

    def call(self, X):
        return self.u(X[:, 0:1], X[:, 1:2])

    # Running the model
    @tf.function
    def u(self, xPhys, yPhys):
        X = tf.concat([xPhys, yPhys], 1)
        X = 2.0 * (X - self.bounds["lb"]) / (self.bounds["ub"] - self.bounds["lb"]) - 1.0
        for l in self.model_layers:
            X = l(X)
        return X

    # Return the first derivatives
    @tf.function
    def du(self, xPhys, yPhys):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(xPhys)
            tape.watch(yPhys)
            c_val = self.u(xPhys, yPhys)
        dcdx_val = tape.gradient(c_val, xPhys)  # c_x
        dcdt_val = tape.gradient(c_val, yPhys)  # c_t
        del tape
        return dcdx_val, dcdt_val

    @tf.function
    def d2u(self, xPhys, yPhys):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(xPhys)
            tape.watch(yPhys)
            dcdx_val, _ = self.du(xPhys, yPhys)
        d2cdx2_val = tape.gradient(dcdx_val, xPhys)  # C_xx
        del tape
        return d2cdx2_val

    """
        Establish the partial derivatives with respect to u
    """
    def du_fc(self, xPhys, yPhys):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(xPhys)
            tape.watch(yPhys)
            u_val = pred_model_stress(tf.concat([xPhys, yPhys], axis=1))
        dudx_fc = tape.gradient(u_val, xPhys)
        del tape
        return dudx_fc

    def d2u_fc(self, xPhys, yPhys):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(xPhys)
            tape.watch(yPhys)
            dudx_fc = self.du_fc(xPhys, yPhys)
        d2udx2_fc = tape.gradient(dudx_fc, xPhys)
        del tape
        return d2udx2_fc

    def d3u_fc(self, xPhys, yPhys):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(xPhys)
            tape.watch(yPhys)
            d2udx2_fc = self.d2u_fc(xPhys, yPhys)
        d3udx3_fc = tape.gradient(d2udx2_fc, xPhys)
        del tape
        return d3udx3_fc

    """"End"""

    @tf.function
    def get_loss(self, Xint, Wint, Yint, XbndDir, WbndDir, YbndDir, Xinit, Yinit, WinitDir, Xbnd_r, Ybnd_r,
                 cycle_number):
        int_loss, bnd_loss, init_loss = self.get_all_losses(Xint, Wint, Yint,
                                                            XbndDir, WbndDir, YbndDir,
                                                            Xinit, Yinit, WinitDir,
                                                            Xbnd_r, Ybnd_r, cycle_number)
        return int_loss + bnd_loss + init_loss

    @tf.function
    def get_all_losses(self, Xint, Wint, Yint, XbndDir_l, WbndDir, YbndDir_l, XinitDir, YinitDir, WinitDir, Xbnd_r,
                       Ybnd_r, cycle_number):

        #  calculate int loss
        xPhys = Xint[:, 0:1]
        tPhys = Xint[:, 1:2]

        c_val = self.u(xPhys, tPhys)
        dcdx_val, dcdy_val = self.du(xPhys, tPhys)
        d2cdx2_val = self.d2u(xPhys, tPhys)

        if cycle_number == 0:
            f_val_diffusion = xPhys ** 3 * dcdy_val - xPhys ** 3 * d2cdx2_val - xPhys ** 2 * dcdx_val
        else:
            # Calculation of partial derivatives
            u_val = tf.stop_gradient(pred_model_stress(tf.concat([xPhys, tPhys], axis=1)))
            dudx_val = self.du_fc(xPhys, tPhys)
            d2udx2_val = self.d2u_fc(xPhys, tPhys)
            d3udx3_val = self.d3u_fc(xPhys, tPhys)
            f_val_diffusion = xPhys ** 3 * dcdy_val - xPhys ** 3 * d2cdx2_val - xPhys ** 2 * dcdx_val + theta * c_val * (
                        2.0 * xPhys ** 2 * d2udx2_val - xPhys * dudx_val + u_val - alfa * xPhys ** 2 * dcdx_val + xPhys ** 3 * d3udx3_val - alfa * xPhys ** 3 * d2cdx2_val) + theta * xPhys * dcdx_val * (
                                          xPhys ** 2 * d2udx2_val + xPhys * dudx_val - u_val - alfa * dcdx_val * xPhys ** 2)  # stress limitations
        f_val_diffusion = tf.reduce_mean(tf.math.square(f_val_diffusion))
        int_loss = f_val_diffusion

        # calculate boundary left loss
        xPhys_bnd_l = XbndDir_l[:, 0:1]
        yPhys_bnd_l = XbndDir_l[:, 1:2]
        c_val_bnd_l = self.u(xPhys_bnd_l, yPhys_bnd_l)
        dcdx_val_l, _ = self.du(xPhys_bnd_l, yPhys_bnd_l)
        if cycle_number == 0:
            f_val_l_d = dcdx_val_l
            f_val_l_d = tf.reduce_mean(tf.math.square(f_val_l_d - YbndDir_l))
        else:
            u_val_bnd_l = tf.stop_gradient(pred_model_stress(tf.concat([xPhys_bnd_l, yPhys_bnd_l], axis=1)))
            dudx_val_l = self.du_fc(xPhys_bnd_l, yPhys_bnd_l)
            d2udx2_val_l = self.d2u_fc(xPhys_bnd_l, yPhys_bnd_l)
            f_val_l_d = tf.reduce_mean(tf.math.square(xPhys_bnd_l ** 2 * dcdx_val_l - theta * c_val_bnd_l * (
                        xPhys_bnd_l ** 2 * d2udx2_val_l + xPhys_bnd_l * dudx_val_l - u_val_bnd_l - alfa * dcdx_val_l * xPhys_bnd_l ** 2)))

        bnd_loss_l = f_val_l_d
        # calculate boundary right loss
        xPhys_bnd_r = Xbnd_r[:, 0:1]
        yPhys_bnd_r = Xbnd_r[:, 1:2]
        c_val_bnd_r = self.u(xPhys_bnd_r, yPhys_bnd_r)
        dcdx_val_r, _ = self.du(xPhys_bnd_r, yPhys_bnd_r)
        if cycle_number == 0:
            f_val_r_d = dcdx_val_r
            f_val_r_d = tf.reduce_mean(tf.math.square(f_val_r_d - Ybnd_r))
        else:
            u_val_bnd_r = tf.stop_gradient(pred_model_stress(tf.concat([xPhys_bnd_r, yPhys_bnd_r], axis=1)))
            dudx_val_r = self.du_fc(xPhys_bnd_r, yPhys_bnd_r)
            d2udx2_val_r = self.d2u_fc(xPhys_bnd_r, yPhys_bnd_r)
            f_val_r_d = tf.reduce_mean(tf.math.square(xPhys_bnd_r ** 2 * dcdx_val_r - theta * c_val_bnd_r * (
                        xPhys_bnd_r ** 2 * d2udx2_val_r + xPhys_bnd_r * dudx_val_r - u_val_bnd_r - alfa * dcdx_val_r * xPhys_bnd_r ** 2) - xPhys_bnd_r ** 2))

        bnd_loss_r = f_val_r_d
        # calculate init loss
        xPhys_init = XinitDir[:, 0:1]
        yPhys_init = XinitDir[:, 1:2]
        c_val_init = self.u(xPhys_init, yPhys_init)

        f_val_init_d = tf.reduce_mean(tf.math.square(c_val_init - YinitDir))

        bnd_loss = bnd_loss_l + bnd_loss_r
        #       均方差初始条件
        init_loss = f_val_init_d

        return int_loss, bnd_loss, init_loss

    # get gradients
    @tf.function
    def get_grad(self, Xint, Wint, Yint, Xbnd_l, Wbnd, Ybnd_l, Xinit, Yinit, WinitDir, Xbnd_r, Ybnd_r, cycle_number):
        with tf.GradientTape() as tape:
            tape.watch(self.trainable_variables)
            L = self.get_loss(Xint, Wint, Yint, Xbnd_l, Wbnd, Ybnd_l, Xinit, Yinit, WinitDir, Xbnd_r, Ybnd_r,
                              cycle_number)
        g = tape.gradient(L, self.trainable_variables)
        return L, g

    # perform gradient descent
    def network_learn(self, Xint, Wint, Yint, Xbnd_l, Wbnd, Ybnd_l, Xinit, Yinit, Winit, Xbnd_r, Ybnd_r, cycle_number):
        xmin = tf.math.reduce_min(Xint[:, 0])
        ymin = tf.math.reduce_min(Xint[:, 1])
        xmax = tf.math.reduce_max(Xint[:, 0])
        ymax = tf.math.reduce_max(Xint[:, 1])
        self.bounds = {"lb": tf.reshape(tf.stack([xmin, ymin], 0), (1, 2)),
                       "ub": tf.reshape(tf.stack([xmax, ymax], 0), (1, 2))}
        for i in range(self.num_epoch):
            L, g = self.get_grad(Xint, Wint, Yint, Xbnd_l, Wbnd, Ybnd_l, Xinit, Yinit, Winit, Xbnd_r, Ybnd_r,
                                 cycle_number)
            self.train_op.apply_gradients(zip(g, self.trainable_variables))
            self.adam_loss_hist.append(L)
            if i % self.print_epoch == 0:
                int_loss, bnd_loss, init_loss = self.get_all_losses(Xint, Wint, Yint,
                                                                    Xbnd_l, Wbnd, Ybnd_l,
                                                                    Xinit, Yinit, Winit, Xbnd_r,Ybnd_r,
                                                                    cycle_number)
                print("Epoch {} loss: {} int_loss:{} bnd_loss:{} init_loss:{} ".format(i, L, int_loss, bnd_loss, init_loss))


class Diffusion_DEM_Stress(tf.keras.Model):
    '''
        Class for calculating diffusion-induced stresses
        DESCRIPTION: This class contains the computation of the PDE loss, the computation of the boundary condition loss and
                     the computation of the initial condition loss for the diffusion-induced stress model of a hollow cylinder.
    '''
    def __init__(self, layers, train_op, num_epoch, print_epoch):
        super(Diffusion_DEM_Stress, self).__init__()
        self.model_layers = layers
        self.train_op = train_op
        self.num_epoch = num_epoch
        self.print_epoch = print_epoch
        self.adam_loss_hist = []

    def call(self, X):
        return self.u(X[:, 0:1], X[:, 1:2])

    # Running the model
    @tf.function
    def u(self, xPhys, yPhys):
        X = tf.concat([xPhys, yPhys], 1)
        X = 2.0 * (X - self.bounds["lb"]) / (self.bounds["ub"] - self.bounds["lb"]) - 1.0
        for l in self.model_layers:
            X = l(X)
        return X

    # Return the first derivatives
    @tf.function
    def du(self, xPhys, yPhys):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(xPhys)
            tape.watch(yPhys)
            u_val = self.u(xPhys, yPhys)
        dudx_val = tape.gradient(u_val, xPhys)  # u_x
        del tape
        return dudx_val

    @tf.function
    def d2u(self, xPhys, yPhys):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(xPhys)
            tape.watch(yPhys)
            dudx_val = self.du(xPhys, yPhys)
        d2udx2_val = tape.gradient(dudx_val, xPhys)  # U_xx
        del tape
        return d2udx2_val

    @tf.function
    def d3u(self, xPhys, yPhys):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(xPhys)
            tape.watch(yPhys)
            d2udx2_val = self.d2u(xPhys, yPhys)
        d3udx3_val = tape.gradient(d2udx2_val, xPhys)  # U_xx
        del tape
        return d3udx3_val

    @tf.function
    def dcdx_fc(self, xPhys, yPhys):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(xPhys)
            tape.watch(yPhys)
            c_val = pred_model(tf.concat([xPhys, yPhys], axis=1))
        dcdx_val = tape.gradient(c_val, xPhys)
        del tape
        return dcdx_val

    @tf.function
    def get_loss(self, Xint, Wint, Yint, XbndDir, WbndDir, YbndDir, Xinit, Yinit, WinitDir, Xbnd_r, Ybnd_r):
        int_loss, bnd_loss, init_loss = self.get_all_losses(Xint, Wint, Yint,
                                                            XbndDir, WbndDir, YbndDir,
                                                            Xinit, Yinit, WinitDir,
                                                            Xbnd_r, Ybnd_r)
        return int_loss + bnd_loss + init_loss


    @tf.function
    def get_all_losses(self, Xint, Wint, Yint, XbndDir_l, WbndDir, YbndDir_l, XinitDir, YinitDir, WinitDir, Xbnd_r,
                       Ybnd_r):

        # calculate int loss
        xPhys = Xint[:, 0:1]
        tPhys = Xint[:, 1:2]
        u_val = self.u(xPhys, tPhys)
        dudx_val = self.du(xPhys, tPhys)
        # d2udx2_val = self.d2u(xPhys,tPhys)
        # dcdx_val = self.dcdx_fc(xPhys,tPhys)

        # strain calculation
        eps_xx_val = dudx_val
        eps_yy_val = tf.divide(u_val, xPhys)

        lam_1 = 1 / (1 - 2 * nu)
        lam_2 = (1 - nu) / (1 + nu)
        lam_3 = nu / (1 + nu)
        c_val = tf.stop_gradient(pred_model(tf.concat([xPhys, tPhys], axis=1)))

        stress_xx_val = lam_1 * (lam_2 * eps_xx_val + lam_3 * eps_yy_val)
        stress_yy_val = lam_1 * (lam_3 * eps_xx_val + lam_2 * eps_yy_val)
        '''
            If the energy method is used then the above line of code is used, 
            if the balance equation is used then the following two lines of code are used, 
            and the modifications should be made with attention to the use of internal parameters (derivatives, etc.).
        '''
        f_val_stress = tf.reduce_sum(((0.5*(eps_xx_val * stress_xx_val + eps_yy_val * stress_yy_val))-lam_1*alfa*c_val/3*(eps_xx_val+eps_yy_val)) * Wint)  # Energy Method

        int_loss = f_val_stress

        # calculate boundary left loss
        xPhys_bnd_l = XbndDir_l[:, 0:1]
        yPhys_bnd_l = XbndDir_l[:, 1:2]

        u_val_bnd_l = self.u(xPhys_bnd_l, yPhys_bnd_l)
        dudx_val_l = self.du(xPhys_bnd_l, yPhys_bnd_l)
        # Strain at the left boundary
        eps_xx_val_l_s = dudx_val_l
        eps_yy_val_l_s = tf.divide(u_val_bnd_l, xPhys_bnd_l)

        # Stress at the left boundary
        c_val_bnd_l = tf.stop_gradient(pred_model(tf.concat([xPhys_bnd_l, yPhys_bnd_l], axis=1)))
        stress_xx_val_l_s = lam_1 * (lam_2 * eps_xx_val_l_s + lam_3 * eps_yy_val_l_s - alfa * c_val_bnd_l / 3)

        bnd_loss_l = tf.reduce_mean(tf.math.square(stress_xx_val_l_s))

        # calculate boundary right loss
        xPhys_bnd_r = Xbnd_r[:, 0:1]
        yPhys_bnd_r = Xbnd_r[:, 1:2]

        u_val_bnd_r = self.u(xPhys_bnd_r, yPhys_bnd_r)
        dudx_val_r = self.du(xPhys_bnd_r, yPhys_bnd_r)

        # Strain at the right boundary
        eps_xx_val_r_s = dudx_val_r
        eps_yy_val_r_s = tf.divide(u_val_bnd_r, xPhys_bnd_r)
        # Stress at the right boundary
        c_val_bnd_r = tf.stop_gradient(pred_model(tf.concat([xPhys_bnd_r, yPhys_bnd_r], axis=1)))
        stress_xx_val_r_s = lam_1 * (lam_2 * eps_xx_val_r_s + lam_3 * eps_yy_val_r_s - alfa * c_val_bnd_r / 3)

        bnd_loss_r = tf.reduce_mean(tf.math.square(stress_xx_val_r_s))

        # calculate init loss
        xPhys_init = XinitDir[:, 0:1]
        yPhys_init = XinitDir[:, 1:2]

        u_val_init = self.u(xPhys_init, yPhys_init)
        f_val_init_s = tf.reduce_mean(tf.math.square(u_val_init - YinitDir))

        bnd_loss = bnd_loss_l + bnd_loss_r
        init_loss = f_val_init_s

        return int_loss, bnd_loss, init_loss

    # get gradients
    @tf.function
    def get_grad(self, Xint, Wint, Yint, Xbnd_l, Wbnd, Ybnd_l, Xinit, Yinit, WinitDir, Xbnd_r, Ybnd_r):
        with tf.GradientTape() as tape:
            tape.watch(self.trainable_variables)
            L = self.get_loss(Xint, Wint, Yint, Xbnd_l, Wbnd, Ybnd_l, Xinit, Yinit, WinitDir, Xbnd_r, Ybnd_r)
        g = tape.gradient(L, self.trainable_variables)
        return L, g

    # perform gradient descent
    def network_learn(self, Xint, Wint, Yint, Xbnd_l, Wbnd, Ybnd_l, Xinit, Yinit, Winit, Xbnd_r, Ybnd_r):
        xmin = tf.math.reduce_min(Xint[:, 0])
        ymin = tf.math.reduce_min(Xint[:, 1])
        xmax = tf.math.reduce_max(Xint[:, 0])
        ymax = tf.math.reduce_max(Xint[:, 1])
        self.bounds = {"lb": tf.reshape(tf.stack([xmin, ymin], 0), (1, 2)),
                       "ub": tf.reshape(tf.stack([xmax, ymax], 0), (1, 2))}
        for i in range(self.num_epoch):
            L, g = self.get_grad(Xint, Wint, Yint, Xbnd_l, Wbnd, Ybnd_l, Xinit, Yinit, Winit, Xbnd_r, Ybnd_r)
            self.train_op.apply_gradients(zip(g, self.trainable_variables))
            self.adam_loss_hist.append(L)
            if i % self.print_epoch == 0:
                int_loss, bnd_loss, init_loss = self.get_all_losses(Xint, Wint, Yint,
                                                                    Xbnd_l, Wbnd, Ybnd_l,
                                                                    Xinit, Yinit, Winit, Xbnd_r, Ybnd_r)
                print("Epoch {} loss: {} int_loss:{} bnd_loss:{} init_loss:{} ".format(i, L, int_loss, bnd_loss, init_loss))


def train_diffusion_model():
    '''
        Setting up Circuit Training
    '''
    # Reset the optimizer
    pred_model.train_op = tf.keras.optimizers.Adam()
    print("Training (ADAM)...")
    pred_model.network_learn(Xint_tf, Wint_tf, Yint_tf,
                             Xbnd_l_tf, Wbnd_tf, Ybnd_l_tf,
                             Xinit_tf, Yinit_tf, Winit_tf,
                             Xbnd_r_tf, Ybnd_r_tf, cycle_number
                             )

    print("Training (BFGS)...")
    loss_func = tfp_function_factory(pred_model,
                                     Xint_tf, Wint_tf, Yint_tf,
                                     Xbnd_l_tf, Wbnd_tf, Ybnd_l_tf,
                                     Xinit_tf, Yinit_tf, Winit_tf,
                                     Xbnd_r_tf, Ybnd_r_tf, cycle_number)
    init_params = tf.dynamic_stitch(loss_func.idx, pred_model.trainable_variables)  # .numpy()
    # train the model with BFGS solver
    results = tfp.optimizer.bfgs_minimize(
        value_and_gradients_function=loss_func, initial_position=init_params,
        max_iterations=500, tolerance=1e-7)
    loss_func.assign_new_model_parameters(results.position)

    pred_model.adam_loss_hist.extend(loss_func.history)
    cumulative_epochs_diffusion.append(len(pred_model.adam_loss_hist))
    _ = pred_model(Xint_tf[:1])
    directory_diffusion = os.path.dirname(model_save_path_diffusion)
    os.makedirs(directory_diffusion,exist_ok=True)
    pred_model.save_weights(model_save_path_diffusion)

def train_stress_model():
    # Reset the optimizer
    pred_model_stress.train_op = tf.keras.optimizers.Adam()
    print("Training (ADAM)...")
    pred_model_stress.network_learn(Xint_tf, Wint_tf, Yint_tf,
                                    Xbnd_l_tf, Wbnd_tf, Ybnd_l_tf,
                                    Xinit_tf, Yinit_tf, Winit_tf,
                                    Xbnd_r_tf, Ybnd_r_tf
                                    )
    print("Training (BFGS)...")
    loss_func_stress = tfp_function_factory(pred_model_stress,
                                            Xint_tf, Wint_tf, Yint_tf,
                                            Xbnd_l_tf, Wbnd_tf, Ybnd_l_tf,
                                            Xinit_tf, Yinit_tf, Winit_tf,
                                            Xbnd_r_tf, Ybnd_r_tf)
    init_params_stress = tf.dynamic_stitch(loss_func_stress.idx, pred_model_stress.trainable_variables)  # .numpy()
    # train the model with BFGS solver
    results_stress = tfp.optimizer.bfgs_minimize(
        value_and_gradients_function=loss_func_stress, initial_position=init_params_stress,
        max_iterations=500, tolerance=1e-7)
    loss_func_stress.assign_new_model_parameters(results_stress.position)

    pred_model_stress.adam_loss_hist.extend(loss_func_stress.history)
    cumulative_epochs_stress.append(len(pred_model_stress.adam_loss_hist))
    _ = pred_model_stress(Xint_tf[:1])
    directory_stress = os.path.dirname(model_save_path_stress)
    os.makedirs(directory_stress, exist_ok=True)
    pred_model_stress.save_weights(model_save_path_stress)

def train_model():
    global cycle_number
    t0 = time.time()
    for i in range(number_of_iterations):
        # Train the first neural network
        train_diffusion_model()
        # Train the second neural network
        train_stress_model()
        cycle_number += 1
        print("No.", cycle_number, "Wheel training complete")
    t1 = time.time()
    print("Total training time", t1 - t0, "seconds")

def load_model_weights(model, path):
    try:
        model.load_weights(path)
        print(f"Model weights loaded from {path}")
        return True
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return False

Xint, Yint, Wint, Xbnd_l, Xbnd_r, Wbnd, Ybnd_l, Ybnd_r, Xinit, Winit, Yinit, domainGeom = geometric_shapes()

#convert the training data to tensors
Xint_tf = tf.convert_to_tensor(Xint)
Yint_tf = tf.convert_to_tensor(Yint)
Wint_tf = tf.convert_to_tensor(Wint)

Xbnd_l_tf = tf.convert_to_tensor(Xbnd_l)
Xbnd_r_tf = tf.convert_to_tensor(Xbnd_r)
Wbnd_tf = tf.convert_to_tensor(Wbnd)
Ybnd_l_tf = tf.convert_to_tensor(Ybnd_l)
Ybnd_r_tf = tf.convert_to_tensor(Ybnd_r)

Xinit_tf = tf.convert_to_tensor(Xinit)
Winit_tf = tf.convert_to_tensor(Winit)
Yinit_tf = tf.convert_to_tensor(Yinit)

'''
    Build a neural network
'''
# The first neural network
tf.keras.backend.set_floatx(data_type)
l1 = tf.keras.layers.Dense(32, "tanh")
l2 = tf.keras.layers.Dense(32, "tanh")
l3 = tf.keras.layers.Dense(32, "tanh")
l4 = tf.keras.layers.Dense(32, "tanh")
l5 = tf.keras.layers.Dense(1, None)
train_op = tf.keras.optimizers.Adam()
num_epoch = 501
num_epoch_print = 500
pred_model = Diffusion_DEM([l1,l2, l5], train_op, num_epoch, num_epoch_print)  # Adam optimizer, 3000 training sessions, prints a loss every 1000 sessions

# The second neural network
l6 = tf.keras.layers.Dense(32, "swish")
l7 = tf.keras.layers.Dense(32, "swish")
l8 = tf.keras.layers.Dense(32, "swish")
l9 = tf.keras.layers.Dense(32, "swish")
l10 = tf.keras.layers.Dense(1, None)
train_op3 = tf.keras.optimizers.Adam()
pred_model_stress = Diffusion_DEM_Stress([l6,l7,l10], train_op3, num_epoch, num_epoch_print) # Adam optimizer, 3000 training sessions, prints a loss every 1000 sessions

# Choose to load or train the model
'''
    Note: The loaded model architecture and the pre-built model architecture should be the same
'''
choice = input("Do you want to (1) Load existing model weights or (2) Train a new model? Enter 1 or 2: ")
if choice == '1':

    # Trying to load weights
    '''
    The .h5 file is used here, so it needs to be trained once to initialize the internal parameters before loading the data
    '''
    pred_model = Diffusion_DEM([l1,l2, l5], train_op, 1,
                               num_epoch_print)
    pred_model.network_learn(Xint_tf, Wint_tf, Yint_tf,
                             Xbnd_l_tf, Wbnd_tf, Ybnd_l_tf,
                             Xinit_tf, Yinit_tf, Winit_tf,
                             Xbnd_r_tf, Ybnd_r_tf, cycle_number
                             )
    pred_model_stress = Diffusion_DEM_Stress([l6,l7, l10], train_op3, 1,
                                             num_epoch_print)
    pred_model_stress.network_learn(Xint_tf, Wint_tf, Yint_tf,
                                    Xbnd_l_tf, Wbnd_tf, Ybnd_l_tf,
                                    Xinit_tf, Yinit_tf, Winit_tf,
                                    Xbnd_r_tf, Ybnd_r_tf
                                    )
    _ = pred_model(Xint_tf[:1])
    _ = pred_model_stress(Xint_tf[:1])
    loaded_diffusion = load_model_weights(pred_model, model_save_path_diffusion)
    loaded_stress = load_model_weights(pred_model_stress, model_save_path_stress)

    if not (loaded_diffusion and loaded_stress):
        print("Failed to load model weights. Training new models.")
        choice = '2'

if choice == '2':
    train_model()

'''
    Prediction 
'''
# numPtsUTest = 2*numPtsU
# numPtsVTest = 2*numPtsV
numPtsUTest = 501
numPtsVTest = 501
xPhysTest, tPhysTest = domainGeom.getUnifIntPts(numPtsUTest, numPtsVTest, [1,1,1,1])
XTest = np.concatenate((xPhysTest,tPhysTest),axis=1).astype(data_type)
XTest_tf = tf.convert_to_tensor(XTest)
YTest = pred_model(XTest_tf).numpy()
cx_Test = YTest[:,0:1]

xPhysTest2D = np.resize(XTest[:,0], [numPtsUTest, numPtsVTest])
yPhysTest2D = np.resize(XTest[:,1], [numPtsUTest, numPtsVTest])

cTest2D = np.resize(cx_Test, [numPtsUTest, numPtsVTest])
plt.contourf(xPhysTest2D, yPhysTest2D, cTest2D, 255, cmap=plt.cm.jet)
plt.colorbar()
plt.title("Calculate the solution of the concentration field") # Calculate the solution of the concentration field
plt.savefig('fig/concentration_field.svg', format='svg', dpi=300)
plt.show()

YTest_stress = pred_model_stress(XTest_tf).numpy()
ux_Test = YTest_stress[:,0:1]
uTest2D = np.resize(ux_Test, [numPtsUTest, numPtsVTest])
plt.contourf(xPhysTest2D, yPhysTest2D, uTest2D, 255, cmap=plt.cm.jet)
plt.colorbar()
plt.title("Calculate the solution of the strain field") # Calculate the solution of the strain field
plt.savefig('fig/strain_field.svg', format='svg', dpi=300)
plt.show()

# 应力场和轴向力
x_stress = tf.convert_to_tensor(xPhysTest.astype(data_type))
t_stress = tf.convert_to_tensor(tPhysTest.astype(data_type))
du_dx_Test = pred_model_stress.du(x_stress,t_stress).numpy()
du_dx_Test = du_dx_Test[:,0:1]
du_dx_Test2D = np.resize(du_dx_Test, [numPtsUTest, numPtsVTest])
xPhysTest = np.resize(xPhysTest, [numPtsUTest, numPtsVTest]).astype(data_type)
sigma3 = 0.3/(1.3*0.4)*du_dx_Test2D+0.3/(1.3*0.4)*uTest2D/xPhysTest-alfa/1.2*cTest2D
plt.contourf(xPhysTest2D, yPhysTest2D, sigma3, 255, cmap=plt.cm.jet)
plt.colorbar()
plt.title("Calculate the solution of the stress field") # Calculate the solution of the strain field
plt.show()

# 计算步长
dx = (1 - 0.5) / (numPtsUTest - 1)
dt = (0.5 - 0) / (numPtsVTest - 1)
# 积分 sigma3
sigma3_in = sigma3 * np.pi * 2*xPhysTest
integral_sigma3_over_time = [np.trapz(sigma3_in[i, :], dx=dx) for i in range(numPtsVTest)]
# 画图
time_steps = np.linspace(0, 0.5, numPtsVTest)
plt.plot(time_steps,integral_sigma3_over_time)
plt.title('Integral of Sigma3 Over Time')
plt.xlabel('Time')
plt.ylabel('Integral of Sigma3')
plt.show()
# 准备数据写入 Excel
results_df = pd.DataFrame({
    'Time': time_steps,
    'Integral of Sigma3': integral_sigma3_over_time
})
excel_file_path = 'Integral_Sigma3_Results.xlsx'
results_df.to_excel(excel_file_path, index=False)
'''
    Loss Plot
'''
print(cumulative_epochs_diffusion)
print(cumulative_epochs_stress)
# Diffusion Loss
plt.figure(dpi=300)
plt.plot(pred_model.adam_loss_hist, label='Diffusion Loss')
for epoch_end in cumulative_epochs_diffusion:
    plt.axvline(x=epoch_end, color='gray', linestyle=':', linewidth=1)
plt.title('Diffusion Loss During Training')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Stress Loss
plt.figure(dpi=300)
plt.plot(pred_model_stress.adam_loss_hist, label='Stress Loss', linestyle='--')
for epoch_end in cumulative_epochs_stress:
    plt.axvline(x=epoch_end, color='gray', linestyle=':', linewidth=1)
plt.title('Stress Loss During Training')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
'''
# Export data to excel

y121 = cTest2D[60:60 + 1, :]
x121 = xPhysTest2D[0:1, :]

y122 = cTest2D[120:120 + 1, :]
x122 = xPhysTest2D[0:1, :]

y123 = cTest2D[180:180 + 1, :]
x123 = xPhysTest2D[0:1, :]

y124 = cTest2D[240:240+ 1, :]
x124 = xPhysTest2D[0:1, :]

y125 = cTest2D[300:300 + 1, :]
x125 = xPhysTest2D[0:1, :]

plt.xlim(0.5, 1)

plt.plot(x121[0], y121[0], x122[0], y122[0], x123[0], y123[0], x124[0], y124[0], x125[0], y125[0])
plt.show()

# Prepare data
data = {
    "x121": x121[0].tolist(),
    "y121": y121[0].tolist(),

    # "x122": x122[0].tolist(),
    "y122": y122[0].tolist(),

    # "x123": x123[0].tolist(),
    "y123": y123[0].tolist(),

    # "x124": x124[0].tolist(),
    "y124": y124[0].tolist(),

    # "x125": x125[0].tolist(),
    "y125": y125[0].tolist()
}

df = pd.DataFrame(data)

df.to_excel("output_data_c.xlsx", index=False)

y221 = uTest2D[60:60 + 1, :]
x221 = xPhysTest2D[0:1,:]

y222 = uTest2D[120:120 + 1, :]
x222 = xPhysTest2D[0:1,:]

y223 = uTest2D[180:180 + 1, :]
x223 = xPhysTest2D[0:1,:]

y224 = uTest2D[240:240 + 1, :]
x224 = xPhysTest2D[0:1,:]

y225 = uTest2D[300:300 + 1, :]
x225 = xPhysTest2D[0:1,:]

plt.xlim(0.5,1)

plt.plot(x221[0],y221[0],x222[0],y222[0],x223[0],y223[0],x224[0],y224[0],x225[0],y225[0])
plt.show()

data = {
    # "x221": x221[0].tolist(),
    "y221": y221[0].tolist(),
    # "x222": x222[0].tolist(),
    "y222": y222[0].tolist(),
    # "x223": x223[0].tolist(),
    "y223": y223[0].tolist(),
    # "x224": x224[0].tolist(),
    "y224": y224[0].tolist(),
    # "x225": x225[0].tolist(),
    "y225": y225[0].tolist()
}

df = pd.DataFrame(data)

df.to_excel("output_data_u.xlsx", index=False)

# 输出导数
y321 = sigma3[60:60 + 1, :]
x321 = xPhysTest2D[0:1,:]

y322 = sigma3[120:120 + 1, :]
x322 = xPhysTest2D[0:1,:]

y323 = sigma3[180:180 + 1, :]
x323 = xPhysTest2D[0:1,:]

y324 = sigma3[240:240 + 1, :]
x324 = xPhysTest2D[0:1,:]

y325 = sigma3[300:300 + 1, :]
x325 = xPhysTest2D[0:1,:]

plt.xlim(0.5,1)

plt.plot(x321[0],y321[0],x322[0],y322[0],x323[0],y323[0],x324[0],y324[0],x325[0],y325[0])
plt.show()

data = {
    # "x221": x221[0].tolist(),
    "y221": y321[0].tolist(),
    # "x222": x222[0].tolist(),
    "y222": y322[0].tolist(),
    # "x223": x223[0].tolist(),
    "y223": y323[0].tolist(),
    # "x224": x224[0].tolist(),
    "y224": y324[0].tolist(),
    # "x225": x225[0].tolist(),
    "y225": y325[0].tolist()
}

df = pd.DataFrame(data)

df.to_excel("output_data_du_dx.xlsx", index=False)

# 将TensorFlow张量转换为NumPy数组
adam_loss_hist_values = [x.numpy() for x in pred_model.adam_loss_hist]
adam_loss_hist_stress_values = [x.numpy() for x in pred_model_stress.adam_loss_hist]

# 创建两个DataFrame
df_adam_loss_hist = pd.DataFrame(adam_loss_hist_values, columns=['adam_loss_hist'])
df_adam_loss_hist_stress = pd.DataFrame(adam_loss_hist_stress_values, columns=['adam_loss_hist_stress'])

# 创建一个Pandas Excel writer使用XlsxWriter作为引擎
with pd.ExcelWriter('loss_data_separate.xlsx') as writer:  # save loss data
    df_adam_loss_hist.to_excel(writer, sheet_name='adam_loss_hist', index=False)
    df_adam_loss_hist_stress.to_excel(writer, sheet_name='adam_loss_hist_stress', index=False)
'''