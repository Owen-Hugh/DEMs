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

'''
    Constructing geometric shapes
    Description: Spatial coordinates 0.5-1.0, temporal coordinates 0.0-1.0, 
                 and create spatio-temporal training points.
'''
xmin = 0.5
xmax = 1
tmin = 0
tmax = 0.5
domainCorners = np.array([[xmin,tmin], [xmin,tmax], [xmax,tmin], [xmax,tmax]])
domainGeom = Quadrilateral(domainCorners)

numPtsU = 80
numPtsV = 80
# Gauss's law of products
numElemU = 20
numElemV = 20
numGauss = 4
boundary_weight = 1
xPhys, tPhys, Wint = domainGeom.getQuadIntPts(numElemU, numElemV, numGauss)
data_type = "float64"

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
plt.title("Boundary collocation and interior integration points")
# plt.show()

'''
    Classes for modeling computational diffusion
    DESCRIPTION: This class contains the computation of the PDE loss, the computation of the boundary condition loss and 
                 the computation of the initial condition loss for the diffusion model of a hollow cylinder.
'''
class Diffusion_DEM(tf.keras.Model):
    def __init__(self, layers, train_op, num_epoch, print_epoch):
        super(Diffusion_DEM, self).__init__()
        self.model_layers = layers
        self.train_op = train_op
        self.num_epoch = num_epoch
        self.print_epoch = print_epoch
        self.adam_loss_hist = []

    def call(self, X):
        u_val, c_val = self.u(X[:, 0:1], X[:, 1:2])
        return tf.concat([u_val, c_val], 1)

    def dirichletBound(Self, X, xPhys, yPhys):
        u_val = X[:, 0:1]
        c_val = X[:, 1:2]
        return u_val, c_val

    # Running the model
    @tf.function
    def u(self, xPhys, yPhys):
        X = tf.concat([xPhys, yPhys], 1)
        X = 2.0 * (X - self.bounds["lb"]) / (self.bounds["ub"] - self.bounds["lb"]) - 1.0
        for l in self.model_layers:
            X = l(X)
        u_val, c_val = self.dirichletBound(X, xPhys, yPhys)
        return u_val, c_val

    # Return the first derivatives
    @tf.function
    def du(self, xPhys, yPhys):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(xPhys)
            tape.watch(yPhys)
            u_val, c_val = self.u(xPhys, yPhys)
        dudx_val = tape.gradient(u_val, xPhys)  # u_x
        dcdx_val = tape.gradient(c_val, xPhys)  # c_x
        dcdy_val = tape.gradient(c_val, yPhys)  # c_t
        del tape
        return dudx_val, dcdx_val, dcdy_val

    def d2u(self, xPhys, yPhys):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(xPhys)
            tape.watch(yPhys)
            dudx_val, dcdx_val, _ = self.du(xPhys, yPhys)
        d2udx2_val = tape.gradient(dudx_val, xPhys)  # U_xx
        d2cdx2_val = tape.gradient(dcdx_val, xPhys)  # C_xx
        del tape
        return d2udx2_val, d2cdx2_val

    def d3u(self, xPhys, yPhys):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(xPhys)
            tape.watch(yPhys)
            d2udx2_val, _ = self.d2u(xPhys, yPhys)
        d3udx3_val = tape.gradient(d2udx2_val, xPhys)
        del tape
        return d3udx3_val

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

        xPhys = Xint[:, 0:1]
        tPhys = Xint[:, 1:2]
        u_val, c_val = self.u(xPhys, tPhys)
        dudx_val, dcdx_val, dcdy_val = self.du(xPhys, tPhys)
        d2udx2_val, d2cdx2_val = self.d2u(xPhys, tPhys)
        d3udx3_val = self.d3u(xPhys, tPhys)

        # lam_1 = E/(1-2*nu)
        lam_1 = 1 / (1 - 2 * nu)
        lam_2 = (1 - nu) / (1 + nu)
        lam_3 = nu / (1 + nu)

        # If DEMs
        # eps_xx_val = dudx_val
        # eps_yy_val = tf.divide(u_val, xPhys)
        # stress_xx_val = lam_1 * (lam_2 * eps_xx_val + lam_3 * eps_yy_val - alfa * 2.0 * c_val / 3)
        # stress_yy_val = lam_1 * (lam_3 * eps_xx_val + lam_2 * eps_yy_val - alfa * 2.0 * c_val / 3)
        # f_val_stress = tf.reduce_sum(0.5 * (eps_xx_val * stress_xx_val + eps_yy_val * stress_yy_val) * Wint)
        # if PINN
        f_val_stress = tf.reduce_mean(tf.math.square(
            xPhys ** 2 * d2udx2_val + xPhys * dudx_val - u_val - xPhys ** 2 * dcdx_val * alfa * 1.3 / (0.7 * 3)))  # PINN

        f_val_diffusion = xPhys ** 3 * dcdy_val - xPhys ** 3 * d2cdx2_val - xPhys ** 2 * dcdx_val + theta * c_val * (
                    2.0 * xPhys ** 2 * d2udx2_val - xPhys * dudx_val + u_val - alfa * xPhys ** 2 * dcdx_val + xPhys ** 3 * d3udx3_val - alfa * xPhys ** 3 * d2cdx2_val) + theta * xPhys * dcdx_val * (
                                      xPhys ** 2 * d2udx2_val + xPhys * dudx_val - u_val - alfa * dcdx_val * xPhys ** 2)   # Stress limits
        # f_val_diffusion = xPhys**3 * dcdy_val - xPhys**3 * d2cdx2_val - xPhys**2 * dcdx_val   # Fick Diffusion
        f_val_diffusion = tf.reduce_mean(tf.math.square(f_val_diffusion))

        int_loss = f_val_stress + f_val_diffusion
        # calculate boundary left
        xPhys_bnd_l = XbndDir_l[:, 0:1]
        yPhys_bnd_l = XbndDir_l[:, 1:2]
        u_val_bnd_l, c_val_bnd_l = self.u(xPhys_bnd_l, yPhys_bnd_l)
        dudx_val_l, dcdx_val_l, _ = self.du(xPhys_bnd_l, yPhys_bnd_l)
        d2udx2_val_l, _ = self.d2u(xPhys_bnd_l, yPhys_bnd_l)
        # 边界处应变
        eps_xx_val_l_s = dudx_val_l
        eps_yy_val_l_s = tf.divide(u_val_bnd_l, xPhys_bnd_l)
        # 边界处应力
        stress_xx_val_l_s = lam_1 * (lam_2 * eps_xx_val_l_s + lam_3 * eps_yy_val_l_s - alfa * c_val_bnd_l / 3)
        # stress_xx_val_l_s = 0.7 / 1.3 * dudx_val_l * xPhys_bnd_l + 0.3 / 1.3 * u_val_bnd_l - alfa * c_val_bnd_l * xPhys_bnd_l / 3
        f_val_l_s = tf.reduce_mean(tf.math.square(stress_xx_val_l_s))
        # 扩散损失
        # f_val_l_d = tf.reduce_mean(tf.math.square(dcdx_val_l - Ybnd_l))  # Fick Diffusion
        f_val_l_d = tf.reduce_mean(tf.math.square(xPhys_bnd_l ** 2 * dcdx_val_l - theta * c_val_bnd_l * (
                    xPhys_bnd_l ** 2 * d2udx2_val_l + xPhys_bnd_l * dudx_val_l - u_val_bnd_l - alfa * dcdx_val_l * xPhys_bnd_l ** 2)))
        bnd_loss_l = f_val_l_s + f_val_l_d
        # calculate boundary right
        xPhys_bnd_r = Xbnd_r[:, 0:1]
        yPhys_bnd_r = Xbnd_r[:, 1:2]
        u_val_bnd_r, c_val_bnd_r = self.u(xPhys_bnd_r, yPhys_bnd_r)
        dudx_val_r, dcdx_val_r, _ = self.du(xPhys_bnd_r, yPhys_bnd_r)
        d2udx2_val_r, _ = self.d2u(xPhys_bnd_r, yPhys_bnd_r)
        # 边界处应变
        eps_xx_val_r_s = dudx_val_r
        eps_yy_val_r_s = tf.divide(u_val_bnd_r, xPhys_bnd_r)
        # 边界处应力
        stress_xx_val_r_s = lam_1 * (lam_2 * eps_xx_val_r_s + lam_3 * eps_yy_val_r_s - alfa * c_val_bnd_r / 3)
        # stress_xx_val_r_s = 0.7 / 1.3 * dudx_val_r * xPhys_bnd_r + 0.3 / 1.3 * u_val_bnd_r - alfa * c_val_bnd_r * xPhys_bnd_r / 3
        f_val_r_s = tf.reduce_mean(tf.math.square(stress_xx_val_r_s))
        # 扩散损失
        # f_val_r_d = tf.reduce_mean(tf.math.square(dcdx_val_r - Ybnd_r))   # Fick Diffusion
        f_val_r_d = tf.reduce_mean(tf.math.square(xPhys_bnd_r ** 2 * dcdx_val_r - theta * c_val_bnd_r * (
                    xPhys_bnd_r ** 2 * d2udx2_val_r + xPhys_bnd_r * dudx_val_r - u_val_bnd_r - alfa * dcdx_val_r * xPhys_bnd_r ** 2) - xPhys_bnd_r ** 2))

        bnd_loss_r = f_val_r_s + f_val_r_d
        # calculate init
        xPhys_init = XinitDir[:, 0:1]
        yPhys_init = XinitDir[:, 1:2]
        u_val_init, c_val_init = self.u(xPhys_init, yPhys_init)
        f_val_init_s = tf.reduce_mean(tf.math.square(u_val_init - YinitDir) * WinitDir)
        f_val_init_d = tf.reduce_mean(tf.math.square(c_val_init - YinitDir) * WinitDir)

        bnd_loss = bnd_loss_l + bnd_loss_r
        init_loss = f_val_init_s + f_val_init_d

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
                print("Epoch {} loss: {}".format(i, L))

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
    Build a neural network and train it
'''
# The first neural network
tf.keras.backend.set_floatx(data_type)
l1 = tf.keras.layers.Dense(32, "tanh")
l2 = tf.keras.layers.Dense(32, "tanh")
l3 = tf.keras.layers.Dense(32, "tanh")
l4 = tf.keras.layers.Dense(32, "tanh")
l5 = tf.keras.layers.Dense(2, None)
train_op = tf.keras.optimizers.Adam()
train_op2 = "BFGS-B"
num_epoch = 5000
print_epoch = 1000
pred_model = Diffusion_DEM([l1,l2,l3,l4,l5], train_op, num_epoch, print_epoch)

'''
    Setting up Circuit Training
'''
print("Training (ADAM)...")
t0 = time.time()
pred_model.network_learn(Xint_tf, Wint_tf, Yint_tf,
                         Xbnd_l_tf, Wbnd_tf, Ybnd_l_tf,
                         Xinit_tf,Yinit_tf,Winit_tf,
                         Xbnd_r_tf,Ybnd_r_tf
                        )
t1 = time.time()
print("Time taken (ADAM)", t1-t0, "seconds")

print("Training (BFGS)...")
loss_func = tfp_function_factory(pred_model,
                                 Xint_tf, Wint_tf, Yint_tf,
                                 Xbnd_l_tf, Wbnd_tf, Ybnd_l_tf,
                                 Xinit_tf,Yinit_tf,Winit_tf,
                                 Xbnd_r_tf,Ybnd_r_tf)
init_params = tf.dynamic_stitch(loss_func.idx, pred_model.trainable_variables)#.numpy()
# train the model with BFGS solver
results = tfp.optimizer.bfgs_minimize(
    value_and_gradients_function=loss_func, initial_position=init_params,
          max_iterations=5000, tolerance=1e-14)
loss_func.assign_new_model_parameters(results.position)
pred_model.adam_loss_hist.extend(loss_func.history)
t2 = time.time()
print("Time taken (BFGS)", t2-t1, "seconds")
print("Time taken (Train)", t2-t0, "seconds")
'''
    Prediction 
'''
numPtsUTest = 501
numPtsVTest = 501
xPhysTest, tPhysTest = domainGeom.getUnifIntPts(numPtsUTest, numPtsVTest, [1,1,1,1])
XTest = np.concatenate((xPhysTest,tPhysTest),axis=1).astype(data_type) # x,t拼接
XTest_tf = tf.convert_to_tensor(XTest) # 转换成张量
YTest = pred_model(XTest_tf).numpy()  # 带入神经网络中预测
ux_Test = YTest[:,0:1]
cx_Test = YTest[:,1:2]
xPhysTest2D = np.resize(XTest[:,0], [numPtsUTest, numPtsVTest])
yPhysTest2D = np.resize(XTest[:,1], [numPtsUTest, numPtsVTest])

uTest2D = np.resize(ux_Test, [numPtsUTest, numPtsVTest])
cTest2D = np.resize(cx_Test, [numPtsUTest, numPtsVTest])
plt.contourf(xPhysTest2D, yPhysTest2D, uTest2D, 255, cmap=plt.cm.jet)
plt.colorbar()
plt.title("Computed solution") # 预测解
plt.show()
plt.contourf(xPhysTest2D, yPhysTest2D, cTest2D, 255, cmap=plt.cm.jet)
plt.colorbar()
plt.title("Computed solution") # 预测解
plt.show()

# 应力场
x_stress = tf.convert_to_tensor(xPhysTest.astype(data_type))
t_stress = tf.convert_to_tensor(tPhysTest.astype(data_type))
du_dx_Test,_,_ = pred_model.du(x_stress,t_stress)
du_dx_Test = du_dx_Test.numpy()
du_dx_Test = du_dx_Test[:,0:1]
du_dx_Test2D = np.resize(du_dx_Test, [numPtsUTest, numPtsVTest])
xPhysTest = np.resize(xPhysTest, [numPtsUTest, numPtsVTest])
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
    # "x121": x121[0].tolist(),
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

df.to_excel("output_data_c_couple.xlsx", index=False)

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

df.to_excel("output_data_u_couple.xlsx", index=False)

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
# 创建两个DataFrame
df_adam_loss_hist = pd.DataFrame(adam_loss_hist_values, columns=['adam_loss_hist'])
# 创建一个Pandas Excel writer使用XlsxWriter作为引擎
with pd.ExcelWriter('loss_data_separate_couple.xlsx') as writer:  # save loss data
    df_adam_loss_hist.to_excel(writer, sheet_name='adam_loss_hist', index=False)
