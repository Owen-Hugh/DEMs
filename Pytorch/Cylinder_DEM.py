import math
import torch
import numpy as np
import time
from network import Network
import matplotlib.pyplot as plt
from utility.Geom_Creat import Quadrilateral
from utility.plot import *
import pandas as pd
# import seaborn as sns
# torch.set_num_threads(32)  
random = 44
torch.manual_seed(random)
torch.cuda.manual_seed(random)
torch.cuda.manual_seed_all(random)
np.random.seed(random)

# parameters set
alfa = 3.497e-6 * 8e-7 * 1e-3 / 7.08e-15
theta = 3.497e-6*1e10/(8.3145*300*3.0*0.4)
nu = 0.3
lam_1 = 1 / (1 - 2 * nu)
lam_2 = (1 - nu) / (1 + nu)
lam_3 = nu / (1 + nu)
n_gauss = 4  
num_elem_x, num_elem_t = 20, 20

def plot_predictions(model, domainGeom, numPtsU=10, numPtsV=10):
    """
    Plots the model predictions over the 2D spatiotemporal domain.
    """
    # Generate a grid of points over the domain
    xPhys, tPhys = domainGeom.getUnifIntPts(numPtsU, numPtsV, [1,1,1,1])
    X_plot = np.concatenate((xPhys, tPhys), axis=1).astype(np.float32)
    X_plot = torch.from_numpy(X_plot).to(next(model.parameters()).device)

    # Make predictions
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        u_pred = model(X_plot).cpu().numpy()

    # Reshape predictions for plotting
    x_vals = xPhys[:, 0].reshape((numPtsU, numPtsV))
    t_vals = tPhys[:, 0].reshape((numPtsU, numPtsV))
    u_vals = u_pred.reshape((numPtsU, numPtsV))

    # Create a contour plot
    plt.figure(figsize=(10, 10))
    plt.contourf(x_vals, t_vals, u_vals, 255, cmap=plt.cm.jet)
    plt.colorbar(label='u(x, t)')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title('PINN Model Predictions')
    plt.show()


class AutomaticWeightedLoss(torch.nn.Module):
    def __init__(self, num=3):  # Adjusting for three losses
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += loss
        return loss_sum
    
class Diffusion:
    def __init__(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cycle_number = 1
        # define neural network
        self.model_diffusion = Network(
            input_size=2,  # Number of neurons in the input layer
            hidden_size=32,  # Number of neurons in hidden layer
            output_size=1,  # Number of neurons in the output layer
            depth=2,  # Number of hidden layers
            act=torch.nn.Tanh  
        ).to(device)  
        self.model_physics = Network(
            input_size=2,  # Number of neurons in the input layer
            hidden_size=32,  # Number of neurons in hidden layer
            output_size=1,  # Number of neurons in the output layer
            depth=2,  # Number of hidden layers
            act=lambda: torch.nn.SiLU()  
        ).to(device) 
        
        # Geometric size (Parameters Setting )
        x_min, x_max = 0.5, 1.0
        t_min, t_max = 0.0, 0.5
        h, k = (1.0-0.5)/(num_elem_x*n_gauss), (0.5-0.0)/(num_elem_t*n_gauss)
        x = torch.arange(x_min, x_max + h, h); t = torch.arange(t_min, t_max + k, k)
        # Create Training Points
        # interior domain
        # Get Gauss-Legendre nodes and weights
        self.nodes, self.weights = np.polynomial.legendre.leggauss(n_gauss)
        # convert to torch tensor
        self.nodes = torch.tensor(self.nodes, dtype=torch.float32)
        self.weights = torch.tensor(self.weights, dtype=torch.float32)
        
        # Generate storage for integration points and weights
        quadPts = []
        
        # Generate integral points in each element
        x_edges = np.linspace(x_min, x_max, num_elem_x + 1)
        t_edges = np.linspace(t_min, t_max, num_elem_t + 1)
        
        for i_x in range(num_elem_x):
            for i_t in range(num_elem_t):
                x_min_elem, x_max_elem = x_edges[i_x], x_edges[i_x + 1]
                t_min_elem, t_max_elem = t_edges[i_t], t_edges[i_t + 1]
                x_mapped = 0.5 * (x_max_elem - x_min_elem) * (self.nodes + 1) + x_min_elem
                t_mapped = 0.5 * (t_max_elem - t_min_elem) * (self.nodes + 1) + t_min_elem
                x_mesh, t_mesh = torch.meshgrid(x_mapped, t_mapped, indexing='ij')
                X_elem = torch.stack([x_mesh.flatten(), t_mesh.flatten()], dim=1)
                scale_factor = (x_max_elem - x_min_elem) * (t_max_elem - t_min_elem) / 4
                weights_elem = torch.outer(self.weights, self.weights).flatten() * scale_factor
                quadPts.append(torch.cat([X_elem, weights_elem.unsqueeze(1)], dim=1))
        
        quadPts = torch.cat(quadPts, dim=0)
    
        self.X_Int = quadPts[:, :2]  
        self.W_Int = quadPts[:, 2]   

        # boundary & Initial
        self.XBnd_Left = torch.stack(torch.meshgrid(torch.tensor([x[0]]), t, indexing='ij')).reshape(2, -1).T
        self.XBnd_Right = torch.stack(torch.meshgrid(torch.tensor([x[-1]]), t, indexing='ij')).reshape(2, -1).T
        self.XInit = torch.stack(torch.meshgrid(x, torch.tensor([t[0]]), indexing='ij')).reshape(2, -1).T
        
        self.plot_training_points() # plot
        
        # Boundary and Initial value
        self.XBnd_Left_Val = torch.zeros(len(self.XBnd_Left)).unsqueeze(1)
        self.XBnd_Right_Val = torch.ones(len(self.XBnd_Right)).unsqueeze(1)
        self.XInit_Val = torch.zeros(len(self.XInit)).unsqueeze(1)
        
        # to GPU
        self.X_Int = self.X_Int.to(device)
        self.XBnd_Left = self.XBnd_Left.to(device)
        self.XBnd_Right = self.XBnd_Right.to(device)
        self.XInit = self.XInit.to(device)
        self.XBnd_Left_Val = self.XBnd_Left_Val.to(device)
        self.XBnd_Right_Val = self.XBnd_Right_Val.to(device)
        self.XInit_Val = self.XInit_Val.to(device)
        self.W_Int = self.W_Int.to(device)
        
        self.X_Int.requires_grad = True
        self.XBnd_Left.requires_grad = True
        self.XBnd_Right.requires_grad = True
        
        self.criterion = torch.nn.MSELoss()

        self.iter_diffusion = 0
        self.iter_physics = 0
        
        self.lbfgs_diffusion = torch.optim.LBFGS(
            self.model_diffusion.parameters(),
            lr=0.01,
            max_iter=1000,
            max_eval=1000,
            history_size=50,
            tolerance_grad=1e-7,
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe",
        )
        self.lbfgs_physics = torch.optim.LBFGS(
            self.model_physics.parameters(),
            lr=0.01,
            max_iter=1000,
            max_eval=1000,
            history_size=50,
            tolerance_grad=1e-7,
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe",
        )
        
        # Set Loss Weighting Function
        self.awl_diffusion = AutomaticWeightedLoss(4)
        self.awl_physics = AutomaticWeightedLoss(4)
        
        # Set Adam Optimizer
        self.optimizer_diffusion = torch.optim.Adam([
            {'params': self.model_diffusion.parameters()},
            {'params': self.awl_diffusion.parameters(), 'weight_decay': 0}
        ])
        self.optimizer_physics = torch.optim.Adam([
            {'params': self.model_physics.parameters()},
            {'params': self.awl_physics.parameters(), 'weight_decay': 0}
        ])
        
    # Plot Method
    def plot_training_points(self):
        plt.figure(figsize=(6, 5))

        plt.scatter(self.X_Int[:, 0], self.X_Int[:, 1],
                    s=5, c='C0', alpha=0.7, label='Interior Gauss Points')

        plt.scatter(self.XBnd_Left[:, 0], self.XBnd_Left[:, 1],
                    s=20, c='C1', marker='s', label='Left Boundary Points')

        plt.scatter(self.XBnd_Right[:, 0], self.XBnd_Right[:, 1],
                    s=20, c='C2', marker='s', label='Right Boundary Points')

        plt.scatter(self.XInit[:, 0], self.XInit[:, 1],
                    s=20, c='C3', marker='^', label='Initial Boundary Points')

        plt.xlabel('x')
        plt.ylabel('t')
        plt.title('Training Points Visualization')
        plt.legend()
        plt.tight_layout()
        plt.show()
        
    def loss_diffusion(self):
        self.optimizer_diffusion.zero_grad()
        # in domain
        c = self.model_diffusion(self.X_Int)[:, 0]
        dc = torch.autograd.grad(inputs=self.X_Int,
                                 outputs=c,
                                 grad_outputs=torch.ones_like(c),
                                 retain_graph=True,
                                 create_graph=True)[0]
        dc_dx = dc[:, 0]
        dc_dt = dc[:, 1]
        dc_dxx = torch.autograd.grad(inputs=self.X_Int,
                                       outputs=dc_dx,
                                       grad_outputs=torch.ones_like(dc_dx),
                                       retain_graph=True,
                                       create_graph=True)[0][:, 0]
        dis = self.model_physics(self.X_Int)[:, 0]  # material a displacement
        dis_dx = torch.autograd.grad(inputs=self.X_Int,
                                       outputs=dis,
                                       grad_outputs=torch.ones_like(dis),
                                       retain_graph=True,
                                       create_graph=True)[0][:, 0]
        dis_dxx = torch.autograd.grad(inputs=self.X_Int,
                                        outputs=dis_dx,
                                        grad_outputs=torch.ones_like(dis_dx),
                                        retain_graph=True,
                                        create_graph=True)[0][:, 0]
        dis_dxxx = torch.autograd.grad(inputs=self.X_Int,
                                         outputs=dis_dxx,
                                         grad_outputs=torch.ones_like(dis_dxx),
                                         retain_graph=True,
                                         create_graph=True)[0][:, 0]
        if self.cycle_number == 1:
            loss_diffusion_pde = self.criterion(self.X_Int[:,0] * dc_dt,self.X_Int[:,0] * dc_dxx + dc_dx)
        else:
            loss_diffusion_pde = self.criterion(self.X_Int[:,0]**3 * dc_dt + theta * dc_dx * (self.X_Int[:,0]**3 * dis_dxx + self.X_Int[:,0]**2 * dis_dx - self.X_Int[:,0] * dis - self.X_Int[:,0]**3 * alfa * dc_dx) + theta * c * (self.X_Int[:,0]**3 * dis_dxxx + 2*self.X_Int[:,0]**2 * dis_dxx - self.X_Int[:,0] * dis_dx + dis - self.X_Int[:,0]**3 * alfa * dc_dxx - self.X_Int[:,0]**2 * alfa * dc_dx),
                                             self.X_Int[:,0]**3 * dc_dxx + self.X_Int[:,0]**2 * dc_dx)
            
        # boundary condition
        c_bnd_left = self.model_diffusion(self.XBnd_Left)[:, 0]
        dc_dx_bnd_left = torch.autograd.grad(inputs=self.XBnd_Left,outputs=c_bnd_left,grad_outputs=torch.ones_like(c_bnd_left),retain_graph=True,create_graph=True)[0][:, 0]
        dis_bnd_left = self.model_physics(self.XBnd_Left)[:, 0]
        dis_dx_bnd_left = torch.autograd.grad(inputs=self.XBnd_Left,outputs=dis_bnd_left,grad_outputs=torch.ones_like(dis_bnd_left),retain_graph=True,create_graph=True)[0][:, 0]
        dis_dxx_bnd_left = torch.autograd.grad(inputs=self.XBnd_Left,
                                               outputs=dis_dx_bnd_left,
                                               grad_outputs=torch.ones_like(dis_dx_bnd_left),
                                               retain_graph=True,
                                               create_graph=True)[0][:, 0]
        if self.cycle_number == 1:
            loss_boundary_left = self.criterion(dc_dx_bnd_left, self.XBnd_Left_Val)
        else:
            loss_boundary_left = self.criterion(self.XBnd_Left[:,0]**2 * dc_dx_bnd_left,
                                            theta * c_bnd_left * (self.XBnd_Left[:,0]**2 * dis_dxx_bnd_left + self.XBnd_Left[:,0] * dis_dx_bnd_left - dis_bnd_left - alfa * self.XBnd_Left[:,0]**2 * dc_dx_bnd_left))
        
        # boundary condition
        c_bnd_right = self.model_diffusion(self.XBnd_Right)[:, 0]
        dc_dx_bnd_right = torch.autograd.grad(inputs=self.XBnd_Right,outputs=c_bnd_right,grad_outputs=torch.ones_like(c_bnd_right),retain_graph=True,create_graph=True)[0][:, 0]
        # boundary displacement predict
        dis_bnd_right = self.model_physics(self.XBnd_Right)[:, 0]
        dis_dx_bnd_right = torch.autograd.grad(inputs=self.XBnd_Right,outputs=dis_bnd_right,grad_outputs=torch.ones_like(dis_bnd_right),retain_graph=True,create_graph=True)[0][:, 0]
        dis_dxx_bnd_right = torch.autograd.grad(inputs=self.XBnd_Right,
                                                outputs=dis_dx_bnd_right,
                                                grad_outputs=torch.ones_like(dis_dx_bnd_right),
                                                retain_graph=True,
                                                create_graph=True)[0][:, 0]
        if self.cycle_number == 1:
            loss_boundary_right = self.criterion(dc_dx_bnd_right, self.XBnd_Right_Val)
        else:
            loss_boundary_right = self.criterion(self.XBnd_Right[:,0]**2 * dc_dx_bnd_right,
                                                theta * c_bnd_right * (self.XBnd_Right[:,0]**2 * dis_dxx_bnd_right + self.XBnd_Right[:,0] * dis_dx_bnd_right - dis_bnd_right - alfa * self.XBnd_Right[:,0]**2 * dc_dx_bnd_right) + self.XBnd_Right[:,0]**2)
        
        # Initial condition
        c_init = self.model_diffusion(self.XInit)[:, 0]
        loss_initial = self.criterion(c_init,self.XInit_Val)
        
        loss_diffusion = self.awl_diffusion(loss_diffusion_pde, loss_boundary_left, loss_boundary_right, loss_initial)
        loss_diffusion.backward()
        
        
        if self.iter_diffusion % 1000 == 0:
            print(f"\nEpoch: {self.iter_diffusion}, Loss: {loss_diffusion.item()}\n")
            print(list((self.awl_diffusion.parameters())))
        self.iter_diffusion = self.iter_diffusion + 1
        
        return loss_diffusion
    
    def loss_physics(self):
        self.optimizer_physics.zero_grad()
        # in domain
        c = self.model_diffusion(self.X_Int)[:, 0]
        dis = self.model_physics(self.X_Int)[:, 0]  # material a displacement
        dis_dx = torch.autograd.grad(inputs=self.X_Int,
                                       outputs=dis,
                                       grad_outputs=torch.ones_like(dis),
                                       retain_graph=True,
                                       create_graph=True)[0][:, 0]

        energy_density = 0.5 * (self.X_Int[:, 0] * dis_dx * lam_1 * (lam_2 * dis_dx * self.X_Int[:, 0] + lam_3 * dis) + dis * lam_1 * (lam_3 * self.X_Int[:, 0] * dis_dx + lam_2 * dis)) - self.X_Int[:, 0] * lam_1 * alfa * c / 3.0 * (self.X_Int[:, 0] * dis_dx + dis)
        weighted_energy_density = energy_density * self.W_Int
        stress_energy = torch.sum(weighted_energy_density)
        # stress_energy = self.criterion(self.X_Int[:,0] ** 2 * dis_dxx + self.X_Int[:,0] * dis_dx, dis + (1+nu)/(1-nu) * alfa * self.X_Int[:,0] **2 * dc_dx / 3)
        # boundary condition & initial condition
        c_bnd_left = self.model_diffusion(self.XBnd_Left)[:, 0]
        # boundary displacement predict
        dis_bnd_left = self.model_physics(self.XBnd_Left)[:, 0]
        dis_dx_bnd_left = torch.autograd.grad(inputs=self.XBnd_Left,outputs=dis_bnd_left,grad_outputs=torch.ones_like(dis_bnd_left),retain_graph=True,create_graph=True)[0][:, 0]
        loss_boundary_physics_left = self.criterion((1-nu)/(1+nu)*self.XBnd_Left[:,0]*dis_dx_bnd_left + nu/(1+nu)*dis_bnd_left, alfa*self.XBnd_Left[:,0]*c_bnd_left/3)

        c_bnd_right = self.model_diffusion(self.XBnd_Right)[:, 0]
        # boundary displacement predict
        dis_bnd_right = self.model_physics(self.XBnd_Right)[:, 0]
        dis_dx_bnd_right = torch.autograd.grad(inputs=self.XBnd_Right,outputs=dis_bnd_right,grad_outputs=torch.ones_like(dis_bnd_right),retain_graph=True,create_graph=True)[0][:, 0]
        loss_boundary_physics_right = self.criterion((1-nu)/(1+nu)*self.XBnd_Right[:,0]*dis_dx_bnd_right + nu/(1+nu)*dis_bnd_right, alfa*self.XBnd_Right[:,0]*c_bnd_right/3)

        dis_init = self.model_physics(self.XInit)[:, 0]
        loss_initial_physics = self.criterion(dis_init,self.XInit_Val)
        
        loss_physics = self.awl_physics(stress_energy, loss_boundary_physics_left, loss_boundary_physics_right, loss_initial_physics)
        loss_physics.backward()
        
        
        if self.iter_physics % 1000 == 0:
            print(f"\nEpoch: {self.iter_physics}, Loss: {loss_physics.item()}\n")
            print(list((self.awl_physics.parameters())))
        self.iter_physics = self.iter_physics + 1
        
        return loss_physics
    
    def train(self):
        self.model_diffusion.train()
        self.model_physics.train()
        total_diffusion_time = 0  
        total_physics_time = 0   
        total_training_time_start = time.time()  
        
        for j in range(10):
            print(f"\nCycle_Number:{self.cycle_number}\n")
            
            # Training Concentration Field
            diffusion_start_time = time.time()
            print("Using the Adam optimizer, train the concentration field")
            for i in range(1001):
                self.optimizer_diffusion.step(self.loss_diffusion)
            print("The L-BFGS optimizer was used to train the concentration field")
            for i in range(1):  
                print(f"L-BFGS Step: {i+1}")
                self.lbfgs_diffusion.step(self.loss_diffusion)
            diffusion_end_time = time.time()
            total_diffusion_time += diffusion_end_time - diffusion_start_time
            
            # Training displacement field
            physics_start_time = time.time()
            print("Training the displacement field using the Adam optimizer")
            for i in range(1001):
                self.optimizer_physics.step(self.loss_physics)
            print("L-BFGS optimizer is used to train the displacement field")
            for i in range(1): 
                print(f"L-BFGS Step: {i+1}")
                self.lbfgs_physics.step(self.loss_physics)
            physics_end_time = time.time()
            total_physics_time += physics_end_time - physics_start_time
            
            self.cycle_number = self.cycle_number + 1
            
        total_training_time_end = time.time() 
       
        print("\n--- Training Time Report ---")
        print(f"Total time for training concentration field (diffusion): {total_diffusion_time:.2f} seconds")
        print(f"Total time for training displacement field (physics): {total_physics_time:.2f} seconds")
        print(f"Total training time: {total_training_time_end - total_training_time_start:.2f} seconds")
        
Diffusion_PINN = Diffusion()
Diffusion_PINN.train()

torch.save(Diffusion_PINN.model_diffusion, 'model_diffusion.pth')
torch.save(Diffusion_PINN.model_physics, 'model_physics.pth')

device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")
model_diffusion_loaded = torch.load('model_diffusion.pth', map_location=device)
model_physics_loaded = torch.load('model_physics.pth', map_location=device)
model_diffusion_loaded.eval() 
model_physics_loaded.eval() 

domainCorners = np.array([[0.5,0], [0.5,0.5], [1.0,0], [1.0,0.5]])
domainGeom = Quadrilateral(domainCorners)

plot_predictions(model_diffusion_loaded, domainGeom, numPtsU=10, numPtsV=10)
plot_predictions(model_physics_loaded, domainGeom, numPtsU=10, numPtsV=10)

# L2 error norm

num_pts_x = 501
num_pts_t = 501
x_vals = np.linspace(0.5, 1.0, num_pts_x)
t_vals = np.linspace(0.0, 0.5, num_pts_t)
XX, TT = np.meshgrid(x_vals, t_vals)  # shape=(501,501)
Xtest_np = np.column_stack((XX.ravel(), TT.ravel()))  # (501*501, 2)
Xtest_torch = torch.tensor(Xtest_np, dtype=torch.float32, device=device)

with torch.no_grad():
    # output: c_pred, u_pred
    out_c = model_diffusion_loaded(Xtest_torch)  # shape=(501*501,1)
    out_u = model_physics_loaded(Xtest_torch)    # shape=(501*501,1)

# resize (501,501)
out_c_2d = out_c.view(num_pts_t, num_pts_x)  # (501,501)
out_u_2d = out_u.view(num_pts_t, num_pts_x)  # (501,501)

# to CPU
c_pred_2d_np = out_c_2d.detach().cpu().numpy()
u_pred_2d_np = out_u_2d.detach().cpu().numpy()

c_data = pd.read_csv('Comsol_solution/zhuc.csv', skiprows=8, header=None, names=['x', 'value'])
n_x = 501
c_data['t_index'] = c_data.index // n_x
c_data['t'] = c_data['t_index'] * 0.001

c_pivot = c_data.pivot(index='t', columns='x', values='value')
c_pivot = c_pivot.sort_index(ascending=True)  
c_pivot = c_pivot.reindex(sorted(c_pivot.columns), axis=1)  
c_ref_2d_np = c_pivot.values  # (501,501)

u_data = pd.read_csv('Comsol_solution/zhuu.csv', skiprows=8, header=None, names=['x', 'value'])
u_data['t_index'] = u_data.index // n_x
u_data['t'] = u_data['t_index'] * 0.001

u_pivot = u_data.pivot(index='t', columns='x', values='value')
u_pivot = u_pivot.sort_index(ascending=True)
u_pivot = u_pivot.reindex(sorted(u_pivot.columns), axis=1)
u_ref_2d_np = u_pivot.values  # (501,501)

diff_c = c_pred_2d_np - c_ref_2d_np
l2_error_c = np.sqrt(np.sum(diff_c**2)) / np.sqrt(np.sum(c_ref_2d_np**2))
print("Concentration field L2 error = {:.2e}".format(l2_error_c))

diff_u = u_pred_2d_np - u_ref_2d_np
l2_error_u = np.sqrt(np.sum(diff_u**2)) / np.sqrt(np.sum(u_ref_2d_np**2))
print("Strain field L2 error        = {:.2e}".format(l2_error_u))