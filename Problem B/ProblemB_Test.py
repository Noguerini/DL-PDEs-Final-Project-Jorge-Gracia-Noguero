import h5py
import pathlib
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import grad, Variable
import matplotlib.pyplot as plt

np.random.seed(67)
torch.manual_seed(67)

DEVICE        = torch.device('cuda')
DATA_TYPE     = torch.float32
LEARNING_RATE = 0.001
ADAM_EPOCHS   = 15000
LBFGS_EPOCHS  = 500
LBFGS_MAX_ITER= 20
LBFGS_LR      = 1.0
W_PDE         = 1.0
LEFT_BOUND    = 0.0
U_LEFT_BOUND  = 1.0
RIGHT_BOUND   = 1.0
U_RIGHT_BOUND = 0.0
N_POINTS_BOUND= 10000
N_POINTS_INT  = 40000
N_POINTS_LBFGS= 400000
LOWER_BOUND   = 0.0
UPPER_BOUND   = 1.0
MU_1          = 10.0
MU_2          = 2.0

DATA_PATH = pathlib.Path(__file__).parent / 'ProblemB_dataset.h5'

with h5py.File(DATA_PATH, 'r') as file:
    mu_field  = np.array(file['mu_field']).reshape(-1, 1)
    x_test    = np.array(file['x_test']).reshape(-1, 2)
    u_test    = np.array(file['u_test']).reshape(-1, 1)

mu_field = torch.tensor(mu_field, dtype=DATA_TYPE, device=DEVICE)

x_test = torch.tensor(x_test, dtype=DATA_TYPE, device=DEVICE)
u_test = torch.tensor(u_test, dtype=DATA_TYPE, device=DEVICE)

def fun_mu(x, mu=mu_field, resolution=128):
    '''The material property field (get values of material field on any given position x)
    Input:
        x: size(N, 2)
    '''
    mu = mu.reshape(1,-1).to(x)
    delta = 1./(resolution-1)
    #
    x_loc = torch.floor(x[...,0] / delta + 0.5).int()
    y_loc = torch.floor(x[...,1] / delta + 0.5).int()
    loc = y_loc * resolution + x_loc
    #
    mu_new = mu[torch.arange(mu.shape[0]).unsqueeze(1), loc]
    
    return mu_new.T

class MLP(nn.Module):
    
    def __init__(self, layers_list):
        super(MLP, self).__init__()
        self.activation = nn.SiLU()
        net = []
        self.hidden_in = layers_list[0]
        for hidden in layers_list[1:]:
            net.append(nn.Linear(self.hidden_in, hidden, dtype=DATA_TYPE))
            self.hidden_in = hidden
        self.net = nn.Sequential(*net)
    
    def forward(self, x):
        h = x
        for net in self.net[:-1]:
            h = net(h)
            h = self.activation(h)
        raw = self.net[-1](h)

        xn = (x[:, 0:1] - LEFT_BOUND) / (RIGHT_BOUND - LEFT_BOUND)
        u = (1.0 - xn) * U_LEFT_BOUND + xn * U_RIGHT_BOUND + xn * (1.0 - xn) * raw

        return u

u_model = MLP([2, 64, 64, 64, 64, 64, 1]).to(DEVICE)

x_left = np.array(LEFT_BOUND).repeat(N_POINTS_BOUND, axis=0)
y_left = np.linspace(LOWER_BOUND, UPPER_BOUND, N_POINTS_BOUND).flatten()
xy_left = np.vstack([x_left, y_left]).T

x_right = np.array(RIGHT_BOUND).repeat(N_POINTS_BOUND, axis=0)
y_right = np.linspace(LOWER_BOUND, UPPER_BOUND, N_POINTS_BOUND).flatten()
xy_right = np.vstack([x_right, y_right]).T

xy_bd = np.vstack([xy_left, xy_right])
xy_bd = torch.tensor(xy_bd, dtype=DATA_TYPE, device=DEVICE)

u_bd = 1.0 - xy_bd[:, 0:1]

class LossClass(object):

    def __init__(self, u_model):
        self.device = DEVICE
        self.u_model = u_model 
        self.getLoss = torch.nn.MSELoss()
        
    def loss_pde(self, x_int=None):
        if x_int is None:
            x_int = np.random.uniform([LEFT_BOUND, LOWER_BOUND], [RIGHT_BOUND, UPPER_BOUND], (N_POINTS_INT, 2))
            x_int = torch.tensor(x_int, dtype=DATA_TYPE)

        xy = Variable(x_int, requires_grad=True).to(self.device)
        u = self.u_model(xy)
        
        du_dx = grad(inputs=xy, outputs=u, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        mu_x = fun_mu(xy.detach())

        energy = 0.5 * (mu_x * du_dx.pow(2).sum(dim=1, keepdim=True)).mean()
        
        return energy
    
    def get_error(self, x_test, u_test):
        x_test = x_test.to(self.device)
        u_test = u_test.to(self.device)
        u = self.u_model(x_test)
        
        return torch.sqrt(torch.sum((u-u_test)**2)/torch.sum(u_test**2))
    
OPTIMIZER = torch.optim.Adam(params=u_model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
SCHEDULER = torch.optim.lr_scheduler.StepLR(OPTIMIZER, step_size=np.int32(ADAM_EPOCHS/10), gamma=0.8)

lossClass = LossClass(u_model)

error_list = []

# Stage 1: Adam
for epoch in range(ADAM_EPOCHS):
    loss_in = lossClass.loss_pde(x_int=None)
    loss_train = W_PDE*loss_in

    OPTIMIZER.zero_grad()
    loss_train.backward()
    OPTIMIZER.step()

    with torch.no_grad():
        error = lossClass.get_error(x_test, u_test)
        error_list.append(error.item())

    SCHEDULER.step()
    if (epoch+1)%500==0:
        print(f'[Adam] Epoch:{epoch}, The loss is:{loss_train.item()}, The L^2 Relative Error is: {error_list[-1]}, lr: {SCHEDULER.optimizer.param_groups[0]["lr"]}')

# Stage 2: L-BFGS
x_int_lbfgs = np.random.uniform([LEFT_BOUND, LOWER_BOUND], [RIGHT_BOUND, UPPER_BOUND], (N_POINTS_LBFGS, 2))
x_int_lbfgs = torch.tensor(x_int_lbfgs, dtype=DATA_TYPE)

LBFGS_OPTIMIZER = torch.optim.LBFGS(
    u_model.parameters(),
    lr=LBFGS_LR,
    max_iter=LBFGS_MAX_ITER,
    history_size=50,
    line_search_fn='strong_wolfe',
)

last_loss = {}

def closure():
    LBFGS_OPTIMIZER.zero_grad()
    loss_in = lossClass.loss_pde(x_int=x_int_lbfgs)
    loss_train = W_PDE*loss_in
    loss_train.backward()
    last_loss['value'] = loss_train.item()
    return loss_train

for epoch in range(LBFGS_EPOCHS):
    LBFGS_OPTIMIZER.step(closure)

    with torch.no_grad():
        error = lossClass.get_error(x_test, u_test)
        error_list.append(error.item())

    if (epoch+1)%50==0:
        print(f'[LBFGS] Epoch:{ADAM_EPOCHS+epoch}, The loss is:{last_loss["value"]}, The L^2 Relative Error is: {error_list[-1]}')

EPOCHS = ADAM_EPOCHS + LBFGS_EPOCHS

plt.figure(figsize=(6, 4))
plt.semilogy(range(EPOCHS), error_list)
plt.axvline(ADAM_EPOCHS, color='k', linestyle='--', linewidth=1, label='Adam -> L-BFGS')
plt.xlabel('Epoch')
plt.ylabel('Relative L2 error')
plt.title('Test error vs. training epoch (Adam + L-BFGS)')
plt.legend()
plt.tight_layout()
plt.show()

X = x_test[:, 0].reshape(128, 128).cpu()
Y = x_test[:, 1].reshape(128, 128).cpu()
u_test_grid = u_test.reshape(128, 128).cpu()

x_te      = x_test.to(DEVICE)
u_te      = u_test.reshape(-1, 1).to(DEVICE)
u_te_norm = torch.linalg.norm(u_te)

final_net = u_model

with torch.no_grad():
    U_pred = final_net(x_te).reshape(128, 128).cpu()

plt.figure(figsize=(5.8, 4.6))
cntr = plt.contourf(X, Y, U_pred, levels=50)
plt.colorbar(cntr); plt.xlabel('x'); plt.ylabel('y')
plt.title(r'Predicted pressure field $u_\mathrm{pred}(x)$ (Deep Ritz)')
plt.tight_layout(); plt.show()

plt.figure(figsize=(5.8, 4.6))
cntr = plt.contourf(X, Y, u_test_grid, levels=50)
plt.colorbar(cntr); plt.xlabel('x'); plt.ylabel('y')
plt.title(r'Reference pressure field $u_\mathrm{true}(x)$')
plt.tight_layout(); plt.show()

plt.figure(figsize=(5.8, 4.6))
cntr = plt.imshow(torch.abs(U_pred - u_test_grid), origin='lower', extent=(0, 1, 0, 1), cmap='inferno')
plt.colorbar(cntr); plt.xlabel('x'); plt.ylabel('y')
plt.title(r'Pointwise absolute error $|u_\mathrm{pred} - u_\mathrm{true}|$')
plt.tight_layout(); plt.show()