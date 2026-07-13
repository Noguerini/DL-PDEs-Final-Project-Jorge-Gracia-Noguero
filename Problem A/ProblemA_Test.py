import h5py
import pathlib
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import grad, Variable
import matplotlib.pyplot as plt

np.random.seed(67)
torch.manual_seed(67)

DEVICE = torch.device('cuda')
DATA_TYPE = torch.float32
LEARNING_RATE = 0.001
EPOCHS = 20000
W_PDE = 1.0
W_OBS = 1.0

DATA_PATH = pathlib.Path(__file__).parent / 'ProblemA_dataset.h5'

with h5py.File(DATA_PATH, 'r') as file:
    x_obs  = np.array(file['x_obs']).reshape(-1, 1)
    u_obs  = np.array(file['u_obs']).reshape(-1, 1)
    x_test = np.array(file['x_test']).reshape(-1, 1)
    k_test = np.array(file['k_test']).reshape(-1, 1)
    u_test = np.array(file['u_test']).reshape(-1, 1)

x_obs = torch.tensor(x_obs, dtype=DATA_TYPE, device=DEVICE)
u_obs = torch.tensor(u_obs, dtype=DATA_TYPE, device=DEVICE)

x_test = torch.tensor(x_test, dtype=DATA_TYPE, device=DEVICE)
k_test = torch.tensor(k_test, dtype=DATA_TYPE, device=DEVICE)
u_test = torch.tensor(u_test, dtype=DATA_TYPE, device=DEVICE)

class MLP(nn.Module):
    
    def __init__(self, layers_list:list, dtype=None):
        super(MLP, self).__init__()
        self.activation = nn.Tanh()
        net = []
        self.hidden_in = layers_list[0]
        for hidden in layers_list[1:]:
            net.append(nn.Linear(self.hidden_in, hidden, dtype=dtype))
            self.hidden_in = hidden
        self.net = nn.Sequential(*net)
    
    def forward(self, x):
        for net in self.net[:-1]:
            x = net(x)
            x = self.activation(x)
        x = self.net[-1](x)

        return x

u_model = MLP([1, 80, 80, 1], dtype=torch.float32).to(DEVICE)
k_model = MLP([1, 80, 80, 1], dtype=torch.float32).to(DEVICE)


leftBound = 0.0
rightBound = 1.0
n_sensors = 500

class LossClass(object):

    def __init__(self, u_model, k_model, f=9.81, L=1.0):
        self.device = DEVICE
        self.u_model = u_model
        self.k_model = k_model
        self.f = f
        self.L = L
        self.getLoss = torch.nn.MSELoss()

    # Enforce positive value of K
    def k(self, x):
        return nn.functional.softplus(self.k_model(x)) + 0.1

    # Enforce boundary conditions
    def u(self, x):
        return x * (self.L - x) * self.u_model(x)

    def loss_obs(self, x_obs, u_obs):
        x_obs = x_obs.to(self.device)
        u = self.u(x_obs)
        loss = self.getLoss(u, u_obs.to(self.device))

        return loss

    def loss_pde(self, x_in):
        x = Variable(x_in, requires_grad=True).to(self.device)
        u = self.u(x)
        k = self.k(x)

        du_dx = grad(inputs=x, outputs=u, grad_outputs=torch.ones_like(u), create_graph=True)[0]

        s = k * du_dx
        ds_dx = grad(inputs=x, outputs=s, grad_outputs=torch.ones_like(s), create_graph=True)[0]

        # Normalize by f so the residual is O(1)
        residual = (- ds_dx - self.f) / self.f

        loss = self.getLoss(residual, torch.zeros_like(residual))
        return loss

    def get_error_k(self, x_test, k_test):
        x_test, k_test = x_test.to(self.device), k_test.to(self.device)
        k = self.k(x_test)
        return torch.sqrt(torch.sum((k - k_test)**2) / torch.sum(k_test**2))

    def get_error_u(self, x_test, u_test):
        x_test, u_test = x_test.to(self.device), u_test.to(self.device)
        u = self.u(x_test)
        return torch.sqrt(torch.sum((u - u_test)**2) / torch.sum(u_test**2))
    
optimizer_u = torch.optim.Adam(params=u_model.parameters(), lr=LEARNING_RATE)
optimizer_k = torch.optim.Adam(params=k_model.parameters(), lr=LEARNING_RATE)

scheduler_u = torch.optim.lr_scheduler.StepLR(optimizer_u, step_size=int(EPOCHS/4), gamma=0.5)
scheduler_k = torch.optim.lr_scheduler.StepLR(optimizer_k, step_size=int(EPOCHS/4), gamma=0.5)

lossClass = LossClass(u_model, k_model)

error_k_list = []
error_u_list = []

for epoch in range(EPOCHS):

    x_col = torch.rand(n_sensors, 1, device=DEVICE)

    loss_obs = lossClass.loss_obs(x_obs, u_obs)
    loss_pde = lossClass.loss_pde(x_col)
    loss_train = W_OBS*loss_obs + W_PDE*loss_pde

    optimizer_k.zero_grad()
    optimizer_u.zero_grad()

    loss_train.backward()

    optimizer_k.step()
    optimizer_u.step()

    with torch.no_grad():
        error_k = lossClass.get_error_k(x_test, k_test)
        error_k_list.append(error_k.item())

        error_u = lossClass.get_error_u(x_test, u_test)
        error_u_list.append(error_u.item())

    scheduler_k.step()
    scheduler_u.step()

    if (epoch+1)%500==0:
        print(f'Epoch:{epoch}, The loss is:{loss_train.item()}, The current L^2 error for k is: {error_k_list[-1]}')
            
k_model.eval()
u_model.eval()

with torch.no_grad():
    k_pred = lossClass.k(x_test).cpu().numpy()
    u_pred = lossClass.u(x_test).cpu().numpy()

print(f"The L_2 error of K is: {error_k_list[len(error_k_list) - 1]}")
print(f"The L_2 error of u is: {error_u_list[len(error_u_list) - 1]}")

plt.figure()
plt.semilogy(range(EPOCHS), error_k_list, label='$k$ rel. $L^2$ error')
plt.semilogy(range(EPOCHS), error_u_list, label='$u$ rel. $L^2$ error')
plt.xlabel('epoch'); plt.ylabel('relative $L^2$ error')
plt.title('Error vs. epoch'); plt.legend(); plt.show()

x_test_np = x_test.cpu().numpy()
k_test_np = k_test.cpu().numpy()

plt.figure()
plt.plot(x_test_np, k_test_np, 'k-',  label='$k_{true}$')
plt.plot(x_test_np, k_pred, 'r--', label='$k_{pred}$')
plt.xlabel('x'); plt.ylabel('k(x)'); plt.title("Young's modulus $k(x)$")
plt.legend(); plt.show()

plt.figure()
plt.plot(x_test_np, np.abs(k_pred - k_test_np))
plt.xlabel('x'); plt.ylabel('$|k_{pred}-k_{true}|$')
plt.title('Pointwise absolute error of $k(x)$'); plt.show()