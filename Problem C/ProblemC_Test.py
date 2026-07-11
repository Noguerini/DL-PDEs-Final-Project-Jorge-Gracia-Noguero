import h5py
import pathlib
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import grad, Variable
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

np.random.seed(67)
torch.manual_seed(67)

DEVICE        = torch.device('cuda')
DATA_TYPE     = torch.float32
LEARNING_RATE = 0.001
EPOCHS        = 50
MODES_1       = 16
MODES_2       = 16
HIDDEN_LIST   = [40, 40, 40]

DATA_PATH = pathlib.Path(__file__).parent / 'ProblemC_dataset.h5'

with h5py.File(DATA_PATH, 'r') as file:
    a_train = np.array(file['a_train'])
    u_train = np.array(file['u_train'])
    a_test  = np.array(file['a_test'])
    u_test  = np.array(file['u_test'])
    X       = np.array(file['X'])
    Y       = np.array(file['Y'])

a_train = torch.tensor(a_train, dtype=DATA_TYPE, device=DEVICE)
u_train = torch.tensor(u_train, dtype=DATA_TYPE, device=DEVICE)
a_test  = torch.tensor(a_test, dtype=DATA_TYPE, device=DEVICE)
u_test  = torch.tensor(u_test, dtype=DATA_TYPE, device=DEVICE)
X       = torch.tensor(X, dtype=DATA_TYPE, device=DEVICE)
Y       = torch.tensor(Y, dtype=DATA_TYPE, device=DEVICE)

class SpectralConvolution(nn.Module):
    
    def __init__(self, in_size, out_size, modes1, modes2):
        super(SpectralConvolution, self).__init__()
        self.in_size = in_size 
        self.out_size = out_size 
        self.modes1 = modes1
        self.modes2 = modes2
        self.scale = 1./(in_size * out_size)
        ctype = torch.complex64
        self.weight1 = nn.Parameter(self.scale * torch.rand(in_size, out_size, self.modes1, self.modes2, dtype=ctype))
        self.weight2 = nn.Parameter(self.scale * torch.rand(in_size, out_size, self.modes1, self.modes2, dtype=ctype))

    def compl_mul_2d(self, input, weight):
        return torch.einsum("bixy, ioxy->boxy", input, weight)

    def forward(self, x):
        batch_size = x.shape[0]
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(batch_size, self.out_size, x.size(-2), x.size(-1)//2+1, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul_2d(x_ft[:, :, :self.modes1, :self.modes2], self.weight1)
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul_2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weight2)
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))

        return x
        
class FNO2D(nn.Module):
    def __init__(self, in_size, out_size, modes1, modes2, hidden_list:list[int], dtype=None):
        super(FNO2D, self).__init__()
        self.hidden_list = hidden_list
        self.activation = nn.ReLU()

        self.fc_in = nn.Linear(in_size, hidden_list[0], dtype=dtype)
        
        conv_net, w_net = [], []
        self.hidden_in = hidden_list[0]
        for hidden in hidden_list[1:]:
            conv_net.append(SpectralConvolution(self.hidden_in, hidden, modes1, modes2))
            w_net.append(nn.Conv2d(self.hidden_in, hidden, kernel_size=1, dtype=dtype))
            self.hidden_in = hidden 
            
        self.spectral_conv = nn.ModuleList(conv_net)
        self.weight_conv = nn.ModuleList(w_net)
        
        self.fc_out0 = nn.Linear(self.hidden_in, 128, dtype=dtype)
        self.fc_out1 = nn.Linear(128, out_size, dtype=dtype)
    
    def forward(self, ax):
        # Enforce boundary condition
        x_coord, y_coord = ax[..., 1], ax[..., 2]
        d = (x_coord * (1 - x_coord) * y_coord * (1 - y_coord)).unsqueeze(-1)
        
        ax = self.fc_in(ax)
        #(b, nx, ny, hidden) -> (b, hidden, nx, ny)
        ax = ax.permute(0, 3, 1, 2)
        
        for conv, weight in zip(self.spectral_conv, self.weight_conv):
            ax1 = conv(ax)
            ax2 = weight(ax)
            ax = self.activation(ax1 + ax2)
            
        # (b, hidden, nx, ny) -> (b, nx, ny, hidden)
        ax = ax.permute(0, 2, 3, 1)
        
        # Output projections
        ax = self.fc_out0(ax)
        ax = self.activation(ax)


        return d * self.fc_out1(ax)

class LossClass(object):

    def __init__(self, u_model):
        self.device = DEVICE
        self.u_model = u_model 
    
    def loss_data(self, ax_batch, u_batch):
        batch_size = u_batch.shape[0]
        ax, u = ax_batch.to(self.device), u_batch.to(self.device)
        u_pred = self.u_model(ax)
        loss = torch.norm(u.reshape(batch_size, -1)-u_pred.reshape(batch_size, -1), 2, 1)
        loss = torch.mean(loss)
        
        return loss 

    def get_error(self, ax, u):
        batch_size = u.shape[0]
        ax, u = ax.to(self.device), u.to(self.device)

        u_pred = self.u_model(ax)
        error = torch.norm(u.reshape(batch_size,-1)-u_pred.reshape(batch_size,-1), 2, 1) / torch.norm(u.reshape(batch_size,-1), 2, 1)

        return torch.mean(error)
    
class MyDataset(Dataset):

    def __init__(self, ax, uT):
        self.ax = ax
        self.uT = uT
    
    def __getitem__(self, index):
        return self.ax[index], self.uT[index]

    def __len__(self):
        return self.ax.shape[0]

a_train = a_train.unsqueeze(-1)
u_train = u_train.unsqueeze(-1)

a_test = a_test.unsqueeze(-1)
u_test = u_test.unsqueeze(-1)

batch_size = a_train.shape[0]
test_size = a_test.shape[0]

X_ext = X.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, -1, -1)
Y_ext = Y.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, -1, -1)

X_test = X.unsqueeze(0).unsqueeze(-1).expand(test_size, -1, -1, -1)
Y_test = Y.unsqueeze(0).unsqueeze(-1).expand(test_size, -1, -1, -1)

ax_train = torch.cat([a_train, X_ext, Y_ext], dim=-1)
ax_test = torch.cat([a_test, X_test, Y_test], dim=-1)

in_channels = ax_train.shape[-1]  # (a, X, Y)
out_channels = u_train.shape[-1]  # (u)
u_model = FNO2D(in_channels, out_channels, MODES_1, MODES_2, HIDDEN_LIST, dtype=DATA_TYPE).to(DEVICE)

loader = DataLoader(MyDataset(ax_train, u_train), batch_size=50, shuffle=True)

OPTIMIZER = torch.optim.Adam(params=u_model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
SCHEDULER = torch.optim.lr_scheduler.StepLR(OPTIMIZER, step_size=int(EPOCHS/4), gamma=0.5)

lossClass = LossClass(u_model)

loss_list, error_list = [], []
for epoch in range(EPOCHS):
    loss = 0.
    for ax_batch, u_batch in loader:
        loss_train = lossClass.loss_data(ax_batch, u_batch)
        
        OPTIMIZER.zero_grad()
        loss_train.backward()
        OPTIMIZER.step()
        
        loss += loss_train

    SCHEDULER.step()
    with torch.no_grad():
        error = lossClass.get_error(ax_test, u_test)
        error_list.append(error.item())

    loss = loss/len(loader)
    loss_list.append(loss.item())
    
    print(f'Epoch:{epoch}, The loss is:{loss.item()}, Current L^2 Error: {error_list[-1]}')

u_model.eval()
with torch.no_grad():
    u_pred0 = u_model(ax_test[:1].to(DEVICE)).squeeze(-1).cpu()[0]
a0 = a_test.squeeze(-1).cpu()[0]
u_true0 = u_test.squeeze(-1).cpu()[0]
abs_err0 = (u_pred0 - u_true0).abs()

X_np = X.cpu().numpy()
Y_np = Y.cpu().numpy()

plt.figure(figsize=(7, 5))
plt.semilogy(range(1, EPOCHS + 1), error_list, label='Test $L^2$ relative error')
plt.xlabel('Epoch')
plt.ylabel('$L^2$ relative error')
plt.title('Error vs. Epoch')
plt.grid(True, which='both', alpha=0.3)
plt.legend()
plt.show()

# Input conductivity field a(x,y)
plt.figure(figsize=(6, 5))
im = plt.imshow(a0, origin='lower', extent=(0, 1, 0, 1))
plt.colorbar(im)
plt.title('Input conductivity $a(x,y)$ (first test instance)')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# Predicted temperature field
plt.figure(figsize=(6, 5))
cs = plt.contourf(X_np, Y_np, u_pred0, levels=40, cmap='jet')
plt.colorbar(cs)
plt.title('Predicted temperature $u_{\\mathrm{pred}}(x,y)$')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# Ground truth temperature field
plt.figure(figsize=(6, 5))
cs = plt.contourf(X_np, Y_np, u_true0, levels=40, cmap='jet')
plt.colorbar(cs)
plt.title('Ground truth temperature $u_{\\mathrm{true}}(x,y)$')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# Pointwise absolute error
plt.figure(figsize=(6, 5))
cs = plt.contourf(X_np, Y_np, abs_err0, levels=40, cmap='jet')
plt.colorbar(cs)
plt.title('Pointwise absolute error $|u_{\\mathrm{pred}} - u_{\\mathrm{true}}|$')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

print(f'Final test L2 relative error after {EPOCHS} epochs: 'f'{error_list[-1]:.4f} ({error_list[-1]*100:.2f}%)')