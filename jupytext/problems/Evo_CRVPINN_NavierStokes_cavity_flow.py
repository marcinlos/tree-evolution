# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% id="qkchBv-we6tx" editable=true slideshow={"slide_type": ""}
import torch
import torch.nn as nn
from typing import Tuple, List, Callable
import matplotlib.pyplot as plt
from functools import partial
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np

from tree_evolution.io import plot_activations, store, load
from tree_evolution.nn import decode_activations

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE

# %% [markdown] id="3K61aHlHrDj5"
# # Parameters

# %% id="_IghnIFdzHfP" editable=true slideshow={"slide_type": ""} tags=["parameters"]
# >>>CHANGE<<<
# This cell must have "parameters" tag
# See https://papermill.readthedocs.io/en/latest/usage-parameterize.html

LEARNING_RATE = 0.005
EPOCHS = 200_000
LAYERS = 4
NEURONS_PER_LAYER = 200
X_POINTS = 50
Y_POINTS = 50
X_PLOT = 100
Y_PLOT = 100
RPINN = 1
N = X_POINTS # we assume X_POINTS and Y_POINTS to be equal
h=1/(N-1)
VELOCITY = 100

PLOT_ACTIVATIONS_EVERY = 1000 # how many epochs between plotting activation functions

ACTIVATIONS = None
OUTPUT_PATH = None
OPERATORS = None

# %% [markdown] id="VGy8iK0irGK8"
# # PINN and loss function definitions

# %% editable=true slideshow={"slide_type": ""}
# >>>CHANGE<<<
# Parse the injected description of the activation functions
ops = load(OPERATORS)
ACTIVATION_MODULES = decode_activations(ACTIVATIONS, ops)

# %% editable=true slideshow={"slide_type": ""}
# >>>CHANGE<<<
# Plot the initial state of the activation functions
plot_activations(ACTIVATION_MODULES)


# %% id="RTz1VMu0fTP-" editable=true slideshow={"slide_type": ""}
class PINN(nn.Module):
    def __init__(self, num_hidden: int = 3, dim_hidden: int = 100, middle_layers = None, act=nn.Tanh(), hard_constraint: Callable = None):

        super().__init__()

        self.hard_constraint = hard_constraint
        self.layer_in = nn.Linear(2, dim_hidden)
        self.layer_out = nn.Linear(dim_hidden, 1)

        self.layers = nn.Sequential(self.layer_in, *middle_layers, self.layer_out)


    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_stack = torch.cat([x, y], dim=1)
        logits = self.layers(x_stack)

        if self.hard_constraint:
            logits = self.hard_constraint(logits, x , y)

        return logits

    def get_device(self) -> torch.device:
        return next(self.parameters()).device

class MultiPINN():
    def __init__(self,  num_hidden: int, dim_hidden: int, pinns: List[PINN], act=nn.Tanh()):
        self.pinns = pinns

    def train(self, loss_function, epochs, lr):
        optimizers = [torch.optim.Adamax(pinn.parameters(), lr=lr) for pinn in self.pinns]
        loss_per_epoch = []
        for epoch in range(epochs):
            loss = loss_function(self.pinns)
            loss.backward()
            for i in range(len(self.pinns)):
                optimizers[i].step()
                optimizers[i].zero_grad()
            loss_per_epoch.append(loss.item())

            if epoch % 100 == 0:
                print(f"Epoch {epoch} \t with loss = {loss.item()}")

            # >>>CHANGE<<<
            # Plot activation functions to observe their changes during the training
            if (epoch + 1) % PLOT_ACTIVATIONS_EVERY == 0:
                plot_activations(ACTIVATION_MODULES)

            if np.isnan(loss.item()):
                break

        return loss_per_epoch

    def to(self, device: torch.device):
        self.pinns = [pinn.to(device) for pinn in self.pinns]
        return self

def f(pinn: PINN, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute the value of the approximate solution from the NN model"""
    return pinn(x, y)

def df(output: torch.Tensor, input: torch.Tensor, order: int = 1) -> torch.Tensor:
    """Compute neural network derivative with respect to input features using PyTorch autograd engine"""
    df_value = output
    for _ in range(order):
        df_value = torch.autograd.grad(
            df_value,
            input,
            grad_outputs=torch.ones_like(df_value),
            create_graph=True,
            retain_graph=True,
        )[0]

    return df_value

def dfdx(pinn: PINN, x: torch.Tensor, y: torch.Tensor, order: int = 1) -> torch.Tensor:
    """Derivative with respect to the spatial variable of arbitrary order"""
    f_value = f(pinn, x, y)
    return df(f_value, x, order=order)

def dfdy(pinn: PINN, x: torch.Tensor, y: torch.Tensor, order: int = 1) -> torch.Tensor:
    """Derivative with respect to the spatial variable of arbitrary order"""
    f_value = f(pinn, x, y)
    return df(f_value, y, order=order)


# %% id="C2o3kc6MK5lk"
if RPINN==1:
  def ff( i,j , n , m ):
    if i==n and j==m : return 1.0
    return 0.
  def dxp( foo , i , j , n , m  ):
      if i+1 > N :
          return 0.
      return ( foo(  i+1 , j , n , m) - foo(  i , j , n , m) )/h
  def dxm( foo , i , j , n , m  ):
      if i-1 <1 :
          return 0
      return ( foo(  i , j , n , m) - foo(  i-1 , j , n , m) )/h
  def dyp( foo , i , j , n , m  ):
      if j+1 > N :
          return 0.
      return ( foo(  i , j+1 , n , m) - foo(  i , j , n , m) )/h
  def dym( foo , i , j , n , m  ):
      if j-1 < 1 :
          return 0.
      return ( foo(  i , j , n , m) - foo(  i , j-1 , n , m) )/h
  M = torch.zeros([N*N,N*N]).to(DEVICE)
  AX = torch.zeros([N*N,N*N]).to(DEVICE)
  AX_m = torch.zeros([N*N,N*N]).to(DEVICE)
  AY = torch.zeros([N*N,N*N]).to(DEVICE)
  AY_m = torch.zeros([N*N,N*N]).to(DEVICE)
  K = torch.zeros([N*N,N*N]).to(DEVICE)
  KX = torch.zeros([N*N,N*N]).to(DEVICE)
  KX_m = torch.zeros([N*N,N*N]).to(DEVICE)
  KY = torch.zeros([N*N,N*N]).to(DEVICE)
  KY_m = torch.zeros([N*N,N*N]).to(DEVICE)
  S = torch.zeros([N*N,N*N]).to(DEVICE)
  S_m = torch.zeros([N*N,N*N]).to(DEVICE)
  lst = [(a, b) for a in range(1,N+1) for b in range(1,N+1)]
  for n,m in lst :
      lst2 = [ (n+a,m+b) for a in [-1,0,1] for b in [-1,0,1]  if n+a>0 and n+a<=N and m+b>0 and m+b<=N ]
      for k,l in lst2 :
          y,x = m+(n-1)*N-1,l+(k-1)*N-1
          for i,j in lst2:
              M[x,y] += ff(i,j,n,m)*ff(i,j,k,l)
              AX[x,y] += dxp( ff ,i,j,n,m)*ff ( i,j,k,l )
              AX_m[x,y] += dxm( ff ,i,j,n,m)*ff ( i,j,k,l )
              AY[x,y] += dyp( ff ,i,j,n,m)*ff ( i,j,k,l )
              AY_m[x,y] += dym( ff ,i,j,n,m)*ff ( i,j,k,l )
              KX[x,y] += dxp( ff ,i,j,n,m)*dxp( ff , i,j,k,l )
              KX_m[x,y] += dxm( ff ,i,j,n,m)*dxm( ff , i,j,k,l )
              KY[x,y] += dyp( ff ,i,j,n,m)*dyp( ff , i,j,k,l )
              KY_m[x,y] += dym( ff ,i,j,n,m)*dym( ff , i,j,k,l )
              S[x,y] += dxp( ff ,i,j,n,m)* dyp( ff , i,j,k,l )
              S_m[x,y] += dxm( ff ,i,j,n,m)* dym( ff , i,j,k,l )
          M[x,y] *= h*h
          AX[x,y] *= h*h
          AX_m[x,y] *= h*h
          AY[x,y] *= h*h
          AY_m[x,y] *= h*h
          KX[x,y] *= h*h
          KX_m[x,y] *= h*h
          KY[x,y] *= h*h
          KY_m[x,y] *= h*h
          S[x,y] *= h*h
          S_m[x,y] *= h*h
  K=KX+KY

  ZERO = torch.zeros( [N*N,N*N] ).to(DEVICE)

  G_sigma = torch.cat( (
  torch.cat( ( KX+2*M , S.t() , ZERO , ZERO ), dim=1),
  torch.cat(( S , KY +2*M, ZERO , ZERO ), dim=1),
  torch.cat(( ZERO , ZERO , KX  +2*M, S.t() ), dim=1),
  torch.cat(( ZERO , ZERO , S , KY +2*M ), dim=1) ), dim=0).to(DEVICE)

  G_u = torch.cat((
      torch.cat( (2*KX_m + KY_m + M , S_m.t() ), dim=1),
      torch.cat(( S_m , 2*KY_m + KX_m + M), dim=1),
  ), dim=0).to(DEVICE)


  G_sigma_u = torch.cat((
      torch.cat( ( AX_m , ZERO ), dim=1),
      torch.cat(( AY_m , ZERO ), dim=1),
      torch.cat(( ZERO , AX_m ), dim=1),
      torch.cat(( ZERO , AY_m ), dim=1),
  ), dim=0).to(DEVICE)

  G_sigma_p = (-1)* torch.cat( ( KX , S , S.t() , KY ), dim=0).to(DEVICE)

  G_p = K+M

  ZERO_2 = torch.zeros( [N*N*2,N*N] ).to(DEVICE)
  G = torch.cat( (
      torch.cat( ( G_sigma , G_sigma_u , G_sigma_p ), dim=1),
      torch.cat( ( G_sigma_u.t() , G_u , ZERO_2 ), dim=1),
      torch.cat( ( G_sigma_p.t() , ZERO_2.t() , G_p ), dim=1),
  ), dim=0).to(DEVICE)

# %% id="eQV56ZcwMgOQ"
if RPINN==1 :
    A_sigma_tau = torch.cat( (
      torch.cat( ( M    , ZERO , ZERO , ZERO ), dim=1),
      torch.cat( ( ZERO ,  M   , ZERO , ZERO ), dim=1),
      torch.cat( ( ZERO , ZERO , M    , ZERO ), dim=1),
      torch.cat( ( ZERO , ZERO , ZERO , M ), dim=1)
    ), dim=0).to(DEVICE)

    A_u_tau = (-1)* torch.cat( (
      torch.cat( ( AX_m , ZERO ), dim=1),
      torch.cat( ( AY_m , ZERO ), dim=1),
      torch.cat( ( ZERO , AX_m ), dim=1),
      torch.cat( ( ZERO , AY_m ), dim=1)
    ), dim=0).to(DEVICE)

    A_p_v = torch.cat( (
        AX,
        AY
    ), dim=0 ).to(DEVICE)

    A_u_q = torch.cat( ( AX_m , AY_m ), dim=1 ).to(DEVICE)

    A_v_sigma = (-1)* torch.cat( (
        torch.cat( ( AX , AY , ZERO , ZERO ), dim=1),
        torch.cat( ( ZERO , ZERO , AX , AY ), dim=1)
    ), dim=0 ).to(DEVICE)

    ZERO_4x1 = torch.zeros( [N*N*4,N*N] ).to(DEVICE)
    ZERO_2x2 = torch.zeros( [N*N*2,N*N*2] ).to(DEVICE)

    A = torch.cat( (
        torch.cat( ( A_sigma_tau , A_u_tau , ZERO_4x1 ), dim=1),
        torch.cat( ( A_v_sigma , ZERO_2x2 , A_p_v ), dim=1),
        torch.cat( ( ZERO_4x1.t() , A_u_q , ZERO ), dim=1)
    ), dim=0 ).to(DEVICE)

# %% id="xypoPBi4Mkj0"
if RPINN==1:
  if False :
    G = torch.cat( (
        torch.cat( ( 2*M , ZERO , ZERO , ZERO , AX_m, ZERO, ZERO ), dim=1),
        torch.cat( ( ZERO , 2*M , ZERO , ZERO , AY_m, ZERO, ZERO ), dim=1),
        torch.cat( ( ZERO , ZERO , 2*M , ZERO , ZERO, AX_m, ZERO ), dim=1),
        torch.cat( ( ZERO , ZERO , ZERO , 2*M  , ZERO, AY_m, ZERO ), dim=1),
        torch.cat( ( AX_m.t() , AY_m.t() , ZERO , ZERO , M +KX_m+KY_m , ZERO , ZERO ), dim=1),
        torch.cat( ( ZERO , ZERO , AX_m.t() , AY_m.t(),ZERO , M +KX_m+KY_m , ZERO ), dim=1),
        torch.cat( ( ZERO , ZERO , ZERO , ZERO , ZERO, ZERO, M ), dim=1)
    ), dim=0).to(DEVICE)

  lst_v = [(1, b) for b in range(1,N+1)]
  lst_v += [(a, 1) for a in range(1,N+1) ]
  lst_v += [(N, b) for b in range(1,N+1)]
  lst_v += [(a, N) for a in range(1,N+1) ]
  lv = lst_v
  lst_v = [ n+(m-1)*N-1 for (n,m) in lst_v  ]
  lst_p = [(1, b) for b in range(1,N+1)]
  lst_p += [(a, 1) for a in range(1,N+1) ]
  lst_p.append( (N,N) )
  lst_p = [ n+(m-1)*N-1 for (n,m) in lst_p ]

  lst_sigma_11 = [(a, 1) for a in range(1,N+1)]
  lst_sigma_12 = [(1, a) for a in range(1,N+1)]
  lst_sigma_21 = [(a, 1) for a in range(1,N+1)]
  lst_sigma_22 = [(1, a) for a in range(1,N+1)]
  lst_sigma_11 = [ n+(m-1)*N-1 for (n,m) in lst_sigma_11 ]
  lst_sigma_12 = [ n+(m-1)*N-1 for (n,m) in lst_sigma_12 ]
  lst_sigma_21 = [ n+(m-1)*N-1 for (n,m) in lst_sigma_21 ]
  lst_sigma_22 = [ n+(m-1)*N-1 for (n,m) in lst_sigma_22 ]

  offset = N*N
  lst_v = np.array( lst_v )
  lst_p = np.array( lst_p )

  lst_sigma_11 = np.array(  lst_sigma_11 )
  lst_sigma_12 = np.array(  lst_sigma_12 )
  lst_sigma_21 = np.array(  lst_sigma_21 )
  lst_sigma_22 = np.array(  lst_sigma_22 )

  lst_G = np.hstack(  [ lst_sigma_11  , lst_sigma_12 + offset, lst_sigma_21 + 2*offset, lst_sigma_22 + 3*offset, lst_v + 4*offset , lst_v + 5*offset , lst_p + 6*offset ] )

  for z in lst_G:
    A[z,:] = 0
    A[:,z] = 0
    A[z,z] = 0

  M_big = torch.cat( (
    torch.cat( ( M , ZERO , ZERO , ZERO , ZERO, ZERO, ZERO ), dim=1),
    torch.cat( ( ZERO , M , ZERO , ZERO , ZERO, ZERO, ZERO ), dim=1),
    torch.cat( ( ZERO , ZERO , M , ZERO , ZERO, ZERO, ZERO ), dim=1),
    torch.cat( ( ZERO , ZERO , ZERO , M , ZERO, ZERO, ZERO ), dim=1),
    torch.cat( ( ZERO , ZERO , ZERO , ZERO , M , ZERO , ZERO ), dim=1),
    torch.cat( ( ZERO , ZERO , ZERO , ZERO , ZERO  , M , ZERO ), dim=1),
    torch.cat( ( ZERO , ZERO , ZERO , ZERO , ZERO, ZERO, M ), dim=1)
  ), dim=0).to(DEVICE)
  G = M_big + A @ A.T / (h*h)

  for z in lst_G:
    G[z,:] = 0
    G[:,z] = 0
    G[z,z] = 1
    A[z,:] = 0
    A[:,z] = 0
    A[z,z] = 1
  GRAM_LU = torch.linalg.lu_factor(G)


# %% id="36BeoW_sM3wv"
def get_points(x_points: int, y_points: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    x_raw = torch.linspace(0, 1, x_points)
    t_raw = torch.linspace(0, 1, y_points)
    x, t = torch.meshgrid(x_raw, t_raw, indexing="ij")
    x = x.to(device).reshape(-1, 1).requires_grad_(True)
    t = t.to(device).reshape(-1, 1).requires_grad_(True)
    return x, t


# %% id="iQny1bH2CMhG"
def calculate_loss(pinns: list[PINN], x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    ux = pinns[0]
    uy = pinns[1]
    p = pinns[2]
    duxdx = pinns[3]
    duxdy = pinns[4]
    duydx = pinns[5]
    duydy = pinns[6]

    u1 =  f(ux, x, y)
    u2 =  f(uy, x, y)
    du1dx = f(duxdx, x, y)
    du1dy = f(duxdy, x, y)
    du2dx = f(duydx, x, y)
    du2dy = f(duydy, x, y)
    uxdotgradux = u1 * du1dx + u2 * du1dy
    uydotgraduy = u1 * du2dx + u2 * du2dy

    d2uxdx = dfdx(duxdx,x,y)
    d2uxdy = dfdy(duxdy,x,y)
    dpdx = dfdx(p,x,y,order=1)

    d2uydx = dfdx(duydx, x, y)
    d2uydy = dfdy(duydy, x, y)
    dpdy = dfdy(p,x,y)

    loss1 = -d2uxdx - d2uxdy + dpdx + uxdotgradux
    loss2 = -d2uydx - d2uydy + dpdy + uydotgraduy
    loss3 = duxdx(x, y) + duydy(x, y)
    loss_duxdx = duxdx(x, y) - dfdx(ux, x, y)
    loss_duxdy = duxdy(x, y) - dfdy(ux, x, y)
    loss_duydx = duydx(x, y) - dfdx(uy, x, y)
    loss_duydy = duydy(x, y) - dfdy(uy, x, y)

    #return loss1.pow(2).mean() + \
    #       loss2.pow(2).mean() + \
    #       loss3.pow(2).mean() + \
    #       loss_duxdx.pow(2).mean() + \
    #       loss_duxdy.pow(2).mean() + \
    #       loss_duydx.pow(2).mean() + \
    #       loss_duydy.pow(2).mean()

    if RPINN==0:
       loss =  loss1.pow(2).mean() + \
           loss2.pow(2).mean() + \
           loss3.pow(2).mean() + \
           loss_duxdx.pow(2).mean() + \
           loss_duxdy.pow(2).mean() + \
           loss_duydx.pow(2).mean() + \
           loss_duydy.pow(2).mean()

    if RPINN==1:
       list_of_tensors = [ loss_duxdx, loss_duxdy, loss_duydx, loss_duydy, loss1, loss2, loss3]

       concatenated = torch.cat(list_of_tensors, dim=0)
       concatenated = concatenated.reshape(-1, 1)
       concatenated[ lst_G ] = 0
       Ginv_loss = torch.linalg.lu_solve(*GRAM_LU, concatenated )
       loss_val = torch.dot(concatenated.reshape(-1), Ginv_loss.reshape(-1))
       loss = loss_val*h*h*h*h

    p_val = f(pinns[2], x, y)

    return loss + p_val.mean().pow(2)


# %% id="LpoZjAu_zZO-"
def get_points(x_points: int, y_points: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    x_raw = torch.linspace(0, 1, x_points)
    t_raw = torch.linspace(0, 1, y_points)
    x, t = torch.meshgrid(x_raw, t_raw, indexing="ij")
    x = x.to(device).reshape(-1, 1).requires_grad_(True)
    t = t.to(device).reshape(-1, 1).requires_grad_(True)
    return x, t


# %% [markdown] id="vewbG4nzrT4y"
# # Plotting utils

# %% id="Xj4oJzfbrVav"
def plot_3d(x, y, z):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)
    # Add a color bar which maps values to colors.
    fig.colorbar(surf)

    return fig


# %% id="zkaiG_fX2f8e"
def plot_color(z: torch.Tensor, x: torch.Tensor, y: torch.Tensor, n_points_x, n_points_t, title, figsize=(8, 6), dpi=100, cmap="viridis"):
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    z_raw = z.detach().cpu().numpy()
    x_raw = x.detach().cpu().numpy()
    y_raw = y.detach().cpu().numpy()
    X = x_raw.reshape(n_points_x, n_points_t)
    Y = y_raw.reshape(n_points_x, n_points_t)
    Z = z_raw.reshape(n_points_x, n_points_t)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    c = ax.pcolormesh(X, Y, Z, cmap=cmap)
    fig.colorbar(c, ax=ax)

    return fig


# %% [markdown] id="aCO2QrzcpnN6"
# # Hard constraint

# %% id="jXOPa9d2rh-5"
# For plotting to demonstrate hard constraints
x = torch.linspace(0, 1, 1000)
y = torch.linspace(0, 1, 1000)
x, y = torch.meshgrid(x, y, indexing='xy')


# %% colab={"base_uri": "https://localhost:8080/", "height": 415} id="6wN8RzK5qouj" outputId="067ca087-c15a-43e9-8877-d267dc5ae132"
def zero_dirichlet(x, y):
    return 20*x*(1-x)*y*(1-y)

z = zero_dirichlet(x, y)
plot_3d(x, y, z)
None


# %% colab={"base_uri": "https://localhost:8080/", "height": 415} id="QLtzLwuDsotN" outputId="226464e4-15e8-42a9-dbcc-40be143f305d"
def force_up_stream(x, y):
    return VELOCITY*torch.exp(-1000*(y-1)**2)

z = force_up_stream(x, y)
plot_3d(x, y, z)
None


# %% colab={"base_uri": "https://localhost:8080/", "height": 438} id="3EPr-k4Iq28Z" outputId="742a18a8-5373-4dad-e7aa-ac6b53145466"
def zero_at_middle(x, y):
    return -torch.exp(-1000*(y-0.5)**2)*torch.exp(-1000*(x-0.5)**2) + 1.0

z = zero_at_middle(x, y)
plot_3d(x, y, z)
zero_at_middle(torch.tensor(0.5), torch.tensor(0.5))


# %% id="jg8MsNwUv1I_"
def ux_constraint(logits, x, y):
    return logits * zero_dirichlet(x, y) + force_up_stream(x, y)

def uy_constraint(logits, x, y):
    return logits * zero_dirichlet(x, y)

def p_constraint(logits, x, y):
    return logits * zero_at_middle(x, y)


# %% [markdown] id="jMm1l9U502Gc"
# # Training

# %% colab={"background_save": true, "base_uri": "https://localhost:8080/"} id="ut1Z_Wxl01KK" outputId="5c9e4ac9-5a05-4b7d-ea22-5362454b1000" editable=true slideshow={"slide_type": ""}
# Create PINNs
middle_layers = []

for act in ACTIVATION_MODULES:
    middle_layers.append(act)
    middle_layers.append(nn.Linear(NEURONS_PER_LAYER, NEURONS_PER_LAYER))

ux_pinn = PINN(LAYERS, NEURONS_PER_LAYER, middle_layers=middle_layers, hard_constraint=ux_constraint)
uy_pinn = PINN(LAYERS, NEURONS_PER_LAYER, middle_layers=middle_layers, hard_constraint=uy_constraint)
p_pinn  = PINN(LAYERS, NEURONS_PER_LAYER, middle_layers=middle_layers, hard_constraint=p_constraint)
duxdx_pinn  = PINN(LAYERS, NEURONS_PER_LAYER, middle_layers=middle_layers)
duxdy_pinn  = PINN(LAYERS, NEURONS_PER_LAYER, middle_layers=middle_layers)
duydx_pinn  = PINN(LAYERS, NEURONS_PER_LAYER, middle_layers=middle_layers)
duydy_pinn  = PINN(LAYERS, NEURONS_PER_LAYER, middle_layers=middle_layers)
pinns = [ux_pinn, uy_pinn, p_pinn, duxdx_pinn, duxdy_pinn, duydx_pinn, duydy_pinn]

multiPINN = MultiPINN(LAYERS, NEURONS_PER_LAYER, pinns=pinns).to(DEVICE)
# Prepare training points
x, y = get_points(X_POINTS, Y_POINTS, DEVICE)
# Prepare loss function with fixed points
loss_function = partial(calculate_loss, x=x, y=y)

# Prepare vector for storing loss values in each epoch
loss_per_epoch = multiPINN.train(loss_function, EPOCHS, LEARNING_RATE)

# %% [markdown] id="j9i5cw8xyV0M"
# # Results

# %% id="GLJU5pp53ZJN"
pinns = multiPINN.to("cpu").pinns

# %% id="KuzHfW_R8Cq1"
plt.yscale("log")
plt.plot(loss_per_epoch)


# %% id="FUekCD-3nRq9"
def clear(x: torch.Tensor) -> torch.Tensor:
    return x.detach().cpu().reshape(-1)


# %% id="rR35bLHL24bT"
x, y = get_points(X_PLOT, Y_PLOT, "cpu")
ux = f(pinns[0], x, y)
uy = f(pinns[1], x, y)
p = f(pinns[2], x, y)
magnitude = ux**2 + uy**2
x_plot = clear(x)
y_plot = clear(y)
ux_plot = clear(ux)
uy_plot = clear(uy)
p_plot = clear(p)
magnitude_plot = clear(magnitude)

# %% id="EHV5i9Gn3scI"
plot_color(ux, x, y, X_PLOT, Y_PLOT, "ux")
plot_color(uy, x, y, X_PLOT, Y_PLOT, "uy")
plot_color(p, x, y, X_PLOT, Y_PLOT, "p")
plot_color(magnitude, x, y, X_PLOT, Y_PLOT, "magnitude")
None

# %% id="KMU9jCWX4QRT" editable=true slideshow={"slide_type": ""}
pinns[2](torch.tensor([0.5]).reshape(-1,1), torch.tensor([0.5]).reshape(-1,1))

# %% editable=true slideshow={"slide_type": ""}
# >>>CHANGE<<<
# Store the results in a file
store(OUTPUT_PATH, loss_per_epoch)
