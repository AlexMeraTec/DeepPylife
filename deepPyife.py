import math
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim

from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import StepLR

POOL_SIZE = 1024
N_CHANNEL = 16
BATCH_SIZE = 8
width = height = int(math.sqrt(POOL_SIZE))

device = torch.device("cpu")

plt.ion()
plt.show()

sobelX = torch.from_numpy(np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).astype(float)).repeat((16, 16, 1, 1)).to(
    device)
sobelY = torch.from_numpy(np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).astype(float)).repeat((16, 16, 1, 1)).to(
    device)
cellId = torch.from_numpy(np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]).astype(float)).repeat((16, 16, 1, 1)).to(device)

filters = [sobelX, sobelY, cellId]


def imshowTensor(x, chan_n=3):
    plt.imshow(x[0, :chan_n, ...].detach().numpy().transpose(1, 2, 0))
    plt.show()
    return


class UpdateGrid(torch.nn.Module):

    def __init__(self):
        super(UpdateGrid, self).__init__()

        self.fc1 = nn.Conv2d(N_CHANNEL * len(filters), 128, (1, 1))
        self.fc2 = nn.Conv2d(128, N_CHANNEL, (1, 1))
        torch.nn.init.zeros_(self.fc2.weight)

    def forward(self, x):
        # Creamos un tensor donde guardar la computación de los filtros de percepción.
        perception = torch.empty((BATCH_SIZE, len(filters) * N_CHANNEL, width, height)).to(device)

        # Computamos los vectores de percepción con cada filtro. 3 filtros x 16 = 48 componentes.
        for f, filt in enumerate(filters):
            perception[:, (f * N_CHANNEL):((f + 1) * N_CHANNEL), :, :] = F.conv2d(x, filt, padding=[1, 1])

        # La red neuronal :)
        dx = self.fc1(perception)
        dx = F.relu(dx)
        dx = self.fc2(dx)

        # Cada célula tiene un 50% de ser actualizada.
        randomMask = torch.from_numpy(np.random.randint(0, 2, (BATCH_SIZE, 1, width, height))).repeat(1, 16, 1, 1).to(
            device)
        x = x + dx * randomMask

        # Limitamos el crecimiento sólo a las células vivas. Vivas: alpha o vecina con alpha > 0.1
        alv_f = torch.from_numpy(np.ones((1, 1, 3, 3)).astype(int)).to(device)
        alive = F.conv2d((x[:, 3:4, :, :] > 0.1).double(), alv_f.double(), padding=1)
        alive = (alive > 0.0)
        alive = alive.repeat(1, 16, 1, 1)

        return x * alive


# Mostrar imagen cada...
debug_img = 10

# Instanciamos nuestro modelo.
updateGrid = UpdateGrid().to(device)

# Cargamos la imagen objetivo a recrear.
target = torch.tensor(np.array(Image.open("corona.png")) / 255.).to(device)

# Declaramos la función de error.
loss_f = nn.MSELoss()

# Creamos el optimizador.
optimizer = optim.Adam(updateGrid.parameters(), lr=1e-3)
scheduler = StepLR(optimizer, step_size=2000, gamma=0.1)

# for p in updateGrid.parameters():
#     p.register_hook(lambda grad: grad / (torch.norm(grad, 2) + 1e-8))

nImg = 0

plt.ion()

for trStep in range(10001):

    # Seleccionamos un número aleatorio de steps.
    n_steps = np.random.randint(64, 96)

    # Inicializamos una rejilla vacía.
    grid = np.zeros((width, height, N_CHANNEL))
    # Y añadimos una célula viva en medio.
    grid[height // 2, width // 2, 3:] = 1.0

    batch_grid = np.repeat(grid[np.newaxis, ...], BATCH_SIZE, axis=0)

    # Creamos el tensor resultado donde iremos guardando la computación.
    result = torch.from_numpy(batch_grid).permute(0, 3, 1, 2).to(device)

    for step in range(n_steps):

        # Cuidamos que ninguna célula sobrepase el rango 0-1.
        result = torch.clamp(updateGrid.forward(result), 0.0, 1.0)

        if (trStep + 1) % debug_img == 0 and step % 2 == 0:
            batch_img = result[0, :4, :, :].detach().cpu().numpy()

            imRes = np.clip(batch_img.transpose(1, 2, 0)[:, :, :4], 0.0, 1.0)
            plt.imshow(imRes)

            plt.title('Tr.Step:' + str(trStep) + '- Step:' + str(step))
            plt.draw()
            # ax.set_facecolor('white')
            plt.pause(0.001)
            plt.clf()
            # nImg += 1
            # plt.imsave('./output_imgs/img_' + str(nImg) + '.png', imRes)
            # plt.show()

    # Limpiamos los gradientes.
    optimizer.zero_grad()
    # Extraemos las componentes RGBA de cada célula.
    output = result[:, :4, :, :].permute(0, 2, 3, 1)
    # Calculamos el Error Cuadrático Medio de la imagen resultante y objetivo.
    loss = loss_f(output, target.repeat((BATCH_SIZE, 1, 1, 1)))
    # Y optimizamos en base a este error.
    loss.backward()
    # Optimizamos un paso.
    optimizer.step()
    scheduler.step()

    print('Tr.Loss ' + str(trStep) + ': ' + str(loss.item())[0:6] + '; lr=' + str(optimizer.param_groups[0]['lr']))
