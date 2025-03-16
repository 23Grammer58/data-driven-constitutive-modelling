import numpy as np
import torch
import torch.nn as nn

# 1. Внутренние точки
n_points = 1000
x = np.random.uniform(0, 1, n_points)
y = np.random.uniform(0, 1, n_points)
coords = np.vstack((x, y)).T  # Размер: (N, 2)

# 2. Объёмные силы (гравитация)
f_body = np.array([0, -9.8])
f_body = np.tile(f_body, (coords.shape[0], 1))  # Размер: (N, 2)

# 3. Точки Дирихле (левая граница)
x_left = np.zeros(100)
y_left = np.linspace(0, 1, 100)
dirichlet_points = np.stack((x_left, y_left), axis=1)  # Размер: (N_D, 2)

# 4. Значения условий Дирихле (фиксированные)
dirichlet_values = np.zeros_like(dirichlet_points)  # Размер: (N_D, 2)

print("coords:", coords.shape)
print("f_body:", f_body.shape)
print("dirichlet_points:", dirichlet_points.shape)
print("dirichlet_values:", dirichlet_values.shape)

coords = torch.tensor(coords, dtype=torch.float32, requires_grad=True)  # Преобразуем coords
f_body = torch.tensor(f_body, dtype=torch.float32)  # Преобразуем f_body
dirichlet_points = torch.tensor(dirichlet_points, dtype=torch.float32)
dirichlet_values = torch.tensor(dirichlet_values, dtype=torch.float32)


def compute_P(F, mu=1.0, lambda_=1.0):
    """
    Вычисляет второй тензор Пиолы-Кирхгоффа для нео-Гуковского материала.

    Args:
        F (torch.Tensor): Тензор градиента деформации (N, 2, 2).
        mu (float): Параметр Ламе.
        lambda_ (float): Параметр Ламе.

    Returns:
        torch.Tensor: Второй тензор Пиолы-Кирхгоффа (N, 2, 2).
    """
    # Вычисляем J = det(F)
    J = torch.det(F)

    # Вычисляем C = F^T F
    C = torch.matmul(F.transpose(1, 2), F)

    # Вычисляем след (trace) C
    tr_C = torch.diagonal(C, dim1=1, dim2=2).sum(dim=1)

    # Гиперупругий потенциал Psi
    Psi = (mu / 2) * (tr_C - 3) - mu * torch.log(J) + (lambda_ / 2) * (torch.log(J) ** 2)

    # Вычисляем производную Psi по C
    grad_Psi = torch.autograd.grad(
        outputs=Psi,
        inputs=C,
        grad_outputs=torch.ones_like(Psi),
        create_graph=True
    )[0]

    # Второй тензор Пиолы-Кирхгоффа: S = 2 * grad_Psi
    S = 2 * grad_Psi

    return S


# PINN model
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 2)  # Outputs u_x, u_y
        )

    def forward(self, x):
        return self.net(x)


# Loss function
def loss_function(model, coords, f_body, dirichlet_points, dirichlet_values):
    # Predictions
    u_pred = model(coords)

    # Compute gradients (automatic differentiation)
    grads = torch.autograd.grad(
        outputs=u_pred, inputs=coords, grad_outputs=torch.ones_like(u_pred),
        create_graph=True
    )[0]
    F = torch.eye(2).unsqueeze(0).repeat(grads.shape[0], 1, 1) + grads.unsqueeze(-1)

    # Compute P and residual of equilibrium
    P = compute_P(F)  # Define hyperelastic P computation
    # Вычисляем дивергенцию P
    div_P = []
    for i in range(P.shape[1]):  # По строкам матрицы P
        div_P_i = torch.autograd.grad(
            outputs=P[:, :, i], inputs=coords, grad_outputs=torch.ones_like(P[:, :, i]),
            create_graph=True
        )[0]
        div_P.append(div_P_i)

    div_P = torch.stack(div_P, dim=-1)
    residual = div_P + f_body

    # Losses
    physics_loss = torch.mean(residual ** 2)
    dirichlet_loss = torch.mean((u_pred[dirichlet_points] - dirichlet_values) ** 2)

    return physics_loss + dirichlet_loss


# Training loop
model = PINN()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(1000):
    optimizer.zero_grad()
    loss = loss_function(model, coords, f_body, dirichlet_points, dirichlet_values)
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

