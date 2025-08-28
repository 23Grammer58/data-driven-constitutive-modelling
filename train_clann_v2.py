#!/usr/bin/env python3
# ================================================================
#  Convex Laplace model:   F  ->  P        (обучение)
#                          ξ  ->  dψ/dξ    (симулятор, ONNX)
# ================================================================
import math, csv, argparse, pathlib, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.autograd import grad
from torch.utils.data import DataLoader, TensorDataset, random_split
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Используем non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
from torch.cuda.amp import autocast, GradScaler
import subprocess
import sys

# ------------------------- 1. выпуклая ICNN ----------------------
class ICNN1(nn.Module):
    """
    Convex MLP (xi_dim=3)  --single softplus layer→  ψ.
    strictly convex, C¹;  beta controls smoothness (beta→∞ → ReLU)
    """
    def __init__(self, hidden: int = 64, beta: float = 10.):
        super().__init__()
        self.beta = beta
        self.Wx   = nn.Linear(3, hidden, bias=True)      # свободные веса
        self.w2   = nn.Parameter(torch.rand(hidden))     # ≥0 via abs
        self.b2   = nn.Parameter(torch.zeros(1))

    def forward(self, xi: torch.Tensor) -> torch.Tensor: # (B,3)→(B,)
        z = F.softplus(self.beta * self.Wx(xi)) / self.beta
        psi = F.linear(z, self.w2.abs()) + self.b2
        return psi.squeeze(-1)

# -------------------- 2. кинематика Laplace (2-D) ----------------
def cholesky_upper(C):                      # SPD (B,2,2) → upper-triangular
    return torch.linalg.cholesky(C).transpose(-2, -1)

def sigma_from_grad(xi, g, U):
    """g = dψ/dξ  (B,3);  U – upper-triangular Cholesky (B,2,2)
       → Σ = dψ/dU  (upper tri, B,2,2)"""
    u11, u22, u12 = U[:,0,0], U[:,1,1], U[:,0,1]
    # компоненты Σ11, Σ22, Σ12
    s11 = g[:,0] / u11 - g[:,2] * (u12 / u11**2)
    s22 = g[:,1] / u22
    s12 = g[:,2] / u11
    Σ = U.new_zeros(U.shape)
    Σ[:,0,0], Σ[:,1,1], Σ[:,0,1] = s11, s22, s12
    return Σ

def vec_uvec(U):                            # (B,2,2) → (B,3)  (ln(u11),ln(u22),u12/u11)
    u11 = U[...,0,0]
    u22 = U[...,1,1]  
    u12 = U[...,0,1]
    xi1 = torch.log(u11)  # ln(u11)
    xi2 = torch.log(u22)  # ln(u22)
    xi3 = u12 / u11       # u12/u11
    return torch.stack([xi1, xi2, xi3], dim=-1)

def mat_from_vec(v):                        # (B,3) → верхний 2×2
    B = v.shape[0]
    M = v.new_zeros(B,2,2)
    u11 = torch.exp(v[:,0])     # u11 = exp(xi1) = exp(ln(u11))
    u22 = torch.exp(v[:,1])     # u22 = exp(xi2) = exp(ln(u22))  
    u12 = v[:,2] * u11          # u12 = xi3 * u11 = (u12/u11) * u11
    M[:,0,0], M[:,1,1], M[:,0,1] = u11, u22, u12
    return M

# ------------------------------------------------------------------
# Sylvester solver  X Rᵀ + R X = C   for upper-triangular R
# returns symmetric X  (B,n,n)
# ------------------------------------------------------------------
def sylv_symm_upper(R: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
    B, n, _ = R.shape
    X = C.new_zeros(B, n, n)
    for i in reversed(range(n)):               # диагонали
        rii = R[:, i, i]
        X[:, i, i] = C[:, i, i] / (2 * rii)

        for j in reversed(range(i)):           # сверхдиагональ
            rhs = C[:, j, i]
            if i + 1 < n:                      # Σ_k>i R_jk X_ki
                rhs -= (R[:, j, i+1:] * X[:, i, i+1:]).sum(-1)
            if j + 1 < i:                      # Σ_k>j X_jk R_ik
                rhs -= (X[:, j, j+1:i] * R[:, i, j+1:i]).sum(-1)
            Xji = rhs / (R[:, j, j] + rii)
            X[:, j, i] = X[:, i, j] = Xji
    return X

def S_from_grad(g, U):
    """g = dψ/dξ  (B,3);  U – upper-triangular Cholesky (B,2,2)
       → S = Sigma  (upper tri, B,2,2)"""
    u11, u22, u12 = U[:,0,0], U[:,1,1], U[:,0,1]
    # компоненты S11, S22, S12
    s11 = (g[:,0] - 2 * u12/u11 * g[:,2]) / u11 + (u12 / (u11 * u22))**2 * g[:,1]
    s12 = 1/u11**2 * g[:,0] - (u12 / (u11 * u22**2)) * g[:,1] 
    s21 = s12
    s22 = 1/u22**2 * g[:,1]
    S = U.new_zeros(U.shape)

    S[:,0,0], S[:,1,1], S[:,0,1] = s11, s22, s12
    return S
    
def S_from_xi(g, xi, U):
    """xi = (xi1, xi2, xi3)  (B,3);  g = dψ/dξ  (B,3)
       → S = dψ/dC  (upper tri, B,2,2)"""

    s11 = torch.exp(-2 * xi[:,0]) * (g[:,0] - 2 * xi[:,2] * g[:,2]) + torch.exp(-2 * xi[:,1]) * g[:,1] * xi[:,2] * xi[:,2] 
    s22 = torch.exp(-2 * xi[:,1]) * g[:,1]
    s12 = -torch.exp(-2 *xi[:,1]) * g[:,1] * xi[:, 2] + torch.exp(- 2 * xi[:, 0]) * g[:,2]

    S = U.new_zeros(U.shape)
    S[:,0,0], S[:,1,1], S[:,0,1] = s11, s22, s12
    return S

    return S_from_grad(g, U)
# -------------------------- 3. модель ----------------------------
class LaplaceModel(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.psi_net = ICNN1(hidden)

    # C (B,2,2) → S, ψ
    def forward(self, C):
        # Если AMP передало сюда тензор float16, переводим в float32
        if C.dtype != torch.float32:
            C = C.float()
        U = cholesky_upper(C)
        xi = vec_uvec(U).requires_grad_(True)

        psi = self.psi_net(xi)

        g   = torch.autograd.grad(psi, xi,
          grad_outputs=torch.ones_like(psi),
          create_graph=True)[0]

        # --- новая формула для вычисления напряжений ---
        S = S_from_xi(g, xi, U)  # ← новый прямой расчёт
        # dpsi_dxi = grad(psi, xi,
        #                 grad_outputs=torch.ones_like(psi),
        #                 create_graph=True)[0]
        

        # Sigma_ut = mat_from_vec(dpsi_dxi)                 # верхний треуг.
        # # full = ut + utᵀ  (но без удвоения диагонали!)
        # Sigma = Sigma_ut + Sigma_ut.transpose(-2, -1) \
        #         - torch.diag_embed(Sigma_ut.diagonal(dim1=-2, dim2=-1))


        # Sigma = mat_from_vec(dpsi_dxi)
        
        

        
        # Sigma = Sigma_ut + Sigma_ut.transpose(-2, -1)   # диагональ удвоена

        # Uinv_Sigma = torch.linalg.solve_triangular(U, Sigma, upper=True)
        # S = 0.5*(Uinv_Sigma + Uinv_Sigma.transpose(-2,-1))

        # --- решаем Sylvester S Uᵀ + U S = 2 Σ -------------------
        # S = sylv_symm_upper(U, 2 * Sigma)      # symmetric (B,n,n)
        # S = sylv_symm_upper(U, Sigma)      # symmetric (B,n,n)

        return S, psi   

    # ξ (B,3) → dψ/dξ   (нужен autograd для обучения)
    def dpsi_dxi(self, xi):
        xi = xi.requires_grad_(True)
        psi = self.psi_net(xi)
        return grad(psi, xi,
                    grad_outputs=torch.ones_like(psi))[0]

# ---------- 4. ручной градиент для ONNX (один слой ICNN1) --------
class DerivWrapper1L(nn.Module):
    """  ξ → ∂ψ/∂ξ  – без autograd; все ops – ONNX-friendly  """
    def __init__(self, base: ICNN1):
        super().__init__()
        self.beta  = base.beta
        self.W1    = base.Wx        # weight (H,3), bias (H)
        # Отсоединяем градиент для w2 перед взятием абсолютного значения
        self.w2pos = nn.Parameter(base.w2.detach().abs(), requires_grad=False)  # (H,)
        
    def forward(self, xi):          # (B,3) → (B,3)
        s  = self.W1(xi)                           # (B,H)
        sig = torch.sigmoid(self.beta * s)         # σ′
        g   = sig * self.w2pos                     # (B,H)
        return torch.matmul(g, self.W1.weight)     # (B,3)
    

class HessianWrapper1L(nn.Module):
    """
    ONNX-friendly: вручную вычисляет гессиан ∂²ψ/∂ξᵢ∂ξⱼ (B,3,3).
    Поддерживает только один слой ICNN1.
    """
    def __init__(self, base: ICNN1):
        super().__init__()
        self.beta = base.beta
        self.W1 = base.Wx            # Linear(3→H), содержит weight и bias
        self.w2pos = nn.Parameter(base.w2.detach().abs(), requires_grad=False)

    def forward(self, xi):  # (B,3) → (B,3,3)
        W = self.W1.weight     # (H,3)
        b = self.W1.bias       # (H,)
        H = W.shape[0]

        s = torch.matmul(xi, W.t()) + b       # (B,H)
        sig = torch.sigmoid(self.beta * s)    # (B,H)
        sig_prime = self.beta * sig * (1 - sig)   # σ′′(s) = βσ(1−σ)

        # g = σ′′(Wx+b) * w2 * w_i * w_j
        B = xi.shape[0]
        H_diag = sig_prime * self.w2pos    # (B,H)

        # теперь собираем гессиан вручную
        H_out = torch.zeros(B, 3, 3, dtype=xi.dtype, device=xi.device)
        for i in range(3):
            for j in range(3):
                # сумма по h: H_ij = Σ_h σ′′_h w2_h W_hi W_hj
                prod = W[:, i] * W[:, j]     # (H,)
                weighted = H_diag * prod     # (B,H)
                H_out[:, i, j] = weighted.sum(dim=1)

        return H_out  # (B,3,3)

# -------------------------- 5. utils -----------------------------
def load_csv(fname: str):
    """CSV с колонками Test,frame,F_xx,F_xy,F_yx,F_yy,P_xx,P_xy,P_yy или C_xx,C_xy,C_yy,P_xx,P_xy,P_yy и опционально dpsidksi1,2,3"""
    data = pd.read_csv(fname)

    # Проверяем, есть ли колонки F или C
    if 'F_xx' in data.columns:
        # --- 1. формируем тензор деформации F (2×2) ---
        Fmat = torch.zeros(len(data), 2, 2, dtype=torch.float32)
        Fmat[:, 0, 0] = torch.tensor(data['F_xx'].values, dtype=torch.float32)
        Fmat[:, 1, 1] = torch.tensor(data['F_yy'].values, dtype=torch.float32)
        if 'F_xy' in data.columns:
            Fmat[:, 0, 1] = torch.tensor(data['F_xy'].values, dtype=torch.float32)
        if 'F_yx' in data.columns:
            Fmat[:, 1, 0] = torch.tensor(data['F_yx'].values, dtype=torch.float32)
        # Вычисляем C = F^T F
        C = torch.matmul(Fmat.transpose(-2,-1), Fmat)
    else:
        # --- 1. формируем тензор C напрямую ---
        C = torch.zeros(len(data), 2, 2, dtype=torch.float32)
        C[:, 0, 0] = torch.tensor(data['C_xx'].values, dtype=torch.float32)
        C[:, 1, 1] = torch.tensor(data['C_yy'].values, dtype=torch.float32)
        if 'C_xy' in data.columns:
            C[:, 0, 1] = torch.tensor(data['C_xy'].values, dtype=torch.float32)
            C[:, 1, 0] = C[:, 0, 1]  # C симметричен

    # --- 2. формируем тензор напряжений PK2 (только симметричные компоненты) ---
    S_vec = torch.zeros(len(data), 3, dtype=torch.float32)
    S_vec[:, 0] = torch.tensor(data['PK2_xx'].values, dtype=torch.float32)
    S_vec[:, 1] = torch.tensor(data['PK2_yy'].values, dtype=torch.float32)
    S_vec[:, 2] = torch.tensor(data['PK2_xy'].values, dtype=torch.float32)

    # --- 3. получаем типы экспериментов ---
    experiment_types = data['Test'].values if 'Test' in data.columns else None

    # --- 4. DataFrame для визуализации диагональных компонент ---
    plot_data = pd.DataFrame({
        'experiment_type': data['Test'] if 'Test' in data.columns else 'unknown',
        'strain_xx': data['C_xx'].values if 'C_xx' in data.columns else data['F_xx'].values,
        'strain_yy': data['C_yy'].values if 'C_yy' in data.columns else data['F_yy'].values,
        'S_exp_xx': data['PK2_xx'].values,
        'S_exp_yy': data['PK2_yy'].values,
        'S_exp_xy': data['PK2_xy'].values
    })

    # --- 5. Опционально: загружаем dpsidksi1,2,3 если есть ---
    dpsidksi = None
    if all(col in data.columns for col in ['dpsidksi1', 'dpsidksi2', 'dpsidksi3']):
        dpsidksi = torch.tensor(data[['dpsidksi1', 'dpsidksi2', 'dpsidksi3']].values, dtype=torch.float32)

    return C, S_vec, experiment_types, plot_data, dpsidksi

def split_dataset(F, S, val_frac=0.1):
    """Разделяет датасет последовательно: первые (1-val_frac) для обучения, последние val_frac для валидации"""
    n_total = len(F)
    n_train = int(n_total * (1 - val_frac))
    
    # Последовательное разделение
    F_train, F_val = F[:n_train], F[n_train:]
    S_train, S_val = S[:n_train], S[n_train:]
    
    train_ds = TensorDataset(F_train, S_train)
    val_ds = TensorDataset(F_val, S_val)
    
    print(f"Разделение датасета: {n_train} образцов для обучения, {n_total - n_train} для валидации")
    return train_ds, val_ds

def split_dataset_by_experiments(F, S, experiment_types, train_types=None, val_types=None):
    """
    Разделяет датасет по типам экспериментов
    
    Args:
        F: тензор деформаций (N, 2, 2)
        S: тензор напряжений (N, 3) 
        experiment_types: список типов экспериментов для каждого образца (N,)
        train_types: список типов для обучения, если None - автоматически первые 70%
        val_types: список типов для валидации, если None - автоматически оставшиеся 30%
    """
    # Преобразуем в numpy для удобства
    exp_types = np.array(experiment_types)
    unique_types = np.unique(exp_types)
    
    print(f"Найдено типов экспериментов: {list(unique_types)}")
    
    # Автоматическое разделение, если не указано
    if train_types is None:
        n_train_types = max(1, int(len(unique_types) * 0.7))  # 70% типов для обучения
        train_types = unique_types[:n_train_types]
        
    if val_types is None:
        if isinstance(train_types, (list, np.ndarray)):
            # Берем все типы, которые не в train_types
            val_types = [t for t in unique_types if t not in train_types]
        else:
            val_types = unique_types[n_train_types:]
    
    print(f"Типы для обучения: {list(train_types)}")
    print(f"Типы для валидации: {list(val_types)}")
    
    # Создаем маски для разделения
    train_mask = np.isin(exp_types, train_types)
    val_mask = np.isin(exp_types, val_types)
    
    # Разделяем данные
    F_train = F[train_mask]
    S_train = S[train_mask]
    F_val = F[val_mask]  
    S_val = S[val_mask]
    
    train_ds = TensorDataset(F_train, S_train)
    val_ds = TensorDataset(F_val, S_val)
    
    print(f"Образцов для обучения: {len(F_train)} (типы: {list(train_types)})")
    print(f"Образцов для валидации: {len(F_val)} (типы: {list(val_types)})")
    
    return train_ds, val_ds

def predict_stresses(model, C):
    """Получает предсказания модели для визуализации (все три компоненты)"""
    model.eval()
    with torch.enable_grad():
        device = next(model.parameters()).device
        C = C.clone().to(device).requires_grad_(True)
        S_pred, _ = model(C)
        S_pred_cpu = S_pred.detach().cpu()
        return (S_pred_cpu[:,0,0].numpy(),
                S_pred_cpu[:,1,1].numpy(),
                S_pred_cpu[:,0,1].numpy())

# -------- 5.b  загрузка CSV с xi и производными -----------------
def load_xi_csv(fname: str):
    """CSV с колонками xi1,xi2,xi3,dpsidksi1,dpsidksi2,dpsidksi3"""
    df = pd.read_csv(fname)
    xi = torch.tensor(df[['xi1', 'xi2', 'xi3']].values, dtype=torch.float32)
    dpsi = torch.tensor(df[['dpsidksi1', 'dpsidksi2', 'dpsidksi3']].values, dtype=torch.float32)
    return xi, dpsi

def load_xi_stretch_csv(fname: str):
    """Загружает данные из xi_stretch файлов с колонками C_xx,C_xy,C_yy,P_xx,P_xy,P_yy"""
    return load_csv(fname)

# ---------------------- 6. обучение одной модели ----------------
def train_one(hidden, train_ds, val_ds,
              epochs=4000, lr=3e-4, batch=256, device='cpu', amp=False,
              xi_data: tuple = None, dpsi_weight: float = 0.0,
              anchor_SI_weight: float = 0.0,
              anchor_dpsi0_weight: float = 0.0,
              anchor_psi0_weight: float = 0.0):
    amp = amp and (device == "cuda")
    net = LaplaceModel(hidden).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    scaler = GradScaler(enabled=amp)
    # Сравниваем все три независимых компоненты: S_xx, S_yy, S_xy
    l2vec = lambda S: torch.stack([S[:,0,0], S[:,1,1], S[:,0,1]], dim=1)

    if xi_data is not None:
        xi, dpsi = xi_data
        xi_loader = DataLoader(TensorDataset(xi, dpsi), batch, shuffle=True, pin_memory=(device=="cuda"))
        xi_iter = iter(xi_loader)

    train_loader = DataLoader(train_ds, batch, shuffle=True, pin_memory=(device=="cuda"))
    best_val, best_state = 1e9, None

    for ep in range(1, epochs+1):
        net.train()
        for Fbatch, Sbatch in train_loader:
            Fbatch = Fbatch.to(device, non_blocking=(device=="cuda"))
            Sbatch = Sbatch.to(device, non_blocking=(device=="cuda"))
            with autocast(enabled=amp):
                S_pred,_ = net(Fbatch)
                loss = F.mse_loss(l2vec(S_pred), Sbatch)

                # добавляем лосс по dpsi/dxi при наличии данных
                if xi_data is not None and dpsi_weight > 0.0:
                    try:
                        xi_batch, dpsi_batch = next(xi_iter)
                    except StopIteration:
                        xi_iter = iter(xi_loader)
                        xi_batch, dpsi_batch = next(xi_iter)
                    xi_batch = xi_batch.to(device, non_blocking=(device=="cuda"))
                    dpsi_batch = dpsi_batch.to(device, non_blocking=(device=="cuda"))
                    dpsi_pred = net.dpsi_dxi(xi_batch)
                    loss_dpsi = F.mse_loss(dpsi_pred, dpsi_batch)
                    loss = loss + dpsi_weight * loss_dpsi

                # якорные слагаемые при C=I и ξ=0
                if anchor_SI_weight > 0.0 or anchor_dpsi0_weight > 0.0 or anchor_psi0_weight > 0.0:
                    Bsize = Fbatch.shape[0]
                    Ibatch = torch.eye(2, dtype=Fbatch.dtype, device=Fbatch.device).unsqueeze(0).repeat(Bsize, 1, 1)
                    S_I, psi_I = net(Ibatch)

                    if anchor_SI_weight > 0.0:
                        # S_I имеет размерность (B, 2, 2), извлекаем компоненты напрямую
                        S_I_vec = torch.stack([S_I[:,0,0], S_I[:,1,1], S_I[:,0,1]], dim=1)
                        zero_S = torch.zeros(Bsize, 3, dtype=S_I.dtype, device=S_I.device)
                        loss_SI = F.mse_loss(S_I_vec, zero_S)
                        loss = loss + anchor_SI_weight * loss_SI

                    if anchor_dpsi0_weight > 0.0:
                        xi_zero = torch.zeros(Bsize, 3, dtype=Fbatch.dtype, device=Fbatch.device)
                        dpsi_zero = net.dpsi_dxi(xi_zero)
                        loss_dpsi0 = F.mse_loss(dpsi_zero, torch.zeros_like(dpsi_zero))
                        loss = loss + anchor_dpsi0_weight * loss_dpsi0

                    if anchor_psi0_weight > 0.0:
                        loss_psi0 = F.mse_loss(psi_I, torch.zeros_like(psi_I))
                        loss = loss + anchor_psi0_weight * loss_psi0
            opt.zero_grad()
            if amp:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward(); opt.step()

        if ep%10==0 or ep==epochs:
            net.eval(); Fval,Sval = val_ds[:]
            Fval = Fval.to(device, non_blocking=(device=="cuda"))
            Sval = Sval.to(device, non_blocking=(device=="cuda"))
            with autocast(enabled=amp):
                S_pred,_ = net(Fval)
                val_loss = F.mse_loss(l2vec(S_pred), Sval).item()
            if val_loss < best_val:
                best_val, best_state = val_loss, net.state_dict()
            print(f"hidden={hidden:4d}  epoch {ep:4d}/{epochs}  "
                  f"train={loss.item():.4e}  val={val_loss:.4e}")
        if loss.item() < 1e-7:
            break
    # Если валидация ни разу не выполнялась (например, epochs < 500), best_state останется None
    if best_state is None:
        best_state = net.state_dict()
    net.load_state_dict(best_state)
    return net, best_val

# -------------------- 7. поиск оптимального hidden ---------------------
def search_width(F, S, experiment_types=None, split_by_experiments=False, train_types=None, val_types=None, widths=(16,32,64,128,256), **train_kw):
    """
    Поиск оптимальной ширины сети
    
    Args:
        F: тензор деформаций
        S: тензор напряжений  
        experiment_types: типы экспериментов (для split_by_experiments)
        split_by_experiments: если True, разделяет по типам экспериментов, иначе последовательно
        train_types: типы для обучения (для split_by_experiments)
        val_types: типы для валидации (для split_by_experiments)
        widths: список ширин для тестирования
    """
    if split_by_experiments and experiment_types is not None:
        train_ds, val_ds = split_dataset_by_experiments(F, S, experiment_types, train_types, val_types)
    else:
        train_ds, val_ds = split_dataset(F, S)
        
    best_net, best_loss, best_w = None, 1e9, None
    for w in widths:
        net, val = train_one(w, train_ds, val_ds, **train_kw)
        if val < best_loss:
            best_loss, best_net, best_w = val, net, w
    print(f"\nbest width = {best_w},  val MSE = {best_loss:.4e}")
    return best_net

def plot_curves(df: pd.DataFrame,
                name: Path = Path("test"),
                columns: dict = {
                    "strain_col_xx": "strain_xx",
                    "strain_col_yy": "strain_yy",
                    "exp_col_xx": "S_exp_xx",
                    "exp_col_yy": "S_exp_yy",
                    "exp_col_xy": "S_exp_xy",
                    "pred_col_xx": "S_pred_xx",
                    "pred_col_yy": "S_pred_yy",
                    "pred_col_xy": "S_pred_xy",
                    "type_col": "experiment_type"
                }) -> None:
    """
    Draws experimental & predicted stress–strain curves on separate figures.
    One colour per experiment type, solid = experiment, dashed = model.
    """


    types = df[columns["type_col"]].unique()
    cmap = plt.get_cmap("tab10")

    # График для xx компонент
    fig1, ax1 = plt.subplots(figsize=(7, 5))
    lines_xx = []  # Сохраняем линии для сортировки легенды
    for i, t in enumerate(types):
        dft = df[df[columns["type_col"]] == t]
        colour = cmap(i)
        # Сначала рисуем предсказание
        pred_line = ax1.plot(dft[columns["strain_col_xx"]], dft[columns["pred_col_xx"]], color=colour, lw=2.0,
                ls="--", label=f"{t} pred")[0]
        # Затем эксперимент
        exp_line = ax1.plot(dft[columns["strain_col_xx"]], dft[columns["exp_col_xx"]], color=colour, lw=2.5,
                label=f"{t} exp")[0]
        lines_xx.extend([pred_line, exp_line])
    
    ax1.set_xlabel("Strain xx")
    ax1.set_ylabel("Stress xx")
    ax1.set_title("XX components")
    ax1.grid(alpha=0.3)
    # Сортируем легенду: сначала все предсказания, потом все эксперименты
    labels = [l.get_label() for l in lines_xx]
    ax1.legend(lines_xx, labels, ncol=2)
    plt.tight_layout()
    plt.savefig(name / 'stress_strain_xx.png', dpi=150, bbox_inches='tight')
    plt.close()

    # График для yy компонент
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    lines_yy = []  # Сохраняем линии для сортировки легенды
    for i, t in enumerate(types):
        dft = df[df[columns["type_col"]] == t]
        colour = cmap(i)
        # Сначала рисуем предсказание
        pred_line = ax2.plot(dft[columns["strain_col_yy"]], dft[columns["pred_col_yy"]], color=colour, lw=2.0,
                ls="--", label=f"{t} pred")[0]
        # Затем эксперимент
        exp_line = ax2.plot(dft[columns["strain_col_yy"]], dft[columns["exp_col_yy"]], color=colour, lw=2.5,
                label=f"{t} exp")[0]
        lines_yy.extend([pred_line, exp_line])
    
    ax2.set_xlabel("Strain yy")
    ax2.set_ylabel("Stress yy")
    ax2.set_title("YY components")
    ax2.grid(alpha=0.3)
    # Сортируем легенду: сначала все предсказания, потом все эксперименты
    labels = [l.get_label() for l in lines_yy]
    ax2.legend(lines_yy, labels, ncol=2)
    plt.tight_layout()
    plt.savefig(name / 'stress_strain_yy.png', dpi=150, bbox_inches='tight')
    plt.close()

    # График для сдвиговых компонент (xy)
    fig3, ax3 = plt.subplots(figsize=(7, 5))
    lines_xy = []
    for i, t in enumerate(types):
        dft = df[df[columns["type_col"]] == t]
        colour = cmap(i)
        # Сначала рисуем предсказание
        pred_line = ax3.plot(dft[columns["strain_col_xx"]], dft[columns["pred_col_xy"]], color=colour, lw=2.0,
                ls="--", label=f"{t} pred")[0]
        # Затем эксперимент
        exp_line = ax3.plot(dft[columns["strain_col_xx"]], dft[columns["exp_col_xy"]], color=colour, lw=2.5,
                label=f"{t} exp")[0]
        lines_xy.extend([pred_line, exp_line])
    
    ax3.set_xlabel("Strain xx")
    ax3.set_ylabel("Shear Stress xy")
    ax3.set_title("XY (Shear) components")
    ax3.grid(alpha=0.3)
    # Сортируем легенду: сначала все предсказания, потом все эксперименты
    labels = [l.get_label() for l in lines_xy]
    ax3.legend(lines_xy, labels, ncol=2)
    plt.tight_layout()
    plt.savefig(name / 'stress_strain_xy.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Графики сохранены: stress_strain_xx.png, stress_strain_yy.png, stress_strain_xy.png")

def load_trained_model(path: Path, device="cpu"):
    """Загружает сохранённую модель LaplaceModel по пути .pth"""
    state = torch.load(path, map_location=device)
    hidden = state["psi_net.Wx.weight"].shape[0]
    net = LaplaceModel(hidden).to(device)
    net.load_state_dict(state)
    net.eval()
    return net




# ----------------------------- 8. main ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/FS_pairs.csv",
                        help="CSV c колонками Test,frame,F_xx,F_xy,F_yx,F_yy,P_xx,P_xy,P_yy или C_xx,C_xy,C_yy,P_xx,P_xy,P_yy")
    parser.add_argument("--xi_stretch_dir", type=str, default=None,
                        help="Директория с xi_stretch файлами (если указана, будет использована вместо --csv)")
    parser.add_argument("--epochs", type=int, default=4000)
    parser.add_argument("--name", default="test")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                        help="Устройство для обучения: cpu / cuda / auto (по умолчанию)")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Размер батча для обучения (по умолчанию: 256)")
    parser.add_argument("--amp", action="store_true",
                        help="Включить mixed precision (AMP) при обучении на CUDA")
    parser.add_argument("--load_model", type=str, default=None,
                        help="Путь к .pth файлу: только построить графики, без обучения")
    parser.add_argument("--xi_csv", type=str, default=None,
                        help="CSV с xi1,xi2,xi3,dpsidksi* для включения dψ/dξ в лосс")
    parser.add_argument("--dpsi_weight", type=float, default=0.0, help="Вес слагаемого лосса по dψ/dξ")
    parser.add_argument("--anchor_SI_weight", type=float, default=0.0,
                        help="Вес якоря для напряжений при C=I: ||S(I)||^2")
    parser.add_argument("--anchor_dpsi0_weight", type=float, default=0.0,
                        help="Вес якоря для производных при ξ=0: ||∂ψ/∂ξ(0)||^2")
    parser.add_argument("--anchor_psi0_weight", type=float, default=0.0,
                        help="Вес нормировки энергии: ψ(0)=0")
    parser.add_argument("--visualisation_dpsi", type=str, default=None,
                        help="Путь к CSV файлу для визуализации производных (например, xi_stretch_1_1_01_txt.csv)")
    args = parser.parse_args()

    dir = Path("experiments") / args.name
    dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)
    
    # Загрузка данных
    if args.xi_stretch_dir:
        # Загружаем все xi_stretch файлы из директории
        stretch_dir = Path(args.xi_stretch_dir)
        all_data = []
        for f in stretch_dir.glob("xi_stretch_*.csv"):
            C, S, exp_types, plot_data, dpsidksi = load_xi_stretch_csv(f)
            all_data.append((C, S, exp_types, plot_data, dpsidksi))
        
        # Объединяем данные
        C = torch.cat([d[0] for d in all_data])
        Svec = torch.cat([d[1] for d in all_data])
        experiment_types = np.concatenate([d[2] if d[2] is not None else ['unknown']*len(d[0]) for d in all_data])
        plot_data = pd.concat([d[3] for d in all_data], ignore_index=True)
        dpsidksi = torch.cat([d[4] for d in all_data])
    else:
        C, Svec, experiment_types, plot_data, dpsidksi = load_csv(args.csv)

    # ---------- формируем данные для дополнительного loss по dψ/dξ ----------
    xi_data = None
    if args.dpsi_weight > 0.0:
        if args.xi_csv:
            xi_data = load_xi_csv(args.xi_csv)
        elif dpsidksi is not None:
            # Получаем xi напрямую из уже загруженного C
            with torch.no_grad():
                U = cholesky_upper(C)
                xi = vec_uvec(U)
            xi_data = (xi, dpsidksi)
        else:
            print("[WARN] dpsi_weight задан, но данные dpsidksi отсутствуют. Дополнительный loss отключён.")
            args.dpsi_weight = 0.0

    # Определяем устройство -----------------------------------------
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    # Если запрошена CUDA, но она недоступна — мягко откатываемся на CPU
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA недоступна (PyTorch собран без поддержки GPU или драйверы не установлены). Переходим на CPU.")
        device = "cpu"
    # Надёжно получаем имя GPU, если она действительно доступна
    _gpu_name = torch.cuda.get_device_name(0) if (device == "cuda" and torch.cuda.is_available()) else ""
    print("Используем устройство:", device, _gpu_name)
    print(f"Размер датасета: {len(C)} образцов")

    if args.load_model is None:
        net = search_width(C, Svec,
                           experiment_types=experiment_types,
                           split_by_experiments=False,
                           train_types=None,
                           val_types=None,  # автоматически возьмутся остальные типы
                           widths=[16],
                           epochs=args.epochs, batch=args.batch_size, device=device,
                           amp=args.amp,
                           xi_data=xi_data,
                           dpsi_weight=args.dpsi_weight,
                           anchor_SI_weight=args.anchor_SI_weight,
                           anchor_dpsi0_weight=args.anchor_dpsi0_weight,
                           anchor_psi0_weight=args.anchor_psi0_weight)
        torch.save(net.state_dict(), dir / (args.name + ".pth"))
        print("Model saved:", dir / (args.name + ".pth"))
    else:
        net_path = Path(args.load_model)
        net = load_trained_model(net_path, device)
        print("Loaded model:", net_path)

    # Добавляем предсказания модели в данные для визуализации
    stress_pred_xx, stress_pred_yy, stress_pred_xy = predict_stresses(net, C)
    plot_data['S_pred_xx'] = stress_pred_xx
    plot_data['S_pred_yy'] = stress_pred_yy
    plot_data['S_pred_xy'] = stress_pred_xy
    
    # Строим графики
    plot_curves(plot_data, name=dir)

    # -------- 8.2  ONNX  ξ→∂ψ/∂ξ  -----------------------
    onnx_path = dir / (args.name + ".onnx")
    wrapper = DerivWrapper1L(net.psi_net).to(device).eval()
    dummy_xi = torch.randn(1,3, device=device)
    torch.onnx.export(wrapper, dummy_xi, onnx_path,
                      input_names=["xi"], output_names=["dpsi_dxi"],
                      opset_version=11) 
    print("ONNX saved:", onnx_path)

    # -------- 8.3  ONNX  ξ→∂²ψ/∂ξ²  -----------------------
    hess_onnx_path = dir / (args.name + "_hessian.onnx")
    hessian_wrapper = HessianWrapper1L(net.psi_net).to(device).eval()
    torch.onnx.export(hessian_wrapper, dummy_xi, hess_onnx_path,
                      input_names=["xi"], output_names=["hessian"],
                      opset_version=11)
    print("ONNX (Hessian) saved:", hess_onnx_path)

    # ---------- Визуализация производных dψ/dξ --------------------
    if args.visualisation_dpsi:
        print(f"Запуск визуализации производных с данными из {args.visualisation_dpsi}")
        model_path = dir / (args.name + ".pth")
        output_dir = dir
        
        cmd = [
            sys.executable, 
            "./tests/test_wrapper_visualization.py",
            "--model", str(model_path),
            "--data", str(args.visualisation_dpsi),
            "--output_dir", str(output_dir),
            "--device", device
        ]
        
        try:
            subprocess.run(cmd, check=True)
            print("Визуализация производных завершена успешно")
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Ошибка при запуске визуализации производных: {e}")
        except FileNotFoundError:
            print(f"[ERROR] Не найден файл test_wrapper_visualization.py")



