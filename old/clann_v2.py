import torch, torch.nn as nn, torch.nn.functional as F
from torch.autograd import grad

# ───────────────────────── 1. Convex MLP ψ(ξ) ──────────────────────────
class ICNN(nn.Module):
    """
    Input-Convex NN с любой глубиной.
    Каждому слою k соответствуют:
        Wx_k : Linear(in_dim, hidden)
        Wz_k_pos : Linear(hidden, hidden)   (веса ≥0 через softplus)
    Завершающий слой: w_pos ≥0, b (скаляр).
    """
    def __init__(self, in_dim=3, hidden=64, layers=3, beta=10.0):
        super().__init__()
        self.beta = beta                              # крутизна softplus

        # первый "чисто-x" слой
        self.Wx0 = nn.Linear(in_dim, hidden, bias=True)

        # стек скрытых слоёв
        self.Wx = nn.ModuleList()
        self.Wz_raw = nn.ModuleList()                 # будут ≥0
        for _ in range(layers - 1):
            self.Wx.append(nn.Linear(in_dim, hidden, bias=True))
            self.Wz_raw.append(nn.Linear(hidden, hidden, bias=False))

        # финальный положительный вес
        self.w2_pos = nn.Parameter(torch.rand(hidden))
        self.b2     = nn.Parameter(torch.zeros(1))

    # --- вспомогательная "положительная" матрица ---
    @staticmethod
    def _pos(W_raw):                                  # ≥0 element-wise
        return F.softplus(W_raw, beta=10.0)

    # -------------------------------------------------------------------
    def forward(self, xi):                            # xi : (B,3)
        z = F.softplus(self.beta * self.Wx0(xi)) / self.beta   # smooth-ReLU

        for Wx_k, Wz_raw in zip(self.Wx, self.Wz_raw):
            z = F.softplus(self.beta * (Wx_k(xi) +
                          F.linear(z, self._pos(Wz_raw.weight)))) / self.beta

        psi = F.linear(z, self.w2_pos.abs()) + self.b2    # (B,1)
        return psi.squeeze(-1)                            # (B,)

# ───────────────────────── 2.  Кинематика (2-D) ─────────────────────────
def cholesky_upper(C):                       # (B,2,2) SPD → U (upper)
    L = torch.linalg.cholesky(C)             # lower
    return L.transpose(-2, -1)

def vec_upper_tri(U):                        # U → (u11,u22,u12)
    return torch.stack([U[...,0,0], U[...,1,1], U[...,0,1]], dim=-1)

def mat_from_vec(v):                         # (B,3) → верхний 2×2
    B = v.shape[0]
    M = v.new_zeros(B,2,2)
    M[:,0,0], M[:,1,1], M[:,0,1] = v[:,0], v[:,1], v[:,2]
    return M

# ───────────────────────── 3.  Модель CLANN ──────────────────────────
class CLANN(nn.Module):
    """
    forward(F)           : (B,2,2) → S (B,2,2), psi (B,)
    derivatives_from_xi  : (B,3)   → dψ/dξ (B,3)
    """
    def __init__(self, hidden=64, layers=3):
        super().__init__()
        self.psi_net = ICNN(in_dim=3, hidden=hidden, layers=layers)

    # ---------- режим обучения / post-proc  F → S ----------
    def forward(self, F):
        C  = torch.matmul(F.transpose(-2,-1), F)       # (B,2,2)
        U  = cholesky_upper(C)
        xi = vec_upper_tri(U)                          # (B,3)
        xi.requires_grad_(True)

        psi = self.psi_net(xi)                         # (B,)
        dpsi_dxi = grad(psi, xi,
                        grad_outputs=torch.ones_like(psi),
                        create_graph=True)[0]          # (B,3)
        Sigma = mat_from_vec(dpsi_dxi)                 # (B,2,2)

        Uinv_Sigma = torch.linalg.solve_triangular(U, Sigma, upper=True)
        S = 0.5 * (Uinv_Sigma + Uinv_Sigma.transpose(-2,-1))
        return S, psi

    # ---------- режим симулятора  ξ → ∂ψ/∂ξ ----------
    @torch.no_grad()
    def derivatives_from_xi(self, xi, create_graph=False):
        xi = xi.requires_grad_(create_graph)
        psi = self.psi_net(xi)
        dpsi_dxi = grad(psi, xi,
                        grad_outputs=torch.ones_like(psi),
                        create_graph=create_graph,
                        retain_graph=create_graph)[0]
        return dpsi_dxi            # (B,3)
    
# ───────────────────────── 4.  Ручные производные CLANN ──────────────────────────
class DerivWrapper1L(nn.Module):
    """ICNN c одним softplus-слоем → dψ/dξ (B,3) без autograd"""
    def __init__(self, base: ICNN, beta: float = 10.0):
        super().__init__()
        self.beta   = beta
        self.W1     = base.Wx0           # Linear 3→H  (weight, bias)
        self.w2_pos = base.w2_pos.abs()  # (H,)  уже ≥0
        self.b2     = base.b2            # скаляр (не нужен для ∂ψ/∂ξ)

    def forward(self, xi):               # xi : (B,3)
        s  = self.W1(xi)                               # (B,H)
        sig = torch.sigmoid(self.beta * s)             # σ′(βs) (B,H)
        g   = sig * self.w2_pos                       # (B,H) elem-wise
        dpsi_dxi = torch.matmul(g, self.W1.weight)     # (B,H)·(H,3)
        return dpsi_dxi                               # (B,3)

