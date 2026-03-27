import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# =========================
# PARÂMETROS # Código em Python
# =========================
S0 = 48.02
K = 48
r = 0.05
sigma = 0.25
T = 1
N = 252


# =========================
# BLACK-SCHOLES
# =========================
def bs_call(S, K, T, r, sigma):
    if T <= 0:
        return max(S - K, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def delta_call(S, K, T, r, sigma):
    if T <= 0:
        return 1.0 if S > K else 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1)


# =========================
# SIMULAÇÃO DO HEDGE
# =========================
dt = T / N
S = S0

V0 = bs_call(S, K, T, r, sigma)
Delta = delta_call(S, K, T, r, sigma)
cash = V0 - Delta * S

S_path = [S]
portfolio_path = []

for t in range(1, N + 1):
    Z = np.random.normal()
    S = S * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z)

    tau = T - t * dt
    cash *= np.exp(r * dt)

    Delta_new = delta_call(S, K, tau, r, sigma)
    cash -= (Delta_new - Delta) * S
    Delta = Delta_new

    portfolio = cash + Delta * S

    S_path.append(S)
    portfolio_path.append(portfolio)

# =========================
# RESULTADOS
# =========================
portfolio = cash + Delta * S
payoff = max(S - K, 0)
error = portfolio - payoff

print(f"Preço da opção (Black-Scholes): R$ {V0:.2f}")
print(f"Erro de replicação: {error:.4e}")
