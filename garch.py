import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. SETUP SIMULAZIONE E PARAMETRI GARCH
# ==========================================
N_sims = 50000  # Scenari paralleli
N_days_per_month = 21  # Giorni lavorativi in un mese
N_months = 36
N_days = N_months * N_days_per_month
dt = 1 / 252  # Step temporale (1 giorno)

# Valori Iniziali e Tassi
S0 = np.array([8.248, 7.908, 10.368])  # BMPS, BPER, BAMI
r = 0.03

# Parametri GARCH(1,1)
long_run_var = (0.35**2) / 252  # Varianza giornaliera di lungo periodo (vol 35%)
alpha = 0.12  # Reattività agli shock
beta = 0.83  # Persistenza della volatilità
omega = long_run_var * (1 - alpha - beta)

# Matrice di Correlazione fissa (Constant Conditional Correlation)
corr = 0.70
corr_matrix = np.array([[1.0, corr, corr], [corr, 1.0, corr], [corr, corr, 1.0]])
L = np.linalg.cholesky(corr_matrix)

# Inizializzazione Vettori (50.000 scenari paralleli per 3 titoli)
prices = np.tile(S0, (N_sims, 1))
variances = np.full((N_sims, 3), long_run_var)
monthly_prices = np.zeros((N_sims, N_months, 3))  # Salva i prezzi a fine mese

np.random.seed(42)

# ==========================================
# 2. MOTORE GARCH: GENERAZIONE PATHS GIORNALIERI
# ==========================================
print("Generazione percorsi GARCH (giornalieri)...")
for t in range(1, N_days + 1):
    # Shock casuali non correlati
    Z = np.random.normal(0, 1, (N_sims, 3))
    # Applico Cholesky per correlare gli shock
    Z_corr = Z.dot(L.T)

    # Calcolo volatilità del giorno corrente
    vols = np.sqrt(variances)
    drift = r * dt - 0.5 * variances
    epsilon = vols * Z_corr  # Shock effettivo della giornata

    # 1. Aggiornamento Prezzi
    prices = prices * np.exp(drift + epsilon)

    # 2. Aggiornamento Varianza GARCH per il giorno successivo
    variances = omega + alpha * (epsilon**2) + beta * variances

    # Registra i prezzi solo alla fine del mese (ogni 21 giorni)
    if t % N_days_per_month == 0:
        month_idx = (t // N_days_per_month) - 1
        monthly_prices[:, month_idx, :] = prices

# ==========================================
# 3. VALUTAZIONE DEL CERTIFICATO (Sui prezzi mensili)
# ==========================================
print("Valutazione del payoff del certificato...")
total_returns = np.zeros(N_sims)
durations = np.full(N_sims, 36)
loss_count = 0
autocall_count = 0

for i in range(N_sims):
    unpaid_coupons = 0
    total_cash = 0
    duration = 36
    autocalled = False

    path = monthly_prices[i]

    for m in range(1, 36 + 1):
        perfs = path[m - 1] / S0
        worst_perf = np.min(perfs)

        # Osservazione Cedola
        if worst_perf >= 0.60:
            coupon_paid = 1.33 + unpaid_coupons
            unpaid_coupons = 0
            total_cash += coupon_paid
        else:
            unpaid_coupons += 1.33

        # Osservazione Autocall (Mesi 9-35, Step-Down decrescente)
        if m >= 9 and m < 36:
            autocall_level = 1.0 - (m - 9) * 0.01
            if worst_perf >= autocall_level:
                total_cash += 100
                duration = m
                autocalled = True
                break

        # Scadenza a 36 mesi
        if m == 36:
            if worst_perf >= 0.60:
                total_cash += 100
            else:
                total_cash += 100 * worst_perf

    total_returns[i] = (total_cash - 100) / 100
    durations[i] = duration
    if total_cash < 100:
        loss_count += 1
    if autocalled:
        autocall_count += 1

# ==========================================
# 4. STAMPA RISULTATI
# ==========================================
mean_return = np.mean(total_returns)
print(f"\n--- RISULTATI GARCH(1,1) SU {N_sims} SCENARI ---")
print(f"Rendimento Medio: {mean_return:.2%}")
print(f"Probabilità di Perdita: {loss_count / N_sims:.2%}")
print(f"Probabilità di Autocall: {autocall_count / N_sims:.2%}")
print(f"Durata Media (mesi): {np.mean(durations):.1f}")

plt.hist(total_returns, bins=50, edgecolor="black", alpha=0.7, color="darkorange")
plt.axvline(
    x=mean_return,
    color="green",
    linestyle="--",
    label=f"Rendimento Medio ({mean_return:.1%})",
)
plt.title("GARCH(1,1) con Volatilità Dinamica")
plt.legend()
plt.show()
