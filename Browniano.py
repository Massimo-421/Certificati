import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. PARAMETRI DI BASE DEL MERCATO E TITOLI
# ==========================================
tickers = ["BMPS.MI", "BPE.MI", "BAMI.MI"]
# Valori Iniziali (Strike) presi dai Final Terms
S0 = np.array([8.248, 7.908, 10.368])

# Parametri stimati per il mercato bancario
vol = 0.35  # Volatilità 35% annualizzata
vols = np.array([vol, vol, vol])
corr = 0.70  # Correlazione fissa al 70%
r = 0.03  # Tasso risk-free stimato al 3%

# Matrice di Correlazione e Covarianza
corr_matrix = np.array([[1.0, corr, corr], [corr, 1.0, corr], [corr, corr, 1.0]])
cov_matrix = np.outer(vols, vols) * corr_matrix
L = np.linalg.cholesky(cov_matrix)  # Decomposizione di Cholesky per gli shock correlati

# ==========================================
# 2. PARAMETRI DELLA SIMULAZIONE
# ==========================================
N_sims = 100000  # Numero di scenari
N_months = 36  # Durata in mesi
dt = 1 / 12  # Step temporale (1 mese)

# Variabili per salvare i risultati
total_returns = []
durations = []
loss_count = 0
autocall_count = 0

np.random.seed(42)  # Per riproducibilità dei risultati

# ==========================================
# 3. MOTORE MONTE CARLO E PAYOFF
# ==========================================
print("Avvio simulazione Monte Carlo Standard...")
for _ in range(N_sims):
    S = S0.copy()
    unpaid_coupons = 0
    total_cash = 0
    duration = 36
    autocalled = False

    for m in range(1, 36 + 1):
        # Generazione shock casuali normali
        Z = np.random.normal(0, 1, len(tickers))

        # Moto Browniano Geometrico (GBM) per calcolare il nuovo prezzo
        S = S * np.exp((r - 0.5 * vols**2) * dt + np.sqrt(dt) * L.dot(Z))

        # Calcolo performance rispetto allo strike
        perfs = S / S0
        worst_perf = np.min(perfs)  # Meccanismo Worst-Of

        # Condizione Pagamento Cedola (Soglia 60%) + Effetto Memoria
        if worst_perf >= 0.60:
            coupon_paid = 1.33 + unpaid_coupons
            unpaid_coupons = 0
            total_cash += coupon_paid
        else:
            unpaid_coupons += 1.33

        # Condizione Autocall (Mesi 9-35, Step-Down decrescente)
        if m >= 9 and m < 36:
            autocall_level = 1.0 - (m - 9) * 0.01  # Scende dell'1% ogni mese
            if worst_perf >= autocall_level:
                total_cash += 100
                duration = m
                autocalled = True
                break

        # Scadenza a 36 Mesi (Barriera Europea al 60%)
        if m == 36:
            if worst_perf >= 0.60:
                total_cash += 100  # Capitale protetto
            else:
                total_cash += 100 * worst_perf  # Perdita sul capitale

    # Salva risultati del singolo scenario
    total_returns.append((total_cash - 100) / 100)
    durations.append(duration)
    if total_cash < 100:
        loss_count += 1
    if autocalled:
        autocall_count += 1

# ==========================================
# 4. STAMPA RISULTATI E GRAFICO
# ==========================================
mean_return = np.mean(total_returns)
print(f"Rendimento Medio: {mean_return:.2%}")
print(f"Probabilità di Perdita: {loss_count / N_sims:.2%}")
print(f"Probabilità di Autocall: {autocall_count / N_sims:.2%}")
print(f"Durata Media (mesi): {np.mean(durations):.1f}")

plt.hist(total_returns, bins=50, edgecolor="black", alpha=0.7)
plt.axvline(
    x=mean_return,
    color="green",
    linestyle="--",
    label=f"Rendimento Medio ({mean_return:.1%})",
)
plt.title("Monte Carlo Standard (100.000 scenari)")
plt.legend()
plt.show()
