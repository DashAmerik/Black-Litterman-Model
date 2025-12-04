import yfinance as yf
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

#define assets
assets = ['AAPL','MSFT','AMZN','GOOGL','META','NVDA','TSLA','BRK-B','JPM','V',
    'JNJ','WMT','PG','MA','HD','XOM','BAC','CVX','ABBV','AVGO',
    'COST','ADBE','KO','PEP','PFE','TMO','CSCO','MRK','ABT','ACN',
    'CRM','DIS','NFLX','INTC','AMD','MCD','NKE','TXN','LIN','UNH',
    'VZ','QCOM','UPS','RTX','IBM','AMAT','CAT','LOW','HON','ORCL']

num_assets = len(assets)

#download data and compute returns
data = yf.download(assets, start="2022-01-01", end="2025-01-01")['Close']
daily_returns = data.pct_change().dropna()

#annualise the statrs
mu_hist = daily_returns.mean() * 252      # historical annualised mean returns
cov_annual = daily_returns.cov() * 252    # annualised covariance matrix

#get the risk free rate
risk_free_rate = yf.Ticker("^TNX").history(period="1d")["Close"].iloc[-1] / 100


#define the market portfolio
market_caps = []

#get the market cap for each ticker
for t in assets:
    info = yf.Ticker(t).info
    mc = info.get("marketCap", None)
    market_caps.append(mc)

#convert market cap into array of floats for better mathematical operations
market_caps = np.array(market_caps, dtype=float)


#check if there are any missing market caps, if there are use equal-weighted assets
if np.any(np.isnan(market_caps)) or np.any(market_caps <= 0):
    print("Some market caps missing. Using equal-weight as market portfolio.")
    w_mkt = np.ones(num_assets) / num_assets
else:
    w_mkt = market_caps / market_caps.sum()

#computes expected return of market portfolio from historical mean
market_return = np.dot(mu_hist, w_mkt) #sum of expected return * market weight over all assets
#computes variance of the market portfolio (@ is for matrix multiplication)
market_var = w_mkt.T @ cov_annual @ w_mkt  #market weights as a row vector * covariance matrix * market weights
#delta is the risk-aversion parameter, if high then market is risk averse, if low market is comfortable with risk
delta = (market_return - risk_free_rate) / market_var

#pi is the vector of "implied expected excess returns"
pi = delta * cov_annual @ w_mkt

#tau is how uncertain we think the markets implied returns are, sensible default
tau = 0.05
tauSigma = tau * cov_annual #scaled-down risk representing how uncertain we think the markets returns are



# Example view: "AAPL is expected to outperform MSFT by 2% per year"
K = 1  # number of views
P = np.zeros((K, num_assets))

idx_aapl = assets.index('AAPL')
idx_msft = assets.index('MSFT')

# Relative view: r_AAPL - r_MSFT = 2%
P[0, idx_aapl] = 1
P[0, idx_msft] = -1

Q = np.array([0.02])  #2% per year outperformance

# View uncertainty Ω (K x K), standard choice:
Omega = np.diag(np.diag(P @ cov_annual @ P.T)) # builds a diagonal matrix out of variances

#computes inverse matrices
inv_tauSigma = np.linalg.inv(tauSigma)
inv_Omega = np.linalg.inv(Omega)

#values for posterior mean formula
middle = inv_tauSigma + P.T @ inv_Omega @ P
right = inv_tauSigma @ pi + P.T @ inv_Omega @ Q

mu_bl = np.linalg.inv(middle) @ right    # posterior excess returns (N,)


def portfolio_performance_bl(weights, bl_excess_returns, cov_matrix):
    portfolio_excess_return = np.dot(bl_excess_returns, weights)
    portfolio_volatility = np.sqrt(weights.T @ cov_matrix @ weights)
    return portfolio_excess_return, portfolio_volatility

def negative_sharpe_bl(weights, bl_excess_returns, cov_matrix):
    excess_return, vol = portfolio_performance_bl(weights, bl_excess_returns, cov_matrix)
    return -(excess_return / vol)

def weight_constraint(weights):
    return np.sum(weights) - 1

bounds = tuple((0, 1) for _ in range(num_assets)) #bound for not short selling
constraints = ({'type': 'eq', 'fun': weight_constraint})

# Start from equal weights
weights_init_bl = np.ones(num_assets) / num_assets

result_bl = minimize(   #finds the asset weights that maximise sharpe ratio subject to constraints
    negative_sharpe_bl,
    weights_init_bl,
    args=(mu_bl, cov_annual),
    method='SLSQP',
    bounds=bounds,
    constraints=constraints
)

bl_weights = result_bl.x #get our optimized weights
bl_weights_percent = bl_weights * 100 #save them as percentages

bl_excess_return, bl_volatility = portfolio_performance_bl(bl_weights, mu_bl, cov_annual)
bl_sharpe = bl_excess_return / bl_volatility #get our sharpe ratio

print("\n Black–Litterman Portfolio")
print("BL Optimized Weights (in %):")
for i, weight in enumerate(bl_weights_percent):
    print(f"{assets[i]}: {weight:.2f}%")

print("\nBL Weights Sum Check:", np.sum(bl_weights_percent))
print("BL Expected Excess Return:", bl_excess_return)
print("BL Volatility:", bl_volatility)
print("BL Sharpe Ratio:", bl_sharpe)


#plot weights
plt.figure(figsize=(14, 6))
x = np.arange(num_assets)
plt.bar(x, bl_weights_percent)
plt.xticks(x, assets, rotation=90)
plt.ylabel("% Portfolio Allocation")
plt.title("Black–Litterman Optimised Portfolio Weights")
plt.tight_layout()
plt.show()
