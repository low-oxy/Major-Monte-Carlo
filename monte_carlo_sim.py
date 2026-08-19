#Importing dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import yfinance as yf
import cupy as cp

def get_data(stocks, start, end):
    """
    parametrs:stocks, start(start date), end(end date)
    return: covariance matrix, mean_returns
    """
    stockData = yf.download(stocks, start=start, end=end, auto_adjust=True)
    stockData = stockData['Close']
    #percentage change from one day to the next
    returns = stockData.pct_change().dropna()
    mean_returns = returns.mean()
    #extracting the covariance matrix           
    cov_matrix = returns.cov()    
    return cov_matrix, mean_returns

#The list of stocks to be used in the simulation
stocklist = ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK', 'HINDUNILVR', 'KOTAKBANK', 'SBIN']
stocks = [stock + '.NS' for stock in stocklist]
endDate = dt.datetime.now()
startDate = endDate - dt.timedelta(days=365)
cov_matrix_cpu, mean_returns_cpu = get_data(stocks, startDate, endDate)
print("Mean returns:\n", mean_returns_cpu)
rng = cp.random.default_rng()
weights = rng.random(len(mean_returns_cpu))
weights /= cp.sum(weights)

#Monet Carlo Simulation
#number of simulations
mc_sims = 50000
#how many time frames to simulate
time_frame = 100
# transfer values to GPU
mean_returns = cp.asarray(mean_returns_cpu.values)
cov_matrix = cp.asarray(cov_matrix_cpu.values)
initial_investment = 1000000

#Calculating the simulated daily returns with help of Cholesky decomposition to introduce correlation between the stocks
Z = rng.standard_normal(size=(mc_sims, time_frame, len(weights)))
L = cp.linalg.cholesky(cov_matrix)
correlated_shocks = Z@L.T
daily_returns = mean_returns + correlated_shocks
portfolio_daily_returns = daily_returns @ weights
portfolio_sims = cp.cumprod(portfolio_daily_returns + 1, axis=1) * initial_investment
#plotting the simulations
portfolio_sims_cpu = cp.asnumpy(portfolio_sims).T

#Plotting the Monte Carlo simulations
plt.plot(portfolio_sims_cpu, color='blue', alpha=0.05)
plt.ylabel('Portfolio Value')
plt.xlabel('Time Frame')
plt.title('Monte Carlo Simulation of Portfolio Value')
plt.show()

#Calculating percentiles for the simulations
percentiles = cp.percentile(portfolio_sims, cp.array([5, 25, 50, 75, 95]), axis=0)
p5, p25, p50, p75, p95 = cp.asnumpy(percentiles)
plt.figure(figsize=(10, 6))
plt.fill_between(range(time_frame), p5, p95, alpha=0.2, color='blue', label='5th-95th percentile')
plt.fill_between(range(time_frame), p25, p75, alpha=0.4, color='blue', label='25th-75th percentile')
plt.plot(p50, color='blue', linewidth=2, label='Median')
plt.ylabel('Portfolio Value')
plt.xlabel('Time Frame')
plt.title('Monte Carlo Simulation of Portfolio Value')
plt.legend()
plt.show()

#Calculating final portfolio values and plotting histogram
final_values = cp.asnumpy(portfolio_sims[:, -1])
plt.figure(figsize=(10, 6))
plt.hist(final_values, bins=100, color='blue', alpha=0.7)
plt.axvline(np.median(final_values), color='red', linestyle='--', label=f'Median: {np.median(final_values):,.0f}')
plt.xlabel('Final Portfolio Value')
plt.ylabel('Frequency')
plt.title(f'Distribution of Final Portfolio Value after {time_frame} days')
plt.legend()
plt.show()

#Calculating Value at Risk (VaR) and Conditional Value at Risk (CVaR)
returns_pct = (final_values - initial_investment) / initial_investment * 100
VaR_95 = np.percentile(final_values, 5)  # 5th percentile = 95% VaR
CVaR_95 = final_values[final_values <= VaR_95].mean()  # average of the worst 5% outcomes
print(f"Initial Investment: ₹{initial_investment:,.0f}")
print(f"Median return: {np.median(returns_pct):.2f}%")
print(f"Median outcome after {time_frame} days: ₹{np.median(final_values):,.0f}")
print(f"95% VaR: {np.percentile(returns_pct, 5):.2f}%  (5% chance of losing more than this)")
print(f"95% VaR: ₹{VaR_95:,.0f} (5% chance of ending below this)")
print(f"Best case (95th pct): {np.percentile(returns_pct, 95):.2f}%")
print(f"Worst case (5th pct): {np.percentile(returns_pct, 5):.2f}%")
print(f"95% CVaR (Expected Shortfall): ₹{CVaR_95:,.0f}")
print(f"Probability of loss: {(final_values < initial_investment).mean()*100:.1f}%")