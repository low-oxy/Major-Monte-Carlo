# Monte Carlo Portfolio Risk Simulator

A GPU-accelerated Monte Carlo simulator that models the future value of a stock
portfolio and quantifies downside risk using **Value at Risk (VaR)** and
**Conditional Value at Risk (CVaR)** — the same style of risk metrics used by
institutional risk desks.

Built on a basket of major NSE (Indian stock exchange) large-caps as a working
example, but works with any set of tickers supported by Yahoo Finance.

## What it does

1. Pulls 1 year of historical daily prices for a set of stocks via `yfinance`
2. Computes each stock's mean daily return and the covariance matrix between them
3. Runs `N` (default 50,000) simulated future price paths using **correlated
   Geometric Brownian Motion** — random daily shocks are correlated across
   assets via Cholesky decomposition of the covariance matrix, so the
   simulation respects how these stocks actually move together, not just
   their individual volatility in isolation
4. Aggregates simulated paths into portfolio value over time, using
   randomly-generated (but normalized) portfolio weights
5. Reports:
   - **Median outcome** — the 50th-percentile simulated result
   - **95% VaR** — the loss threshold you'd expect to exceed only 5% of the time
   - **95% CVaR (Expected Shortfall)** — the average loss *given* that you land
     in that worst 5% tail
   - **Probability of loss** — the share of simulations ending below the
     initial investment

## Example output

```
--- Risk Report (100-day horizon, 50,000 simulations) ---
Initial Investment:      ₹1,000,000
Median Outcome:          ₹964,602  (-3.54%)
95% VaR:                 ₹832,179  (-16.78%)
95% CVaR (Exp. Shortfall): ₹803,010
95th Percentile (upside): +11.71%
Probability of Loss:     65.6%
```

## Why GPU acceleration

At 50,000 simulations × 100 time steps × 8 assets, the simulation is fully
vectorized as batched tensor operations (`Z @ L.T`, matrix-vector products,
cumulative products) rather than a Python `for` loop over simulations. This
lets the entire Monte Carlo run execute as a handful of large GPU kernel
calls via [CuPy](https://cupy.dev/) — a NumPy-compatible array library — instead
of tens of thousands of small sequential operations.

The script automatically detects whether CuPy is available and falls back to
plain NumPy on CPU-only machines, so it runs anywhere — just faster with a GPU.

## Installation

```bash
git clone https://github.com/<your-username>/portfolio-risk-simulator.git
cd portfolio-risk-simulator
pip install -r requirements.txt
```

For GPU acceleration, additionally install CuPy matching your CUDA version
(see comments in `requirements.txt`), e.g.:

```bash
pip install cupy-cuda13x
```

## Usage

```bash
pip install --upgrade yfinance
python monte_carlo_sim.py
```

## Method notes

- Daily returns are modeled as **normally distributed and correlated**
  across assets (multivariate Geometric Brownian Motion) — a standard
  simplifying assumption for this class of simulation. Real markets exhibit
  fatter tails and volatility clustering that this model doesn't capture.
- Mean returns and covariance are estimated from a **trailing 1-year window**
  of historical data and held constant across the simulated horizon; they
  are not forecasts and will shift with a different lookback period.
- Portfolio weights in the example are randomly generated for demonstration.
  A natural extension is mean-variance (Markowitz) optimization to select
  weights that maximize return for a given risk level — see **Future Work**.
## Requirements

- Python 3.10+
- See `requirements.txt`. GPU support requires an NVIDIA GPU with a matching
  CUDA driver (CUDA 12.x or 13.x) — see [CuPy installation
  docs](https://docs.cupy.dev/en/stable/install.html) for details.

## License

MIT
