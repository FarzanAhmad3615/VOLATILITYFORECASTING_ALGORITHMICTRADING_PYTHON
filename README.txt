# VOLATILITYFORECASTING_ALGORITHMICTRADING_PYTHON
This README uses the **STAR** (Situation, Task, Action, Result) method to document your stock price prediction project based on the code provided and the resulting output.

---

# Multi-Model Stock Price Forecasting Project

Situation

Financial markets exhibit highly non-linear and stochastic behavior, making price prediction a significant challenge. To address this, I developed a comparative analysis framework to evaluate how traditional statistical models (ARIMA, VARMAX), volatility-focused models (GARCH), and deep learning architectures (LSTM) perform on the same time-series dataset.

Task

The primary objective was to build an end-to-end Python pipeline to:

* Ingest and preprocess historical stock data (Close prices and percentage returns).
* Implement and optimize four distinct modeling approaches.
* Resolve critical numerical stability issues (such as `LinAlgError` in state-space models).
* Evaluate performance using standard metrics (MAE, RMSE, MAPE) and visual trend comparison.

ActioN

I implemented several advanced data engineering and modeling techniques to ensure robust results:

* **Numerical Stability:** Fixed `LinAlgError` in the **VARMAX** model by implementing a `StandardScaler` to harmonize features with different magnitudes and using `ssm.initialize_diffuse()` to bypass unstable matrix decompositions.
* **Statistical Modeling:** Configured an **ARIMA(5,1,0)** model for univariate forecasting and a **GARCH(1,1)** model using the `arch` library to analyze return volatility.
* **Deep Learning:** Developed an **LSTM** network using TensorFlow/Keras, utilizing a sliding window sequence generator (`seq_len=5`) and `MinMaxScaler` for optimized gradient descent.
* **Robust Optimization:** Switched to the `powell` optimizer for multivariate convergence and implemented custom inverse-transform logic to ensure all predictions were evaluated on the original price scale.

Result

The comparative analysis yielded the following performance outcomes (as seen in the project output):

| Model | MAE | RMSE | MAPE |
| --- | --- | --- | --- |
| **LSTM** | **1.4552** | **1.7973** | **1.4533%** |
| **ARIMA** | 2.8053 | 3.6023 | 2.7842% |
| **VARMAX** | 4.8912 | 6.0239 | 4.8598% |

**Key Findings:**

* **LSTM Superiority:** The LSTM model was the clear winner, successfully capturing the non-linear "up-trend" in the latter half of the test set, whereas traditional models (ARIMA/VARMAX) tended to revert to a mean or flat-line forecast.
* **GARCH Insights:** Effectively modeled volatility on returns, though it is fundamentally designed for risk assessment rather than direct price level prediction.

---

### **How to Run**

1. Ensure `stock_data.csv` is in the directory with a `Date` and `Close` column.
2. Install dependencies: `pip install pandas numpy matplotlib statsmodels arch scikit-learn tensorflow`.
3. Execute the script to generate the **Model Comparison Plot**.
<img width="1195" height="638" alt="Screenshot 2026-01-01 154140" src="https://github.com/user-attachments/assets/ede44b9d-713a-4c71-9d5c-62129b12e8af" />
