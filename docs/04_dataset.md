# 04 - Dataset

## Name and source

**California Housing** (1990 U.S. census, block-group level).
- Loader documentation: <https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html>
- Original source: StatLib repository. The dataset was introduced by R. Kelley Pace and Ronald Barry,
  "Sparse Spatial Autoregressions", *Statistics & Probability Letters* 33 (1997) 291-297.

The dataset fits this project well. Pace and Barry built it to show **fast log-determinant computation**
for large spatial models, which is the same computational bottleneck that MEMe targets.

## License

The data is distributed through StatLib and scikit-learn for research and educational use. scikit-learn
does not state a specific license for the data. It is U.S. census-derived aggregate data with no personal
information.

## Size

- 20,640 rows (block groups), 8 numeric features, 1 continuous target.
- The download is a small compressed archive, far under the 200 MB limit. scikit-learn caches it locally
  after the first call.

## Features and target

| Feature | Meaning |
|---|---|
| MedInc | median income in block group |
| HouseAge | median house age |
| AveRooms | average rooms per household |
| AveBedrms | average bedrooms per household |
| Population | block-group population |
| AveOccup | average household members |
| Latitude | latitude |
| Longitude | longitude |

**Target:** `MedHouseVal`, the median house value in units of $100,000. It is capped at 5.0 in the
source data.

## Exact loading call

```python
from sklearn.datasets import fetch_california_housing
data = fetch_california_housing(as_frame=True)
X, y = data.data, data.target   # (20640, 8), (20640,)
```

## Preprocessing

1. Apply `log1p` to the heavy-tailed `Population`, `AveRooms`, `AveBedrms` and `AveOccup`. RBF kernels
   are sensitive to extreme outliers.
2. Standardise features with `StandardScaler` fitted on the **training split only**.
3. Centre and scale the target with the training mean and std for the GP. Predictions are mapped back to
   original units, and all reported metrics are in original units ($100k).
4. Rows are not removed. The 5.0 cap is documented as a limitation.

## Split plan

- Shuffle with `random_state=42`. Split into **70 % train / 10 % validation / 20 % test**, giving
  14,447 / 2,064 / 4,129 rows (as produced by `src/data.py`).
- **Log-det benchmark matrices:** a fixed random subset of n = 2000 training rows (seed 0). Exact
  eigenvalues act as ground truth.
- **GP training subset:** a separate fixed subset of n = 2000 training rows (seed 1). An exact GP on all
  14k rows would need a 1.6 GB matrix and minutes of Cholesky per grid point, so a subset keeps the full
  run under 15 minutes.
- **Validation split:** a sanity check of the selected hyper-parameters, reported in the metrics.
- **Test split:** used only once, for the final RMSE, MAE, R^2 and NLL of every model.
- **Baselines** (mean predictor, Ridge) are trained on the full 14,448-row training split. This is
  generous to the baselines.
