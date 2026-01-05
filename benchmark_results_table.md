| Benchmark | Model | GraphEmbeddings | R² | MAE | RMSE | RMSLE | Num Features
| --- | --- | --- | --- | --- | --- | --- | --- |
| **airbnb** | DeepRegressor | **Yes** | 0.3475 | 50.16 | 71.12 | 0.4696 | 146 |
|  | DeepRegressor | No | 0.3064 | 53.48 | 73.33 | 0.4980 | 18 |
|  | LinearRegression | **Yes** | 0.2523 | 54.81 | 76.13 | 0.5329 |146 |
|  | LinearRegression | No | 0.2140 | 57.88 | 78.06 | 0.5401 | 18 |
| **chicago_crime** | DeepRegressor | Yes | 0.9650 | 7.20 | 10.38 | 0.5189 | 159 |
|  | DeepRegressor | **No** | 0.9710 | 5.69 | 9.44 | 0.3376 | 31 |
|  | LinearRegression | Yes | 0.9783 | 5.54 | 8.16 | 0.3411 | 159 |
|  | LinearRegression | **No** | 0.9783 | 5.43 | 8.17 | 0.3069 | 31 |
| **king_county** | DeepRegressor | Yes | 0.7723 | 57,484.81 | 78,540.41 | 0.1619 | 218 |
|  | DeepRegressor | **No** | 0.7793 | 55,863.03 | 77,325.97 | 0.1573 | 90 |
|  | LinearRegression | **Yes** | 0.7637 | 59,079.40 | 80,002.59 | 0.1738 | 218 |
|  | LinearRegression | No | 0.7629 | 59,351.87 | 80,149.93 | 0.1739 | 90 |
| **philadelphia_crime** | DeepRegressor | **Yes** | 0.8766 | 10.26 | 21.61 | 0.7016 | 158 |
|  | DeepRegressor | No | 0.7658 | 9.90 | 29.78 | 0.5523 | 30 |
|  | LinearRegression | Yes | 0.8740 | 8.69 | 21.85 | 0.5919 | 158 |
|  | LinearRegression | **No** | 0.8760 | 7.96 | 21.67 | 0.5056 | 30 |
| **san_francisco_crime** | DeepRegressor | Yes | 0.9587 | 30.84 | 69.93 | 0.45834 | 147 |
|  | DeepRegressor | **No** | 0.9609 | 30.40 | 68.02 | 0.5440 | 19 |
|  | LinearRegression | Yes | 0.9608 | 28.99 | 68.08 | 0.5782 | 147 |
|  | LinearRegression | **No** | 0.9634 | 24.07 | 65.83 | 0.3768 | 19 |
