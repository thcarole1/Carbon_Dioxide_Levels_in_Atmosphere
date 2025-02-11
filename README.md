# Carbon_Dioxide_Levels_in_Atmosphere

The carbon dioxide record from Mauna Loa Observatory, known as the “Keeling Curve,” is the world’s longest unbroken record of atmospheric carbon dioxide concentrations.<br>
Scientists make atmospheric measurements in remote locations to sample air that is representative of a large volume of Earth’s atmosphere and relatively free from local influences.<br>

**Source Kaggle** : https://www.kaggle.com/datasets/ucsandiego/carbon-dioxide

## Framing the problem
### What could be the business objectives?
#### Regulatory Compliance and Reporting
•	**Objective**: Help organizations meet emission standards and comply with regulations. <br><br>
•	**Applications**:

- Forecast emissions trends to ensure compliance with future regulations.
- Generate reports for governments or stakeholders showcasing emission reduction efforts.
- Aid in carbon trading by quantifying emission reductions.

### What is the current solution (if it exists) ?
Predicting carbon dioxide (CO₂) levels is a critical task for environmental monitoring and climate action.<br><br>
Based on the data at hand, which likely includes historical CO₂ emissions and time, several existing solutions and approaches can be applied to predict CO₂ levels. <br><br>
Here’s an overview:

#### Statistical and Classical Time Series Models
These models are well-suited for datasets with clear temporal patterns and trends:
##### Autoregressive Integrated Moving Average (ARIMA)
•	**Description**: A classical time series model used to predict CO₂ levels based on past values.<br>
•	**Use Case**: Suitable for linear trends and seasonality in CO₂ emissions.<br><br>
•	**Advantages**:
- Good for stationary data.
- Provides interpretable coefficients.<br>
•	**Challenges**: May not handle complex patterns or external factors well.

#### Deep Learning Models
Deep learning methods are powerful for capturing complex patterns in large datasets.
##### Long Short-Term Memory Networks (LSTMs)
• **Description**: A type of recurrent neural network (RNN) designed for sequential data, ideal for time series forecasting.<br>
•	**Use Case**: Predicts future CO₂ levels by learning long-term dependencies in emission trends.<br>
•	**Advantages**:
- Handles temporal dependencies well.
- Works with non-linear trends.<br>
•	**Challenges**: Computationally intensive and requires large datasets.

### What is the type of learning ?
With the gathered information, we are ready to design our system.<br>
This is a **time series forecasting** task.


## Selecting a performance measure
The next step is to choose a performance measure: <br>
•	**Short-term predictions**: Use MAE, RMSE, or MASE for clear and interpretable results.<br>
•	**Scale-independent metrics**: Use MAPE, sMAPE, or Theil’s U for datasets with varying scales or multiple regions.

We'll focus on:

### Mean Absolute Error (MAE)

•	**Description**: The average of absolute differences between actual and predicted values.<br><br>
•	**Formula**: <br>
$$MAE  = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$$

•	**Use Case**:
- Easy to interpret as it represents the average error in the same unit as the data (e.g., metric tons of CO₂).
- Sensitive to the scale of the data but not overly penalized by large errors.


### Mean Squared Error (MSE)
•	**Description**: The average of squared differences between actual and predicted values.<br><br>
•	**Formula**:<br>
$$MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$
•	**Use Case**:
- Penalizes large errors more heavily than MAE.
- Useful for models where large prediction deviations are particularly undesirable.

### Root Mean Squared Error (RMSE)
•	**Description**: The square root of MSE, representing the error in the same units as the original data.<br><br>
•	**Formula**:<br>
 $$RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$
 •	**Use Case**:
- Highlights significant prediction errors.
- Popular in time series forecasting due to interpretability.

### Mean Absolute Percentage Error (MAPE)
•	**Description**: The average percentage error between actual and predicted values.<br><br>
•	**Formula**: <br>
$$MAPE=\frac{1}{n} \sum_{i=1}^{n} \left| \frac{y_i - \hat{y}_i}{y_i} \right| \times 100 $$

•	**Use Case**:
- Provides a percentage-based measure, making it scale-independent.
- Becomes problematic when actual values are near zero (dividing by zero).
