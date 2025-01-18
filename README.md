# Stock Price Prediction Using Recurrent Neural Networks (RNNs)

## Overview
This project predicts the next day's stock opening price using the past three days' stock features (Open, High, Low prices, and Volume) with a Recurrent Neural Network (RNN) architecture. The solution involves feature engineering, data preprocessing, and implementing RNNs with Gated Recurrent Units (GRUs).

---

## Objective
- To predict the opening stock price for the next day using time-series data.
- To evaluate model performance using Mean Squared Error (MSE) and visualizations.

---

## Dataset

### Description
The dataset consists of daily stock prices with the following features:
- **Open**: Opening stock price
- **High**: Highest price of the day
- **Low**: Lowest price of the day
- **Volume**: Total trading volume for the day

### Feature Engineering
- A new dataset is created with 12 features (last three days' Open, High, Low prices, and Volume) and the target variable as the next day’s opening price.

### Data Splits
- Training Dataset: 70%
- Test Dataset: 30%

---

## Methodology

### Preprocessing
1. **Feature Selection**:
   - Extract relevant columns (Open, High, Low, Volume) from the raw dataset.
2. **Feature Engineering**:
   - Convert 2D data into a 1D feature vector for the last three days.
3. **Scaling**:
   - Apply Min-Max Scaling to normalize the features between 0 and 1 for improved model performance.
   - Save the scaling model for consistency during testing.
4. **Data Splitting**:
   - Train-test split (70%-30%) using `train_test_split` from scikit-learn.
5. **Saving Data**:
   - Save processed data into `train_data_RNN.csv` and `test_data_RNN.csv`.

---

### Model Architecture
The Recurrent Neural Network (RNN) employs GRUs to capture temporal dependencies in the data:
1. **Input Layer**:
   - Input shape: `(batch_size, 12, 1)` representing 12 features over time.
2. **GRU Layers**:
   - Multiple GRU layers with 128 and 254 units to learn temporal dependencies.
   - Return sequences enabled for stacked GRUs.
3. **Dropout Layers**:
   - Dropout of 0.2 to reduce overfitting.
4. **Output Layer**:
   - Dense layer with one neuron for regression.

### Model Training
- **Loss Function**: Huber Loss
- **Optimizer**: Adam
- **Metrics**: Mean Squared Error (MSE)
- **Epochs**: 100
- **Batch Size**: 32

---

### Model Evaluation
- **Loss and MSE**:
   - Evaluated using the test dataset.
- **Visual Comparison**:
   - Compare actual vs. predicted stock prices.
   - Partial dataset plots (first 100 points) for better visualization.

---

## Results
- **Test Set MSE**: 18.42
- The model successfully predicts stock prices with a low MSE, achieving close proximity to real values.

---

## Code Structure
1. **`train_RNN.py`**:
   - Creates training and testing datasets.
   - Builds and trains the RNN model using GRUs.
   - Saves the trained model and Min-Max scaling object.
2. **`test_RNN.py`**:
   - Loads the pre-trained model and scaling object.
   - Preprocesses test data.
   - Evaluates the model on the test set and visualizes results.

---

## Future Work
- Experiment with advanced architectures like LSTMs, Transformer-based models, or Attention mechanisms.
- Integrate additional features like moving averages or industry indicators.
- Perform hyperparameter tuning for optimal architecture and training configuration.


