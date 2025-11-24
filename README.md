# Ethereum Price Prediction using LSTM

**Forecasting Ethereum (ETH-USD) trends using Long Short-Term Memory (LSTM)**

[Open in Colab](https://colab.research.google.com/github/Jarukit-Jack/project_deep_learningvv/blob/main/deep_learn_eth.ipynb)

---

##  Project Overview

This project implements a Long Short-Term Memory (LSTM) neural network to predict Ethereum prices. It utilizes historical price data fetched via the `yfinance` library to train a time-series forecasting model. This project demonstrates end-to-end deep learning workflow capability, from data ingestion to model deployment and visualization. 

###  Key Features

-  Real-Time Data Extraction: Fetches live ETH-USD data directly from Yahoo Finance APIs.
-  Robust Architecture: Implements a 2-layer LSTM with Dropout regularization (0.3) to prevent overfitting.
-  Data Preprocessing: Utilizes MinMaxScaler for optimal time-series normalization.
-  Sequence Prediction: Forecasts future prices based on a sliding window of 60 days.
-  Performance Evaluation: Includes comprehensive visualization and metric calculation (MSE, RMSE, MAE).

---

##  Model Architecture

```
Input: (batch_size, sequence_length=60, features=5)
   ↓
LSTM Layer 1 (hidden_dim=64, dropout=0.3)
   ↓
LSTM Layer 2 (hidden_dim=64, dropout=0.3)
   ↓
Dropout (p=0.3)
   ↓
Fully Connected Layer (64 → 1)
   ↓
Output: Predicted Price
```

###  Hyperparameters

| Parameter | Value |
|-----------|-------|
| Input Dimension | 5 (Open, High, Low, Close, Volume) |
| Hidden Dimension | 64 |
| Number of Layers | 2 |
| Dropout Rate | 0.3 |
| Learning Rate | 0.00001 |
| Batch Size | 256 |
| Max Epochs | 750 |
| Sequence Length | 60 days |
| Train/Test Split | 80/20 |

---

##  Installation

### Method 1: Using UV (Recommended)

```bash
# Install UV (if not already installed)
curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh

# Install dependencies
uv pip install -r requirements.txt
```

### Method 2: Standard pip

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

---

##  Usage

### 1. เปิด Jupyter Notebook

```bash
jupyter notebook deep_learn_eth.ipynb
```

### 2. หรือรันใน Google Colab

Click the "Open in Colab" badge at the top of this README to execute the notebook in the cloud.

### 3. Project Structure

```
project_deep_learning/
│
├── deep_learn_eth.ipynb          
├── requirements.txt           
├── README.md                    
├── best_lstm_model.pth         
├── LSTM_model_architecture.png 
└── LSTM_model_architecture     
```

---

## 📊 Workflow

### StepA: Dependency Setup
Installation of core libraries: `torch`, `yfinance`, `tqdm`, `torchvision`, `torchsummary`

### StepB: Data Preparation
1. Data Extraction: Fetch ETH-USD data from Yahoo Finance.
2. Normalization: Apply MinMaxScaler to scale data between 0 and 1.
3. Sequencing: Create sliding window sequences (60 days history).
4. Splitting: Divide data into Training (80%) and Testing (20%) sets.

### StepC: Model Design
Construction of the LSTM architecture:
- Stacked LSTM layers (2 layers).
- Dropout layers
- Fully Connected output layer

### StepD: Configuration
- Loss Function: MSE (Mean Squared Error)
- Optimizer: Adam Optimizer.
- Loaders: Batched DataLoaders for training and validation.

### StepE: Training
- Training loop implementation with Early Stopping.
- Automated saving of the best model weights.
- Real-time tracking of Training and Validation Loss.

### StepF: Evaluation
- Inference on the Test Set.
- Metric calculation: MSE, MAE, RMSE, MAPE
- Visualization: Plotting Predicted Prices vs. Actual Prices.

---

##  Results

The model successfully identifies price trends using the following input features:
- **Open Price**: ราคาเปิด
- **High Price**: ราคาสูงสุด
- **Low Price**: ราคาต่ำสุด
- **Close Price**: ราคาปิด (Target)
- **Volume**: ปริมาณการซื้อขาย

---

##  Tech Stack

- **PyTorch**: Framework สำหรับ Deep Learning
- **yfinance**: ดึงข้อมูลการเงินจาก Yahoo Finance
- **NumPy & Pandas**: การจัดการข้อมูล
- **scikit-learn**: MinMaxScaler และ metrics
- **Matplotlib**: การแสดงผล
- **TQDM**: Progress bar
- **TorchSummary**: แสดงสถาปัตยกรรมโมเดล

---

##  References

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [yfinance Documentation](https://pypi.org/project/yfinance/)
- [Time Series Forecasting Guide](https://pytorch.org/tutorials/beginner/transformer_tutorial.html)

---

##  Authors

<table>
  <tr>
    <td align="center">
      <b>Parinya Aobaun</b><br>
      std_id: 6610502145
    </td>
    <td align="center">
      <b>Jarukit Phonwattananuwong</b><br>
      std_id 6610505306
    </td>
  </tr>
</table>

---

##  Academic Context

This project is submitted as part of the curriculum for:

Deep Learning (01204466-65)
Semester 1, Academic Year 2025

Kasetsart University, Bang Khen Campus
Instructor: Asst. Prof. Dr. Paruj Rattanaworabhan

---

##  License

This project is created for educational purposes.

---

##  Contribution
Contributions are welcome! If you find an issue or want to improve the model:

1. Fork the project.
2. Create a Feature Branch (git checkout -b feature/AmazingFeature).
3. Commit your changes (git commit -m 'Add some AmazingFeature').
4. Push to the branch (git push origin feature/AmazingFeature).
5. Open a Pull Request.

---

##  Disclaimer

This model is created for educational and research purposes only. It should not be used for financial decision-making or actual trading. Cryptocurrency investments carry high risks; always consult a financial expert before investing.

---

<div align="center">

**Made with ❤️ by FJ TEAM**

Support us by ⭐ this project.

</div>
