Machine Learning Project for Heart Disease Prediction
This repository contains a machine learning workflow for binary classification of heart disease risk using structured clinical data. The project is developed in a Jupyter Notebook and can be executed both locally with Jupyter and online with Google Colab.

The main goal is to compare a classical machine learning model, Logistic Regression, with a simple neural network implemented in PyTorch, focusing on data preprocessing, feature engineering, model training, and performance evaluation.

Project Objective
The task is to predict the target variable HeartDisease:

0: no heart disease
1: heart disease
This is a supervised binary classification problem. The project follows a complete machine learning pipeline:

Dataset loading and inspection
Missing value analysis
Data imputation
Categorical feature encoding
Train-test split
Numerical feature scaling
Model training
Model evaluation
Comparison between Logistic Regression and Neural Network
Dataset Overview
The dataset contains 918 observations and 12 original columns:

Age
Sex
ChestPainType
RestingBP
Cholesterol
FastingBS
RestingECG
MaxHR
ExerciseAngina
Oldpeak
ST_Slope
HeartDisease
The target column is HeartDisease, while the remaining columns are used as predictive features.

Notebook Compatibility
The project can be run in:

Jupyter Notebook
JupyterLab
Google Colab
When running the notebook in Google Colab, upload the dataset file and update the path if needed:

df = pd.read_csv('/content/heart.csv')
When running locally with Jupyter, place heart.csv in the project folder and use:

df = pd.read_csv('heart.csv')
Data Preprocessing
Missing Values
The dataset contains missing values in:

RestingBP: 1 missing value
Cholesterol: 172 missing values
The missing values are handled using median imputation.

Median imputation is used because it is more robust than the mean when a feature contains outliers. This is especially relevant for Cholesterol, where the distribution shows several extreme values.

Categorical Encoding
The dataset contains nominal categorical variables:

Sex
ChestPainType
RestingECG
ExerciseAngina
ST_Slope
Since these variables do not have a natural ordinal relationship, One-Hot Encoding is applied using pd.get_dummies().

The option drop_first=True is used to reduce redundant information and avoid perfect multicollinearity in the encoded dataset.

After encoding, the dataset expands from 12 columns to 16 columns.

Train-Test Split
The dataset is split into training and test sets using:

train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
The split produces:

Training set: 734 samples
Test set: 184 samples
The parameter stratify=y preserves the original class distribution in both training and test sets. This is important because the model should be evaluated on a test set that reflects the same balance between positive and negative cases.

Feature Scaling
Numerical features are standardized using StandardScaler.

Scaling is applied only to continuous numerical variables, while binary and one-hot encoded columns are left unchanged.

The scaler is fitted only on the training set:

scaler.fit_transform(X_train[numerical_cols])
Then the same transformation is applied to the test set:

scaler.transform(X_test[numerical_cols])
This avoids data leakage, because the test set must not influence the preprocessing parameters learned from the training data.

Model 1: Logistic Regression
Logistic Regression is used as the baseline model for this binary classification task.

The model is trained with:

LogisticRegression(
    random_state=67,
    solver='liblinear',
    max_iter=1000
)
Why Logistic Regression?
Logistic Regression is a strong baseline for this type of problem because:

it is designed for binary classification
it is computationally efficient
it performs well on small and medium-sized tabular datasets
it is easier to interpret than more complex models
it is less prone to overfitting compared to neural networks on small datasets
Training Process
The model is trained on the scaled training data:

model.fit(X_train_scaled, y_train)
Predictions are generated on the test set:

y_pred = model.predict(X_test_scaled)
The model is evaluated using:

Accuracy
Precision
Recall
F1-score
Confusion Matrix
Logistic Regression Results
The Logistic Regression model achieved:

Accuracy: 0.8913
Macro Avg Precision: 0.89
Macro Avg Recall: 0.89
Macro Avg F1-score: 0.89
Confusion Matrix:

[[71 11]
 [ 9 93]]
In this medical classification context, recall is particularly important because false negatives may represent patients with heart disease incorrectly classified as healthy.

Model 2: Neural Network with PyTorch
The second model is a feedforward neural network implemented with PyTorch.

Before training, the processed Pandas dataframes are converted into PyTorch tensors:

X_train_tensor = torch.tensor(X_train_scaled.astype(float).values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
Boolean columns are converted to floating-point values because PyTorch layers require numerical tensor inputs.

Neural Network Architecture
The neural network contains:

Input layer with one neuron for each feature
Hidden layer with 16 neurons
ReLU activation function
Dropout layer with probability 0.3
Output layer with 1 neuron
Sigmoid activation for binary classification
Architecture:

class DiseaseNN(nn.Module):
    def __init__(self, input_size):
        super(DiseaseNN, self).__init__()
        self.layer_1 = nn.Linear(input_size, 16)
        self.relu_1 = nn.ReLU()
        self.dropout_1 = nn.Dropout(0.3)
        self.output_layer = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer_1(x)
        x = self.relu_1(x)
        x = self.dropout_1(x)
        x = self.output_layer(x)
        x = self.sigmoid(x)
        return x
Loss Function and Optimizer
The model uses:

BCELoss as the loss function
Adam as the optimizer
Learning rate: 0.001
Epochs: 250
BCELoss is appropriate because the output is a probability between 0 and 1 produced by the Sigmoid activation function.

The optimizer updates the network weights through backpropagation:

loss.backward()
optimizer.step()
Neural Network Training Process
For each epoch, the model performs:

Forward pass
Loss calculation
Gradient reset
Backpropagation
Weight update
Training loop:

for epoch in range(epochs):
    outputs = model_nn_pytorch(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
After training, the model is switched to evaluation mode:

model_nn_pytorch.eval()
Predicted probabilities are converted into binary predictions using a threshold of 0.5:

y_pred_nn = (y_pred_proba_nn > 0.5).int().numpy()
Neural Network Results
The neural network achieved approximately:

Accuracy: 0.875
Macro Avg Precision: 0.87
Macro Avg Recall: 0.87
Macro Avg F1-score: 0.87
Confusion Matrix:

[[71 11]
 [12 90]]
Model Comparison
Model	Accuracy	Macro Precision	Macro Recall	Macro F1-score
Logistic Regression	0.8913	0.89	0.89	0.89
Neural Network	0.8750	0.87	0.87	0.87
The Logistic Regression model performs slightly better than the neural network across the main evaluation metrics.

This result is meaningful because the dataset is relatively small, with only 918 samples. Neural networks usually perform best when trained on larger datasets, where they can learn more complex and non-linear patterns. In this case, the additional complexity of the neural network does not improve predictive performance.

The Logistic Regression model is also more interpretable and stable, which is an important advantage in a healthcare-related classification problem.

Technical Interpretation
The results suggest that the relationship between the input features and the target variable can be captured effectively by a simpler linear model.

The neural network introduces more parameters and higher model complexity, but this does not lead to better generalization on the test set. This may indicate:

limited dataset size
possible overfitting risk
insufficient need for non-linear modeling
higher sensitivity to initialization and hyperparameters
For this specific dataset, Logistic Regression provides the best balance between performance, simplicity, and interpretability.

Technologies Used
Python
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn
PyTorch
Jupyter Notebook
Google Colab
How to Run
Option 1: Jupyter Notebook
Install the required libraries:

pip install pandas numpy matplotlib seaborn scikit-learn torch jupyter
Start Jupyter Notebook:

jupyter notebook
Open:

Machine learning project for heart disease.ipynb
Option 2: Google Colab
Open Google Colab
Upload the notebook
Upload heart.csv
Make sure the dataset path is:
df = pd.read_csv('/content/heart.csv')
Run all cells
Conclusion
This project demonstrates a complete machine learning workflow for heart disease prediction using tabular clinical data.

The most important result is that the simpler Logistic Regression model outperforms the PyTorch neural network on this dataset. This highlights an important machine learning principle: a more complex model is not always better, especially when the dataset is small and the relationships between variables can be modeled effectively with simpler methods.

For this reason, Logistic Regression is the preferred model in this project due to its stronger performance, lower complexity, and better interpretability.

Disclaimer
This project is intended for educational and analytical purposes only. It is not intended to provide medical diagnosis or clinical decision support.
Questo rende la Regressione Logistica una scelta adatta per un primo approccio predittivo su questo dataset.
