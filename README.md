# Neural Network Setups for Student Performance Prediction
## Dataset Overview:

The dataset used for this analysis comprises 10,000 student records, with a variety of
features influencing student performance. These features include:

- Hours Studied: Total hours dedicated to studying by each student.
- Previous Scores: Scores achieved by students in prior assessments.
- Sleep Hours: Average daily hours of sleep for each student.
- Sample Question Papers Practiced: Number of sample question papers attempted by
students.
- Extracurricular Activities: Indicates whether students engage in extracurricular
activities (Yes or No).

The target variable is the Performance Index, which ranges from 10 to 100. The objective is
to develop neural network models with distinct configurations to find the most effective
setup for predicting student performance accurately. (the feature “Extracurricular Activities”
is dropped because it is not so meaningful for prediction and it has the String3 type.)

## Objective

The goal is to find the most effective neural network configuration for accurately predicting student performance based on the given features.

## Experimental Setup:
Three different neural network configurations were tested: (Dropout: Applied with a rate of
0.5 and Training Epochs: 1000 are the same for all.)

1. **Setup 1:**
Two hidden layers with 13 and 10 neurons respectively, employing ReLU
activation function, Learning Rate: 0.01
2. **Setup 2:**
Two hidden layers with 13 and 20 neurons respectively, employing ReLU
activation function, Learning Rate: 0.005
3. **Setup 3:**
Single hidden layer with 13 neurons, employing ReLU activation function,
Learning Rate: 0.005

For all configurations:
- **Training Epochs**: 1000.

## Results

### R2 Scores Comparison

- R2 score measures the proportion of the variance in the target variable that is predictable from the independent variables. Higher R2 scores indicate better predictive capability.

- **R2 scores achieved by each setup:**
  - **Setup 1**: 0.8780410724752464
  - **Setup 2**: 0.9244793374960797
  - **Setup 3**: 0.9697640208866516
- **Setup 3** attains the highest R2 score of **0.970**, signifying it has the best predictive performance.

### Conclusion

Setup 3 emerges as the most effective configuration, offering the highest predictive accuracy with an R2 score of **0.970**. In summary, the experimentation underscores the significance of thoughtful model architecture selection and hyperparameter tuning in achieving optimal predictive performance for student performance prediction.
