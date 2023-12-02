# Credit Approval Using ANN

## Overview
This project demonstrates the process of data handling, preprocessing, and neural network modeling using Python libraries such as Pandas, NumPy, and TensorFlow. The script includes functions for normalizing data, handling categorical and continuous variables, and constructing a sequential neural network model for binary classification.

## Dependencies
- Pandas
- NumPy
- scikit-learn
- TensorFlow

## Description
The script performs the following major tasks:
1. **Data Handling**: Reading a dataset (`crx.data`), handling missing values, and separating features into categorical and continuous variables.
2. **Data Preprocessing**: 
   - Normalizing continuous variables.
   - Encoding categorical variables using one-hot encoding.
   - Splitting the dataset into training and testing sets.
3. **Neural Network Construction**: 
   - Building a Sequential model with multiple Dense layers.
   - Applying ReLU and Sigmoid activations.
4. **Model Training and Evaluation**: 
   - Compiling the model with Adam optimizer and binary crossentropy loss.
   - Training the model on the training set.
   - Evaluating the model's performance on the testing set.

### Functions
- `normalize(matrix, multiplier)`: Normalizes the values in each column of the input matrix.

### Data Files
- `crx.data`: The primary dataset used for the project.

### Output Files
- `samples.csv`: Contains normalized and encoded feature variables.
- `labels.csv`: Contains target labels for the dataset.

## Usage
1. Ensure all dependencies are installed.
2. Run the script to process the data, build and train the model, and evaluate its performance.
3. The model's loss and accuracy on the test set are printed as output.

## Notes
- The script includes detailed comments explaining each step of the process.
- Random seed values are set for reproducibility.
- The script is modular and can be adapted for similar binary classification tasks.
