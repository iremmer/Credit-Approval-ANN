# Import sys module
import sys

# Import pandas library for data manipulation
import pandas as pd

# Import train_test_split function for splitting data
from sklearn.model_selection import train_test_split

# Import numpy for numerical computing
import numpy as np

# Import Sequential class for building a neural network
from tensorflow.keras.models import Sequential

# Import necessary classes for defining model architecture
from tensorflow.keras.layers import Flatten, Dense, Activation

# Import random module for generating random numbers (usage unclear)
import random


def normalize(matrix, multiplier):
    """
    Normalizes the values in each column of the input matrix to a range between 0 and the `multiplier` argument.

    Parameters:
    matrix (numpy.ndarray): Input matrix to be normalized
    multiplier (float): Value to which the normalized values are scaled. For example, if multiplier = 1, the normalized values will range between 0 and 1.

    Returns:
    numpy.ndarray: Normalized matrix, with each column having values in the range of 0 to `multiplier`
    """
    
    # Iterate through each column in the matrix
    for i in range(matrix.shape[1]):
        # Find the maximum and minimum values in the current column
        column_max = np.max(matrix[:, i])
        column_min = np.min(matrix[:, i])
        
        # Compute the range of the current column, taking care to handle the case where max = min
        vector_range = 1 if column_max == column_min else column_max - column_min
        
        # Normalize the current column to a range between 0 and `multiplier`
        matrix[:, i] = (matrix[:, i] - column_min) / (vector_range) * multiplier
        
    # Return the normalized matrix
    return matrix

# Define a dictionary called categorical_set with keys 'A1' through 'A15' and empty lists as values
categorical_set = {'A1': [], 'A2': [], 'A3': [], 'A4': [], 'A5': [], 'A6': [], 'A7': [], 'A8': [], 'A9': [], 'A10': [], 'A11': [], 'A12': [], 'A13': [], 'A14': [], 'A15': []}

# Define a set called continuous containing column names of the continuous variables in the dataset
continuous = {'A2', 'A3', 'A8', 'A11', 'A14', 'A15'}

# Define a list called categoricals containing column names of the categorical variables in the dataset
categoricals = ['A1', 'A4', 'A5', 'A6', 'A7', 'A9', 'A10', 'A12', 'A13']

# Open the file 'crx.data' in read mode and read all the lines into a list called lines
with open('crx.data', 'r') as data_file:
    lines = data_file.readlines()

# Close the file 'crx.data'
data_file.close()

# Initialize an empty list called labels and a list called attributes containing the keys of categorical_set
labels = []
attributes = list(categorical_set.keys())

# Initialize a list called previous_line with values that will be used as replacements for missing values ('?')
previous_line = ['a', '64.08', '0.165', 'u', 'g', 'ff', 'ff', '0', 't', 't', '01', 'f', 'g', '00232', '100']

# Iterate through each line in the list 'lines'
for line in lines:
    # Split the line into a list called 'record'
    record = line[:-1].split(',')

    # Initialize a counter to keep track of the number of missing values in the current record
    no_question_marks = 0
    
    # Iterate through each attribute in 'attributes'
    for i in range(len(attributes)):
        # Get the key and value for the current attribute
        key = attributes[i]
        value = record[i]

        # If the value is missing ('?'), replace it with the corresponding value from the previous line and increment the counter
        if value == '?':
            value = previous_line[i]
            no_question_marks += 1

        # Convert the value to a float if it corresponds to a continuous variable, otherwise leave it as a string
        value = -np.float32(value) if key in continuous else value

        # Append the value to the corresponding list in 'categorical_set'
        categorical_set[key].append(value)

    # Append 1 to 'labels' if the last element in the current record is '+', otherwise append 0
    labels.append(1) if record[-1] == '+' else labels.append(0)
    
    # If there were missing values in the current record, replace 'previous_line' with the current record
    # so that missing values in the next record can be replaced with the corresponding values from the current record
    previous_line = record if no_question_marks == 0 else previous_line

# Convert the dictionary 'categorical_set' into a pandas DataFrame called 'data'
data = pd.DataFrame(categorical_set)

# Encode the categorical variables in 'data' using one-hot encoding and store the result in a new DataFrame called 'encoded_data'
encoded_data = pd.get_dummies(data, categoricals)

# Get the column names of the encoded variables
encoded_columns = encoded_data.columns

# Normalize the values in 'encoded_data' to a range between 0 and 1 and store the result in a numpy array
normalized_data = normalize(encoded_data.to_numpy(), 1)

# Convert the normalized numpy array to a pandas DataFrame called 'encoded_data'
encoded_data = pd.DataFrame(normalized_data)

# Set the column names of 'encoded_data' to be the same as the column names of the original encoded variables
encoded_data.columns = encoded_columns

# Convert the list 'labels' into a pandas DataFrame called 'labels'
labels = pd.DataFrame(labels)

# Write the normalized encoded variables to a CSV file called 'samples.csv' without including the index column
encoded_data.to_csv('samples.csv', index=False)

# Write the labels to a CSV file called 'labels.csv' without including the index column
labels.to_csv('labels.csv', index=False)

# Read the CSV file 'samples.csv' into a pandas DataFrame called 'encoded_data'
encoded_data = pd.read_csv('samples.csv')

# Read the CSV file 'labels.csv' into a pandas DataFrame called 'labels'
labels = pd.read_csv('labels.csv')

# Set the seed for the random number generator to ensure reproducibility
random.seed(150)

# Split the encoded variables and labels into training and testing sets using the 'train_test_split' function from scikit-learn
# The 'encoded_data' DataFrame contains the encoded variables, and the 'labels' DataFrame contains the labels
# The training set will contain 80% of the data, and the testing set will contain 20% of the data
# The random state is set to 2164 to ensure reproducibility of the results
x_train, x_test, y_train, y_test = train_test_split(encoded_data, labels, test_size=0.2, random_state=2164)

# Define a sequential model with 5 dense layers
# The first 4 layers each have 1024, 256, 128, and 32 neurons, respectively, and use ReLU activation
# The final layer has a single neuron and uses sigmoid activation
model = Sequential([
    Dense(1024, activation='relu'),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model with Adam optimizer, binary crossentropy loss, and accuracy as the evaluation metric
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model using the training set with 7 epochs and a batch size of 60
model.fit(x_train, y_train, epochs=7, batch_size=60)

# Evaluate the trained model on the testing set using the 'evaluate' method
# The 'evaluate' method returns the loss and accuracy of the model on the testing set
results = model.evaluate(x_test, y_test, verbose=0)

# Print the loss and accuracy of the model on the testing set
print('test loss, test acc:', results)

