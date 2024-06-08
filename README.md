# GA Optimization

A package for genetic algorithm-based feature selection.

## Overview

GA Optimization is a Python package that leverages genetic algorithms to perform feature selection. It helps in identifying the most relevant features for a machine learning model, thereby improving model performance and reducing overfitting.

## How It Works

Genetic Algorithms (GAs) are inspired by the process of natural selection and are used to find approximate solutions to optimization and search problems. Here's a brief overview of how the GA in this package works:

### Initialization

The GA starts by initializing a population of individuals. Each individual represents a potential solution and is encoded as a list of binary values (genes). Each gene in the individual corresponds to a feature in the dataset. A value of `1` indicates that the feature is selected, while a value of `0` indicates that the feature is not selected.

### Fitness Evaluation

The fitness of each individual is evaluated using a fitness function. In this package, the fitness function trains a Random Forest classifier using only the selected features and evaluates its accuracy on the training data. The accuracy score is used as the fitness value.

### Selection

Selection is the process of choosing individuals from the current population to create offspring for the next generation. This package uses tournament selection, where a subset of individuals is chosen at random, and the best individual from this subset is selected.

### Crossover (Recombination)

Crossover is the process of combining two parent individuals to create offspring. This package uses two-point crossover, where two points are selected on the parent individuals' genes, and the genes between these points are swapped.

### Mutation

Mutation introduces random changes to an individual's genes to maintain genetic diversity within the population. This package uses flip-bit mutation, where each gene has a probability of being flipped (i.e., changed from `0` to `1` or from `1` to `0`).

### Genetic Algorithm Process

The GA iterates through the following steps for a fixed number of generations or until a stopping criterion is met:

1. **Fitness Evaluation**: Evaluate the fitness of each individual in the population.
2. **Selection**: Select individuals to create offspring.
3. **Crossover**: Apply crossover to create offspring.
4. **Mutation**: Apply mutation to the offspring.
5. **Replacement**: Replace the old population with the new offspring.

### Using Random Forest as a Classifier

In this package, a Random Forest classifier is used as the underlying model to evaluate the fitness of each individual. Random Forest is an ensemble learning method that constructs multiple decision trees and outputs the mode of the classes (classification) or mean prediction (regression) of the individual trees. It is chosen for its robustness and ability to handle large datasets with high dimensionality.

## Installation

You can install the package via pip:

bash
pip install ga-optimization

Alternatively, you can clone the repository and install the dependencies:


git clone https://github.com/suraj385/ga_optimization.git
cd ga_optimization
pip install -r requirements.txt

Usage

Here is a basic example of how to use the GA Optimization package:


import sys
import os
import pandas as pd
from optimization.feature_extraction import prepare_data, setup_deap, run_ga, save_selected_data

# Parameters
data_path = "train_data.csv"
target_column = "Class"
sample_frac = 0.2  # Adjust the fraction of data to use for training
population_size = 10 #adjust accordingly to your requirement
generations = 2 #just fro testing purpose , you can use more !

# Load the original data
original_data = pd.read_csv(data_path)

# Sample the data
sampled_data = original_data.sample(frac=sample_frac, random_state=42)

# Prepare the data
X, y = prepare_data(sampled_data, target_column)

# Setup DEAP
toolbox = setup_deap(X)

# Run the genetic algorithm to select features
selected_columns = run_ga(toolbox, X, y, sampled_data.columns, population_size=population_size, generations=generations)

# Save the data with selected features applied to the original data
save_selected_data(original_data, selected_columns, target_column, output_path="train_data_selected1.csv")


License

This project is licensed under the MIT License - see the LICENSE file for details.
