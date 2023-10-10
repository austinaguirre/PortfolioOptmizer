import pandas as pd
import numpy as np
import random
import scipy.stats as stats

# Define some constants
POPULATION_SIZE = 150
GENERATIONS = 500
MUTATION_PROBABILITY = 0.08
CROSSOVER_PROBABILITY = 0.8

# Load the price data
prices_df = pd.read_csv('prices.csv')

# Preprocess the data to calculate daily returns
returns_df = prices_df.pivot(index='Date', columns='Ticker', values='Close').pct_change().dropna()


# Define the fitness function
def fitness_function(weights):
    # Calculate the portfolio returns
    portfolio_returns = returns_df.dot(weights)

    # Calculate the Sharpe ratio
    sharpe_ratio = np.sqrt(252) * portfolio_returns.mean() / portfolio_returns.std()

    return sharpe_ratio


# Define the genetic algorithm
def genetic_algorithm(population_size, generations, mutation_probability, crossover_probability):
    # Initialize the population
    population = [np.random.uniform(0, 1, len(returns_df.columns)) for i in range(population_size)]

    # Evaluate the fitness of each individual in the population
    fitness = [fitness_function(weights) for weights in population]

    # Iterate through generations
    for i in range(generations):
        print("Generation: ", i)

        # Select parents for reproduction
        parents = [population[np.argmax(fitness)]]
        for j in range(population_size - 1):
            parent1 = population[np.random.choice(range(population_size), p=fitness / np.sum(fitness))]
            parent2 = population[np.random.choice(range(population_size), p=fitness / np.sum(fitness))]
            child = parent1.copy()
            # Perform crossover
            if np.random.rand() < crossover_probability:
                crossover_point = np.random.randint(len(child))
                child[crossover_point:] = parent2[crossover_point:]
            # Perform mutation
            if np.random.rand() < mutation_probability:
                mutation_point = np.random.randint(len(child))
                child[mutation_point] = np.random.uniform(0, 1)
            parents.append(child)

        # Evaluate the fitness of the offspring
        fitness = [fitness_function(weights) for weights in parents]

        # Select the fittest individuals for the next generation
        population = [parents[i] for i in np.argsort(fitness)[-population_size:]]
        fitness = [fitness[i] for i in np.argsort(fitness)[-population_size:]]

    # Return the best individual found
    return population[np.argmax(fitness)]


# Run the genetic algorithm
best_weights = genetic_algorithm(POPULATION_SIZE, GENERATIONS, MUTATION_PROBABILITY, CROSSOVER_PROBABILITY)

# Calculate the portfolio returns using the best weights
portfolio_returns = returns_df.dot(best_weights)

# Print the Sharpe ratio of the portfolio
sharpe_ratio = np.sqrt(252) * portfolio_returns.mean() / portfolio_returns.std()
print("Sharpe Ratio: ", sharpe_ratio)
print("best weights", best_weights)


# Load the GSPC data
gspc_df = pd.read_csv('GSPC.csv')
# Calculate daily returns
gspc_df['Returns'] = gspc_df['Price'].pct_change()
# Calculate the Sharpe ratio of the GSPC
gspc_sharpe_ratio = np.sqrt(252) * gspc_df['Returns'].mean() / gspc_df['Returns'].std()
print("Sharpe Ratio of GSPC: ", gspc_sharpe_ratio)



# Generate random weights and calculate Sharpe ratio
random_weights = np.random.uniform(0, 1, len(returns_df.columns))
random_sharpe_ratio = fitness_function(random_weights)
# Repeat the process to get a distribution of Sharpe ratios for randomly selected portfolios
random_sharpe_ratios = []
for i in range(1000):
    random_weights = np.random.uniform(0, 1, len(returns_df.columns))
    random_sharpe_ratio = fitness_function(random_weights)
    random_sharpe_ratios.append(random_sharpe_ratio)
# Calculate the p-value of the portfolio's Sharpe ratio compared to the distribution of randomly selected Sharpe ratios
p_value = stats.percentileofscore(random_sharpe_ratios, sharpe_ratio) / 100
# Print the p-value
print("p-value: ", p_value)


