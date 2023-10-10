"""
geneticAlgorithmOptimizer.py

1) Read in .csv price data
2) Optimize Portfolio with Genetic Algorithm
3) Refine Genetic Algorithm Solution via Simulated Annealing Algorithm

"""


"""
=============================================================================
Import Modules
-----------------------------------------------------------------------------
"""
import os
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import random
from random import choice

from pandas_datareader import data as wb
from scipy import stats

# Custom Modules
import sys
root_path = os.getcwd()
# Add project folders to root path
sys.path.append(root_path + '/AIPortfolioOptimizer')
sys.path.append(root_path + '/AIPortfolioOptimizer/data')
sys.path.append(root_path + '/AIPortfolioOptimizer/src')
sys.path.append(root_path + '/AIPortfolioOptimizer/utilities')
from loadTickerPriceData import load_price_data


"""
==============================================================================
Classes
------------------------------------------------------------------------------
"""
class Stock():
    """
    Class to generate a stock object with the following attributes:
    """
    def __init__(self, Date, Ticker, High, Low, Open, Close, AdjClose, Volume):
        self.date = Date
        self.ticker = Ticker
        self.high = High
        self.low = Low
        self.open = Open
        self.close = Close
        self.adjclose = AdjClose
        self.volume = Volume

class IndividualPortfolio():
    """
    Class to generate an individual portfolio object for use in the genetic algorithm:
    """
    def __init__(self, stock_data, generation=0, random_chromosome=True, chromosome=None):
        self.stock_data = stock_data
        self.generation = generation
        self.total_return = 0
        self.total_risk = 0
        self.sharpe_ratio = 0
        self.chromosome = []
        self.symbol_list = []
        self.returns = []

        # Create a random chromosome
        if random_chromosome:
            self.chromosome = [random.choice(['0', '1']) for _ in range(len(stock_data))]
        elif chromosome is not None:
            self.chromosome = chromosome
        else:
            raise ValueError('Either random_chromosome or chromosome must be provided.')

    def fitness(self):
        """
        Calculates the fitness (Sharpe Ratio) of the individual portfolio object. Higher is better.
        """
        # Reset values for each fitness calculation
        self.total_return = 0
        self.total_risk = 0
        self.sharpe_ratio = 0
        # print('Info: => Performing Fitness Calcs...')
        # Build weights from chromosome
        weights = pd.Series([int(gene) for gene in self.chromosome])
        # initialize lists
        mean_daily_log_return = []
        all_daily_log_returns = []
        returns = []
        
        for i, stock in enumerate(self.stock_data):
            # Calculate annualized return of stock i
            daily_log_returns =  np.log(stock.adjclose / stock.adjclose.shift(1)) 
            daily_log_returns.replace(np.nan,0.000, inplace = True)                                
            daily_log_returns.replace(np.inf,0.000, inplace = True)
            daily_log_returns.replace(-np.inf,0.000, inplace = True) 
            # Store daily log returns in a list
            all_daily_log_returns.append(daily_log_returns)
            # Concatenate all daily log returns into a single dataframe for cov() calculation
            df_all_daily_log_returns = pd.concat(all_daily_log_returns, axis=1)
            
            mean_daily_log_return = daily_log_returns.mean()
            annualized_return = (1 + mean_daily_log_return)** 252 - 1
            # Calculate annualized risk of stock i
            annualized_risk = np.sqrt(252) * daily_log_returns.std()
            # Calculate weighted return of stock i
            weighted_return = weights[i] * annualized_return
            self.returns.append(weighted_return)
        
        #Calculate total portfolio return and risk
        self.total_return = np.sum(self.returns)
        self.total_risk = np.sqrt(np.dot(weights, np.dot(df_all_daily_log_returns.cov(), weights)))

        # Calculate Sharpe Ratio
        self.sharpe_ratio = self.total_return / self.total_risk

    def crossover(self, other_individual_portfolio):
        """
        Creates a new individual portfolio object by crossing over two individuals.
        """
        # print('Info: => Performing Crossover...')
        cutoff = round(random.random() * len(self.chromosome))
        # print('Crossover Cutoff: ', cutoff)
        child1 = other_individual_portfolio.chromosome[0:cutoff] + self.chromosome[cutoff::]
        child2 = self.chromosome[0:cutoff] + other_individual_portfolio.chromosome[cutoff::]
        
        children = [IndividualPortfolio(self.stock_data, self.generation + 1), 
                    IndividualPortfolio(self.stock_data, self.generation + 1)]
        children[0].chromosome = child1
        children[1].chromosome = child2
        return children

    def mutation(self, rate):
        """
        Mutates an individual portfolio object.
        """
        # Debug / check
        # print('Info: => Mutating...')
        # print('Before: ', self.chromosome)
        for i in range(len(self.chromosome)):
            if random.random() < rate:
                if self.chromosome[i] == '1':
                    self.chromosome[i] = '0'
                else:
                    self.chromosome[i] = '1'
        # Debug / check
        # print('After: ', self.chromosome)
        return self.chromosome

    def create_symbol_list(self):
        """
        Creates a list of symbols from the chromosome.
        """
        # print('Info: => Creating Symbol List...')
        # CLear symbol_list to eliminate duplicates
        self.symbol_list.clear()
        counter = 0
        for i, gene in enumerate(self.chromosome):
            if gene == '1':
                self.symbol_list.append(self.stock_data[i].ticker)
                counter += 1
                if counter == len(self.chromosome):
                    break
        return self.symbol_list

class GeneticAlgorithm():
    """
    Class to generate a genetic algorithm object for use in optimizing a portfolio.
    """
    def __init__(self, population_size):
        self.population_size = population_size
        self.population = []
        self.generation = 0
        self.best_portfolio = None
        self.list_of_portfolios = []

    def initialize_population(self, stock_data):
        """
        Creates the initial population of individual stock portfolios.  The population size is 
        determined by the user.
        """
        for i in range(self.population_size):
            self.population.append(IndividualPortfolio(stock_data))
        self.best_portfolio = self.population[0]

    def best_individual_portfolio(self, individual_portfolio):
        """
        Get the best individual_portfolio in the population.
        """
        if individual_portfolio.sharpe_ratio > self.best_portfolio.sharpe_ratio:
            self.best_portfolio = individual_portfolio

    def order_population(self):
        """
        Orders the population by fitness.
        """
        self.population = sorted(self.population, key=lambda population: population.sharpe_ratio, reverse=True)

    def sum_fitness(self):
        """
        Sums the fitness of the entire population.
        """
        sum = 0
        for individual_portfolio in self.population:
            sum += individual_portfolio.sharpe_ratio
        return sum

    def select_parent(self, sum_fitness):
        """
        Selects a parent for crossover.
        """
        parent = -1
        pick = random.random() * sum_fitness
        current = 0.0
        i = 0
        # print('*** pick value:', pick)
        while i < len(self.population) and abs(current) < abs(pick):
            # print('i:', i, ' - current: ', current)
            current += self.population[i].sharpe_ratio
            parent += 1
            i += 1
        return parent

    def visualize_generation(self):
        best = self.population[0]
        print('\nGeneration: ', self.population[0].generation, 
              '\nSharpe Ratio: ', best.sharpe_ratio,
              '\nTotal Return (Annualized): ', best.total_return,
              '\nTotal Risk (Annualized): ', best.total_risk,
              '\nChromosome: ', best.chromosome,
              '\n')

    def run(self, mutation_rate, num_of_generations, stock_data):
        """
        Runs the genetic algorithm.
        """
        #Initialize the population
        print('\nInfo: => Initializing Population...\n')
        self.initialize_population(stock_data)
        # Evaluate the Initial Population
        for individual_portfolio in self.population:
            individual_portfolio.fitness()
        # Order the population by fitness
        self.order_population()
        # Visualize the initial population
        print('\nInfo: => Visualizing Initial Population...')
        self.visualize_generation()
        
        # Start evaluation loop
        for generation in range(num_of_generations):
            #Generate Parents, Crossover, and Mutate
            sum_total_fitness = self.sum_fitness()
            new_population = []
            #Apply Genetic Operators
            for new_individual_portfolios in range(0, self.population_size, 2):
                #Select Parents
                parent1 = self.select_parent(sum_total_fitness)
                parent2 = self.select_parent(sum_total_fitness)
                #Crossover
                children = self.population[parent1].crossover(self.population[parent2])
                #Mutation
                children[0].mutation(mutation_rate)
                children[1].mutation(mutation_rate)
                #Add children to new population
                new_population.append(children[0])
                new_population.append(children[1])
                
            #Replace the old population with the new population
            self.population = list(new_population)
            
            #Evaluate the new population
            for individual_portfolio in self.population:
                individual_portfolio.fitness()
            #Order the population by fitness
            self.order_population()
            # Visualize the NEw Generation
            self.visualize_generation()
            # Get the best individual portfolio
            best = self.population[0]
            # Update list of portfolios
            self.list_of_portfolios.append(best.sharpe_ratio)
            # Update the best portfolio
            self.best_individual_portfolio(best)
        
        # Print the best portfolio
        print('\n*** GENETIC ALGORITHM BEST SOLUTION Generation: ', self.best_portfolio.generation, 
            '\nBest Sharpe Ratio: ', self.best_portfolio.sharpe_ratio,
            '\nTotal Return (Annualized): ', self.best_portfolio.total_return,
            '\nTotal Risk (Annualized): ', self.best_portfolio.total_risk,
            '\nChromosome: ', self.best_portfolio.chromosome,
            '\nStock List: ', self.best_portfolio.create_symbol_list(),
            '\nNumber of Stocks: ', len(self.best_portfolio.create_symbol_list()),
            '\n')
        
        return self.best_portfolio

class SimulatedAnnealing:
    def __init__(self, ga_solution, temperature=5.0, cooling_rate=0.95):
        self.original = ga_solution
        self.temperature = temperature
        self.cooling_rate = cooling_rate
        
        self.initial_portfolio = None
        # self.best_solution = None

    def initialize_portfolio(self, stock_data):
        """
        Initialize the portfolio from the GA solution to be used in the Simulated Annealing.
        """
        self.initial_portfolio = IndividualPortfolio(stock_data, random_chromosome=False, chromosome=self.original.chromosome)

    def neighbor(self, stock_data):
        """
        Returns a neighbor solution.
        """
        neighbor = IndividualPortfolio(stock_data, random_chromosome=False, chromosome=self.original.chromosome)
        neighbor.mutation(0.2)
        neighbor.fitness()
        return neighbor

    def run(self, stock_data):
        """
        Runs the Simulated Annealing Algorithm.
        """
        #Initialize the population
        print('\nInfo: => Initializing Starting Portfolio from GA Solution...\n')
        self.initialize_portfolio(stock_data)
        self.initial_portfolio.fitness()
        best_solution = self.initial_portfolio
        print('\nInfo: => Confirm SA Solution Initialized by GA Result...')
        print('Initial Sharpe Ratio: ', best_solution.sharpe_ratio)
        print('Initial Total Return (Annualized): ', best_solution.total_return)
        print('Initial Total Risk (Annualized): ', best_solution.total_risk)
        print('Initial Chromosome: ', best_solution.chromosome, '\n')
        
        print('Info: => Running Simulated Annealing Algorithm...\n')
        current_solution = self.initial_portfolio
        
        # Initialize variables to keep track of sharpe ratio, total returns, and total risk
        # for the Simulated Annealing Algorithm solution.
        best_sharpe_ratio = best_solution.sharpe_ratio
        best_total_return = best_solution.total_return
        best_total_risk = best_solution.total_risk
        best_chromosome = best_solution.chromosome
        # Print output counter
        p_counter = 0
        while self.temperature > 0.1:
            # Get a neighbor solution
            new_neighbor = self.neighbor(stock_data)
            # Calculate the value of the current solution and the new neighbor solution
            current_solution.fitness()
            current_value = current_solution.sharpe_ratio
            new_neighbor.fitness()
            new_neighbor_value = new_neighbor.sharpe_ratio
            
            # Decide whether to accept the neighbor solution
            if new_neighbor_value > current_value:
                # print('SA Loop: New Neighbor SR > Current SR')
                print('SA Update: New Neighbor SR: ', new_neighbor_value, ' > Current SR: ', current_value)
                current_solution = new_neighbor
                # Update best variables if new solution is better than previous best
                if new_neighbor.sharpe_ratio > best_sharpe_ratio:
                    # print('Updating Best SR, Return & Risk to New Neighbor Values')
                    best_sharpe_ratio = new_neighbor.sharpe_ratio
                    best_total_return = new_neighbor.total_return
                    best_total_risk = new_neighbor.total_risk
                    best_chromosome = new_neighbor.chromosome
            else:
                # print('SA Loop: Calculating Best Acceptance Probability...')
                acceptance_probability = np.exp((new_neighbor_value - current_value) / self.temperature)
                if random.random() < acceptance_probability:
                    current_solution = new_neighbor
            # Decrease the temperature
            self.temperature *= self.cooling_rate
            
            # Print status message
            if p_counter % 10 == 0: 
                print('SA running...', 'temperature = ', self.temperature)
            # Increment print counter
            p_counter += 1
        
        # Update the best solution using the best variables
        best_solution = IndividualPortfolio(stock_data, random_chromosome=False, chromosome=current_solution.chromosome)
        best_solution.fitness()
        best_solution.sharpe_ratio = best_sharpe_ratio
        best_solution.total_return = best_total_return
        best_solution.total_risk = best_total_risk
        best_solution.chromosome = best_chromosome
                
        # Print the best portfolio
        print('\n*** SIMULATED ANNEALING ALGORITHM SOLUTION: \n',
            'Best Sharpe Ratio: ', best_solution.sharpe_ratio, '\n'
            'Best Total Return (Annualized): ', best_solution.total_return, '\n'
            'Best Total Risk (Annualized): ', best_solution.total_risk, '\n'
            'Best Chromosome: ', best_solution.chromosome, '\n'
            'Best Stock List: ', best_solution.create_symbol_list(), '\n'
            'Best Number of Stocks: ', len(best_solution.create_symbol_list()))    
        
        return best_solution

"""
==============================================================================
Functions
------------------------------------------------------------------------------
"""
def clear_terminal():
    """
    Clear terminal window.
    """
    if os.name == 'nt':
        os.system('cls')
    else:
        os.system('clear')

def create_stock_data(data):
    """
    Creates a list of Stock objects from a data dataframe.

    Parameters:
    data : pandas.DataFrame
        A dataframe containing price data for one or more stocks.

    Returns:
    list
        A list of Stock objects.
    """
    stock_data = []
    for col in data.columns:
        if col[0] == 'Adj Close':
            ticker = col[1]
            date = data.index
            high = data['High', ticker]
            low = data['Low', ticker]
            open_price = data['Open', ticker]
            close = data['Close', ticker]
            adjclose = data['Adj Close', ticker]
            volume = data['Volume', ticker]
            stock_data.append(Stock(date, ticker, high, low, open_price, close, adjclose, volume))
    return stock_data


"""
==============================================================================
MAIN PROGRAM
------------------------------------------------------------------------------
"""
def main():
    #Use for code development and debugging
    #Comment out when not needed.
    clear_terminal()

    # Specify root path for csv data files
    relative_path = 'AIPortfolioOptimizer/data' 
    # filename = 'PriceData DOW30 From 01 Jan, 2020 To 19 Mar, 2023.csv'
    # filename = 'PriceData DOW30 From 01 Jan, 2013 To 12 Mar, 2023.csv'
    filename = 'PriceData SP500 From 01 Jan, 2018 To 21 Apr, 2023.csv'
    file_path = os.path.join(root_path, relative_path, filename)

    # Load csv data file and format data frame
    data = load_price_data(file_path)

    # Create list of Stock objects for each stock in data loaded from csv file
    stock_data = create_stock_data(data)

    # *** Run GA ***
    print('\n\n*** GENETIC ALGORITHM STARTED')
    population_size = 5        # Number of individuals in the population   
    mutation_rate = 0.01        # Mutation rate
    num_of_generations = 5    # Number of generations to evolve
    ga = GeneticAlgorithm(population_size)
    result_ga = ga.run(mutation_rate, num_of_generations, stock_data)

    print('*** GENETRIC ALGORITHM COMPLETE')
    print(20 * '^^^')
    print()

    # Plot GA results
    plt.figure(figsize=(12, 6))
    plt.plot(ga.list_of_portfolios, color='red', label='Sharpe Ratio')
    plt.xlabel('Generation')
    plt.ylabel('Sharpe Ratio')
    plt.title('Sharpe Ratio (Fitness) vs. Generation')
    plt.grid(True)
    plt.show()

    # *** Run SA ***
    print('\n\n*** SIMULATED ANNEALING ALGORITHM STARTED')
    sa = SimulatedAnnealing(result_ga)
    result_sa = sa.run(stock_data)

    
    print('*** SIMULATED ANNEALING ALGORITHM COMPLETE')
    print(20 * '^^^')


if __name__ == '__main__':
    main()
