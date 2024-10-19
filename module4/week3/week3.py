'''

                                            AI VIET NAM - COURSE 2024
                                                M04 - Exercises
                                    (Genetic Algorithm and Its Applications)
                                            Ngày 14 tháng 10 năm 2024

'''

import numpy as np
import matplotlib.pyplot as plt
import random
random.seed(0)

#---------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------ Bai tap 1 ----------------------------------------------#
#---------------------------------------------------------------------------------------------------------------#

def load_data_from_file(fileName="advertising.csv"):
    data = np.genfromtxt(fileName, dtype=None, delimiter=',', skip_header=1)
    features_X = data[:, :3]
    features_X = np.concatenate([np.ones((features_X.shape[0], 1)), features_X], axis=1)
    sales_y = data[:, 3]
    return features_X, sales_y

features_X, sales_Y = load_data_from_file(fileName="advertising.csv")

# print(features_X[:5,:])
''' 
[[  1.  230.1  37.8  69.2]
 [  1.   44.5  39.3  45.1]
 [  1.   17.2  45.9  69.3]
 [  1.  151.5  41.3  58.5]
 [  1.  180.8  10.8  58.4]]
''' # => Question 2: A

# print(sales_Y.shape) # (200,) => Question 3: B

#---------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------ Bai tap 2 ----------------------------------------------#
#---------------------------------------------------------------------------------------------------------------#

def create_individual(n=4, bound=10):
    individual = [(random.random() - 0.5)*bound for _ in range(n)]
    return individual

individual = create_individual() # (n,)
# print(individual)

#---------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------ Bai tap 3 ----------------------------------------------#
#---------------------------------------------------------------------------------------------------------------#

def compute_loss(individual):
    theta = np.array(individual)
    y_hat = features_X.dot(theta)
    loss  = np.multiply((y_hat - sales_Y), (y_hat - sales_Y)).mean()
    return loss

def compute_fitness(individual):
    loss = compute_loss(individual)
    fitness_value = 0
    fitness_value = 1/(loss + 1)
    return fitness_value

# features_X, sales_Y = load_data_from_file()
# individual = [4.09, 4.82, 3.10, 4.02]
# fitness_score = compute_fitness(individual)
# print(fitness_score) # 1.0185991537088997e-06 => Question 4: C

#---------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------ Bai tap 4 ----------------------------------------------#
#---------------------------------------------------------------------------------------------------------------#

def crossover(individual1, individual2, crossover_rate = 0.9):
    individual1_new = individual1.copy()
    individual2_new = individual2.copy()
    for i in range(len(individual)):
        if random.random() < crossover_rate:
            individual1_new[i] = individual2[i]
            individual2_new[i] = individual1[i]
    return individual1_new, individual2_new

#question 5
individual1 = [4.09, 4.82, 3.10, 4.02]
individual2 = [3.44, 2.57, -0.79, -2.41]

individual1, individual2 = crossover(individual1, individual2, 2.0)
print("individual1: ", individual1)
print("individual2: ", individual2)

# individual1:  [3.44, 2.57, -0.79, -2.41]
# individual2:  [4.09, 4.82, 3.1, 4.02]
# => Question 5: D

#---------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------ Bai tap 5 ----------------------------------------------#
#---------------------------------------------------------------------------------------------------------------#

def mutate(individual, mutation_rate = 0.05):
    individual_m = individual.copy()
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual_m[i] = individual[i] + 0.05
    return individual_m

#Question 6
before_individual = [4.09, 4.82, 3.10, 4.02]
after_individual = mutate(individual, mutation_rate = 2.0)
print(before_individual == after_individual)
# => Question 6: False => A

#---------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------ Bai tap 6 ----------------------------------------------#
#---------------------------------------------------------------------------------------------------------------#

def initializePopulation(m):
  population = [create_individual() for _ in range(m)]
  return population

#---------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------ Bai tap 7 ----------------------------------------------#
#---------------------------------------------------------------------------------------------------------------#

def selection(sorted_old_population, m):
    index1 = random.randint(0, m-1)
    while True:
        index2 = random.randint(0, m-1)
        if (index2 != index1):
            break

    individual_s = sorted_old_population[index1]
    if index2 > index1:
        individual_s = sorted_old_population[index2]

    return individual_s

#---------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------ Bai tap 8 ----------------------------------------------#
#---------------------------------------------------------------------------------------------------------------#


def create_new_population(old_population, elitism=2, gen=1):
    m = len(old_population)
    sorted_population = sorted(old_population, key=compute_fitness)

    if gen%1 == 0:
        print("Best loss:", compute_loss(sorted_population[m-1]), "with chromsome: ", sorted_population[m-1])

    new_population = []
    while len(new_population) < m-elitism:
        # selection
        individual1 = selection(sorted_population, m)
        individual2 = selection(sorted_population, m)

        # crossover
        individual1_crossover, individual2_crossover = crossover(individual1, individual2)

        # mutation
        individual1_mutation = mutate(individual1_crossover)
        individual2_mutation = mutate(individual2_crossover)

        new_population.append(individual1_mutation)
        new_population.append(individual2_mutation)

        # copy elitism chromosomes that have best fitness score to the next generation
    for ind in sorted_population[m-elitism:]:
        new_population.append(ind)

    return new_population, compute_loss(sorted_population[m-1])

#Question 7
individual1 = [4.09, 4.82, 3.10, 4.02]
individual2 = [3.44, 2.57, -0.79, -2.41]
old_population = [individual1, individual2]
new_population, _ = create_new_population(old_population, elitism=2, gen=1)
# Best loss: 123415.051528805 with chromsome:  [3.44, 2.57, -0.79, -2.41] 
# => Question 7: A

#---------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------ Bai tap 9 ----------------------------------------------#
#---------------------------------------------------------------------------------------------------------------#

def run_GA():
  n_generations = 100
  m = 600
  features_X, sales_Y = load_data_from_file()
  population = initializePopulation(m)
  losses_list = []
  for i in range(n_generations):
    new_population, loss = create_new_population(population, elitism=2, gen=1)
    population = new_population
    losses_list.append(loss)
  return population, losses_list

#---------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------ Bai tap 10 ----------------------------------------------#
#---------------------------------------------------------------------------------------------------------------#

def visualize_loss(losses_list):
    plt.plot(losses_list)
    plt.xlabel("generation")
    plt.ylabel("losses")

population, losses_list = run_GA()
# Best loss: 2.9084840080684007 with chromsome:  [4.219090077794494, 0.056304233883703786, 0.13440891490750329, -0.013534081349910743]

# visualize_loss(losses_list)

#---------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------ Bai tap 11 ----------------------------------------------#
#---------------------------------------------------------------------------------------------------------------#


def visualize_predict_gt(population):
  # visualization of ground truth and predict value
  sorted_population = sorted(population, key=compute_fitness)
  print(sorted_population[-1])
  theta = np.array(sorted_population[-1])

  estimated_prices = []
  for feature in features_X:
    y_hat = feature.dot(theta)
    estimated_prices.append(y_hat)

  fig, ax = plt.subplots(figsize=(10, 6))
  plt.xlabel('Samples')
  plt.ylabel('Price')
  plt.plot(sales_Y, c='green', label='Real Prices')
  plt.plot(estimated_prices, c='blue', label='Estimated Prices')
  plt.legend()
  plt.show()

visualize_predict_gt(population)

plt.show()