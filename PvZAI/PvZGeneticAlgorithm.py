import random
from PlantsVsZombies.source import constants as c

class PvZGeneticAlgorithm:
    def __init__(self, population_size, generations, mutation_rate, crossover_rate, alpha, beta, theta):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.alpha = alpha
        self.beta = beta
        self.theta = theta
        self.current_generation = 0

    def initialize_population(self):
        return [self.generate_chromosome() for _ in range(self.population_size)]

    def generate_chromosome(self):
        return [random.random() for _ in range(c.NUM_OF_FEATURES)]

    def uniform_crossover(self, parent1, parent2, p=0.5):
        child1, child2 = [], []
        for gene1, gene2 in zip(parent1, parent2):
            if random.random() < p:
                child1.append(gene1)
                child2.append(gene2)
            else:
                child1.append(gene2)
                child2.append(gene1)
        return child1, child2

    def arithmetic_crossover(self, parent1, parent2, alpha=0.5):
        child1 = [alpha * gene1 + (1 - alpha) * gene2 for gene1, gene2 in zip(parent1, parent2)]
        child2 = [(1 - alpha) * gene1 + alpha * gene2 for gene1, gene2 in zip(parent1, parent2)]
        return child1, child2

    def sbx_crossover(self, parent1, parent2, eta=1):
        child1, child2 = [], []
        for gene1, gene2 in zip(parent1, parent2):
            if random.random() <= 0.5:
                beta = (2 * random.random()) ** (1 / (eta + 1))
            else:
                beta = (1 / (2 * (1 - random.random()))) ** (1 / (eta + 1))
            child1.append(0.5 * ((1 + beta) * gene1 + (1 - beta) * gene2))
            child2.append(0.5 * ((1 - beta) * gene1 + (1 + beta) * gene2))
        return child1, child2

    def crossover(self, parent1, parent2):
        if random.random() < self.crossover_rate:
            # Assign higher probability to arithmetic crossover
            method = random.choices(
                [self.arithmetic_crossover, self.uniform_crossover, self.sbx_crossover],
                weights=[0.7, 0.15, 0.15],
                k=1
            )[0]
            return method(parent1, parent2)
        return parent1, parent2

    def mutate(self, chromosome):
        return [gene if random.random() > self.mutation_rate else random.random() for gene in chromosome]

    def select_parents(self, population, fitnesses, tournament_size=3):
        tournament = random.sample(list(zip(population, fitnesses)), tournament_size)
        return max(tournament, key=lambda x: x[1])[0]

    def evolve(self, evaluate_fitness):
        population = self.initialize_population()
        old_mutation = self.mutation_rate
        for i in range(self.generations):
            self.current_generation = i + 1
            fitnesses = [evaluate_fitness(chrom) for chrom in population]

            # Pair chromosomes with their fitness
            population_fitness = list(zip(population, fitnesses))

            # Sort based on fitness (descending order)
            sorted_population = sorted(population_fitness, key=lambda x: x[1], reverse=True)

            # Elitism: carry forward top 10%
            elite_count = max(1, int(0.1 * self.population_size))  # Ensure at least one elite
            elites = [chrom for chrom, fit in sorted_population[:elite_count]]
            new_population = elites.copy()

            # Generate the rest of the new population
            while len(new_population) < self.population_size:
                parent1 = self.select_parents(population, fitnesses)
                parent2 = self.select_parents(population, fitnesses)
                child1, child2 = self.crossover(parent1, parent2)
                new_population.extend([self.mutate(child1), self.mutate(child2)])

            # Trim the population if it exceeds the population size
            population = new_population[:self.population_size]
            print(f"Generation {i + 1} done.")
            self.mutation_rate *= 0.97

        # Final evaluation to find the best chromosome
        fitnesses = [evaluate_fitness(chrom) for chrom in population]
        best_chromosome = population[fitnesses.index(max(fitnesses))]
        self.mutation_rate = old_mutation
        return best_chromosome
