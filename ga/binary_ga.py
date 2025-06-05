import numpy as np
from ga.crossover import binary_1_point, binary_2_point

class BinaryGeneticAlgorithm:
    """
    My take on a binary-coded genetic algorithm for finding optimal solutions.
    Uses binary strings to represent solutions and evolves them through generations.
    """
    def __init__(self, func, bounds, n_bits=16, pop_size=50, generations=100, mutation_rate=0.01, crossover_rate=0.8, crossover_type='1-point'):
        self.func = func
        self.bounds = bounds  # Each dimension's min and max values
        self.n_bits = n_bits
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.crossover_type = crossover_type
        self.dim = len(bounds)
        self.chrom_length = self.n_bits * self.dim

    def _decode(self, chromosome):
        # Convert binary strings back to real numbers
        decoded = []
        for i, (low, high) in enumerate(self.bounds):
            bits = chromosome[i*self.n_bits:(i+1)*self.n_bits]
            integer = int(''.join(str(int(b)) for b in bits), 2)
            value = low + (high - low) * integer / (2**self.n_bits - 1)
            decoded.append(value)
        return np.array(decoded)

    def _init_population(self):
        # Create a random bunch of binary strings
        return np.random.randint(0, 2, (self.pop_size, self.chrom_length))

    def _fitness(self, population):
        # Figure out how good each solution is
        decoded = np.array([self._decode(ind) for ind in population])
        return -self.func(decoded[:, 0], decoded[:, 1])  # We're looking for the minimum

    def _select(self, population, fitness):
        # Pick the winners for the next round
        idx = np.random.choice(np.arange(self.pop_size), size=self.pop_size, p=fitness/fitness.sum())
        return population[idx]

    def _crossover(self, parent1, parent2):
        # Mix and match the best parts of two solutions
        if self.crossover_type == '1-point':
            return binary_1_point(parent1, parent2)
        else:
            return binary_2_point(parent1, parent2)

    def _mutate(self, chromosome):
        # Add some randomness to keep things interesting
        mutation_mask = np.random.rand(self.chrom_length) < self.mutation_rate
        chromosome[mutation_mask] = 1 - chromosome[mutation_mask]
        return chromosome

    def run(self):
        # Let's evolve some solutions!
        population = self._init_population()
        best_fitness = []
        for gen in range(self.generations):
            fitness = self._fitness(population)
            # Time to pick the winners
            fitness_shifted = fitness - fitness.min() + 1e-6
            selected = self._select(population, fitness_shifted)
            # Mix things up
            next_pop = []
            for i in range(0, self.pop_size, 2):
                p1, p2 = selected[i], selected[(i+1)%self.pop_size]
                if np.random.rand() < self.crossover_rate:
                    c1, c2 = self._crossover(p1, p2)
                else:
                    c1, c2 = p1.copy(), p2.copy()
                next_pop.extend([self._mutate(c1), self._mutate(c2)])
            population = np.array(next_pop)[:self.pop_size]
            best_fitness.append(fitness.max())
        # Find our champion
        fitness = self._fitness(population)
        best_idx = np.argmax(fitness)
        best_solution = self._decode(population[best_idx])
        return best_solution, -fitness[best_idx], best_fitness
