import numpy as np
from ga.crossover import arithmetic_crossover, blx_alpha_crossover

class RealGeneticAlgorithm:
    """
    A genetic algorithm that works directly with real numbers - no binary conversion needed!
    Perfect for smooth optimization problems where we want precise solutions.
    """
    def __init__(self, func, bounds, pop_size=50, generations=100, mutation_rate=0.01, crossover_rate=0.8, crossover_type='arithmetic', blx_alpha=0.5):
        self.func = func
        self.bounds = bounds  # The playground boundaries for each dimension
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.crossover_type = crossover_type
        self.blx_alpha = blx_alpha
        self.dim = len(bounds)

    def _init_population(self):
        # Create our initial group of random solutions
        pop = []
        for _ in range(self.pop_size):
            ind = np.array([np.random.uniform(low, high) for (low, high) in self.bounds])
            pop.append(ind)
        return np.array(pop)

    def _fitness(self, population):
        # Score each solution - lower is better
        return -self.func(population[:, 0], population[:, 1])

    def _select(self, population, fitness):
        # Choose the best performers for the next generation
        idx = np.random.choice(np.arange(self.pop_size), size=self.pop_size, p=fitness/fitness.sum())
        return population[idx]

    def _crossover(self, parent1, parent2):
        # Blend two solutions together
        if self.crossover_type == 'arithmetic':
            return arithmetic_crossover(parent1, parent2)
        else:
            return blx_alpha_crossover(parent1, parent2, self.blx_alpha)

    def _mutate(self, individual):
        # Shake things up a bit with random changes
        for i, (low, high) in enumerate(self.bounds):
            if np.random.rand() < self.mutation_rate:
                individual[i] = np.random.uniform(low, high)
        return individual

    def run(self):
        # Let's find the best solution!
        population = self._init_population()
        best_fitness = []
        for gen in range(self.generations):
            fitness = self._fitness(population)
            # Pick our champions
            fitness_shifted = fitness - fitness.min() + 1e-6
            selected = self._select(population, fitness_shifted)
            # Create the next generation
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
        # Return our best solution
        fitness = self._fitness(population)
        best_idx = np.argmax(fitness)
        best_solution = population[best_idx]
        return best_solution, -fitness[best_idx], best_fitness
