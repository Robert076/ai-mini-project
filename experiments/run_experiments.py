import numpy as np
from functions.griewank import griewank
from functions.ackley import ackley
from ga.binary_ga import BinaryGeneticAlgorithm
from ga.real_ga import RealGeneticAlgorithm
import os

def run_experiments():
    """
    Run all combinations of (function, representation, crossover) for 30 runs each.
    Save results for later analysis.
    """
    os.makedirs('experiments/results', exist_ok=True)
    n_runs = 30
    generations = 100
    pop_size = 50
    mutation_rate = 0.01
    crossover_rate = 0.8
    n_bits = 16
    bounds_griewank = [(-600, 600), (-600, 600)]
    bounds_ackley = [(-32.768, 32.768), (-32.768, 32.768)]

    configs = [
        # (function, bounds, name)
        (griewank, bounds_griewank, 'griewank'),
        (ackley, bounds_ackley, 'ackley')
    ]
    results = {}
    for func, bounds, fname in configs:
        # Binary GA
        for crossover_type in ['1-point', '2-point']:
            key = f'{fname}_binary_{crossover_type}'
            results[key] = []
            for run in range(n_runs):
                ga = BinaryGeneticAlgorithm(
                    func, bounds, n_bits=n_bits, pop_size=pop_size, generations=generations,
                    mutation_rate=mutation_rate, crossover_rate=crossover_rate, crossover_type=crossover_type
                )
                best_sol, best_fit, fit_hist = ga.run()
                results[key].append(best_fit)
            np.save(f'experiments/results/{key}.npy', np.array(results[key]))
        # Real GA
        for crossover_type in ['arithmetic', 'blx']:
            key = f'{fname}_real_{crossover_type}'
            results[key] = []
            for run in range(n_runs):
                ga = RealGeneticAlgorithm(
                    func, bounds, pop_size=pop_size, generations=generations,
                    mutation_rate=mutation_rate, crossover_rate=crossover_rate,
                    crossover_type='arithmetic' if crossover_type=='arithmetic' else 'blx',
                    blx_alpha=0.5
                )
                best_sol, best_fit, fit_hist = ga.run()
                results[key].append(best_fit)
            np.save(f'experiments/results/{key}.npy', np.array(results[key]))
    print('All done! Your results are ready to explore in the results folder.')

if __name__ == '__main__':
    run_experiments()
