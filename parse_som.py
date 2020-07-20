import os
import json
from concurrent.futures import ProcessPoolExecutor, wait

from problem.mtsp import MTSPProblem
from initialisers.som import SOMCircle, SOM
from solvers.genetic_algorithm import GeneticAlgorithm
from config.settings import INPUTS_DIR, OUTPUTS_DIR


def parse_som_solutions():
    input_files = ['eil51', 'berlin52', 'eil76', 'rat99']
    salesmen_versions = [2, 3, 5, 7]

    data_results = {}
    for input_test in input_files:
        input_file = os.path.join(INPUTS_DIR, input_test + '.tsp')
        problem = MTSPProblem(input_file)
        if input_test not in data_results:
            data_results[input_test] = {}

        for salesman in salesmen_versions:
            sol_dir = os.path.join(INPUTS_DIR, 'som_solutions', input_test, str(salesman))
            distances = []
            for run in range(1, 301):
                sol_file = os.path.join(sol_dir, str(run))
                with open(sol_file, 'r') as h:
                    solution = json.loads(h.read())
                solution = [[1, *item, 1] for item in solution]
                distance = problem.solution_cost(solution, 'min-max')
                distances.append(distance)
            data_results[input_test][str(salesman)] = distances

    for input_test, s in data_results.items():
        results_path = os.path.join('som_results', input_test + '.csv')
        with open(results_path, 'w') as h:
            h.write('run,2,3,5,7\n')
            for c, values in enumerate(zip(s['2'], s['3'], s['5'], s['7'])):
                h.write('{},{}\n'.format(c + 1, ','.join(map(str, values))))


if __name__ == '__main__':
    parse_som_solutions()
    # generate_som_solutions()
    # run_genetic_with_som()
    # run_genetic_no_som()
    # solution_eil51_2 = [
    #     [1, 32, 11, 38, 5, 49, 10, 39, 33, 45, 15, 37, 17, 44, 42, 19, 40, 41, 13, 25, 18, 4, 47, 12, 46, 51, 27, 1],
    #     [1, 22, 2, 16, 50, 9, 30, 34, 21, 29, 20, 35, 36, 3, 28, 31, 8, 26, 7, 23, 43, 24, 14, 6, 48, 1]
    # ]
    # solution_eil51_2_our = [
    #     [1, 22, 2, 16, 50, 21, 29, 20, 35, 36, 3, 28, 31, 26, 8, 48, 23, 7, 43, 24, 14, 25, 18, 6, 1],
    #     [1, 32, 11, 38, 5, 49, 9, 34, 30, 10, 39, 33, 45, 15, 37, 17, 44, 42, 19, 40, 41, 13, 4, 47, 12, 46, 51, 27, 1]
    # ]
    # solution_eil51_5 = [
    #     [1, 11, 10, 39, 33, 45, 15, 44, 17, 47, 51, 27, 1],
    #     [1, 48, 6, 14, 24, 43, 23, 7, 26, 8, 31, 28, 1],
    #     [1, 4, 41, 40, 19, 42, 37, 12, 46, 1],
    #     [1, 32, 2, 16, 50, 9, 30, 34, 21, 29, 20, 35, 36, 3, 22, 1],
    #     [1, 38, 49, 5, 25, 13, 18, 1]
    # ]
    # problem = MTSPProblem('eil51.tsp')
    # print(problem.solution_cost(solution_eil51_2_our, 'min-max'))
