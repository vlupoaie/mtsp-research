import os
import json
from concurrent.futures import ProcessPoolExecutor, wait

from problem.mtsp import MTSPProblem
from initialisers.som import SOMCircle, SOM
from solvers.genetic_algorithm import GeneticAlgorithm
from config.settings import INPUTS_DIR, OUTPUTS_DIR


def generate_som_solutions():
    # input_files = ['eil51.tsp', 'berlin52.tsp', 'eil76.tsp', 'rat99.tsp']
    # input_files = ['gr202.tsp']
    input_files = ['eil76.tsp']
    neurons_multiplier = 3
    salesmen_versions = [2, 3, 5, 7]
    salesmen_versions = [5]
    population_size = 1

    tasks = {}
    with ProcessPoolExecutor(max_workers=1) as executor:

        # schedule tasks to be executed
        for input_file in input_files:
            # make solutions directory for this input file
            solutions_dir = os.path.join(INPUTS_DIR, 'som_solutions', input_file.split('.')[0])
            try:
                os.makedirs(solutions_dir)
            except FileExistsError:
                pass

            # initialise problem using input file
            problem = MTSPProblem(input_file)
            data = problem.get_data(normalized=True)
            neurons = (len(data) - 1) * neurons_multiplier

            for salesmen in salesmen_versions:
                # make salesmen directory
                salesmen_dir = os.path.join(solutions_dir, str(salesmen))
                try:
                    os.mkdir(salesmen_dir)
                except FileExistsError:
                    pass

                # compute max rotations and rotation incrementation
                max_rotation = 1 / salesmen
                rotation_chunk = max_rotation / population_size

                for count in range(population_size):
                    print('Running {} with {} salesmen ({})'.format(input_file, salesmen, count + 1))

                    # get circle center coordinates
                    nodes = data.values()
                    min_x = min(node[0] for node in nodes)
                    max_x = max(node[0] for node in nodes)
                    min_y = min(node[1] for node in nodes)
                    max_y = max(node[1] for node in nodes)
                    x_center = (max_x - min_x) / 2
                    y_center = (max_y - min_y) / 2

                    # initialise neurons on circle
                    init = SOMCircle(neurons=neurons, salesmen=salesmen, center=(x_center, y_center),
                                     radius=0.5, depot=data[1], rotation=rotation_chunk * count)

                    # run som and get solution tours
                    som = SOM(problem, salesmen, init, learning_rate=0.2, iterations=5000, plots=10)
                    tasks[executor.submit(som.run)] = {
                        'output_file': os.path.join(salesmen_dir, str(count + 1))
                    }

        # wait for tasks to finish and consume results
        while tasks:
            done, not_done = wait(tasks, timeout=10)
            if not done:
                continue

            # process results
            for task in done:
                result = task.result()
                print('Task done: {} - {}'.format(tasks[task], result[0]))

                # save solution
                with open(tasks[task]['output_file'], 'w') as h:
                    h.write(json.dumps(result))

                # remove finished task
                del tasks[task]


def run_genetic_with_som():
    # input_files = ['eil51.tsp', 'berlin52.tsp', 'eil76.tsp', 'rat99.tsp']
    # input_files = ['berlin52.tsp', 'eil76.tsp', 'rat99.tsp']
    # input_files = ['eil76.tsp', 'rat99.tsp']
    input_files = ['rat99.tsp']
    # input_files = ['ali535.tsp']
    # input_files = ['gr202.tsp']
    salesmen_versions = [2, 3, 5, 7]
    salesmen_versions = [7]
    # salesmen_versions = [2]
    population_size = 100

    tasks = {}
    with ProcessPoolExecutor(max_workers=1) as executor:

        # schedule tasks to be executed
        for input_file in input_files:
            solutions_dir = os.path.join(INPUTS_DIR, 'som_solutions', input_file.split('.')[0])

            # initialise problem using input file
            problem = MTSPProblem(input_file)
            test_name = input_file.split('.')[0]

            for salesmen in salesmen_versions:
                salesmen_dir = os.path.join(solutions_dir, str(salesmen))

                # get som solutions
                population = []
                for count in range(population_size):
                    solution_file = os.path.join(salesmen_dir, str(count + 1))
                    with open(solution_file, 'r') as h:
                        population.append(json.loads(h.read()))

                # run genetic algorithm multiple times and save each result
                for count in range(1):
                    genetic = GeneticAlgorithm(problem=problem, salesmen=salesmen, population=population,
                                               plots=10, prints=50, target='min-max')
                    tasks[executor.submit(genetic.run)] = {
                        'test_name': test_name,
                        'salesmen': salesmen
                    }

        # wait for tasks to finish and consume results
        while tasks:
            done, not_done = wait(tasks, timeout=10)
            if not done:
                continue

            # process results
            for task in done:
                result = task.result()
                print('Task done: {} - {}'.format(tasks[task], result[0]))

                # save solution
                solution_file = os.path.join(OUTPUTS_DIR + '2', 'genetic_{}_{}'.format(tasks[task]['test_name'],
                                                                                       tasks[task]['salesmen']))
                with open(solution_file, 'a') as h:
                    h.write(json.dumps(result) + '\n')

                # remove finished task
                del tasks[task]


def run_genetic_no_som():
    input_files = ['eil51.tsp', 'berlin52.tsp', 'eil76.tsp', 'rat99.tsp']
    # input_files = ['berlin52.tsp', 'eil76.tsp', 'rat99.tsp']
    # input_files = ['eil76.tsp', 'rat99.tsp']
    input_files = ['rat99.tsp']
    # input_files = ['ali535.tsp']
    # input_files = ['gr202.tsp']
    salesmen_versions = [2, 3, 5, 7]
    salesmen_versions = [7]
    # salesmen_versions = [2]
    population_size = 100

    tasks = {}
    with ProcessPoolExecutor(max_workers=1) as executor:

        # schedule tasks to be executed
        for input_file in input_files:
            # initialise problem using input file
            problem = MTSPProblem(input_file)
            test_name = input_file.split('.')[0]

            for salesmen in salesmen_versions:
                # run genetic algorithm multiple times and save each result
                for count in range(1):
                    genetic = GeneticAlgorithm(problem=problem, salesmen=salesmen, population_size=population_size,
                                               plots=10, prints=50, target='min-max')
                    tasks[executor.submit(genetic.run)] = {
                        'test_name': test_name,
                        'salesmen': salesmen
                    }

        # wait for tasks to finish and consume results
        while tasks:
            done, not_done = wait(tasks, timeout=10)
            if not done:
                continue

            # process results
            for task in done:
                result = task.result()
                print('Task done: {} - {}'.format(tasks[task], result[0]))

                # save solution
                solution_file = os.path.join(OUTPUTS_DIR + '_simple', 'genetic_{}_{}'.format(tasks[task]['test_name'],
                                                                                             tasks[task]['salesmen']))
                with open(solution_file, 'a') as h:
                    h.write(json.dumps(result) + '\n')

                # remove finished task
                del tasks[task]


if __name__ == '__main__':
    # generate_som_solutions()
    run_genetic_with_som()
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
