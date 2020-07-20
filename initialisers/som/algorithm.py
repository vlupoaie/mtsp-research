import random
from math import pi, cos, sin, log, exp
import matplotlib.pyplot as plt

from problem.mtsp import MTSPProblem
from utils.utils import euclidean_distance

COLORS = ['red', 'green', 'purple', 'black', 'yellow', 'blue', 'brown']


class SOMCircle:
    def __init__(self, neurons, salesmen, center, radius, depot, rotation=None,
                 distribution=None, tour_distribution=None, plot=False):
        self.neurons = neurons
        self.center = center
        self.radius = radius
        self.depot = [*depot, 0]
        self.plot = plot

        if distribution is None:
            self.distribution = [1 / neurons for _ in range(neurons)]
        else:
            if round(sum(distribution), 10) != 1:
                raise ValueError('Distribution must sum up to 1')
            self.distribution = distribution

        if rotation is None:
            self.rotation = 0
        else:
            if not (0 <= rotation < 1):
                raise ValueError('Rotation must be between 0 and 1')
            self.rotation = rotation

        if tour_distribution is None:
            self.tour_distribution = []
            remaining = neurons
            for i in range(salesmen):
                value = remaining // (salesmen - i)
                self.tour_distribution.append(value)
                remaining -= value
        else:
            if not len(tour_distribution) == salesmen:
                raise ValueError('Tour distribution must be a list of salesmen count integers')
            self.tour_distribution = tour_distribution

    def get_neurons(self):
        neurons = []
        angle_percent = self.rotation

        # generate nodes in a circle using the distribution
        neurons_counter = 0
        for tour_count, this_tour in enumerate(self.tour_distribution):
            for neuron_count in range(this_tour):
                angle = angle_percent * pi * 2
                x = cos(angle) * self.radius + self.center[0]
                y = sin(angle) * self.radius + self.center[1]
                neurons.append([x, y, tour_count + 1])
                angle_percent += self.distribution[neurons_counter]
                neurons_counter += 1
            neurons.append(self.depot)

        if self.plot:
            self.plot_initializer(neurons)

        return neurons

    @staticmethod
    def plot_initializer(neurons):
        for node in neurons:
            plt.plot(node[0], node[1], 'o', color=COLORS[node[2] % len(COLORS)])
        plt.axes().set_aspect('equal', 'datalim')
        plt.show()


class SOM:
    def __init__(self, problem, salesmen, neurons_initializer, iterations=20000, learning_rate=0.6,
                 min_learning_rate=0.01, plots=0, save_plots=False, push=0):
        self.problem = problem
        self.data = problem.get_data(normalized=True)
        if not plots:
            self.plot_each = iterations + 1
        else:
            self.plot_each = iterations // plots
        self.save_plots = save_plots
        self.push = push
        self.neurons = neurons_initializer.get_neurons()
        self.salesmen = salesmen
        self.iterations = iterations
        self.initial_learning_rate = learning_rate
        self.learning_rate = learning_rate
        self.decay_constant = self.iterations / log(len(self.neurons))
        self.decay_learning_rate = self.iterations / log(self.learning_rate / min_learning_rate)
        self.solution = []

    def run(self):
        # run algorithm and update neurons
        iterations_count = 0
        while iterations_count < self.iterations:
            if not iterations_count % self.plot_each and self.plot_each <= self.iterations:
                self.plot()
            chosen_index = random.randint(2, len(self.data))
            chosen_node = self.data[chosen_index]
            bmu_index, bmu = self.get_best_matching_unit(chosen_node)

            self.update_neurons(chosen_node, bmu_index, bmu, iterations_count)
            iterations_count += 1

        # compute final solution
        self.compute_solution()

        # self.problem.plot_solution(self.solution)

        return self.solution

    def get_best_matching_unit(self, node):
        return min(enumerate(filter(lambda k: k[2], self.neurons)), key=lambda x: euclidean_distance(x[1], node))

    def update_neurons(self, chosen_node, bmu_index, bmu, iteration):
        radius = len(self.neurons) * exp(-iteration / self.decay_constant)

        # update each neuron in the neighbourhood
        for index, neuron in enumerate(filter(lambda k: k[2], self.neurons)):
            distance = min(abs(index - bmu_index), abs(index - bmu_index + len(self.neurons)),
                           abs(index - bmu_index - len(self.neurons)))
            if distance >= radius:
                continue

            sign = -1 if bmu[2] != neuron[2] and self.push and self.push < iteration else 1

            distance_coefficient = exp(- distance ** 2 / (2 * (radius / 10) ** 2))
            neuron[0] += sign * distance_coefficient * self.learning_rate * (chosen_node[0] - neuron[0])
            neuron[1] += sign * distance_coefficient * self.learning_rate * (chosen_node[1] - neuron[1])

        # update learning rate
        self.learning_rate = self.initial_learning_rate * exp(-iteration / self.decay_learning_rate)
        print(self.learning_rate)

    def compute_solution(self):
        self.solution = []
        for index, node in self.data.items():
            if index == 1:
                continue
            best_neuron = None
            best_distance = None
            for count, neuron in enumerate(self.neurons):
                if not neuron[2]:
                    continue
                distance = euclidean_distance(node, neuron)
                if not best_neuron or distance < best_distance:
                    best_distance = distance
                    best_neuron = count
            self.solution.append((index, best_neuron))

        # add depots between nodes
        for count, depot in enumerate(self.neurons):
            if depot[2]:
                continue
            self.solution.append((1, count))

        # sort solution according to neuron index
        self.solution.sort(key=lambda x: x[1])

        # split solution in salesmen
        temp_solution = []
        last_index = 0
        for index, node in enumerate(self.solution):
            if node[0] == 1:
                temp_solution.append([item[0] for item in self.solution[last_index:index]])
                last_index = index + 1
        self.solution = temp_solution

    def plot(self):
        # plot nodes
        for index, node in self.data.items():
            if index == 1:
                continue
            plt.plot(node[0], node[1], 'bo', markersize=4)

        # plot neurons
        route_x = [self.neurons[-1][0]] + [item[0] for item in self.neurons]
        route_y = [self.neurons[-1][1]] + [item[1] for item in self.neurons]
        plt.setp(plt.plot(route_x, route_y), color='red')

        plt.axes().set_aspect('equal', 'datalim')
        plt.show()


if __name__ == '__main__':
    problem = MTSPProblem('eil76.tsp')
    salesmen = 7

    for i in range(8):
        data = problem.get_data(normalized=True)
        init = SOMCircle((len(data) - 1) * 4, salesmen, (0.5, 0.5), 0.5, data[1], rotation=0.3 + i * 0.01)
        som = SOM(problem, salesmen, init, learning_rate=1, iterations=10, plots=1, push=0)
        som.run()
    exit()

    data = problem.get_data(normalized=True)
    init = SOMCircle((len(data) - 1) * 4, salesmen, (0.5, 0.5), 0.5, data[1], rotation=0.3)

    som = SOM(problem, salesmen, init, learning_rate=1, iterations=3000, plots=10, push=0)
    solution = som.run()
    problem.plot_solution(solution)

    print('Total sum:', problem.solution_cost(solution))
    print('Min-max cost:', problem.solution_cost(solution, target='min-max'))
