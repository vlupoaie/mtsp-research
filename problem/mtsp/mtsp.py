import os
import re
import random
import matplotlib.pyplot as plt

from config.settings import INPUTS_DIR
from utils.utils import euclidean_distance

COLORS = ['red', 'green', 'purple', 'black', 'yellow', 'blue', 'brown']


class MTSPProblem:
    def __init__(self, input_file=None, nodes=50):
        # get data
        if input_file:
            input_file = os.path.join(INPUTS_DIR, input_file)
            self.data = self.read_input(input_file)
        else:
            self.data = self.generate_data(nodes)

        # parse and normalize data
        self.normalized_data, self.scale = self.normalize_data(self.data)

        self.distance_cache = {}

    @staticmethod
    def read_input(input_file):
        # read input file
        with open(input_file, 'r') as handle:
            lines = [line.strip() for line in handle.readlines()]

        # parse nodes coordinates
        data = {}
        for i, line in enumerate(lines):
            node = re.findall('\s*\d+\s*(\d+(?:\.\d+)?)\s*(\d+(?:\.\d+)?)', line)[0]
            node = tuple(map(float, node))
            data[i + 1] = node

        return data

    @staticmethod
    def generate_data(nodes):
        # generate random data in a square
        data = {}
        already_generated = set()
        for i in range(nodes + 1):
            x = random.randint(0, 250)
            y = random.randint(0, 250)
            while (x, y) in already_generated:
                x = random.randint(0, 250)
                y = random.randint(0, 250)
            already_generated.add((x, y))
            data[i + 1] = (x, y)

        return data

    @staticmethod
    def normalize_data(data):
        normalized_data = {}

        nodes = data.values()
        min_x = min(node[0] for node in nodes)
        max_x = max(node[0] for node in nodes)
        min_y = min(node[1] for node in nodes)
        max_y = max(node[1] for node in nodes)

        # translate all the nodes in the first quadrant
        for index, node in data.items():
            normalized_data[index] = [node[0] - min_x, node[1] - min_y]

        # update max values
        max_x -= min_x
        max_y -= min_y

        # scale all the nodes keeping the ratio
        scale = max(max_x, max_y)
        for index, node in list(normalized_data.items()):
            normalized_data[index] = [node[0] / scale, node[1] / scale]

        return normalized_data, scale

    def get_data(self, normalized=False):
        if not normalized:
            return self.data
        else:
            return self.normalized_data

    def get_distance(self, x, y):
        if x > y:
            x, y = y, x
        if (x, y) in self.distance_cache:
            return self.distance_cache[(x, y)]
        distance = euclidean_distance(self.data[x], self.data[y])
        self.distance_cache[(x, y)] = distance
        return distance

    def solution_cost(self, solution, target='sum'):
        if target == 'sum':
            return self._sum_cost(solution)
        elif target == 'min-max':
            return self._min_max_cost(solution)
        else:
            raise ValueError("Unknown target cost function")

    @staticmethod
    def solution_cost_precomputed(solution, target='sum'):
        if target == 'sum':
            return sum(item[0] for item in solution)
        elif target == 'min-max':
            return max(item[0] for item in solution)
        else:
            raise ValueError("Unknown target cost function")

    def tour_cost(self, tour):
        cost = self.get_distance(1, tour[0])
        for counter in range(1, len(tour)):
            cost += self.get_distance(tour[counter - 1], tour[counter])
        cost += self.get_distance(1, tour[-1])
        return cost

    def _sum_cost(self, solution):
        cost = 0
        for tour in solution:
            if not tour:
                continue
            cost += self.tour_cost(tour)
        return cost

    def _min_max_cost(self, solution):
        cost = None
        for tour in solution:
            if not tour:
                continue
            temp_cost = self.tour_cost(tour)

            if not cost or temp_cost > cost:
                cost = temp_cost
        return cost

    def plot_solution(self, solution):
        # plot nodes
        for index, node in self.data.items():
            if index == 1:
                plt.plot(node[0], node[1], 'o', markersize=2, color='black')
                continue
            plt.plot(node[0], node[1], 'bo', markersize=4)

        # plot tours
        for counter, tour in enumerate(solution):
            route_x, route_y = list(zip(self.data[1], *(self.data[i] for i in tour), self.data[1]))
            plt.setp(plt.plot(route_x, route_y), color=COLORS[counter % len(COLORS)])

        plt.axes().set_aspect('equal', 'datalim')
        plt.show()
