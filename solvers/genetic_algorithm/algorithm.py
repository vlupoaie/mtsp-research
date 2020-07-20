import random
from copy import deepcopy


class GeneticAlgorithm:
    def __init__(self, problem, salesmen, generations=2500, population=None, population_size=100, keep_best=0.25,
                 crossover_chance=0.4, mutation_chance=0.1, plots=0, prints=0, two_opt=2500, save_plots=False,
                 target='sum'):
        self.problem = problem
        self.data = problem.get_data(normalized=True)
        self.target = target
        if not plots:
            self.plot_each = generations + 1
        else:
            self.plot_each = generations // plots
        if not prints:
            self.print_each = generations + 1
        else:
            self.print_each = generations // prints
        self.save_plots = save_plots
        self.two_opt_each = two_opt
        self.salesmen = salesmen
        self.generations = generations
        if population:
            self.population_size = len(population)
            self.population = self.initialize_population(population)
        else:
            self.population_size = population_size
            self.population = self.generate_population()
        self.crossover_chance = crossover_chance
        self.mutation_chance = mutation_chance
        self.keep_best = int(keep_best * population_size)
        self.best = deepcopy(sorted(self.population, key=lambda x: x[0], reverse=True)[:self.keep_best])

    def run(self):
        for generation in range(1, self.generations + 1):
            # select best individuals from the population
            self.selection()

            # add best individuals back
            self.population += deepcopy(self.best)

            # cross over chromosomes from an individual
            for individual in self.population:
                if random.random() < self.crossover_chance:
                    self.external_mutation(individual)

            # do internal mutations for each chromosome of an individual
            for individual in self.population:
                changed = False
                for chromosome_count in range(self.salesmen):
                    if random.random() < self.mutation_chance:
                        changed = True
                        individual[1][chromosome_count][1:] = \
                            self.internal_mutation(individual[1][chromosome_count][1:])
                if changed:
                    individual[0] = self.get_fitness(individual[1])

            # run 2-opt local search once in a while
            if not generation % self.two_opt_each or generation == 1:
                self.two_opt(only_best=False)

            # remember new best
            self.best = deepcopy(sorted(self.population + self.best, key=lambda x: x[0], reverse=True)[:self.keep_best])

            # print current generation and best individual so far
            best_individual = self.best[0]
            if (not generation % self.print_each or generation == 1) and self.print_each <= self.generations:
                print('Generation {}. Best distance so far {:.2f}'.format(
                    generation, self.problem.solution_cost_precomputed(best_individual[1], target=self.target)))

            # plot best model
            if (not generation % self.plot_each or generation == 1) and self.plot_each <= self.generations:
                self.plot_best(best_individual)

        full_solution = [[1, *tour[1:], 1] for tour in self.best[0][1]]
        return self.problem.solution_cost_precomputed(self.best[0][1], target=self.target), full_solution

    def initialize_population(self, population):
        new_population = []
        for chromosomes in population:
            new_chromosomes = []
            for tour in chromosomes:
                new_tour = [0, *tour]
                new_chromosomes.append(new_tour)
            fitness = self.get_fitness(new_chromosomes)
            new_population.append([fitness, new_chromosomes])

        return new_population

    def generate_population(self):
        population = []
        for i in range(self.population_size):
            chromosomes = self.generate_chromosomes()
            fitness = self.get_fitness(chromosomes)
            population.append([fitness, chromosomes])

        return population

    def generate_chromosomes(self):
        # assign each salesman an almost average number of cities in a random order
        chromosomes = [[]]
        while not all(chromosomes):
            chromosomes = self.random_partition(self.salesmen, range(2, len(self.data) + 1))
        new_chromosomes = []
        for chromosome in chromosomes:
            random.shuffle(chromosome)
            new_chromosome = [0, *chromosome]
            new_chromosomes.append(new_chromosome)

        return new_chromosomes

    @staticmethod
    def random_partition(k, iterable):
        partitions = [[] for _ in range(k)]
        for value in iterable:
            x = random.randrange(k)
            partitions[x].append(value)
        return partitions

    def get_fitness(self, chromosomes):
        for chromosome in chromosomes:
            tour_cost = self.problem.tour_cost(chromosome[1:])
            chromosome[0] = tour_cost
        solution_cost = self.problem.solution_cost_precomputed(chromosomes, target=self.target)
        return self.reverse_fitness(solution_cost)

    @staticmethod
    def reverse_fitness(fitness):
        return 5000 / fitness

    def selection(self):
        # wheel of fortune selection
        total_fitness = sum(subject[0] for subject in self.population)

        # compute partial selection probabilities
        partial_fitness = 0
        selection_probabilities = []
        for subject in self.population:
            partial_fitness += subject[0]
            selection_probabilities.append(partial_fitness / total_fitness)

        selected = []
        for _ in range(self.population_size):
            random_number = random.random()

            # retrieve selected member
            for counter, value in enumerate(selection_probabilities):
                if random_number < value:
                    selected.append(counter)
                    break

        # save new population keeping same number of individuals
        selected_population = []
        for x in selected:
            # selected_population.append(deepcopy(self.population[x]))
            selected_population.append(self.population[x])
        self.population = selected_population

    def internal_mutation(self, chromosome):
        individual_chance = self.mutation_chance / len(chromosome)
        new_chromosome = list(chromosome)

        # randomly choose a type of mutation
        for gene_count in range(len(chromosome) - 2):
            if random.random() > individual_chance:
                continue

            # do mutation
            mutation_type = random.choice(['reverse', 'switch', 'change place'])
            if mutation_type == 'reverse':
                next_gene = random.choice(range(gene_count + 2, len(chromosome) + 1))
                new_chromosome = chromosome[:gene_count] + list(reversed(chromosome[gene_count:next_gene])) + \
                    chromosome[next_gene:]
            elif mutation_type == 'switch':
                next_gene = random.choice(range(len(chromosome)))
                new_chromosome[gene_count], new_chromosome[next_gene] = \
                    new_chromosome[next_gene], new_chromosome[gene_count]
            elif mutation_type == 'change place':
                next_gene = random.choice(range(len(chromosome) + 1))
                old_gene = chromosome[gene_count]
                new_chromosome = chromosome[:gene_count] + chromosome[gene_count + 1:]
                new_chromosome.insert(next_gene, old_gene)
            break

        return new_chromosome

    def external_mutation(self, individual):
        chromosomes = individual[1]

        selected = []

        for x in range(self.salesmen):
            random_number = random.random()
            if random_number < self.crossover_chance:
                selected.append(x)
        random.shuffle(selected)

        special_crossover = random.random()
        # if special_crossover < 0.2:
        #     longest_index = sorted(range(self.salesmen), key=lambda k: chromosomes[k][0], reverse=True)[0]
        #     try:
        #         selected.remove(longest_index)
        #     except ValueError:
        #         pass
        #     selected.insert(0, longest_index)
        #     try:
        #         selected.append(random.choice([item for item in range(self.salesmen) if item not in selected]))
        #     except IndexError:
        #         pass
        if special_crossover < 0.1:
            selected.sort(key=lambda k: chromosomes[k][0], reverse=True)

        if len(selected) % 2 != 0:
            selected = selected[:-1]

        count_selected = 0
        while count_selected < len(selected):
            # crossing chromosomes
            x = selected[count_selected]
            y = selected[-count_selected - 1]

            first = chromosomes[x][1:]
            second = chromosomes[y][1:]

            # generate random crossing point
            first_start = random.choice(range(0, len(first)))
            first_end = random.choice(range(first_start + 1, len(first) + 1))
            second_start = random.choice(range(0, len(second)))
            second_end = random.choice(range(second_start + 1, len(second) + 1))

            # get parts to switch
            mid_part_first = first[first_start:first_end]
            mid_part_second = second[second_start:second_end]

            # rebuild chromosomes with switched parts
            temp_first = first[:first_start] + mid_part_second + first[first_end:]
            temp_second = second[:second_start] + mid_part_first + second[second_end:]

            # remember new chromosomes
            chromosomes[x][1:] = temp_first
            chromosomes[y][1:] = temp_second

            # remove these two
            count_selected += 2

        # recompute fitness
        individual[0] = self.get_fitness(chromosomes)

    def two_opt(self, only_best=True):
        if only_best:
            candidates = self.best
        else:
            candidates = self.population

        for individual in candidates:
            changed = False

            for tour in individual[1]:
                # insert depot at the beginning and the end
                parsed_tour = tour[1:]
                parsed_tour.insert(0, 1)
                parsed_tour.insert(len(parsed_tour), 1)

                # start checking for improvements
                tour_length = len(parsed_tour)
                for first_index in range(tour_length - 2):
                    for second_index in range(first_index + 2, tour_length - 1):
                        first_distance = \
                            self.problem.get_distance(parsed_tour[first_index], parsed_tour[first_index + 1]) + \
                            self.problem.get_distance(parsed_tour[second_index], parsed_tour[second_index + 1])
                        second_distance = \
                            self.problem.get_distance(parsed_tour[first_index], parsed_tour[second_index]) + \
                            self.problem.get_distance(parsed_tour[first_index + 1], parsed_tour[second_index + 1])
                        if first_distance > second_distance:
                            changed = True
                            parsed_tour[first_index + 1:second_index + 1] = \
                                reversed(parsed_tour[first_index + 1:second_index + 1])

                del parsed_tour[0]
                del parsed_tour[-1]
                tour[1:] = parsed_tour

            # recompute fitness if individual was changed
            if changed:
                individual[0] = self.get_fitness(individual[1])

    def plot_best(self, best_individual=None):
        if not best_individual:
            best_individual = self.best[0]
        self.problem.plot_solution([item[1:] for item in best_individual[1]])
