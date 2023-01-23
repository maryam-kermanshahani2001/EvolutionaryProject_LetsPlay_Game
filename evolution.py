import copy
import json

from player import Player
import numpy as np
from config import CONFIG


class Evolution():

    def __init__(self, mode):
        self.mode = mode
        # bonus part I added this variable
        self.generation = 0

    # calculate fitness of players
    def calculate_fitness(self, players, delta_xs):
        for i, p in enumerate(players):
            p.fitness = delta_xs[i]

    def mutate(self, child):

        # TODO
        # child: an object of class `Player`
        import random
        p = 0.8
        # TODO
        # child: an object of class `Player`
        for i in range(len(child.nn.w)):
            gaussian_noise = np.random.normal(size=child.nn.w[i].shape)
            rand = random.uniform(0, 1)

            if rand < p:
                child.nn.w[i] += gaussian_noise

            gaussian_noise = np.random.normal(size=child.nn.b[i].shape)
            rand = random.uniform(0, 1)

            if rand < p:
                child.nn.b[i] += gaussian_noise
        return child

    def generate_new_population(self, num_players, prev_players=None):
        # TODO
        # num_players example: 150
        # prev_players: an array of `Player` objects
        # TODO (additional): a selection method other than `fitness proportionate`
        # TODO (additional): implementing crossover

        import random, copy
        # create random generation in the first round
        if prev_players is None:
            return [Player(self.mode) for _ in range(num_players)]

        # in rounds other than the first one:
        else:
            # Q tournoment method for choosing children 
            Q = 10
            parents_list = []
            children = []

            for _ in range(num_players):
                random_players = random.sample(prev_players, Q)
                best_player = max(random_players, key=lambda x: x.fitness)
                parents_list.append(copy.deepcopy(best_player))

            for _ in range(num_players):
                parents = random.sample(parents_list, 2)
                child = Player('helicopter')

                # w1
                above_rate = np.vsplit(parents[0].nn.w[0], 2)
                below_rate = np.vsplit(parents[1].nn.w[0], 2)
                child.nn.w[0] = np.concatenate((above_rate[0], below_rate[1]), axis=0)

                # w2
                above_rate = np.hsplit(parents[0].nn.w[1], 2)
                below_rate = np.hsplit(parents[1].nn.w[1], 2)
                child.nn.w[1] = np.concatenate((above_rate[0], below_rate[1]), axis=1)

                # b1
                above_rate = np.vsplit(parents[0].nn.b[0].reshape(parents[0].nn.b[0].shape[0], 1), 2)
                below_rate = np.vsplit(parents[1].nn.b[0].reshape(parents[1].nn.b[0].shape[0], 1), 2)
                child.nn.b[0] = np.concatenate((above_rate[0], below_rate[1]), axis=0)

                # b2
                child.nn.b[1] = parents[1].nn.b[1]

                children.append(self.mutate(child))
            new_players = children
            return new_players

    def next_population_selection(self, players, num_players):
        # TODO
        # num_players example: 100
        # players: an array of `Player` objects

        # TODO (additional): a selection method other than `top-k`
        # TODO (additional): plotting

        import heapq, random, copy

        # next_pop = []

        result = players
        # selection_method = "roulette wheel"
        result = self.roulette_wheel(players, num_players)
        score_list = [player.fitness for player in players]
        min_score = float(np.min(score_list))
        max_score = float(np.max(score_list))
        avg_score = float(np.mean(score_list))
        self.save_to_file_for_plotting(min_score, max_score, avg_score)

        return result

    def roulette_wheel(self, players, parent_numbers):
        probabilities = self.calculate_sum_of_probs(players)

        results = []
        for rand in np.random.uniform(low=0, high=1, size=parent_numbers):
            for i, probability in enumerate(probabilities):
                if rand <= probability:
                    results.append(copy.deepcopy(players[i]))
                    break

        return results

    def calculate_sum_of_probs(self, players):
        sum_of_fitness = 0
        for player in players:
            sum_of_fitness += player.fitness
        probs = []
        for p in players:
            probs.append(p.fitness / sum_of_fitness)
        # turn it to cumulative probability
        for i in range(1, len(players)):
            probs[i] += probs[i - 1]
        return probs

    def save_to_file_for_plotting(self, min_score, max_score, avg_score):
        if self.generation != 0:
            # print("Hello")

            fit_res = json.load(open("fit_res.json", "r"))
            fit_res['max_score'].append(max_score)
            fit_res['min_score'].append(min_score)
            fit_res['avg_score'].append(avg_score)

            json.dump(fit_res, open("fit_res.json", "w"))

        else:
            fit_res = {'max_score': [max_score], 'min_score': [min_score], 'avg_score': [avg_score]}

            json.dump(fit_res, open("fit_res.json", "w"))

            # writer = open('fitness_results.json', 'w')
            # writer.write(json_version)

            # writer.write(fitness_results_dict+"\n")
            # Further file processing goes here
        self.generation += 1
