import random
from abc import ABCMeta, abstractmethod
from copy import deepcopy
from collections import deque
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from numpy import argmax
from tabulate import tabulate
from statistics import mean


class TabuSearch:
    # __metaclass__ = ABCMeta

    cur_steps = None

    tabu_size = None
    tabu_list = None

    initial_state = None
    current = None
    best = None

    max_steps = None
    max_score = None

    def __init__(self, data, initial_state, tabu_size, max_steps, max_score=None, max_group=4888, min_group=0):
        """
        :param initial_state: initial state, should implement __eq__ or __cmp__
        :param tabu_size: number of states to keep in tabu list
        :param max_steps: maximum number of steps to run algorithm for
        :param max_score: score to stop algorithm once reached
        """
        self.initial_state = initial_state
        self.data = data
        self.max_group = max_group
        self.min_group = min_group
        if isinstance(tabu_size, int) and tabu_size > 0:
            self.tabu_size = tabu_size
        else:
            raise TypeError('Tabu size must be a positive integer')

        if isinstance(max_steps, int) and max_steps > 0:
            self.max_steps = max_steps
        else:
            raise TypeError('Maximum steps must be a positive integer')

        if max_score is not None:
            if isinstance(max_score, (int, float)):
                self.max_score = float(max_score)
            else:
                raise TypeError('Maximum score must be a numeric type')

    # def __str__(self):
    #     return ('TABU SEARCH: \n' +
    #             'CURRENT STEPS: %d \n' +
    #             'BEST SCORE: %f \n' +
    #             'BEST MEMBER: %s \n\n') % \
    #            (self.cur_steps, self._score(self.best), str(self.best))
    #
    # def __repr__(self):
    #     return self.__str__()

    def _clear(self):
        """
        Resets the variables that are altered on a per-run basis of the algorithm
        :return: None
        """
        self.cur_steps = 0
        self.tabu_list = deque(maxlen=self.tabu_size)
        self.current = self.initial_state
        self.best = self.initial_state

    #    @abstractmethod
    def _score(self, solution):
        """
        Returns objective function value of a state
        :param solution: a solution
        :return: objective function value of solution
        """
        success_rate = solution[35] * 10
        manip_rate = solution[36]
        obj = success_rate + manip_rate
        return obj

    #    @abstractmethod
    def _neighborhood(self):
        """
        Returns list of all members of neighborhood of current state, given self.current
        :return: list of members of neighborhood
        """
        #print("current: ", self.current)
        group_number = self.current[0]
        if group_number == self.max_group:
            group_list = [arm for arm in self.data if arm[0] == group_number or arm[0] == self.min_group]
        else:
            group_list = [arm for arm in self.data if arm[0] == group_number or arm[0] == group_number + 1]
        #print("group_list", group_list)
        if not bool(group_list):
            group_list = self.current
        return group_list

    def _best(self, neighborhood):
        """
        Finds the best member of a neighborhood
        :param neighborhood: a neighborhood
        :return best member of neighborhood
        """
        #print("neighborhood ", neighborhood)
        return neighborhood[argmax([self._score(x) for x in neighborhood])]

    def run(self, verbose=True):
        """
        Conducts tabu search
        :param verbose: indicates whether or not to print progress regularly
        :return: best state and objective function value of best state
        """
        track = []
        track_score = []
        self._clear()
        for i in range(self.max_steps):
            self.cur_steps += 1

            if ((i + 1) % 100 == 0) and verbose:
                print(self)

            neighborhood = self._neighborhood()
            neighborhood_best = self._best(neighborhood)
            track.append(self.current)
            track_score.append(self._score(self.current))
            while True:
                if all([x in self.tabu_list for x in neighborhood]):
                    print("TERMINATING - NO SUITABLE NEIGHBORS")
                    return self.best, self._score(self.best)
                if neighborhood_best in self.tabu_list:
                    if self._score(neighborhood_best) > self._score(self.best):
                        self.tabu_list.append(neighborhood_best)
                        self.best = deepcopy(neighborhood_best)
                        track.append(self.current)
                        track_score.append(self._score(self.current))
                        break
                    else:
                        neighborhood.remove(neighborhood_best)
                        neighborhood_best = self._best(neighborhood)
                else:
                    self.tabu_list.append(neighborhood_best)
                    self.current = neighborhood_best
                    if self._score(self.current) > self._score(self.best):
                        self.best = deepcopy(self.current)
                    break

            if self.max_score is not None and self._score(self.best) > self.max_score:
                print("TERMINATING - REACHED MAXIMUM SCORE")
                return self.best, self._score(self.best)
        print("TERMINATING - REACHED MAXIMUM STEPS")
        return self.best, self._score(self.best), track, track_score


if __name__ == '__main__':
    # import data
    all_data = pd.read_csv("grouped_data.csv")
    all_data = all_data.drop("Unnamed: 0", axis=1)
    columns_names = all_data.keys()

    # Define configuration number for each robotic arm (index)
    all_data['Configuration index'] = all_data.groupby(['Joint2 type_pitch', 'Joint2 type_pris', 'Joint2 type_roll',
                                                        'Joint3 type_pitch', 'Joint3 type_pris', 'Joint3 type_roll',
                                                        'Joint4 type_pitch', 'Joint4 type_pris', 'Joint4 type_roll',
                                                        'Joint5 type_pitch', 'Joint5 type_pris', 'Joint5 type_roll',
                                                        'Joint6 type_pitch', 'Joint6 type_pris', 'Joint6 type_roll',
                                                        'Joint2 axis_y', 'Joint2 axis_z', 'Joint3 axis_x',
                                                        'Joint3 axis_y',
                                                        'Joint3 axis_z', 'Joint4 axis_x', 'Joint4 axis_y',
                                                        'Joint4 axis_z',
                                                        'Joint5 axis_x', 'Joint5 axis_y', 'Joint5 axis_z',
                                                        'Joint6 axis_x',
                                                        'Joint6 axis_y', 'Joint6 axis_z']).ngroup()

    all_data = all_data[['Configuration index', 'Link2 length', 'Link3 length', 'Link4 length', 'Link5 length',
                         'Link6 length', 'Joint2 type_pitch', 'Joint2 type_pris', 'Joint2 type_roll',
                         'Joint3 type_pitch', 'Joint3 type_pris', 'Joint3 type_roll',
                         'Joint4 type_pitch', 'Joint4 type_pris', 'Joint4 type_roll',
                         'Joint5 type_pitch', 'Joint5 type_pris', 'Joint5 type_roll',
                         'Joint6 type_pitch', 'Joint6 type_pris', 'Joint6 type_roll',
                         'Joint2 axis_y', 'Joint2 axis_z', 'Joint3 axis_x', 'Joint3 axis_y',
                         'Joint3 axis_z', 'Joint4 axis_x', 'Joint4 axis_y', 'Joint4 axis_z',
                         'Joint5 axis_x', 'Joint5 axis_y', 'Joint5 axis_z', 'Joint6 axis_x',
                         'Joint6 axis_y', 'Joint6 axis_z', 'Success_Rates', 'Manipulability_Rates']]

    print("Maximum number of Configuration index: \n", all_data['Configuration index'].max())
    print("Minimum number of Configuration index: \n", all_data['Configuration index'].min())

    random.seed(456)
    np.random.seed(456)

    '''Initial Run'''
    initial_state = all_data.sample().values.tolist()
    initial_state = [element for sublist in initial_state for element in sublist]
    all_data = all_data.values.tolist()
    tabu_size = 50
    max_steps = 200
    tabu = TabuSearch(all_data, initial_state, tabu_size, max_steps)
    best_sol, best_score, track, track_score= tabu.run()
    print("Best solution found: \n", best_sol)
    print("Best objective function score: \n", best_score)
    print("track: ", track)
    print("track score ", track_score)

    df = pd.DataFrame(best_sol).T
    df.columns= ['Configuration index', 'Link2 length', 'Link3 length', 'Link4 length', 'Link5 length',
                         'Link6 length', 'Joint2 type_pitch', 'Joint2 type_pris', 'Joint2 type_roll',
                         'Joint3 type_pitch', 'Joint3 type_pris', 'Joint3 type_roll',
                         'Joint4 type_pitch', 'Joint4 type_pris', 'Joint4 type_roll',
                         'Joint5 type_pitch', 'Joint5 type_pris', 'Joint5 type_roll',
                         'Joint6 type_pitch', 'Joint6 type_pris', 'Joint6 type_roll',
                         'Joint2 axis_y', 'Joint2 axis_z', 'Joint3 axis_x', 'Joint3 axis_y',
                         'Joint3 axis_z', 'Joint4 axis_x', 'Joint4 axis_y', 'Joint4 axis_z',
                         'Joint5 axis_x', 'Joint5 axis_y', 'Joint5 axis_z', 'Joint6 axis_x',
                         'Joint6 axis_y', 'Joint6 axis_z', 'Success_Rates', 'Manipulability_Rates']
    print(tabulate(df, headers='keys', tablefmt='psql'))

    x = [i[0] for i in track]
    print(x)
    y = track_score
    plt.plot(x, y, 'ro')
    plt.title("Observed robotic configurations- Tabu search")
    plt.xlabel("Configuration Index")
    plt.ylabel("Objective Function Value")
    plt.show()

    ''' hyper parameter tuning'''
    tabu_size = [50,100,150,200]
    max_steps = [50,70,100,120]
    comb_array = []
    av_res_array = []
    for i in range(len(tabu_size)):
        for j in range(len(max_steps)):
            comb = "tabu size: "+str(tabu_size[i]) + "max steps: " + str(max_steps[j])
            comb_array.append(comb)
            res_array = []
            for k in range(10):
                initial_state = all_data.sample().values.tolist()
                initial_state = [element for sublist in initial_state for element in sublist]
                tabu = TabuSearch(all_data.values.tolist(), initial_state, tabu_size[i], max_steps[j])
                best_sol, best_score, track, track_score= tabu.run()
                df = pd.DataFrame(best_sol).T
                df.columns = ['Configuration index', 'Link2 length', 'Link3 length', 'Link4 length', 'Link5 length',
                              'Link6 length', 'Joint2 type_pitch', 'Joint2 type_pris', 'Joint2 type_roll',
                              'Joint3 type_pitch', 'Joint3 type_pris', 'Joint3 type_roll',
                              'Joint4 type_pitch', 'Joint4 type_pris', 'Joint4 type_roll',
                              'Joint5 type_pitch', 'Joint5 type_pris', 'Joint5 type_roll',
                              'Joint6 type_pitch', 'Joint6 type_pris', 'Joint6 type_roll',
                              'Joint2 axis_y', 'Joint2 axis_z', 'Joint3 axis_x', 'Joint3 axis_y',
                              'Joint3 axis_z', 'Joint4 axis_x', 'Joint4 axis_y', 'Joint4 axis_z',
                              'Joint5 axis_x', 'Joint5 axis_y', 'Joint5 axis_z', 'Joint6 axis_x',
                              'Joint6 axis_y', 'Joint6 axis_z', 'Success_Rates', 'Manipulability_Rates']
                print("Best solution found for population size " + str(max_steps[j]) + " and generation number "+ str(tabu_size[i]) + " is: \n")
                print(tabulate(df, headers='keys', tablefmt='psql'))
                print("Best objective function score for " + str(max_steps[j])  + " and generation number "+ str(tabu_size[i]) + " is: \n", best_score)
                res_array.append(best_score)
            av_res_array.append(mean(res_array))
            # print("Track solution for " + str(population_num[j])  + " and generation number "+ str(generation_num[i]) + " is: \n", np.squeeze(track))
            # print("Score's Track for " + str(population_num[j])  + " and generation number "+ str(generation_num[i]) + " is: \n", np.squeeze(track_score))

            df = pd.DataFrame(best_sol).T
            df.columns = ['Configuration index', 'Link2 length', 'Link3 length', 'Link4 length', 'Link5 length',
                          'Link6 length', 'Joint2 type_pitch', 'Joint2 type_pris', 'Joint2 type_roll',
                          'Joint3 type_pitch', 'Joint3 type_pris', 'Joint3 type_roll',
                          'Joint4 type_pitch', 'Joint4 type_pris', 'Joint4 type_roll',
                          'Joint5 type_pitch', 'Joint5 type_pris', 'Joint5 type_roll',
                          'Joint6 type_pitch', 'Joint6 type_pris', 'Joint6 type_roll',
                          'Joint2 axis_y', 'Joint2 axis_z', 'Joint3 axis_x', 'Joint3 axis_y',
                          'Joint3 axis_z', 'Joint4 axis_x', 'Joint4 axis_y', 'Joint4 axis_z',
                          'Joint5 axis_x', 'Joint5 axis_y', 'Joint5 axis_z', 'Joint6 axis_x',
                          'Joint6 axis_y', 'Joint6 axis_z', 'Success_Rates', 'Manipulability_Rates']
            print("The chosen solution in table format: \n")
            print(tabulate(df, headers='keys', tablefmt='psql'))

            # x = [i[0] for i in np.squeeze(track)]
            # print(x)
            # y = np.squeeze(track_score)
            # plt.plot(x, y, 'ro')
            # plt.title("Observed robotic configurations in first generation - PSO")
            # plt.xlabel("Configuration Index")
            # plt.ylabel("Objective Function Value")
            # plt.show()

    greed_best_score_index = np.argmax(av_res_array)
    best_comb = comb_array[greed_best_score_index]
    best_score_found = av_res_array[greed_best_score_index]
    print("Best combination found: \n",best_comb)
    print("Best avarage score achieved: \n", best_score_found)

    '''Start from the same initial solution'''
    starting_points = all_data.sample(n=90).values.tolist()
    print(starting_points)
    array_sol = []
    array_score = []
    for i in range(len(starting_points)):
        initial_state = starting_points[i]
        tabu = TabuSearch(all_data.values.tolist(), initial_state, tabu_size=100, max_steps=120)
        best_sol, best_score, track, track_score= tabu.run()
        print("Best solution found: \n", best_sol)
        print("Best objective function score: \n", best_score)
        array_sol.append(best_sol)
        array_score.append(best_score)

    print("Average score obtained by Tabu Search: ",mean(array_score))
    start_conf = [i[0] for i in starting_points]
    conf_sol = [i[0] for i in array_sol]
    print("The selected arm configurations: ",conf_sol)

    plt.plot(start_conf,conf_sol)
    plt.show()

    x= [i for i in range(90)]
    y= array_score
    plt.plot(x,y)
    plt.title("Scores obtained in defined starting points - TS")
    plt.xlabel("Iteration number")
    plt.ylabel("Objective function score")
    plt.show()
