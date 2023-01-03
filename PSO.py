import random
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import animation
from tabulate import tabulate
from statistics import mean


class PSO:
    def __init__(self, data, population_size, generation, max_score=None, max_group=4888, min_group=0, w=0.1):
        """
    :param population_size: number of particles in population
    :param generation: number of generations to run algorithm for
    :param max_score: score to stop algorithm once reached
    """
        self.origin_data = data
        self.data = data.values.tolist()
        self.population = population_size
        self.max_group = max_group
        self.min_group = min_group
        self.w = w
        if isinstance(population_size, int) and population_size > 0:
            self.population_size = population_size
        else:
            raise TypeError('population size must be a positive integer')

        if isinstance(generation, int) and generation > 0:
            self.generation = generation
        else:
            raise TypeError('generation must be a positive integer')

        if max_score is not None:
            if isinstance(max_score, (int, float)):
                self.max_score = float(max_score)
            else:
                raise TypeError('Maximum score must be a numeric type')

    def _fitness_function(self, particle):
        """
    Returns objective function value of a state
    :param particle: a solution
    :return: objective function value of solution
    """
        print("curr particle",particle)
        print(type(particle))
        success_rate = particle[35] * 10
        manip_rate = particle[36]
        obj = success_rate + manip_rate
        return obj

    def _update_velocity(self, particle, velocity, pbest, gbest, w_min=0.1, max=0.5, c=0.1):
        # Randomly generate r1, r2 and inertia weight from normal distribution
        r1 = random.uniform(0, max)
        r2 = random.uniform(0, max)
        #w = random.uniform(w_min, max)
        w = self.w
        c1 = c
        c2 = c
        # Calculate new velocity
        new_velocity = w * velocity + c1 * r1 * (pbest[0] - particle[0]) + c2 * r2 * (gbest[0] - particle[0])
        print("veloc", int(new_velocity))
        return int(new_velocity)

    def _update_position(self, particle, velocity):
        # Move particles by adding velocity
        """
        Returns list of all members of neighborhood of current state, given self.current
        :return: list of members of neighborhood
        """
        # print("current: ", particle)
        # print("velocity", velocity)
        group_number = particle[0] + velocity
        # print("group number", group_number)
        if group_number >= self.max_group:
            group_list = [arm for arm in self.data if arm[0] == group_number or arm[0] == velocity]
        elif group_number <= self.min_group:
            group_list = [arm for arm in self.data if arm[0] == self.min_group or arm[0] == velocity]
        else:
            group_list = [arm for arm in self.data if arm[0] == group_number or arm[0] == group_number + 1]
        # if not bool(group_list):
        #     group_list = particle
        new_particle = group_list[random.randint(0, len(group_list) - 1)]
        print("new particle",new_particle)
        return new_particle

    def _initiate_particles(self):
        """
    :return: list of particles
    """
        # print(self.origin_data)
        particles = [self.origin_data.sample().values.squeeze().tolist() for i in range(self.population)]
        return particles

    def pso_run(self, fitness_criterion=1, init=None):
        # search tracking
        track = []
        track_score = []
        tracks = []
        track_scores = []
        # Initialisation
        # Population
        if init is None:
            particles = self._initiate_particles()
        else:
            particles = init
        track.append(particles)
        # Particle's best position
        pbest_position = particles
        # Fitness
        pbest_fitness = [self._fitness_function(p) for p in particles]
        track_score.append(pbest_fitness)
        # Index of the best particle
        gbest_index = np.argmax(pbest_fitness)
        # Global best particle position
        gbest_position = pbest_position[gbest_index]
        # Velocity (starting from 0 speed)
        velocity = [0 for i in range(self.population)]
        # Loop for the number of generation
        for t in range(self.generation):
            tracks.append(particles)
            track_scores.append(pbest_fitness)
            # Stop if the average fitness value reached a predefined success criterion
            if np.average(pbest_fitness) <= fitness_criterion:
                break
            else:
                for n in range(self.population):
                    # Update the velocity of each particle
                    velocity[n] = self._update_velocity(particles[n], velocity[n], pbest_position[n], gbest_position)
                    # Move the particles to new position
                    particles[n] = self._update_position(particles[n], velocity[n])
            # Calculate the fitness value
            pbest_fitness = [self._fitness_function(p) for p in particles]
            # Find the index of the best particle
            gbest_index = np.argmax(pbest_fitness)
            # Update the position of the best particle
            gbest_position = pbest_position[gbest_index]

        # Print the results
        print('Global Best Position: ', gbest_position)
        print('Best Fitness Value: ', max(pbest_fitness))
        print('Average Particle Best Fitness Value: ', np.average(pbest_fitness))
        print('Number of Generation: ', t)
        return gbest_position, pbest_fitness, track, track_score, tracks, track_scores


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
    pso_search = PSO(all_data, population_size=50, generation=200)
    best_sol, best_score, track, track_score, tracks, track_scores = pso_search.pso_run()
    print("Best solution found: \n", best_sol)
    print("Best objective function score: \n", max(best_score))
    print("Track solution: \n", np.squeeze(track))
    print("Score's Track: \n", np.squeeze(track_score))

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
    print(tabulate(df, headers='keys', tablefmt='psql'))

    x = [i[0] for i in np.squeeze(track)]
    print(x)
    y = np.squeeze(track_score)
    plt.plot(x, y, 'ro')
    plt.title("Observed robotic configurations in first generation - PSO")
    plt.xlabel("Configuration Index")
    plt.ylabel("Objective Function Value")
    plt.show()

    x = [i[0][0] for i in np.squeeze(tracks)]
    print(x)
    y = np.squeeze(track_scores)
    plt.plot(x, y, 'ro')
    plt.title("Observed robotic configurations in all generations - PSO")
    plt.xlabel("Configuration Index")
    plt.ylabel("Objective Function Value")
    plt.show()

    ''' hyper parameter tuning'''
    generation_num = [50, 100, 150, 200]
    population_num = [30, 50, 70, 90]
    ws = [0.9,1,1.1,1.2]
    comb_array = []
    av_res_array = []
    for i in range(len(generation_num)):
        for j in range(len(population_num)):
            for l in range(len(ws)):
                comb = "generation: " + str(generation_num[i]) + "population: " + str(population_num[j]) + "W: " + str(ws[j])
                comb_array.append(comb)
                res_array = []
                for k in range(10):
                    pso_search = PSO(all_data, population_size=population_num[j], generation=generation_num[i], w=ws[i])
                    best_sol, best_score, track, track_score, tracks, track_scores = pso_search.pso_run()
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
                    print("Best solution found for population size " + str(
                        population_num[j]) + " and generation number " + str(generation_num[i]) + " is: \n")
                    print(tabulate(df, headers='keys', tablefmt='psql'))
                    print("Best objective function score for " + str(population_num[j]) + " and generation number " + str(
                        generation_num[i]) + " is: \n", max(best_score))
                    res_array.append(max(best_score))
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

        x = [i[0] for i in np.squeeze(track)]
        print(x)
        y = np.squeeze(track_score)
        plt.plot(x, y, 'ro')
        plt.title("Observed robotic configurations in first generation - PSO")
        plt.xlabel("Configuration Index")
        plt.ylabel("Objective Function Value")
        plt.show()

    greed_best_score_index = np.argmax(av_res_array)
    best_comb = comb_array[greed_best_score_index]
    best_score_found = av_res_array[greed_best_score_index]
    print("Best combination found: \n", best_comb)
    print("Best avarage score achieved: \n", best_score_found)

    '''Start from the same initial solution'''
    starting_points = all_data.sample(n=90).values.squeeze().tolist()
    print(starting_points)

    pso_search = PSO(all_data, population_size=90, generation=150)
    best_sol, best_score, track, track_score, tracks, track_scores = pso_search.pso_run(init=starting_points)
    print("The score obtained by PSO: ", max(best_score))
    print("The selected arm configurations: ", best_sol)

    x= [i for i in range(90)]
    y= best_score
    plt.plot(x,y)
    plt.title("Scores of the last generation - PSO")
    plt.xlabel("Iteration number")
    plt.ylabel("Objective function score")
    plt.show()
