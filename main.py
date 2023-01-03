import matplotlib.pyplot as plt
import pandas as pd
import random
import numpy as np
from tabulate import tabulate


def initial_solution(data):
    sol = data.sample()
    return sol


def objective(solution):
    success_rate = (solution.loc[:, 'Success_Rates'].values[0]) * 10
    manip_rate = solution.loc[:, "Manipulability_Rates"].values[0]
    obj = success_rate + manip_rate
    return obj


def next_solution(pre_solution, data, random_jump, max_group_number=4888):
    next_sol = data.sample()
    return next_sol


def BaseLine_search(n_iterations, data):
    # search tracking
    track = []
    track_score = []
    # generate an initial point
    best = initial_solution(data=data)
    print("initial solution: \n", best)
    # evaluate the initial point
    best_eval = objective(best)
    print("initial solution objective score: \n", best_eval)
    # current working solution
    curr, curr_eval = best, best_eval
    track.append(curr.values.squeeze().tolist())
    track_score.append(curr_eval)
    # run the algorithm
    for i in range(n_iterations):
        print("Iteration number: ", i)
        jump_indicator = 10
        # take a step
        candidate = next_solution(pre_solution=curr, data=data, random_jump=jump_indicator)
        # evaluate candidate point
        candidate_eval = objective(candidate)
        track.append(candidate.values.squeeze().tolist())
        track_score.append(candidate_eval)
        # check for new best solution - Maximization
        if candidate_eval > best_eval:
            # store new best point
            best, best_eval = candidate, candidate_eval
            # report progress
            print("Best solution so far: \n", best)
            print("Best score so far: \n", best_eval)
        curr = candidate
        # difference between candidate and current point evaluation
        diff = candidate_eval - curr_eval
    return best, best_eval, diff, track, track_score


def calc_objective_all_data(data):
    # for index, row in data.iterrows():
    #     solution = data.loc[index:index]
    #     data['Objective Score'] = (solution.loc[:, 'Success_Rates'].values[0]) * 10 + solution.loc[:, "Manipulability_Rates"].values[0]
    data = data.assign(Objective_Score=lambda x: x.Success_Rates * 10 + x.Manipulability_Rates)
    return data


def plots(all_data):
    df = calc_objective_all_data(all_data)
    with pd.option_context('display.max_columns', None):  # more options can be specified also
        print(df)
    # Comparision:
    # Searching space visualization
    x = np.arange(0, 4889)
    y = all_data[['Configuration index']].groupby(['Configuration index']).value_counts()
    z = df[['Objective_Score', 'Configuration index']].groupby(['Configuration index']).mean()

    ax = plt.axes(projection='3d')
    ax.scatter3D(x, y, z, c=z, cmap='Greens')
    ax.set_xlabel('Configuration index')
    ax.set_ylabel('Robotic arms number')
    ax.set_zlabel('Objective Score')
    plt.title("Search Space")
    plt.show()



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

    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    # 	print(all_data['Manipulability_Rates'])

    # print(all_data['Configuration index'].sort_values())
    # all_data.to_csv(r"C:\Users\azrie\PycharmProjects\pythonProject\Computational_intelligence\all_data.csv")

    random.seed(456)
    np.random.seed(456)
    plots(all_data)

    '''Baseline'''
    best_base, best_base_eval, diff, track, track_score = BaseLine_search(n_iterations=20, data=all_data)
    print("Evaluation for initial hyper parameters - Base line")
    print("Best solution: ", best_base)
    print("Best evaluation: ", best_base_eval)
    print("Different in objective function between the two last solutions: ", diff)
    print("track: ", track)
    print("track score ", track_score)

    print(tabulate(best_base, headers='keys', tablefmt='psql'))

    x = [i[36] for i in track]
    print(x)
    y = track_score
    plt.plot(x, y, 'ro')
    plt.title("Observed robotic configurations in search")
    plt.xlabel("Configuration Index")
    plt.ylabel("Objective Function Value")
    plt.show()


