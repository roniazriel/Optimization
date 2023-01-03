from numpy.random import rand
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from tabulate import tabulate
from statistics import mean

def objective(solution):
	success_rate = (solution.loc[:, 'Success_Rates'].values[0])*10
	manip_rate = solution.loc[:, "Manipulability_Rates"].values[0]
	obj = success_rate + manip_rate
	return obj

def initial_solution(data):
	sol = data.sample()
	return sol

def next_solution(pre_solution, data , max_group_number):
	group_number = pre_solution.iloc[0]['Configuration index']
	group_df = data.loc[(data['Configuration index'] == group_number) | (data['Configuration index'] == group_number+1), :] #create df of the group only
	if group_df is None and group_number < max_group_number: # if there is no other arm in the group
		group_number +=1
		group_df = data.loc[(data['Configuration index'] == group_number) | (data['Configuration index'] == group_number+1), :]
	if group_df is None and group_number >= max_group_number: # if there is no other arm in the group and we reached thw maximum group number
		group_number = random.randint(0, max_group_number)
		group_df = data.loc[(data['Configuration index'] == group_number) | (data['Configuration index'] == group_number+1), :]

	next_sol = pd.DataFrame()
	i=1
	next_sol_empty= True
	while next_sol_empty:
		for ind in group_df.index:  # iterate over the group
			# print(group_df, ind)
				next_sol = group_df.sample()
				next_sol_empty =False
				break

		group_number +=1
		group_df = data.loc[(data['Configuration index'] == group_number) | (data['Configuration index'] == group_number+1), :]
		i+=1
	return next_sol

def simulated_annealing(n_iterations, temp , data , init=None):
	track = []
	track_score = []
	# generate an initial point
	if init is None:
		best = initial_solution(data=data)
	else:
		best=init
	print("initial solution: \n" ,best)
	track.append(best.values.squeeze().tolist())
	# evaluate the initial point
	best_eval = objective(best)
	track_score.append(best_eval)
	print("initial solution objective score: \n", best_eval)
	# current working solution
	curr, curr_eval = best, best_eval
	# run the algorithm
	for i in range(n_iterations):
		# print("Iteration number: ", i)
		# take a step
		candidate = next_solution(pre_solution= curr,data=data,max_group_number=4288)
		track.append(candidate.values.squeeze().tolist())
		# evaluate candidate point
		candidate_eval = objective(candidate)
		track_score.append(candidate_eval)
		# check for new best solution - Maximization
		if candidate_eval > best_eval:
			# store new best point
			best, best_eval = candidate, candidate_eval
			# report progress
			print( "Best solution so far: \n", best)
			print("Best score so far: \n", best_eval)
		curr = candidate
		# difference between candidate and current point evaluation
		diff = candidate_eval - curr_eval
		# calculate temperature for current epoch
		t = temp / float(i + 1)
		# calculate metropolis acceptance criterion
		metropolis = np.exp(-diff / t)
		# check if we should keep the new point
		if diff < 0 or rand() < metropolis:
			# store the new current point
			curr, curr_eval = candidate, candidate_eval
	print(best, best_eval)
	return best, best_eval, track,track_score

if __name__ == '__main__':
	# import data
	all_data = pd.read_csv("grouped_data.csv")
	all_data = all_data.drop("Unnamed: 0", axis=1)
	columns_names = all_data.keys()
	print(columns_names)

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

	'''Initial Run'''
	best, best_eval, track, track_score = simulated_annealing(n_iterations=100,temp=100, data=all_data)
	print("Best solution: ")
	# with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
	# 	print(best)
	print(tabulate(best, headers='keys', tablefmt='psql'))
	print("Best evaluation: ", best_eval)
	print("track: ", track)
	print("track score ", track_score)

	x = [i[36] for i in track]
	print(x)
	y = track_score
	plt.plot(x, y, 'ro')
	plt.title("Observed robotic configurations- Simulated Annealing")
	plt.xlabel("Configuration Index")
	plt.ylabel("Objective Function Value")
	plt.show()


	''' hyper parameter tuning'''
	iteration_num = [100,120,150,170]
	init_temp = [50,70,100,120]
	comb_array = []
	av_res_array = []
	for i in range(len(iteration_num)):
		for j in range(len(init_temp)):
			comb = "iteration number: " + str(iteration_num[i]) + "initial temperature: " + str(init_temp[j])
			comb_array.append(comb)
			res_array = []
			for k in range(10):
				best, best_eval, track,track_score = simulated_annealing(n_iterations= iteration_num[i] , temp= init_temp[j], data=all_data)
				print("Best solution found for iteration number " + str(
					iteration_num[i]) + " and initial temperature " + str(init_temp[j]) + " is: \n")
				print(tabulate(best, headers='keys', tablefmt='psql'))
				print("Best objective function score for iteration number " + str(
					iteration_num[i]) + " and initial temperature " + str(init_temp[j]) + " is: \n", best_eval)
				res_array.append(best_eval)
			av_res_array.append(mean(res_array))
			# print("Track solution for " + str(population_num[j])  + " and generation number "+ str(generation_num[i]) + " is: \n", np.squeeze(track))
			# print("Score's Track for " + str(population_num[j])  + " and generation number "+ str(generation_num[i]) + " is: \n", np.squeeze(track_score))

			print("The chosen solution in table format: \n")
			print(tabulate(best, headers='keys', tablefmt='psql'))

		x = [i[36] for i in np.squeeze(track)]
		print(x)
		y = np.squeeze(track_score)
		plt.plot(x, y, 'ro')
		plt.title("Observed robotic configurations - SA")
		plt.xlabel("Configuration Index")
		plt.ylabel("Objective Function Value")
		plt.show()

	greed_best_score_index = np.argmax(av_res_array)
	best_comb = comb_array[greed_best_score_index]
	best_score_found = av_res_array[greed_best_score_index]
	print("Best combination found: \n", best_comb)
	print("Best average score achieved: \n", best_score_found)

	'''Start from the same initial solution'''

	array_sol = []
	array_score = []
	for i in range(90):
		starting_point = all_data.sample()
		best, best_eval, track,track_score = simulated_annealing(n_iterations=170 , temp= 100, data=all_data, init=starting_point)
		print("Best solution found: \n", best)
		print("Best objective function score: \n", best_eval)
		array_sol.append(best.values.squeeze())
		array_score.append(best_eval)

	print("Average score obtained by Simulated Annealing: ", mean(array_score))
	conf_sol = [i[36] for i in array_sol]
	print("The selected arm configurations: ", conf_sol)

	x = [i for i in range(90)]
	y = array_score
	plt.plot(x, y)
	plt.title("Scores obtained in defined starting points - SA")
	plt.xlabel("Iteration number")
	plt.ylabel("Objective function score")
	plt.show()

