"""
This file contains code for running all experiments in the paper

A. Kolobov, S. Bubeck, J. Zimmert. "Online Learning for Active Cache Synchronization." ICML-2020.

To reproduce the experiments in Figures 1 and 2 in the main text and Figures 3 and 4 in the Appendix,
run exp1(), exp2(), exp1a(), and exp2a() from this file, respectively. See the comments for these
methods for more details.

Method param_search() allows inspecting the performance of the algorithms from the experiments with 
different hyperparameter value combinations. Hyperparamete ranges for the paper's experiments are 
provided in that method's comments.
"""


from abc import ABC, abstractmethod
import scipy.stats
from scipy.stats import poisson
import numpy as np;
import scipy.integrate as integrate
import scipy.special as special
from scipy.optimize import minimize
import pprint
import time;
import math;
import matplotlib.pyplot as plt
import sys

np.set_printoptions(threshold=sys.maxsize)


PROBE_PLAY_IDX = 0
SYNC_PLAY_IDX = 1


# Abstract class for a cost generation process	
class CostGen(ABC):
	@abstractmethod
	def __init__(self, num_arms, distrib_params):
		super().__init__()
	
	# The instantaneous cost
	@abstractmethod
	def c(self, intervals):
		pass
	

	"""Computes the average infinite-horizon cost of a play-rate-parameterized policy for the specified subset of arms.

	Parameters
	----------
	r : float array
		Play rates to use for all arms.

	Returns
	-------
	float
		Policy cost for the specified subset of arms, normalized by the number of arms in arms_filter.
	"""
	@abstractmethod	
	def J(self, r):
		pass


	"""Simulates arm plays for selected arms until a fixed horizon (epoch_start_time + epoch_len) and records the generated cost samples.

	Parameters
	----------
	r : float array
		Play rates to use for all arms.
	arms_latest_play_times : int array 
		The timestamps of the latest simulated play times for all arms.
		NOTE: arms_latest_play_times contents are modified by this method!
	epoch_start_time : int
		The start time of the epoch during which the arm plays are to be simulated.
	epoch_len : int
		The length of the epoch during which the arm plays are to be simulated.
	epsilon : float
		The probability of doing a probe play between any two consecutive sync plays.

	Returns
	-------
	array of lists
		An array of lists of [probe play cost, sync play cost] pairs.
	"""
	def sample_costs(self, r, arms_latest_play_times, epoch_start_time, epoch_len, epsilon):
		histories = np.empty(len(r), dtype = object)
		for k in range(len(r)):
			arm_k_play_hist = list()
			"""
			For arm k, each iteration of this loop schedules a sync play time (1 / r[k]) after the previous
			sync play, until the time of the next scheduled sync play is past the end of the current scheduling
			epoch (epoch_start_time + epoch_len). With probability epsilon it also schedules a probe play between
			the previous sync play and the one scheduled in this iteration, at a time chosen uniformly between 
			the two.
			"""
			while arms_latest_play_times[k] + 1/r[k] <= epoch_start_time + epoch_len:
				# With prob. epsilon, schedule a probe play.
				if np.random.binomial(1, epsilon, 1) > 0:
					probe_play_time = np.random.uniform(0, 1/r[k])
					# probe_and_sync_play_times's time stamps are relative to the previous scheduled sync play
					probe_and_sync_play_times = np.array([probe_play_time, 1/r[k]])
					# Sample a cost for the scheduled probe play and sync play.
					values = self.c(k, probe_and_sync_play_times)
					arm_k_play_hist.append(values)
				else:
					# If we happened to schedule no probe play before the next sync play, insert the "0" indicator 
					# instead of a probe play timestamp, and the indicator "-1" cost for it.
					probe_and_sync_play_times = np.array([0, 1/r[k]])
					values = self.c(k, probe_and_sync_play_times)
					values[PROBE_PLAY_IDX] = -1
					arm_k_play_hist.append(values)#list(values))
				arms_latest_play_times[k] += 1/r[k]	
			histories[k] = arm_k_play_hist
		return histories


	"""Estimates the gradient of the cost functions for the selected arms.

	Parameters
	----------
	r : float array
		Play rates to use for all arms.
	arms_latest_play_times : int array 
		The timestamps of the latest simulated play times for all arms.
		NOTE: arms_latest_play_times contents are modified by this method!
	epoch_start_time : int
		The start time of the epoch during which the arm plays are to be simulated.
	epoch_len : int
		The length of the epoch during which the arm plays are to be simulated.
	epsilon : float
		The probability of doing a probe play between any two consecutive sync plays.

	Returns
	-------
	est_grad_J : array of floats
		An array representing the gradient. Value of 0 indicates that this dimension hasn't been reestimated
	arms_with_new_grad : array of ints
		An array of arm indices whose partial derivatives got estimated in this function call. 
		All other arms' partial derivative estimates are 0 and should be ignored.
	"""
	def estimate_grad_J(self, r, arms_latest_play_times, epoch_start_time, epoch_len, epsilon):
		est_grad_J = np.zeros_like(r)
		histories = self.sample_costs(r, arms_latest_play_times, epoch_start_time, epoch_len, epsilon)
		arms_with_new_grad = []
		for k in range(len(r)):
			sum_est_grad_k = 0
			# For each sync play, compute a J_k gradient estimate and add them all up.
			for h in range(len(histories[k])):
				if histories[k][h][PROBE_PLAY_IDX] != -1:
					sum_est_grad_k += 1 / (epsilon * r[k]) * (histories[k][h][PROBE_PLAY_IDX] - histories[k][h][SYNC_PLAY_IDX])
				else:
					sum_est_grad_k += 0
			
			# Average the gradient estimates and record the arms for which gradient estimates have been computed
			if len(histories[k]) > 0: 		
				est_grad_J[k] = (sum_est_grad_k / len(histories[k]))
				arms_with_new_grad.append(k)
			else:
				est_grad_J[k] = 0
			
		return est_grad_J / len(arms_with_new_grad), arms_with_new_grad
		

def sigmoid(x):
	return 1/(1 + np.exp(-x))


class SublinPoly(CostGen):
	A_K = 0
	P_K = 1
	NOISE = 2
	"""
	For each arm, initialization uses a prior to choose a distribution over time-dependent cost functions. 
	Namely, arm k will have a distribution over "capped" polynomials  
	
			c_k(t) = a_k * (t^p_k)
	 
	where
		
		- a_k will be sampled from Uniform(mu_c_k - mu_c_k * noise, mu_c_k + mu_c_k * noise) at the cost function query time (in method c(.)),
		  "noise" from [0,1] is a parameter shared by all arms, and mu_c_k is a parameter chosen at initialization time from the prior Uniform(0, 1).
	
		- p_k is a parameter chosen at initialization time from the prior sigmoid(scale * Uniform[0,1]), where "scaling" is a parameter shared by all arms.
		  I.e., p_k is from (0, 1), but p_k values > 0.5 are more likely.
	"""
	def __init__(self, num_arms, distrib_params):
		assert(distrib_params["noise"] <= 1)
		self.params = np.zeros((num_arms, 3))
		# mu_c_k, uniform in [0,1)
		self.params[:, SublinPoly.A_K] =  np.random.rand(num_arms,)
		# p_k, biased towards exponents between 0.5 and 1
		self.params[:, SublinPoly.P_K] = sigmoid(distrib_params["scaling"] * np.random.rand(num_arms,))
		self.params[:, SublinPoly.NOISE] = distrib_params["noise"]

		super().__init__(num_arms, distrib_params)


	def c(self, arm, intervals):
		# c(t) = a_k*t^p_k, where p_k is in (0, 1)
		noisy_scaling = np.random.uniform(self.params[arm, SublinPoly.A_K] - self.params[arm, SublinPoly.NOISE] * self.params[arm, SublinPoly.A_K], self.params[arm, SublinPoly.A_K] + self.params[arm, SublinPoly.NOISE] * self.params[arm, SublinPoly.A_K]) 
		return noisy_scaling * intervals ** self.params[arm, SublinPoly.P_K]


	def J(self, r):
		# J(r) = sum_{k in arms_filter}[1 / |arms_filter| * r_k * a_k * (1 / p_k + 1) * (1 / r_k)^(p_k + 1)]
		pol_cost = 1 / self.params.shape[0] * np.dot(r, self.params[:, SublinPoly.A_K] * (1 / (self.params[:, SublinPoly.P_K] + 1) * ((1 / r) ** (self.params[:, SublinPoly.P_K] + 1))))
		return pol_cost



class BinaryPoisson(CostGen):
	IMPORTANCE = 0
	CHANGE_RATE = 1

	def __init__(self, num_arms, distrib_params):
		self.params = np.zeros((num_arms, 2))
		# Web page importance scores; set them all to 1.0 for simplicity.
		self.params[:, BinaryPoisson.IMPORTANCE] = np.full((num_arms,), 1.0)
		# Web page change rates. Sample them uniformly from the interval [chrate_lo, chrate_hi].
		self.params[:, BinaryPoisson.CHANGE_RATE] = np.random.uniform(low=distrib_params['chrate_lo'], high=distrib_params['chrate_hi'], size=(num_arms,))
		super().__init__(num_arms, distrib_params)

	
	def c(self, arm, intervals):
		assert(len(intervals) == 2)
		"""
		Assume that the values in the "intervals" array are sorted in the 
		increasing order. This will ensure that we will essentially sample
		a monotonically increasing function of inteval samples.
		"""
		results = np.zeros((len(intervals),))
		samples = (0 if intervals[PROBE_PLAY_IDX] == 0 else poisson.rvs(self.params[arm, BinaryPoisson.CHANGE_RATE] * intervals[PROBE_PLAY_IDX], size=1))
		results[PROBE_PLAY_IDX] = self.params[arm, BinaryPoisson.IMPORTANCE] * (1.0 if samples > 0 else 0.0)

		if intervals[SYNC_PLAY_IDX] < intervals[PROBE_PLAY_IDX]:
			raise ValueError("Intervals aren't in the increasing order of length")
		elif intervals[SYNC_PLAY_IDX] == intervals[PROBE_PLAY_IDX]:
			results[SYNC_PLAY_IDX] = samples
		else:
			samples = samples + poisson.rvs(self.params[arm, 1] * (intervals[SYNC_PLAY_IDX] - intervals[PROBE_PLAY_IDX]), size=1)
			results[SYNC_PLAY_IDX] = self.params[arm, BinaryPoisson.IMPORTANCE] * (1.0 if samples > 0 else 0.0)
		
		return results

	
	def _C_exp(self, r):
		# The expectation of a binary indicator over a Poisson distribution = mu_k * e^(delta_k / r_k) + 1 / r_k - 1 / delta_k for each arm (page) k, where mu_k is the page importance and delta_k is the page change rate.
		return self.params[:, BinaryPoisson.IMPORTANCE] * (1.0 / r + np.exp(- self.params[:, BinaryPoisson.CHANGE_RATE] / r) / self.params[:, BinaryPoisson.CHANGE_RATE] - 1.0 / self.params[:, BinaryPoisson.CHANGE_RATE])


	def J(self, r):
		# J(r) = sum_{k in arms_filter}[1 / |arms_filter| * r_k * _C_exp(arms_filter, r)]
		pol_cost = 1.0 / self.params.shape[0] * np.dot(r, self._C_exp(r))
		return pol_cost


"""Computes the optimal policy cost. Since the policy cost function and constraint region is convex, this can be done via simple convex optimization.

Parameters
----------
arms : int array
	Indices corresponding to arms that should be taken into account in this computation.
c : CostGen
	A description of a family of cost-generating processes.
rmin : float
	Minimum allowed arm sync play rate.
rmax : float
	Maximum allowed arm sync play rate.
B : float
	Maximum total sync play rate.

Returns
-------
array of lists
	An array of lists of [probe play cost, sync play cost] pairs.
""" 
def compute_opt_policy_cost(c, rmin, rmax, B):
	num_arms = c.params.shape[0]
	bnds=[[rmin, rmax] for k in range(num_arms)]
	obj_func = (lambda r : c.J(r))

	start_time = time.time()
	argm = minimize(obj_func, x0 = np.full((num_arms, 1), B / num_arms), args=(), method="SLSQP", bounds=bnds, constraints={"fun": (lambda r : B - sum(r)), "type": "ineq"}, tol=None, callback=None, options=None)
	end_time = time.time()

	return c.J(argm.x) 


"""The MirrorSync algorithm. See the ICML-2020 paper for a detailed description.

Parameters
----------
c : CostGen
	A description of a family of cost-generating processes.
num_arms : int
	Number of arms.
learning_rate : float
	Learning rate.
num_rounds : int
	Number of learning rounds (\mathcal{T}_max in the ICML-2020 paper).
epsilon : float
	The fraction of total bandwidth allocated to probe plays.
B_frac : float
	Maximum total sync play rate as a fraction of the number of arms.
rmin : float
	Minimum allowed arm sync play rate.
rmax : float
	Maximum allowed arm sync play rate.
silent: boolean
	If True, suppresses most of the output.

Returns
-------
array of floats
	Policy costs before the first learning round and at the end of each learning round, (num_rounds + 1) values in total.
""" 
def mirror_sync(c, num_arms = 100, learning_rate = 0.05, num_rounds = 50, epsilon = 0.1, B_frac = 0.2, rmin = 0.001, rmax = 100, silent = True):
	print("Running MirrorSync")

	"""
	Since we will be doing exploratory probe plays in addition to regular sync plays, we need to adjust the maximum play rate, which will apply to sync plays only.
	"""
	rmax = 1 / (1 + epsilon) * rmax
	B_total = B_frac * num_arms
	"""
	The total play rate constraint will apply only to sync plays too, we we need to adjust it to account for probe plays as well, as in the case with rmax.
	"""
	B = 1 / (1 + epsilon) * B_total
	results = np.zeros((num_rounds + 1,))
	est_grad_J = np.zeros((num_arms,))
	r = np.zeros((num_arms,))
	r = mirror_descent_step(r, est_grad_J / num_arms, learning_rate, rmin, rmax, B)
	J = c.J(r)

	if not silent:
		print('Initial J value is ', J)	
	
	results[0] = J
	
	for i in range(num_rounds):
		est_grad_J, arms_with_new_grads = c.estimate_grad_J(r, np.zeros_like(r), 0, 1 / rmin, epsilon)
		assert num_arms == len(arms_with_new_grads), "In MirrorSync, every arm must get a new gradient estimate in every round! Num arms: %r, num arms with new gradient estimates: %r" % (len(arms), len(arms_temp))
		r = mirror_descent_breg_step(r, est_grad_J, range(num_arms), learning_rate, rmin, rmax, B)
		J = c.J(r)
		results[i + 1] = J
		if (not silent) or (i == num_rounds - 1):
			print('Update round %r: J_round is %r'%(i + 1, J))
			
	if not silent:
		print('Per-update-round Js are ')
		pprint.pprint(results)
		
	return results


"""The AsynMirrorSync/AsyncPSGDSync algorithm. See the ICML-2020 paper for a detailed description.

Parameters
----------
c : CostGen
	A description of a family of cost-generating processes.
alg : string
	'ms' for AsynMirrorSync, 'psgd' for AsyncPSGDSync.
num_arms : int
	Number of arms.
learning_rate : float
	Learning rate.
num_rounds : int
	Number of equivalent MirrorSync learning rounds (\mathcal{T}_max in the ICML-2020 paper).
epsilon : float
	The fraction of total bandwidth allocated to probe plays.
update_cycle_length : int
	Length of time between rate update attempts.
	NOTE: For ease of comparison to MirrorSync's performance, we assume that (1 / rmin) is a multiple of update_cycle_length. 
B_frac : float
	Maximum total sync play rate as a fraction of the number of arms.
rmin : float
	Minimum allowed arm sync play rate.
rmax : float
	Maximum allowed arm sync play rate.
silent: boolean
	If True, suppresses most of the output.

Returns
-------
array of floats
	Policy costs before the first learning round and at the end of each learning round, (num_rounds + 1) values in total.
"""
def async_mirror_sync(c, alg, num_arms = 100, learning_rate = 0.05, num_rounds = 50, epsilon = 0.1, update_cycle_length = 2, B_frac = 0.2, rmin = 0.001, rmax = 100, silent = True):
	if alg == 'ms':
		alg_name = 'AsyncMirrorSync'
	elif alg == 'psgd':
		alg_name = 'AsyncPSGDSync'
	else:
		raise ValueError('Invalid flavor of AsyncMirrorSync %r.'%(alg))
	
	print('Running', alg_name)

	rmax = 1 / (1 + epsilon) * rmax
	B_total = B_frac * num_arms
	B = 1 / (1 + epsilon) * B_total
	if not silent:
		print('Update cycle length: %r'%(update_cycle_length))
	"""
	We will record policy cost after every update_cycle_length, which can be much smaller than MirrorSync's round length = 1 / rmin.
	For ease of comparison to MirrorSync's performance, we assume that (1 / rmin) is a multiple of update_cycle_length.
	"""
	results = np.zeros((num_rounds * math.ceil((1 / rmin) / update_cycle_length) + 1,))
	r = np.zeros((num_arms,))
	r = mirror_descent_step(r, np.zeros((num_arms,)) / num_arms, learning_rate, rmin, rmax, B)
	J = c.J(r)
	if not silent:
		print('Current J value: %r'%(J))
	results[0] = J
	cycle_num = 1
	t_now = 0
	time_of_prev_update = t_now
	arms_latest_play_times = np.full_like(r, t_now)

	# Repeat this for the amount of time equivalent to num_rounds MirrorSync rounds.
	while t_now < num_rounds * (1 / rmin):
		t_now += update_cycle_length

		# Estimate the new gradient for those arms that have generated enough cost samples since the previous time their play rate was updated.
		est_grad_J_local, arms_with_new_grad = c.estimate_grad_J(r, arms_latest_play_times, time_of_prev_update, t_now - time_of_prev_update, epsilon)	
		"""
		NOTE: the total bandwidth available to the arms with new gradient estimates for re-distribution in this cycle will be their current total
		play rate, _not_ the total bandwidth constraint. 
		"""
		B_current_update = sum(r[arms_with_new_grad])
		
		if (len(arms_with_new_grad) > 0):
			r_new = np.zeros_like(r)

			# AsyncMirrorSync and AsyncPSGDSync differ only in the algorithm they use to update the play rates (mirror descent for the former, projected SGD for the latter.)
			if (alg == 'ms'):
				r_new[arms_with_new_grad] = mirror_descent_breg_step(r, est_grad_J_local, arms_with_new_grad, learning_rate, rmin, rmax, B_current_update)
			elif (alg == 'psgd'):
				r_new[arms_with_new_grad] = projected_sgd_step(r, est_grad_J_local, arms_with_new_grad, learning_rate, rmin, rmax, B_current_update)
			else:
				raise ValueError('Invalid or no optimizer specified.')
							
			for k in arms_with_new_grad:
				# We do this to make sure that the last scheduled play is in the next cycle. Note that this play doesn't contribute to gradient estimation.
				if np.random.binomial(1, epsilon, 1) > 0:
					arms_latest_play_times[k] = t_now
				elif (1 / r_new[k]) < (1 / r[k]) and arms_latest_play_times[k] + (1 / r_new[k]) >= t_now:
					arms_latest_play_times[k] = arms_latest_play_times[k] + 1 / r_new[k]
				else:
					arms_latest_play_times[k] = arms_latest_play_times[k] + 1 / r[k]
				r[k] = r_new[k]
		
		time_of_prev_update = t_now
		J = c.J(r)
		results[cycle_num] = J
		if (not silent) or (t_now >= num_rounds * (1 / rmin)):
			print('Update cycle %r: J_cycle is %r'%(cycle_num, J))
		cycle_num += 1

	if not silent:
		print('Per-update-cycle Js are ')
		pprint.pprint(results)
	
	return results


def log_barrier(r):
	return np.sum(-np.log(r))


def bregman_div(r, r_prev):
	return np.sum(-np.log(r/r_prev) + r/r_prev - 1)


"""One step of lazy mirror descent with log barrier function.

Parameters
----------
curr_r : float array
	Current sync play rates.
est_grad_J : float array
	Gradient estimate at curr_r.
learning_rate : float
	Learning rate.
rmin : float
	Minimum allowed arm sync play rate.
rmax : float
	Maximum allowed arm sync play rate.
B : float
	Maximum total sync play rate.

Returns
-------
float array
	New sync play rates.
"""
def mirror_descent_step(curr_r, est_grad_J, learning_rate, rmin, rmax, B):
	assert(len(curr_r) == len(est_grad_J))
	bnds=[[rmin, rmax] for i in range(len(curr_r))]
	obj_func = (lambda r : learning_rate * np.dot(est_grad_J, r) + log_barrier(r))

	start_time = time.time()
	argm = minimize(obj_func, x0 = curr_r, args=(), method="SLSQP", bounds=bnds, constraints={"fun": (lambda r : B - sum(r)), "type": "ineq"}, tol=None, callback=None, options=None)
	end_time = time.time()

	return argm.x


"""One step of mirror descent with log barrier function.

Parameters
----------
curr_r : float array
	Current sync play rates.
est_grad_J : float array
	Gradient estimate at curr_r.
arms : int array
	The arms to which this function call should apply 
	(mirror_descent_breg_step can be invoked on only a subset of all arms). 
learning_rate : float
	Learning rate.
rmin : float
	Minimum allowed arm sync play rate.
rmax : float
	Maximum allowed arm sync play rate.
B : float
	Maximum total sync play rate.

Returns
-------
float array
	New sync play rates for arms whose indices are specified by the "arms" parameter.
"""
def mirror_descent_breg_step(curr_r, est_grad_J, arms, learning_rate, rmin, rmax, B):
	est_grad_J_filtered = est_grad_J[arms]
	r_prev_filtered = curr_r[arms]
	bnds=[[rmin, rmax] for i in arms]
	obj_func = (lambda r : learning_rate * np.dot(est_grad_J_filtered, r) + bregman_div(r, r_prev_filtered))

	start_time = time.time()
	argm = minimize(obj_func, x0 = curr_r[arms], args=(), method="SLSQP", bounds=bnds, constraints={"fun": (lambda r : B - sum(r)), "type": "ineq"}, tol=None, callback=None, options=None)
	end_time = time.time()

	return argm.x


"""One step of projected SGD. It performs 1 SGD update and projects the result into the constraint region.

Parameters
----------
curr_r : float array
	Current sync play rates.
est_grad_J : float array
	Gradient estimate at curr_r.
arms : int array
	The arms to which this function call should apply 
	(projected_sgd_step can be invoked on only a subset of all arms). 
learning_rate : float
	Learning rate.
rmin : float
	Minimum allowed arm sync play rate.
rmax : float
	Maximum allowed arm sync play rate.
B : float
	Maximum total sync play rate.

Returns
-------
float array
	New sync play rates for arms whose indices are specified by the "arms" parameter.
"""
def projected_sgd_step(curr_r, est_grad_J, arms, learning_rate, rmin, rmax, B):
	est_grad_J_filtered = est_grad_J[arms]
	r_prev_filtered = curr_r[arms]
	r_new = r_prev_filtered - learning_rate * est_grad_J_filtered
	bnds=[[rmin, rmax] for i in arms]
	obj_func = (lambda r : np.sum(np.square(r_new - r)))

	start_time = time.time()
	argm = minimize(obj_func, x0 = r_new, args=(), method="SLSQP", bounds=bnds, constraints={"fun": (lambda r : B - sum(r)), "type": "ineq"}, tol=None, callback=None, options=None)
	end_time = time.time()

	return argm.x





#def param_search(alg='MirrorSync', learning_rates=[5, 7, 9, 11, 13, 15, 17, 19], update_cycle_lengths=[40], num_arms=100, horizon=30, epsilon=0.2, B_frac=0.4, rmin=0.025, rmax=3):
#def param_search(alg='AsyncMirrorSync', learning_rates=[0.5, 1, 3, 6, 9, 11], update_cycle_lengths=[8, 10, 20, 40], num_arms=100, horizon=30, epsilon=0.2, B_frac=0.4, rmin=0.025, rmax=3):
#def param_search(alg='AsyncPSGDSync', learning_rates=[0.04, 0.06, 0.08, 0.1, 0.2, 0.3, 0.4], update_cycle_lengths=[8, 10, 20, 40], num_arms=100, horizon=30, epsilon=0.2, B_frac=0.4, rmin=0.025, rmax=3):

poly_params = { "scaling" : 5, "noise" : 0.1 }
binary_Poisson_params = { "chrate_lo" : 0.005, "chrate_hi" : 5 }

exp1_num_arms = 100
"""
Run this method with the default parameter values to produce the results in Figure 1 of the ICML-2020 paper 
(MirrorSync vs. AsyncMirrorSync evaluated on polynomial cost-generating processes).

See the description of the exp1helper for parameter explanations.

The method prints strings with experiment results that can be used to produce plots such as in the paper's figures using pgfplots.
"""
def exp1(cost_fun=SublinPoly, distrib_params=poly_params, num_arms=exp1_num_arms, num_runs=150, update_cycle_length=20, horizon=240, l_r_ms= 2.7, l_r_ams=1.6, epsilon=0.05, B_frac=0.4, rmin=0.025, rmax=3):
	exp1helper(cost_fun, distrib_params, num_arms, num_runs, update_cycle_length, horizon, l_r_ms, l_r_ams, epsilon, B_frac, rmin, rmax)


"""
Run this method with the default parameter values to produce the results in Figure 3 of the ICML-2020 paper 
(MirrorSync vs. AsyncMirrorSync evaluated on Poisson-process-based cost-generating processes).

See the description of the exp1helper for parameter explanations.

The method prints strings with experiment results that can be used to produce plots such as in the paper's figures using pgfplots.
"""
def exp1a(cost_fun=BinaryPoisson, distrib_params=binary_Poisson_params, num_arms=exp1_num_arms, num_runs=150, update_cycle_length=8, horizon=240, l_r_ms=5, l_r_ams=1.3, epsilon=0.05, B_frac=0.4, rmin=0.025, rmax=6):
	exp1helper(cost_fun, distrib_params, num_arms, num_runs, update_cycle_length, horizon, l_r_ms, l_r_ams, epsilon, B_frac, rmin, rmax)


"""A helper method for running the experiments in Figures 1 and 3 of the ICML-2020 paper that compare MirrorSync's and AsyncMirrorSync's performance.

Parameters
----------
cost_fun : a descendant class of CostGen
	A class describing a family of cost-generating processes.
distrib_params : dictionary
	Parameters for the cfamily of cost-generating processes given by cost_fun.
num_arms : int
	Number of arms.
num_runs : int
	Number of runs of each algorithm to perform. The two algorithms' runs are paired, i.e., in each run both algorithms are executed on the same
	cost_fun instance initialized for that run. 
update_cycle_length : int
	Length of time between rate update attempts of AsyncMirrorSync.
	NOTE: For ease of comparison to MirrorSync's performance, we assume that (1 / rmin) is a multiple of update_cycle_length.
horizon : int
	Number of equivalent MirrorSync learning rounds (\mathcal{T}_max for MirrorSync in the ICML-2020 paper).
l_r_ms : float
	Learning rate for MirrorSync.
l_r_ams : float
	Learning rate for AsyncMirrorSync.
epsilon : float
	The fraction of total bandwidth allocated to probe plays.
B_frac : float
	Maximum total sync play rate as a fraction of the number of arms.
rmin : float
	Minimum allowed arm sync play rate.
rmax : float
	Maximum allowed arm sync play rate.

Returns
-------
None
	The method prints strings with experiment results that can be used to produce plots such as in the paper's figures using pgfplots.
"""
def exp1helper(cost_fun, distrib_params, num_arms, num_runs, update_cycle_length, horizon, l_r_ms, l_r_ams, epsilon, B_frac, rmin, rmax):
	STEP_MULT = (1 / rmin) / update_cycle_length
	# A list of arrays that will hold the policy cost after every policy update round of every run of MirrorSync.
	ms_runs = []
	# A list of arrays that will hold the policy cost after every policy update round of every run of AsyncMirrorSync.
	ams_runs = []
	# A list that will hold the optimal epsilon-exploration policy cost for every run. It can vary from run to run due to stochastic cost_fun initialization.
	opt_epsilon_cost = []
	
	for run in range(num_runs):
		print('\n\n=========== RUN #%r ============'%(run + 1)) 
		"""
		Initialize a family of cost-generating processes.
		NOTE: the initialization may be stochastic, and therefore is done for every run separately.
		"""
		c = cost_fun(num_arms, distrib_params)
		opt_epsilon_cost_curr = compute_opt_policy_cost(c, rmin, rmax / (1 + epsilon), B_frac * num_arms / (1 + epsilon))
		opt_epsilon_cost.append(opt_epsilon_cost_curr)
		print('Optimal epsilon-exploration periodic policy cost: %r'%(opt_epsilon_cost_curr))

		results_ms = mirror_sync(c, num_arms, l_r_ms, horizon, epsilon, B_frac, rmin, rmax, silent=True)
		results_ams = async_mirror_sync(c, 'ms', num_arms, l_r_ams, horizon, epsilon, update_cycle_length, B_frac, rmin, rmax, silent=True)

		ms_runs.append(results_ms)
		ams_runs.append(results_ams)

		if (run + 1) % 10 == 0 or (run + 1) == num_runs:
			print('\n\n********** RESULTS AFTER %r RUNS *************'%(run + 1))
			ms_ms, ms_hs = process_results(ms_runs)
			ms_plot_string = ''
			for i in range(ms_ms.shape[0]):
				ms_plot_string += '(' + str(i) + ',' + str(ms_ms[i]) + ')+=(0,' + str(ms_hs[i]) + ')-=(0,' + str(ms_hs[i]) + ')\n'
			print('MirrorSync:\n', ms_plot_string, '\n\n')

			ams_ms, ams_hs = process_results(ams_runs)
			ams_plot_string = ''
			for i in range(ams_ms.shape[0]):
				ams_plot_string += '(' + str(i / STEP_MULT) + ',' + str(ams_ms[i]) + ')+=(0,' + str(ams_hs[i]) + ')-=(0,' + str(ams_hs[i]) +')\n'
			print('AsyncMirrorSync:\n', ams_plot_string, '\n\n')
			
			opt_epsilon_cost_string = ''
			opt_epsilon_cost_mean = np.mean(opt_epsilon_cost)

			for i in range(horizon + 1):
				opt_epsilon_cost_string += '(' + str(i) + ',' + str(opt_epsilon_cost_mean) + ')\n'
			print('Optimal epsilon-exploration periodic policy cost:\n', opt_epsilon_cost_string, '\n\n')
			print('*************************************************')



exp2_num_arms = 100
"""
Run this method with the default parameter values to produce the results in Figure 2 of the ICML-2020 paper 
(AsyncMirrorSync vs. AsyncPSGDSync evaluated on polynomial cost-generating processes).

See the description of the exp2helper for parameter explanations.

The method prints strings with experiment results that can be used to produce plots such as in the paper's figures using pgfplots.
"""
def exp2(cost_fun=SublinPoly, distrib_params=poly_params, num_arms=exp2_num_arms, num_runs=150, update_cycle_length_psgd=20, update_cycle_length_ams=20, horizon=240, l_r_psgd= 0.08, l_r_ams=1.6, epsilon=0.05, B_frac=0.4, rmin=0.025, rmax=3):
	exp2helper(cost_fun, distrib_params, num_arms, num_runs, update_cycle_length_psgd, update_cycle_length_ams, horizon, l_r_psgd, l_r_ams, epsilon, B_frac, rmin, rmax)


"""
Run this method with the default parameter values to produce the results in Figure 4 of the ICML-2020 paper 
(AsyncMirrorSync vs. AsyncPSGDSync evaluated on Poisson-process-based cost-generating processes).

See the description of the exp2helper for parameter explanations.

The method prints strings with experiment results that can be used to produce plots such as in the paper's figures using pgfplots.
"""
def exp2a(cost_fun=BinaryPoisson, distrib_params=binary_Poisson_params, num_arms=exp2_num_arms, num_runs=150, update_cycle_length_psgd=40, update_cycle_length_ams=8, horizon=240, l_r_psgd= 0.5, l_r_ams=1.3, epsilon=0.05, B_frac=0.4, rmin=0.025, rmax=6):
	exp2helper(cost_fun, distrib_params, num_arms, num_runs, update_cycle_length_psgd, update_cycle_length_ams, horizon, l_r_psgd, l_r_ams, epsilon, B_frac, rmin, rmax)


"""A helper method for running the experiments in Figures 2 and 4 of the ICML-2020 paper that compare AsyncMirrorSync's and AsyncPSGDSync's performance.

Parameters
----------
cost_fun : a descendant class of CostGen
	A class describing a family of cost-generating processes.
distrib_params : dictionary
	Parameters for the cfamily of cost-generating processes given by cost_fun.
num_arms : int
	Number of arms.
num_runs : int
	Number of runs of each algorithm to perform. The two algorithms' runs are paired, i.e., in each run both algorithms are executed on the same
	cost_fun instance initialized for that run. 
update_cycle_length_psgd : int
	Length of time between rate update attempts of AsyncPSGDSync.
	NOTE: For ease of comparison, we assume that (1 / rmin) is a multiple of update_cycle_length_psgd.
update_cycle_length_ams : int
	Length of time between rate update attempts of AsyncMirrorSync.
	NOTE: For ease of comparison, we assume that (1 / rmin) is a multiple of update_cycle_length_ams.
horizon : int
	Number of equivalent MirrorSync learning rounds (\mathcal{T}_max for MirrorSync in the ICML-2020 paper).
l_r_psgd : float
	Learning rate for AsyncPSGDSync.
l_r_ams : float
	Learning rate for AsyncMirrorSync.
epsilon : float
	The fraction of total bandwidth allocated to probe plays.
B_frac : float
	Maximum total sync play rate as a fraction of the number of arms.
rmin : float
	Minimum allowed arm sync play rate.
rmax : float
	Maximum allowed arm sync play rate.

Returns
-------
None
	The method prints strings with experiment results that can be used to produce plots such as in the paper's figures using pgfplots.
"""
def exp2helper(cost_fun, distrib_params, num_arms, num_runs, update_cycle_length_psgd, update_cycle_length_ams, horizon, l_r_psgd, l_r_ams, epsilon, B_frac, rmin, rmax):
	STEP_MULT_MD = (1 / rmin) / update_cycle_length_ams
	STEP_MULT_SGD = (1 / rmin) / update_cycle_length_psgd
	# A list of arrays that will hold the policy cost after every policy update round of every run of AsyncPSGDSync.
	psgd_runs = []
	# A list of arrays that will hold the policy cost after every policy update round of every run of AsyncMirrorSync.
	ams_runs = []
	# A list that will hold the optimal epsilon-exploration policy cost for every run. It can vary from run to run due to stochastic cost_fun initialization.
	opt_epsilon_cost = []
	
	for run in range(num_runs):
		print('\n\n=========== RUN #%r ============'%(run + 1)) 
		"""
		Initialize a family of cost-generating processes.
		NOTE: the initialization may be stochastic, and therefore is done for every run separately.
		"""
		c = cost_fun(num_arms, distrib_params)
		opt_epsilon_cost_curr = compute_opt_policy_cost(c, rmin, rmax / (1 + epsilon), B_frac * num_arms/(1 + epsilon))
		opt_epsilon_cost.append(opt_epsilon_cost_curr)
		print('Optimal epsilon-exploration periodic policy cost: %r'%(opt_epsilon_cost_curr))

		results_psgd = async_mirror_sync(c, 'psgd', num_arms, l_r_psgd, horizon, epsilon, update_cycle_length_psgd, B_frac, rmin, rmax, silent=True)
		results_ams = async_mirror_sync(c, 'ms', num_arms, l_r_ams, horizon, epsilon, update_cycle_length_ams, B_frac, rmin, rmax, silent=True)

		psgd_runs.append(results_psgd)
		ams_runs.append(results_ams)

		if (run + 1) % 10 == 0 or (run + 1) == num_runs:
			print('\n\n********** RESULTS AFTER %r RUNS *************'%(run + 1))	
			psgd_ms, psgd_hs = process_results(psgd_runs)
			psgd_plot_string = ''
			for i in range(psgd_ms.shape[0]):
				psgd_plot_string += '(' + str(i / STEP_MULT_SGD) + ',' + str(psgd_ms[i]) + ')+=(0,' + str(psgd_hs[i]) + ')-=(0,' + str(psgd_hs[i]) + ')\n'
			print('AsyncPSGDSync:\n', psgd_plot_string, '\n\n')

			ams_ms, ams_hs = process_results(ams_runs)
			ams_plot_string = ''
			for i in range(ams_ms.shape[0]):
				ams_plot_string += '(' + str(i / STEP_MULT_MD) + ',' + str(ams_ms[i]) + ')+=(0,' + str(ams_hs[i]) + ')-=(0,' + str(ams_hs[i]) + ')\n'
			print('AsyncMirrorSync:\n', ams_plot_string, '\n\n')

			opt_epsilon_cost_string = ''
			opt_epsilon_cost_mean = np.mean(opt_epsilon_cost)

			for i in range(horizon + 1):
				opt_epsilon_cost_string += '(' + str(i) + ',' + str(opt_epsilon_cost_mean) + ')\n'
			print('Optimal epsilon-exploration periodic policy cost: ', opt_epsilon_cost_string, '\n\n')
			print('*************************************************')


# Compute means and confidence intervals. Adapted from 
# https://github.com/microsoft/Optimal-Freshness-Crawl-Scheduling/blob/master/LambdaCrawlExps.py
def mean_confidence_interval(data, confidence = 0.95):
	a = 1.0 * data
	n = len(a)
	m, se = np.mean(a), scipy.stats.sem(a)
	h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
	return m, h


def process_results(runs):
	runs = np.array(runs)
	ms = np.zeros_like(runs[0,:])
	hs = np.zeros_like(ms)
	
	for j in range(runs.shape[1]):
		m, h = mean_confidence_interval(runs[:,j])
		ms[j] = m
		hs[j] = h
		
	return ms, hs


"""Hyperparameter search for the experiments in the ICML-2020 paper. This method performs one run of the specified algorithm for a fixed seed 
for every combination of values in learning_rates and update_cycle_lengths lists, keeping the other specified parameters such as horizon fixed.

Parameters
----------
cost_fun : a descendant class of CostGen
	A class describing a family of cost-generating processes.
distrib_params : dictionary
	Parameters for the cfamily of cost-generating processes given by cost_fun.
alg : string
	Algorithm name: 'MirrorSync', 'AsyncMirrorSync' or 'AsyncPSGDSync'
learning_rates : float array
	Learning rates to try.
update_cycle_lengths_psgd : int array
	Update cycle lengths to try. For MirrorSync, this list should contain just one entry equal to (1 / rmin).
num_arms : int
	Number of arms.
horizon : int
	Number of equivalent MirrorSync learning rounds (\mathcal{T}_max for MirrorSync in the ICML-2020 paper).
epsilon : float
	The fraction of total bandwidth allocated to probe plays.
B_frac : float
	Maximum total sync play rate as a fraction of the number of arms.
rmin : float
	Minimum allowed arm sync play rate.
rmax : float
	Maximum allowed arm sync play rate.

Returns
-------
None
	The method prints, for every hyperparameter combination, policy costs after every play rate update and, at the end of hyperparameter search,
	policy cost vs. number of play rate update plots for all hyperparameter combinations.
"""
# ***** Parameter ranges for Figures 1 and 2 in the ICML-2020 paper.******
#param_search(cost_fun=SublinPoly, distrib_params=poly_params, alg='MirrorSync', learning_rates=[0.5, 1, 2, 2.3, 2.5, 2.7, 3, 3.3, 4, 5, 6, 7, 9, 11, 13, 15, 17], update_cycle_lengths=[40], num_arms=100, horizon=240, epsilon=0.05, B_frac=0.4, rmin=0.025, rmax=3)
#param_search(cost_fun=SublinPoly, distrib_params=poly_params, alg='AsyncMirrorSync', learning_rates=[1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7], update_cycle_lengths=[8, 10, 20, 40], num_arms=100, horizon=240, epsilon=0.05, B_frac=0.4, rmin=0.025, rmax=3)
#param_search(cost_fun=SublinPoly, distrib_params=poly_params, alg='AsyncPSGDSync', learning_rates=[0.04, 0.06, 0.08, 0.1, 0.2, 0.3, 0.4], update_cycle_lengths=[8, 10, 20, 40], num_arms=100, horizon=240, epsilon=0.05, B_frac=0.4, rmin=0.025, rmax=3)	

# ***** Parameter ranges for Figures 3 and 4 in the ICML-2020 paper.******
#param_search(cost_fun=BinaryPoisson, distrib_params=binary_Poisson_params, alg='MirrorSync', learning_rates=[3, 3.3, 4, 5, 6, 7, 9, 11, 13], update_cycle_lengths=[40], num_arms=100, horizon=240, epsilon=0.05, B_frac=0.4, rmin=0.025, rmax=6)
#param_search(cost_fun=BinaryPoisson, distrib_params=binary_Poisson_params, alg='AsyncMirrorSync', learning_rates=[1.1, 1.0, 1.3, 1.2, 0.9, 0.8], update_cycle_lengths=[8, 10, 20, 40], num_arms=100, horizon=240, epsilon=0.05, B_frac=0.4, rmin=0.025, rmax=6)
#param_search(cost_fun=BinaryPoisson, distrib_params=binary_Poisson_params, alg='AsyncPSGDSync', learning_rates=[0.08, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], update_cycle_lengths=[8, 10, 20, 40], num_arms=100, horizon=240, epsilon=0.05, B_frac=0.4, rmin=0.025, rmax=6)
def param_search(cost_fun, distrib_params, alg, learning_rates, update_cycle_lengths, num_arms, horizon, epsilon, B_frac, rmin, rmax):
	assert(alg == 'MirrorSync' or alg == 'AsyncMirrorSync' or alg == 'AsyncPSGDSync')
	figctr = 0 
	print("Evaluating ", alg)

	for learning_rate in learning_rates:
		tried_this_learning_rate = False

		for update_cycle_length in update_cycle_lengths:
			# The seed is fixed for the purposes of hyperparameter tuning.
			np.random.seed(0)
			c = cost_fun(num_arms, distrib_params)
			success = True

			opt_epsilon_cost_curr = compute_opt_policy_cost(c, rmin, rmax/(1 + epsilon), B_frac * num_arms/(1 + epsilon))
			print('Optimal epsilon-exploration periodic policy cost: %r'%(opt_epsilon_cost_curr))

			results = []
			
			#try:
			if alg == 'MirrorSync':
				if not tried_this_learning_rate:
					results =  mirror_sync(c, num_arms, learning_rate, horizon, epsilon, B_frac, rmin, rmax, True)
					tried_this_learning_rate = True
				else:
					continue
			elif alg == 'AsyncMirrorSync':
				results = async_mirror_sync(c, 'ms', num_arms, learning_rate, horizon, epsilon, update_cycle_length, B_frac, rmin, rmax, True)
			elif alg == 'AsyncPSGDSync':
				results = async_mirror_sync(c, 'psgd', num_arms, learning_rate, horizon, epsilon, update_cycle_length, B_frac, rmin, rmax, True)
			#except:
			#	success = False
				
			if success:
				plt.figure(figctr)
				
				if alg == 'MirrorSync':
					print("Results for learning rate = ", learning_rate, " are: ")
					plt.title("eta = " + str(learning_rate))
				else:
					print("Results for learning rate = ", learning_rate, ", update cycle length = ", update_cycle_length, " are: ")
					plt.title("eta = " + str(learning_rate) + ", update_cycle_length = " + str(update_cycle_length))
					
				print(results)
				x = np.arange(len(results)) / len(results) * horizon
				plt.plot(x, results)
				plt.draw()
			else:
				if alg == 'MirrorSync':
					print("Results for learning rate = ", learning_rate, " are: the convex optimizer is unstable for this parameter combination and RNG seed, trying another parameter combination")
				else:
					print("Results for learning rate = ", learning_rate, ", update cycle length = ", update_cycle_length, " are: the algorithm is unstable for this parameter combination, trying another one")
			
			figctr += 1
			print('\n\n')
	plt.show()
