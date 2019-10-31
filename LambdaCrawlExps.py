import numpy as np;
import scipy as sp;
from scipy import optimize;
import math;
import time;
import sys;
import scipy.stats
import json
from operator import itemgetter


# This file contains the code for running the experiments in the paper
#
# A. Kolobov, Y. Peres, C. Lu, E. Horvitz. "Staying up to Date with Online Content Changes Using Reinforcement Learning for Scheduling." NeurIPS-2019.

epsilon_learn_global = 0.0001

# Learns sources' Poisson change rates from observation histories. 
#
# The curr_time parameter is supposed to be a time relative to the same timestamp as the timestamps in histories. The suffix_length parameter is the
# length of history suffix immediately preceding curr_time telling this method to use only thehistory interval from [curr_time - suffix, curr_time]
# for learning the change rates.
def LearnChRates(observations_incompl_obs, ch_rates_incompl_obs_est, observations_compl_obs, \
    ch_rates_compl_obs_est, epsilon_learn, suffix_length, curr_time):
    for w in range(len(ch_rates_incompl_obs_est)):
        ch_rates_incompl_obs_est[w] = LearnChRates_IncomplObs(observations_incompl_obs[w], suffix_length, curr_time, epsilon_learn)
    for w in range(len(ch_rates_compl_obs_est)):
        ch_rates_compl_obs_est[w] = LearnChRates_ComplObs(observations_compl_obs[w], suffix_length, curr_time)



# Learns a sources' Poisson change rate from the sources' observation history assuming complete change observations.
# 
# The training_data parameter is an array of timestamps when changes were detected, sorted in the increasing order. The timestamps are assumed to be
# relative to some time point in the past. The suffix_length parameter is the length of history suffix immediately preceding curr_time telling this
# method to use only thehistory interval from [curr_time - suffix, curr_time] for learning the change rates. The method uses Equation 11 from the
# NeurIPS-2019 paper for learning the change rate.
def LearnChRates_ComplObs(training_data, suffix_length, curr_time):
    # We use a type of smoothing whereby we assume we start with some number (not necessarily an integer) of imaginary change observations over an
    # imaginary period of time. Their values we used for the experiments in the NeurIPS-2019 paper are below.
    smoothing_count_term = 0.5
    smoothing_interval_term = 0.5
    total_changes = 0
    time_so_far = 0

    i = len(training_data) - 1 
    while (i >= 0 and training_data[i] > curr_time): 
        i -= 1

    if (i < 0):
        # the observation history before curr_time is empty
        return smoothing_count_term / (smoothing_interval_term + min(curr_time, suffix_length))
    elif (curr_time - training_data[i] > suffix_length):
        # there were no observations in the time interval [curr_time - suffix_length, curr_time]
        return smoothing_count_term / (smoothing_interval_term + suffix_length)
    else:
        time_so_far = curr_time - training_data[i]
        
    while (time_so_far < suffix_length and i >= 0):
        # We assume we are fresh w.r.t. every source when we start crawling
        previous_interval = (training_data[i] - training_data[i - 1]) if (i > 0) else training_data[i]
        total_changes += 1
        time_so_far = min(time_so_far + previous_interval, suffix_length)
        i -= 1
        
    return ((total_changes + smoothing_count_term) / (time_so_far + smoothing_interval_term))


# Learns a source's Poisson change rate from the source's observation history assuming incomplete change observations.
# 
# The training_data parameter is a 2D array of history data for a given source, where training_data[i][0] is the timestamp of the i-th crawl happened,
# assumed to be relative to some time point in the past, and training_data[i][0] is a 0/1 flag indicating where that crawl detected any changes. The 
# suffix_length parameter is the length of history suffix immediately preceding curr_time telling this method to use only the history interval from 
# [curr_time - suffix, curr_time] for learning the change rates. The method uses Equation 10 from the NeurIPS-2019 paper for learning the change rate.
def LearnChRates_IncomplObs(training_data, suffix_length, curr_time, epsilon):
    # We use a type of smoothing whereby we assume we start with some number of inter-crawl intervals (not necessarily an integer) of various lengths 
    # that detected a change and some number that didn't. We used one interval of each kind, whose length in both cases is the same, shown below.
    smoothing_term = 0.5
    time_so_far = 0

    # Initialize the sets of inter-observation intervals assuming we had one imaginary interval after which we observed
    # a change and one imaginary interval after which we didn't, with the length of either interval equal to
    # the smoothing term.
    list_ch_training_intervals = [smoothing_term]
    list_no_ch_training_intervals = [smoothing_term]

    i = len(training_data) - 1
    while (i >= 0 and training_data[i][0] > curr_time): 
        i -= 1

    if (i < 0):
        # The observation history before curr_time is empty. Return the change rate estimate 
        # assuming we had one imaginary interval after which we observed a change and 
        # one imaginary interval after which we didn't, with the length of either interval equal to
        # the smoothing term.
        return math.log(2) / smoothing_term
    
    # Count back the history suffix starting with the time of the latest observation (crawl) that happened before curr_time. We can't use the time
    # interval between the latest crawl and the current time for training purposes, because we haven't collected any training data (observations) 
    # in that interval, and hence, due to the incompleteness of our observation history, don't know whether any changes have happened since the 
    # latest observation
    while (time_so_far < suffix_length and i >= 0):
        previous_interval = (training_data[i][0] - training_data[i - 1][0]) if (i > 0) else training_data[i][0]
        # Note that after the next line, it may be that time_so_far + previous_interval > suffix_length.
        # This is fine -- to make learning less biased, we need to either use the last observation that falls into  
        # [reference_time - suffix_length, reference_time] *and* the *entire* intra-observation interval
        # preceding it, or use neither that observation nor that interval. We choose to do the former.
        if training_data[i][1] == 1:
            list_ch_training_intervals.append(previous_interval)
        else:
            list_no_ch_training_intervals.append(previous_interval)

        time_so_far += previous_interval
        i -= 1

    max_ch_interval = max(list_ch_training_intervals)
    min_ch_interval = min(list_ch_training_intervals)
    sum_ch_intervals = sum(list_ch_training_intervals)
    sum_no_ch_intervals = sum(list_no_ch_training_intervals)

    # Based on the MLE for incomplete observation history, it's easy to verify that the bounds always bracket 
    # the change rate estimate. However, they affect the exponent of e during learning, and when they are  
    # too loose, cause numerical errors. Therefore, we bound the range of change rate's values based on reasonable
    # guesses based on prior knowledge.
    ch_rate_lb = 0.0005 # math.log(sum_ch_intervals / sum_no_ch_intervals + 1) / max_ch_interval;
    ch_rate_ub = 7 # math.log(sum_ch_intervals / sum_no_ch_intervals + 1) / min_ch_interval;

    return BisectionSearchMonotoneDecr(DataLLDerivative, (list_ch_training_intervals, list_no_ch_training_intervals), ch_rate_lb, ch_rate_ub, epsilon)


# Computes the data log likelihood of a history of incomplete change observations under a given Poisson change rate.
def DataLLDerivative(params, ch_rate):
    term_with_changes = 0
    term_without_changes = 0
    list_ch_training_intervals = params[0]
    list_no_ch_training_intervals = params[1]

    for interval in list_ch_training_intervals:
        term_with_changes += (interval / (math.exp(ch_rate * interval) - 1))

    for interval in list_no_ch_training_intervals:
        term_without_changes += interval

    return (term_with_changes - term_without_changes)


# Evaluates the Lagrangian of the negative harmonic policy cost under for a given lambda.
#
# See Proposition 2 and equation system 5 in the NeurIPS-2019 paper for details.
def PolicyValueLagr_NoObs(params, lambd):
    sum = 0
    importances = params[0]
    ch_rates = params[1]
    bandwidth = params[2]

    # We are assuming importances and cr_rates have the same length
    for i in range(len(importances)):
        sum += (-ch_rates[i] + math.sqrt(ch_rates[i]**2 + 4 * importances[i] * ch_rates[i] / lambd)) / 2

    return sum - bandwidth


# Bisection search for the root of a 1-variable function parameterized by known parameter values (param) and assumed to be monotonically decreasing
# on the interval [var_lb_init, var_ub_init].
#
# The assumption that the function is monotonically decreasing knowledge lets us save one function evaluation.
def BisectionSearchMonotoneDecr(func, params, var_lb_init, var_ub_init, epsilon):
    var_lb = var_lb_init
    var_ub = var_ub_init
    var_center = (var_ub + var_lb) / 2

    num_it = 0
    while ((var_ub - var_lb) /  2 > epsilon):
        num_it += 1
        func_val = func(params, var_center)
        if (func_val == 0):
            return var_center
        elif (func_val > 0):
            var_lb = var_center
        else:
            var_ub = var_center
        var_center = (var_ub + var_lb) / 2
    return var_center


# LambdaCrawl for sources with incomplete change observations.
#
# See the pseudocode in Algorithm 1 in the NeurIPS-2019 paper for details.
def LambdaCrawl_IncomplObs(importances, ch_rates, bandwidth, epsilon):
    if (bandwidth == 0 or importances.shape[0] == 0):
        return np.zeros_like(importances, dtype=float)

    min_importance = min(importances)
    max_importance = max(importances)
    min_ch_rate = min(ch_rates)
    max_ch_rate = max(ch_rates)
    lambda_lb = len(importances)**2 * min_importance * min_ch_rate / (len(importances) * max_ch_rate * bandwidth + bandwidth**2)
    lambda_ub = len(importances)**2 * max_importance * max_ch_rate / (len(importances) * min_ch_rate * bandwidth + bandwidth**2)
    lambd = BisectionSearchMonotoneDecr(PolicyValueLagr_NoObs, (importances, ch_rates, bandwidth), lambda_lb, lambda_ub, epsilon)
    crawl_rates = np.zeros_like(importances, dtype=float)

    for i in range(len(crawl_rates)):
        crawl_rates[i] = (-ch_rates[i] + math.sqrt(ch_rates[i]**2 + 4 * importances[i] * ch_rates[i] / lambd)) / 2

    return crawl_rates


# LambdaCrawl approximation for sources with incomplete change observations that assumes importance_w/change_rate_w = c for a fixed (unknown) c
# for all sources w.
#
# See Proposition 9 in the NeurIPS-2019 paper for details.
def LambdaCrawlApprox_IncomplObs(importances, ch_rates, bandwidth, epsilon):
    crawl_rates = np.zeros_like(importances, dtype=float)
    sum_imp = sum(importances)

    for w in range(len(importances)):
        crawl_rates[w] = importances[w] * bandwidth / sum_imp
        
    return crawl_rates


# LambdaCrawl for sources with complete change observations.
#
# See the pseudocode in Algorithm 2 in the NeurIPS-2019 paper for details.
def LambdaCrawl_ComplObs(importances, ch_rates, bandwidth):
    if (bandwidth == 0):
        return np.zeros_like(importances, dtype=float)

    crawl_probs = np.zeros_like(importances, dtype=float)
    crawl_probs_approx = np.empty_like(importances, dtype=float) 
    remaining_bandwidth = bandwidth

    num_it = 0
    while True:
        num_it += 1
        saturated_a_constraint = False
        # For the calculation of the denominator later in this loop, we will need the sum of importances, *but only of those sources
        # for which we haven't determined crawl_probs[w] = 1 in previous iterations*. Sources for which we have determined this have
        # been excluded from consideration entirely.
        sum_importances = 0
        for w in range(len(importances)):
            if crawl_probs[w] == 0:
                sum_importances += importances[w]

        for w in range(len(importances)):
            # under any acceptable solution, the crawl probability of every source is > 0, so crawl_probs[i] = 0
            # meanst that we haven't determined the final value for this source yet 
            if crawl_probs[w] == 0:
                p_hat_w = bandwidth * importances[w] / (ch_rates[w] * sum_importances)
                # if the probability constraint p_w <= 1 is saturated for source w...
                if (p_hat_w >= 1.0):
                    crawl_probs[w] = 1.0
                    # indicate that we don't need to reconsider the crawl probability value of source w in subsequent iterations
                    crawl_probs_approx[w] = -1
                    remaining_bandwidth -= (crawl_probs[w] * ch_rates[w])
                    saturated_a_constraint = True
                else:
                    crawl_probs_approx[w] = p_hat_w

        # if we didn't saturate any constraints in the last iteration, the non-negative values in 
        # crawl_probs_approx are the final crawl probability values of the corresponding sources
        if saturated_a_constraint == False:
            for w in range(len(importances)):
                if (crawl_probs_approx[w] != -1):
                    crawl_probs[w] = crawl_probs_approx[w]
            break
        # otherwise, repeat with remaining bandwidth and sources that don't have crawl_probs_approx[w] = -1 yet
        else:
            bandwidth = remaining_bandwidth

    return crawl_probs


# Computes the harmonic policy cost for URLs with incomplete observations.
#
# See Proposition 1/Equation 4 in the NeurIPS-2019 paper.
def HarmonicPolicyCost_IncomplObs(importances, ch_rates, crawl_rates):
    if (len(importances) == 0):
        return 0
        
    cost_incompl_obs = 0
    for w in range(len(importances)):
        if crawl_rates[w] == 0:
            cost_incompl_obs = math.inf
            break
        cost_incompl_obs -= importances[w] * math.log(crawl_rates[w] / (crawl_rates[w] + ch_rates[w]))
    return cost_incompl_obs / len(importances)


# Computes the harmonic policy cost for URLs with complete observations.
#
# See the formula in Proposition 4 in the NeurIPS-2019 paper.
def HarmonicPolicyCost_ComplObs(importances, ch_rates, crawl_probs):
    if (len(importances) == 0):
        return 0
        
    cost_compl_obs = 0
    for w in range(len(importances)):
        if crawl_probs[w] == 0:
            cost_compl_obs = math.inf
            break
        cost_compl_obs -= importances[w] * math.log(crawl_probs[w])
    return cost_compl_obs / len(importances)


# Computes the harmonic policy cost for a mix of URLs with complete and incomplete observations.
#
# See the Equation 8 in the NeurIPS-2019 paper.
def HarmonicPolicyCost(crawl_rates_incompl_obs, crawl_probs_compl_obs, importances_incompl_obs, ch_rates_incompl_obs, \
    importances_compl_obs, ch_rates_compl_obs):
    
    cost_incompl_obs = HarmonicPolicyCost_IncomplObs(importances_incompl_obs, ch_rates_incompl_obs, crawl_rates_incompl_obs)
    cost_compl_obs = HarmonicPolicyCost_ComplObs(importances_compl_obs, ch_rates_compl_obs, crawl_probs_compl_obs)
    
    return ((len(importances_incompl_obs) * cost_incompl_obs + len(importances_compl_obs) * cost_compl_obs) \
        / (len(importances_incompl_obs) + len(importances_compl_obs)))


# Computes the binary policy cost for URLs with incomplete observations.
#
# See Equation 12 in the NeurIPS-2019 paper's supplement.
def BinaryPolicyCost_IncomplObs(importances, ch_rates, crawl_rates):
    if (len(importances) == 0):
        return 0
        
    cost_incompl_obs = 0
    for w in range(len(importances)):
        cost_incompl_obs += importances[w] * ch_rates[w] / (crawl_rates[w] + ch_rates[w])
    return cost_incompl_obs / len(importances)
    

# Computes the binary policy cost for URLs with complete observations.
#
# See Equation 13 in the NeurIPS-2019 paper's supplement.
def BinaryPolicyCost_ComplObs(importances, ch_rates, crawl_probs):
    if (len(importances) == 0):
        return 0
        
    cost_compl_obs = 0
    for w in range(len(importances)):
        cost_compl_obs += importances[w] * (1 - crawl_probs[w])
    return cost_compl_obs / len(importances)


# Computes the binary policy cost for a mix of URLs with complete and incomplete observations.
def BinaryPolicyCost(crawl_rates_incompl_obs, crawl_probs_compl_obs, importances_incompl_obs, ch_rates_incompl_obs, \
    importances_compl_obs, ch_rates_compl_obs):
    cost_incompl_obs = BinaryPolicyCost_IncomplObs(importances_incompl_obs, ch_rates_incompl_obs, crawl_rates_incompl_obs)
    cost_compl_obs = BinaryPolicyCost_ComplObs(importances_compl_obs, ch_rates_compl_obs, crawl_probs_compl_obs)
    return ((len(importances_incompl_obs) * cost_incompl_obs + len(importances_compl_obs) * cost_compl_obs) \
        / (len(importances_incompl_obs) + len(importances_compl_obs)))


# Computes the harmonic policy cost for a given bandwidth split across complete- and incomplete-change-history URLs. 
#
# See Algorithm 3 in the NeurIPS-2019 paper. The return of the function in the pseudocode is the negative of SplitEval_JStar.
def SplitEval_JStar(bandwidth_compl_obs, solver_x_incompl_obs, importances_incompl_obs, ch_rates_incompl_obs, epsilon_incompl_obs, importances_compl_obs, ch_rates_compl_obs, bandwidth):
    if (bandwidth_compl_obs > bandwidth):
        raise ValueError('SplitEval_JStar ERROR: bandwidth allocation to sources with complete observations exceeds total bandwidth! Bandwidth allocation to sources with complete observations: ', bandwidth_compl_obs, ", total bandwidth: ", bandwidth)
    
    crawl_rates_incompl_obs = solver_x_incompl_obs(importances_incompl_obs, ch_rates_incompl_obs, bandwidth - bandwidth_compl_obs, epsilon_incompl_obs)
    J_incompl_obs = 0
    for w in range(len(importances_incompl_obs)):
        if crawl_rates_incompl_obs[w] == 0:
            J_incompl_obs = math.inf
            break
        else:
            J_incompl_obs -= (importances_incompl_obs[w] * math.log(crawl_rates_incompl_obs[w] / (crawl_rates_incompl_obs[w] + ch_rates_incompl_obs[w])))
    
    crawl_probs_compl_obs = LambdaCrawl_ComplObs(importances_compl_obs, ch_rates_compl_obs, bandwidth_compl_obs)
    J_compl_obs = 0
    for w in range(len(importances_compl_obs)):
        if crawl_probs_compl_obs[w] == 0:
            J_compl_obs = math.inf
            break
        else:
            J_compl_obs -= (importances_compl_obs[w] * math.log(crawl_probs_compl_obs[w]))

    return  J_incompl_obs + J_compl_obs


# Implements the LambdaCrawl family of algorithms.
#
# See Algorithm 3 in the NeurIPS-2019 paper. The implementation can use either the optimal LambdaCrawlApprox_IncomplObs or the approximate
# LambdaCrawlApprox_IncomplObs for handling the incomplete-change-observation sources.
def LambdaCrawl_X(solver_x_incompl_obs, importances_incompl_obs, ch_rates_incompl_obs, epsilon_incompl_obs, importances_compl_obs, ch_rates_compl_obs, bandwidth):
    # We use a minimization routine here, so SplitEval_JStar returns the value of J* for a given split, _not_ of \overline{J}^* = -J^* 
    # as in LambdaCrawl's description in the paper.
    result = sp.optimize.minimize_scalar(SplitEval_JStar, bounds=(0, min(bandwidth, sum(ch_rates_compl_obs))), \
        args=(solver_x_incompl_obs, importances_incompl_obs, ch_rates_incompl_obs, epsilon_incompl_obs, importances_compl_obs, ch_rates_compl_obs, bandwidth), \
        method='bounded', options={'xatol': 0.005 * bandwidth})
    
    if result.success:
        bandwidth_compl_obs = result.x
        crawl_rates_incompl_obs = solver_x_incompl_obs(importances_incompl_obs, ch_rates_incompl_obs, bandwidth - bandwidth_compl_obs, epsilon_incompl_obs) 
        crawl_probs_compl_obs = LambdaCrawl_ComplObs(importances_compl_obs, ch_rates_compl_obs, bandwidth_compl_obs)
        return (crawl_rates_incompl_obs, crawl_probs_compl_obs)
    else:
        raise ValueError('LambdaCrawl ERROR: bounded minimization failed')


# Implements LambdaCrawl proper. See LambdaCrawl_X for details.
def LambdaCrawl(importances_incompl_obs, ch_rates_incompl_obs, epsilon_incompl_obs, importances_compl_obs, ch_rates_compl_obs, bandwidth):
    return LambdaCrawl_X(LambdaCrawl_IncomplObs, importances_incompl_obs, ch_rates_incompl_obs, epsilon_incompl_obs, importances_compl_obs, ch_rates_compl_obs, bandwidth)


# Implements LambdaCrawlApprox, i.e., LambdaCrawl that uses the approximation from Proposition 9 in the NeurIPS-2019 paper to handle the 
# incomplete-change-observation sources. See LambdaCrawl_X for details.
def LambdaCrawlApprox(importances_incompl_obs, ch_rates_incompl_obs, epsilon_incompl_obs, importances_compl_obs, ch_rates_compl_obs, bandwidth):
    return LambdaCrawl_X(LambdaCrawlApprox_IncomplObs, importances_incompl_obs, ch_rates_incompl_obs, epsilon_incompl_obs, importances_compl_obs, ch_rates_compl_obs, bandwidth)


# Implements the BinaryLambdaCrawl family of algorithms.
#
# This is a generalization of the algorithm from Y. Azar, E. Horvitz, E. Lubetzky, Y. Peres, D. Shahaf. "Tractable near-optimal policies for crawling."
# PNAS-2018, which the NeurIPS-2019 paper refers to as BinaryLambdaCrawl. That algorithm optimizes the binary policy cost (see that paper for 
# details) and as a result may fail to allocate any crawl rate to some of the sources. BinaryLambdaCrawl(epsilon) (see Subsection 9.3 in the 
# supplement of the NeurIPS-2019 paper) is a modification of BinaryLambdaCrawl that forces it to allocate some bandwidth even to pages that 
# BinaryLambdaCrawl would otherwise crawl-starve. In this family, BinaryLambdaCrawl(0.0) corresponds to the original BinaryLambdaCrawl. 
# BinaryLambdaCrawl(0.4) has the best performance on the NeurIPS-2019 paper's dataset w.r.t. the harmonic policy cost, of all 
# BinaryLambdaCrawl(epsilon) with epsilon in {0.0, 0.1,...,1}.
#
# The wrapper handles sources with complete and incomplete change observations. BinaryLambdaCrawl doesn't know how to handle the former in any special
# way; it simply treats them as if their observation history was incomplete. See LambdaCrawlBinary_Epsilon_Helper for most of this algorithm's logic.
def LambdaCrawlBinary_Epsilon(importances_incompl_obs, ch_rates_incompl_obs, epsilon, importances_compl_obs, ch_rates_compl_obs, bandwidth):
    if (len(importances_compl_obs) > 0):
        sys.exit("ERROR: LambdaCrawlBinary_Epsilon doesn't know how to handle complete observation histories, but importances_compl_obs is nonempty")
    imps_and_chrates = np.column_stack((importances_incompl_obs, ch_rates_incompl_obs))
    crawl_rates = np.zeros_like(imps_and_chrates[:,0], dtype=float)
    min_crawl_rate = bandwidth / imps_and_chrates.shape[0] * epsilon
    LambdaCrawlBinary_Epsilon_Helper(imps_and_chrates, crawl_rates, bandwidth, min_crawl_rate)
    return crawl_rates, []


# The main part of BinaryLambdaCrawl implementation. 
#
# See Section 9.3 of the NeurIPS-2019 paper's supplement and Y. Azar, E. Horvitz, E. Lubetzky, Y. Peres, D. Shahaf. "Tractable near-optimal policies
# for crawling." PNAS-2018 for details.
def LambdaCrawlBinary_Epsilon_Helper(imps_and_chrates, crawl_rates, bandwidth, min_crawl_rate): 
    idxs_and_value_ratios = []
    for w in range(imps_and_chrates.shape[0]):
        idxs_and_value_ratios.append((imps_and_chrates[w, 0] *  imps_and_chrates[w, 1] / (imps_and_chrates[w, 1] + min_crawl_rate)**2, w))
    
    r = 0
    for w in range(len(idxs_and_value_ratios)):
        r += math.sqrt(imps_and_chrates[w, 0] * imps_and_chrates[w, 1])
    s = sum(imps_and_chrates[:, 1])
    idxs_and_value_ratios.sort(key=itemgetter(0))
    rem_bandwidth = bandwidth
    for w in range(len(idxs_and_value_ratios)):
        if (imps_and_chrates[idxs_and_value_ratios[w][1], 0] * imps_and_chrates[idxs_and_value_ratios[w][1], 1] \
                / (imps_and_chrates[idxs_and_value_ratios[w][1], 1] + min_crawl_rate)**2 <= (r / (rem_bandwidth + s ))**2):
            r -= math.sqrt(imps_and_chrates[idxs_and_value_ratios[w][1], 0] \
                * imps_and_chrates[idxs_and_value_ratios[w][1], 1])
            s -= imps_and_chrates[idxs_and_value_ratios[w][1], 1]
            crawl_rates[idxs_and_value_ratios[w][1]] = min_crawl_rate
            rem_bandwidth -= crawl_rates[idxs_and_value_ratios[w][1]]
        else:
            # NOTE: this clause kicks in at every iteration after some iteration M. It doesn't alternate with the clause above.
            crawl_rates[idxs_and_value_ratios[w][1]] = \
                math.sqrt(imps_and_chrates[idxs_and_value_ratios[w][1], 0] \
                * imps_and_chrates[idxs_and_value_ratios[w][1], 1]) \
                * (rem_bandwidth + s) / r - imps_and_chrates[idxs_and_value_ratios[w][1], 1]


# Implements the RL version of the LambdaCrawl family of algorithms.
#
# See Algorithm 4 in the NeurIPS-2019 paper. The implementation can use either LambdaCrawl itself, or LambdaCrawlApprox, its approximate version,
# or LambdaCrawlBinary that minimizes binary policy cost as solver_x, the algorithm for handling incomplete-change-observation sources.
def LambdaLearnAndCrawl_X(solver_x, importances_incompl_obs, ch_rates_incompl_obs_actual, epsilon_incompl_obs, \
    importances_compl_obs, ch_rates_compl_obs_actual, bandwidth, epsilon_learn, epoch_length, suffix_len, horizon):

    start_total = time.time()
    curr_time = 0
    changes_incompl_obs = np.empty(len(importances_incompl_obs), dtype=np.object)
    for i in range(len(changes_incompl_obs)): changes_incompl_obs[i] = []
    crawls_incompl_obs = np.empty(len(importances_incompl_obs), dtype=np.object)
    for i in range(len(crawls_incompl_obs)): crawls_incompl_obs[i] = []
    observations_incompl_obs = np.empty(len(importances_incompl_obs), dtype=np.object)
    for i in range(len(observations_incompl_obs)): observations_incompl_obs[i] = []
    changes_compl_obs = np.empty(len(importances_compl_obs), dtype=np.object)
    for i in range(len(changes_compl_obs)): changes_compl_obs[i] = []
    crawls_compl_obs = np.empty(len(importances_compl_obs), dtype=np.object)
    for i in range(len(crawls_compl_obs)): crawls_compl_obs[i] = []
    observations_compl_obs = changes_compl_obs # since the observation sequence is complete, we observe every change
    ch_rates_incompl_obs_est = [1.0] * len(ch_rates_incompl_obs_actual)
    ch_rates_compl_obs_est = [1.0] * len(ch_rates_compl_obs_actual)

    idealized_policy_costs_harmonic = []
    idealized_policy_costs_binary = []

    while curr_time < horizon:
        start_sch = time.time()
        # Compute the optimal policy parameters given the current change rate estimates.
        crawl_rates_incompl_obs, crawl_probs_compl_obs = solver_x(importances_incompl_obs, ch_rates_incompl_obs_est, \
            epsilon_incompl_obs, importances_compl_obs, ch_rates_compl_obs_est, bandwidth)
        # Apply Propositions 1 & 4 from the NeurIPS-2019 paper to find the (asymptotic) harmonic policy cost.
        idealized_policy_cost_harmonic = HarmonicPolicyCost(crawl_rates_incompl_obs, crawl_probs_compl_obs, importances_incompl_obs,\
                ch_rates_incompl_obs_actual, importances_compl_obs, ch_rates_compl_obs_actual)
        # Apply Equations 12 & 13 from the NeurIPS-2019 paper's supplement to find the (asymptotic) binary policy cost.
        idealized_policy_cost_binary = BinaryPolicyCost(crawl_rates_incompl_obs, crawl_probs_compl_obs, importances_incompl_obs,\
                ch_rates_incompl_obs_actual, importances_compl_obs, ch_rates_compl_obs_actual)
        
        print("Idealized harmonic policy cost: ", idealized_policy_cost_harmonic)
        idealized_policy_costs_harmonic.append(idealized_policy_cost_harmonic)
        idealized_policy_costs_binary.append(idealized_policy_cost_binary)
        end_sch = time.time()
        start_ext = time.time()
        # Advance the simulation time by until the horizon: simulate source changes for sources of both types until the horizon, then, given these 
        # changes, simulate crawls from the current policy using policy parameters computed above, and finally generate observations until the
        # horizon, given the generated changes and crawls. Note that for extending the observation history of complete-observation 
        # sources, the scheduling policy doesn't matter.
        ExtendChangeHistory(changes_incompl_obs, ch_rates_incompl_obs_actual, curr_time, epoch_length)
        ExtendChangeHistory(changes_compl_obs, ch_rates_compl_obs_actual, curr_time, epoch_length)
        ExtendCrawlHistory_IncomplObs(crawls_incompl_obs, crawl_rates_incompl_obs, curr_time, epoch_length)
        ExtendCrawlHistory_ComplObs(changes_compl_obs, crawls_compl_obs, crawl_probs_compl_obs, curr_time, epoch_length)
        ExtendObsHistory_IncomplObs(changes_incompl_obs, crawls_incompl_obs, observations_incompl_obs, curr_time, epoch_length)
        end_ext = time.time()
        curr_time = min(curr_time + epoch_length, horizon)
        start_learn = time.time()
        # Re-learn the change rates from the extended observation data. There is actually no need to learn for the incomplete-observation sources 
        # in case we use LambdaLearnAndCrawlApprox, but we learn them anyway.
        LearnChRates(observations_incompl_obs, ch_rates_incompl_obs_est, observations_compl_obs, ch_rates_compl_obs_est, \
            epsilon_learn, math.inf, curr_time)
        end_learn = time.time()
        #print('\tLearning took {} seconds'.format(end_learn - start_learn))

    end_total = time.time()
    total_time = end_total - start_total
    print('RL took {} seconds in total'.format(total_time))
    policy_cost =  EvalMixedTrace(importances_incompl_obs, changes_incompl_obs, crawls_incompl_obs, importances_compl_obs, changes_compl_obs, \
        crawls_compl_obs, horizon)
    return policy_cost, total_time, idealized_policy_costs_harmonic, idealized_policy_costs_binary


# Implements LambdaLearnAndCrawl. See LambdaLearnAndCrawl_X for details.
def LambdaLearnAndCrawl(importances_incompl_obs, ch_rates_incompl_obs_actual , epsilon_incompl_obs, \
    importances_compl_obs, ch_rates_compl_obs_actual, bandwidth, epsilon_learn, epoch_length, suffix_len, horizon):
    print("Running LambdaLearnAndCrawl...")
    return LambdaLearnAndCrawl_X(LambdaCrawl, importances_incompl_obs, ch_rates_incompl_obs_actual , epsilon_incompl_obs, \
        importances_compl_obs, ch_rates_compl_obs_actual, bandwidth, epsilon_learn, epoch_length, suffix_len, horizon)


# Implements LambdaLearnAndCrawlApprox. See LambdaLearnAndCrawl_X for details.
def LambdaLearnAndCrawlApprox(importances_incompl_obs, ch_rates_incompl_obs_actual , epsilon_incompl_obs, \
    importances_compl_obs, ch_rates_compl_obs_actual, bandwidth, epsilon_learn, epoch_length, suffix_len, horizon):
    print("Running LambdaLearnAndCrawlApprox...")
    return LambdaLearnAndCrawl_X(LambdaCrawlApprox, importances_incompl_obs, ch_rates_incompl_obs_actual , epsilon_incompl_obs, \
        importances_compl_obs, ch_rates_compl_obs_actual, bandwidth, epsilon_learn, epoch_length, suffix_len, horizon)


# Implements LambdaLearnAndCrawlBinary. See LambdaLearnAndCrawl_X for details.
def LambdaLearnAndCrawlBinary(importances_incompl_obs, ch_rates_incompl_obs_actual , epsilon_incompl_obs, \
    importances_compl_obs, ch_rates_compl_obs_actual, bandwidth, epsilon_learn, epoch_length, suffix_len, horizon):
    print("Running LambdaLearnAndCrawlBinary...")
    # LambdaCrawlBinary doesn't know how to deal with pages with complete change observations, so put all pages into the "incomplete change
    # observations bucket.
    importances_incompl_obs_all = np.concatenate((importances_incompl_obs, importances_compl_obs))
    ch_rates_incompl_obs_actual_all = np.concatenate((ch_rates_incompl_obs_actual, ch_rates_compl_obs_actual))
    return LambdaLearnAndCrawl_X(LambdaCrawlBinary_Epsilon, importances_incompl_obs_all, ch_rates_incompl_obs_actual_all, epsilon_incompl_obs, \
        [], [], bandwidth, epsilon_learn, epoch_length, suffix_len, horizon)


# Computes the n-th Harmonic number.
def Harmonic(n):
    # Not very efficient for large n, but we don't expect n to be large under LambdaCrawl
    return sum(1/k for k in range(1, n + 1))


# Computes the time-averaged harmonic penalty for a LambdaCrawl policy given a history of changes for a mixed set of sources with complete and
# incomplete change observations. 
#
# Note that we need the history of changes themselves, not of observations, to do this evaluation. See EvalTrace for more details.
def EvalMixedTrace(importances_incompl_obs, changes_incompl_obs, crawls_incompl_obs, importances_compl_obs, changes_compl_obs, \
    crawls_compl_obs, horizon):
    return (len(importances_incompl_obs) * EvalTrace(importances_incompl_obs, changes_incompl_obs, crawls_incompl_obs, horizon) + \
        len(importances_compl_obs) * EvalTrace(importances_compl_obs, changes_compl_obs, crawls_compl_obs, horizon)) / (len(importances_incompl_obs) + len(importances_compl_obs))


# Computes the time-averaged harmonic penalty for a LambdaCrawl policy.
def EvalTrace(importances, ch_hists, crawl_hists, horizon):
    if (len(importances) == 0):
        return 0
    # This function returns the harmonic *cost* of a trace. I.e., the lower value it returns, the better
    J_pi = 0
    for w in range(len(importances)):
        # If the source never changed during the observation period, its contribution to the penalty is 0
        if (not ch_hists[w]):
            continue

        # Otherwise, if there were changes but no crawls, count the number of changes before the horizon 
        if (not crawl_hists[w]):
            num_changes = 0
            for t in range(len(ch_hists[w])): 
                if (ch_hists[w][t] <= horizon): 
                    num_changes += 1 
                else:
                    break
            J_pi += (importances[w] * Harmonic(num_changes) / horizon)
            continue

        curr_num_changes = 0
        running_penalty = 0
        curr_change_time_idx = 0
        curr_crawl_time_idx = 0

        while (curr_change_time_idx < len(ch_hists[w]) and ch_hists[w][curr_change_time_idx] <= horizon):
            # To continue the loop below we need to have either (a) unprocessed crawls s.t. their crawl times are after the current 
            # change time (but before the horizon), or (b) no such crawls, but the time horizon hasn't been reached yet.        
            while (((curr_crawl_time_idx >= len(crawl_hists[w]) or crawl_hists[w][curr_crawl_time_idx] > horizon) and ch_hists[w][curr_change_time_idx] <= horizon) or \
                (curr_crawl_time_idx < len(crawl_hists[w]) and crawl_hists[w][curr_crawl_time_idx] <= horizon and ch_hists[w][curr_change_time_idx] <= crawl_hists[w][curr_crawl_time_idx])):
                # If the time of a crawl coincides with the time of a change exactly (this can happen in the case of sources with complete
                # change observations), don't count this change -- we assume it is picked up immediately. Just advance to the next change
                # time. 

                if (curr_crawl_time_idx >= len(crawl_hists[w]) or \
                    not(ch_hists[w][curr_change_time_idx] == crawl_hists[w][curr_crawl_time_idx])):
                    curr_num_changes += 1

                
                if (curr_crawl_time_idx < len(crawl_hists[w]) and \
                    ch_hists[w][curr_change_time_idx] == crawl_hists[w][curr_crawl_time_idx]):
                    curr_change_time_idx += 1
                    break

                curr_change_time_idx += 1
                
                if (curr_change_time_idx >= len(ch_hists[w])):
                    break   

            # tally the changes we missed
            running_penalty += (0 if curr_num_changes == 0 else Harmonic(curr_num_changes))
            curr_num_changes = 0
            curr_crawl_time_idx += 1

        J_pi += (importances[w] * running_penalty / horizon)

    return J_pi / len(importances)
    

# Extends the change history for a set of sources from curr_time up to horizon.
#
# To extend the history for a given source, samples change times from the source's Poisson change process with a given rate parameter. 
def ExtendChangeHistory(ch_hists, ch_rates, curr_time, horizon):
    for w in range(len(ch_hists)):
        if (len(ch_hists[w]) == 0): 
            ch_hists[w].append(np.random.exponential(1.0 / ch_rates[w]))
        while (ch_hists[w][-1] < curr_time + horizon):
            ch_hists[w].append(ch_hists[w][-1] + np.random.exponential(1.0 / ch_rates[w]))


# Extends the observation history for sources with incomplete change observations.
# 
# This method assumes that both the change and crawl history for sources with incomplete observations has *already* been extended
# until (or just past) curr_time + horizon. This means that for each such source both its change history and its crawl history is
# assumed to contain at least one element.
#
# WARNING: DO NOT CALL THIS METHOD TWICE WITH THE SAME (curr_time, horizon) PAIR. calling this method twice with the same (curr_time, horizon) may
# reappend existing observations and thereby invalidate the observation history.
def ExtendObsHistory_IncomplObs(ch_hists, crawl_hists, observations_incompl_obs, curr_time, horizon):
    for w in range(len(observations_incompl_obs)):
        # First, find the last crawl in the time interval of interest, [curr_time, curr_time + horizon]. It's possible that the latest 
        # scheduled crawl overall is after curr_time+horizon and therefore hasn't happened yet -- ignore it 
        i = -1
        while (abs(i) <= len(crawl_hists[w]) and crawl_hists[w][i] > curr_time + horizon):
            i -= 1

        if (abs(i) > len(crawl_hists[w]) or crawl_hists[w][i] <= curr_time):
            # No crawls happened during time interval [curr_time, curr_time + horizon], so no new observations
            continue;

        i_last_crawl_in_interval = i

        # Now, find the first crawl in the interval [curr_time, curr_time + horizon]
        while (abs(i) <= len(crawl_hists[w]) and crawl_hists[w][i] > curr_time):
            i -= 1

        # Go back to i before the last subtraction. That's the index of the first crawl timestamp in the interval
        # [curr_time, curr_time + horizon]
        i_first_crawl_in_interval = i + 1

        # For each crawl between and including these two, we need to determine whether we would observe any changes since the previous
        # crawl. Note that i_crawl is always non-positive
        for i_crawl in range(i_first_crawl_in_interval, i_last_crawl_in_interval + 1):
            previous_crawl_time = crawl_hists[w][i_crawl - 1] if (abs(i_crawl - 1) <= len(crawl_hists[w])) else 0

            # Find the index of the timestamp of the latest change before the latest crawl
            j = -1
            while (abs(j) <= len(ch_hists[w]) and ch_hists[w][j] > crawl_hists[w][i_crawl]):
                j -= 1
            
            if (abs(j) > len(ch_hists[w])):
                observations_incompl_obs[w].append((crawl_hists[w][i_crawl], 0))
            else:
                # If this timestamp is after the previous crawl time, we know that there has been at least one change since the previous
                # crawl (although there may have been more -- we wouldn't be able to tell the difference based on this observation!) 
                # and record this in our observation history. Conversely, we also know that if this timestamp is before the previous crawl
                # time, there could not have been any change since the previous crawl, so we would observe no changes 
                if (ch_hists[w][j] > previous_crawl_time):
                    observations_incompl_obs[w].append((crawl_hists[w][i_crawl], 1))
                else:
                    observations_incompl_obs[w].append((crawl_hists[w][i_crawl], 0))    


# Extends the crawl history for sources with complete change observations.
#
# This method assumes that the change history has *already* been extended up to curr_time + horizon
# For every change in the interval [curr_time, curr_time + horizon] it then decides whether to crawl
# at that change's time or not.
#
# WARNING! DO NOT CALL THIS METHOD TWICE WITH THE SAME (curr_time, horizon) PAIR. Calling this method twice on the same change history for the same 
# curr_time, horizon will resample a crawl for every change and append the sampled timestamps onto the crawl history, thereby invalidating it. 
def ExtendCrawlHistory_ComplObs(ch_hists, crawl_hists, crawl_probs_compl_obs, curr_time, horizon):
    for w in range(len(crawl_hists)):
        # Find the index of the first change timestamp in [curr_time, curr_time + horizon]
        i = 0
        while ((abs(i - 1) <= len(ch_hists[w])) and ch_hists[w][i - 1] > curr_time):
            i -= 1
        
        while (i < 0 and ch_hists[w][i] <= curr_time + horizon):
            if (np.random.binomial(1, crawl_probs_compl_obs[w]) == 1):
                crawl_hists[w].append(ch_hists[w][i])
            i += 1


# Extends the crawl history for sources with incomplete change observations.
#
# This method assumes that the change history has *already* been extended up to curr_time + horizon
#
# WARNING! DO NOT CALL THIS METHOD TWICE WITH THE SAME (curr_time, horizon) PAIR. Calling this method twice on the same change history for the same 
# curr_time, horizon will resample crawls and append the sampled timestamps onto the crawl history, thereby invalidating it. 
def ExtendCrawlHistory_IncomplObs(crawl_hists, crawl_rates_incompl_obs, curr_time, horizon):
    for w in range(len(crawl_hists)):
        if (not crawl_hists[w]):
            crawl_hists[w].append(np.random.exponential(1.0 / crawl_rates_incompl_obs[w]))
        else:
            # Remove all crawl times, sampled using a previous crawl rate, that are scheduled after the first timestamp
            # sampled using the current crawl rate
            first_sample = curr_time + np.random.exponential(1.0 / crawl_rates_incompl_obs[w])
            while (crawl_hists[w] and crawl_hists[w][-1] > first_sample):
                del crawl_hists[w][-1]

            crawl_hists[w].append(first_sample)

        while (curr_time + horizon > crawl_hists[w][-1]):
            crawl_hists[w].append(crawl_hists[w][-1] + np.random.exponential(1.0 / crawl_rates_incompl_obs[w]))


# Pre-processes the dataset from the NeurIPS-2019 paper (https://github.com/microsoft/Optimal-Freshness-Crawl-Scheduling, see the README for the #
# dataset format) for consumption by the scheduling algorithms. Produces two files, one for URLS with complete and one for URLs with incomplete 
# change observations. Each file has the tab-separated format
#
# Imp_1     ChangeRate_1
# ...       ...
# Imp_N     ChangeRate_N
#
# where Imp_i and ChangeRate_i are the importance score and change rate of some URL. For complete-change-observation URLs both of these pieces
# of data are already in the dataset itself. For incomplete-change-observation URLs the change rates need to be learned from the dataset's crawl 
# and observation history.
def ProcessRawData(imps_data_file, changes_data_file, change_rates_compl_obs_data_file, \
                                out_incompl_obs_file="imps_and_chrates_incompl.txt", out_compl_obs_file="imps_and_chrates_compl.txt", delimiter='\t'):
    # Dictionary format: key is the URL ID, value is a triplet [importance, change rate, flag with value 1 if the URL has a complete observation
    # history available and 0 otherwise]
    url_data = dict()
    print("Reading the importance scores data...")
    with open(imps_data_file) as imps_data:
        line = imps_data.readline()
        while line:
            tokens = line.strip().split(delimiter, 1)
            url_data[int(tokens[0])] = [int(tokens[1]), 0, 0]   
            line = imps_data.readline()

    print("Learning the change rates for URLs with incomplete change observations...")
    num_high_chrates = 0;
    with open(changes_data_file) as changes_data:
        line = changes_data.readline()
        num_lines = 1
        while line:
            tokens = line.strip().split(delimiter, 2)
            training_intervals = np.array(json.loads(tokens[2]))
            # This is a reference to training_intervals, so operations on it will change training_intervals's contents!
            training_timestamps = training_intervals
            for i in range(1, training_timestamps.shape[0]):
                training_timestamps[i, 0] += training_timestamps[i - 1, 0]
                
            # Here we apply Equation 10 from the NeurIPS-2019 paper to learn URLs' Poisson change rates from histories.
            learned_rate = LearnChRates_IncomplObs(training_timestamps, math.inf, sum(training_timestamps[:,0]), epsilon_learn_global)
            url_data[int(tokens[0])][1] = learned_rate
                
            if (num_lines % 10000 == 0):
                print("Processed {} URLs".format(num_lines))
            
            num_lines += 1
            line = changes_data.readline()

    print("Processing the change rates for URLs with complete change observations...")
    
    with open(change_rates_compl_obs_data_file) as change_rates_compl_obs_data:
        line = change_rates_compl_obs_data.readline()
        while line:
            tokens = line.strip().split(delimiter, 1)
            id = int(tokens[0])
            url_data[id][1] = float(tokens[1])
            url_data[id][2] = 1
            line = change_rates_compl_obs_data.readline()
    
    print("Outputting data to files...")
    incompl_obs_out = open(out_incompl_obs_file,"w+")
    num_incompl_obs_records = 0
    compl_obs_out = open(out_compl_obs_file,"w+")
    num_compl_obs_records = 0
    
    for record in url_data.values():
        if (record[2] == 0):
            incompl_obs_out.write("%d\t%f\n" % (record[0], record[1]))
            num_incompl_obs_records += 1
        elif (record[2] == 1):
            compl_obs_out.write("%d\t%f\n" % (record[0], record[1]))
            num_compl_obs_records += 1
            
    print("A total of {} records written, {} for incomplete- and {} for complete-change-observation URLs.".format(num_incompl_obs_records + num_compl_obs_records, num_incompl_obs_records, num_compl_obs_records))


# This method computes the policy and its harmonic and binary costs for LambdaCrawl, LambdaCrawlApprox, BinaryLambdaCrawl(0.0),
# BinaryLambdaCrawl(0.4), UniformCrawl, and ChangeRateProportionalCrawl, assuming known change rates. The inputs are 2D arrays containing importance-
# change rate pairs for incomplete- and complete-change-observation sources.
def ExperimentHelper(importances_and_ch_rates_incompl_obs, importances_and_ch_rates_compl_obs):
    # We can choose to subsample and experiment with a subset of the whole dataset.
    size = importances_and_ch_rates_incompl_obs.shape[0] + importances_and_ch_rates_compl_obs.shape[0]
    # Determine the fraction of the overall dataset to sample in the current iteration
    sample_frac = size / (importances_and_ch_rates_incompl_obs.shape[0] + importances_and_ch_rates_compl_obs.shape[0])
    
    if (sample_frac >= 1.0):
        importances_and_ch_rates_incompl_obs_partial = importances_and_ch_rates_incompl_obs
        importances_and_ch_rates_compl_obs_partial = importances_and_ch_rates_compl_obs
    else:
        # Sample this fraction of a subset of the sources with incomplete change observations
        if (importances_and_ch_rates_incompl_obs.shape[0] > 0):
            importances_and_ch_rates_incompl_obs_partial = importances_and_ch_rates_incompl_obs[np.random.choice( \
                importances_and_ch_rates_incompl_obs.shape[0], int(sample_frac * importances_and_ch_rates_incompl_obs.shape[0]), replace=False),:]
        else:
            importances_and_ch_rates_incompl_obs_partial = np.empty([0,2])
        
        # Sample this fraction of a subset of the sources with complete change observations
        if (importances_and_ch_rates_compl_obs.shape[0] > 0):
            importances_and_ch_rates_compl_obs_partial = importances_and_ch_rates_compl_obs[np.random.choice( \
                importances_and_ch_rates_compl_obs.shape[0], int(sample_frac * importances_and_ch_rates_compl_obs.shape[0]), replace=False),:]
        else:
            importances_and_ch_rates_compl_obs_partial = np.empty([0,2])
    
    importances_incompl_obs = importances_and_ch_rates_incompl_obs_partial[:,0]
    ch_rates_incompl_obs = importances_and_ch_rates_incompl_obs_partial[:,1]
    importances_compl_obs = importances_and_ch_rates_compl_obs_partial[:,0]
    ch_rates_compl_obs = importances_and_ch_rates_compl_obs_partial[:,1]
    
    bandwidth = 0.2 * (importances_and_ch_rates_incompl_obs_partial.shape[0] + importances_and_ch_rates_compl_obs_partial.shape[0])
    epsilon_incompl_obs = 0.1
    importances = np.concatenate((importances_and_ch_rates_incompl_obs_partial[:,0], importances_and_ch_rates_compl_obs_partial[:,0]))
    ch_rates = np.concatenate((importances_and_ch_rates_incompl_obs_partial[:,1], importances_and_ch_rates_compl_obs_partial[:,1]))
    print("\n*****Running on a set of ", importances_and_ch_rates_incompl_obs_partial.shape[0], \
        " sources with incomplete observations and ", importances_and_ch_rates_compl_obs_partial.shape[0], \
        " sources with complete observations.", sep='')
        
    #==================================================================================
    # Random
    print("\n*****Running UniformCrawl.")
    crawl_rates =  np.ones_like(importances) * bandwidth / len(importances)
    policy_cost_harmonic = HarmonicPolicyCost(crawl_rates, [], importances, ch_rates, [], [])
    policy_cost_binary = BinaryPolicyCost(crawl_rates, [], importances, ch_rates, [], [])
    print("\tHarmonic policy cost was ", policy_cost_harmonic, ".\n\tBinary policy cost was ", policy_cost_binary, ".", sep='')
    #===================================================================================
    
    #==================================================================================
    # Change-rate-proportional
    print("\n")
    print("*****Running ChangeRateProportionalCrawl on a set of ", importances_and_ch_rates_incompl_obs_partial.shape[0], \
        " sources with incomplete observations and ", importances_and_ch_rates_compl_obs_partial.shape[0], \
        " sources with complete observations.", sep='')
    crawl_rates = ch_rates * bandwidth / sum(ch_rates)  
    policy_cost_harmonic = HarmonicPolicyCost(crawl_rates, [], importances, ch_rates, [], [])
    policy_cost_binary = BinaryPolicyCost(crawl_rates, [], importances, ch_rates, [], [])
    print("\tHarmonic policy cost was ", policy_cost_harmonic, ".\n\tBinary policy cost was ", policy_cost_binary, ".", sep='')
    #===================================================================================

    #===================================================================================
    # Epsilon-lambda-crawl-binary with the best parameter value for harmonic-cost performance on the NeurIPS-2019 paper's dataset.
    print("\n")
    print("*****Running BinaryLambdaCrawl-Epsilon (**all sources are treated as having incomplete change observations**).")
    epsilon = 0.4
    crawl_rates_incompl_obs, crawl_probs_compl_obs = LambdaCrawlBinary_Epsilon(importances, ch_rates, epsilon, [], [], bandwidth)
    policy_cost_harmonic = HarmonicPolicyCost(crawl_rates_incompl_obs, [], importances, ch_rates, [], [])
    policy_cost_binary = BinaryPolicyCost(crawl_rates_incompl_obs, [], importances, ch_rates, [], [])
    print("\tHarmonic policy cost with epsilon=", epsilon," was: ", policy_cost_harmonic, ".\n\tBinary policy cost with epsilon=", \
                                                                                                    epsilon," was: ", policy_cost_binary, ".", sep='')
    #===================================================================================
        
    #===================================================================================
    # Vanilla Epsilon-lambda-crawl-binary
    print("\n")
    print("*****Running BinaryLambdaCrawl-Epsilon (**all sources are treated as having incomplete change observations**).")
    epsilon = 0.0
    crawl_rates_incompl_obs, crawl_probs_compl_obs = LambdaCrawlBinary_Epsilon(importances, ch_rates, epsilon, [], [], bandwidth)
    policy_cost_harmonic = HarmonicPolicyCost(crawl_rates_incompl_obs, [], importances, ch_rates, [], [])
    policy_cost_binary = BinaryPolicyCost(crawl_rates_incompl_obs, [], importances, ch_rates, [], [])
    print("\tHarmonic policy cost with epsilon=", epsilon," was: ", policy_cost_harmonic, ".\n\tBinary policy cost with epsilon=", \
                                                                                                    epsilon," was: ", policy_cost_binary, ".", sep='')
    #===================================================================================

    #==================================================================================================
    # LambdaCrawlApprox
    print("\n")
    print("*****Running LambdaCrawlApprox.")
    crawl_rates_approx_incompl_obs, crawl_probs_approx_compl_obs = LambdaCrawlApprox(importances_incompl_obs, ch_rates_incompl_obs, \
                                                                            epsilon_incompl_obs, importances_compl_obs, ch_rates_compl_obs, bandwidth)
    policy_cost_harmonic = HarmonicPolicyCost(crawl_rates_approx_incompl_obs, crawl_probs_approx_compl_obs, importances_incompl_obs, \
                                                                                    ch_rates_incompl_obs, importances_compl_obs, ch_rates_compl_obs)
    policy_cost_binary = BinaryPolicyCost(crawl_rates_approx_incompl_obs, crawl_probs_approx_compl_obs, importances_incompl_obs, \
                                                                                    ch_rates_incompl_obs, importances_compl_obs, ch_rates_compl_obs)
    print("\tHarmonic policy cost was: ", policy_cost_harmonic, ".\n\tBinary policy cost was: ", policy_cost_binary, ".", sep='')
    #===============================================================================================================================
    
    #===============================================================================================================================
    # LambdaCrawlApprox ignoring complete change observations
    print("\n")
    print("*****Running LambdaCrawlApprox (**all sources are treated as having incomplete change observations**).")
    crawl_rates_approx_incompl_obs, crawl_probs_approx_compl_obs = LambdaCrawlApprox(importances, ch_rates, epsilon_incompl_obs, [], [], bandwidth)
    policy_cost_harmonic = HarmonicPolicyCost(crawl_rates_approx_incompl_obs, crawl_probs_approx_compl_obs, importances, ch_rates, [], [])
    policy_cost_binary = BinaryPolicyCost(crawl_rates_approx_incompl_obs, crawl_probs_approx_compl_obs, importances, ch_rates, [], [])
    print("\tHarmonic policy cost was: ", policy_cost_harmonic, ".\n\tBinary policy cost was: ", policy_cost_binary, ".", sep='')
    #===============================================================================================================================
    
    #===============================================================================================================================
    # LambdaCrawl
    print("\n")
    print("*****Running LambdaCrawl.")
    crawl_rates_incompl_obs, crawl_probs_compl_obs = LambdaCrawl(importances_incompl_obs, ch_rates_incompl_obs, epsilon_incompl_obs, \
                                                                                                importances_compl_obs, ch_rates_compl_obs, bandwidth)
    policy_cost_harmonic = HarmonicPolicyCost(crawl_rates_incompl_obs, crawl_probs_compl_obs, importances_incompl_obs, ch_rates_incompl_obs, \
                                                                                                            importances_compl_obs, ch_rates_compl_obs)
    policy_cost_binary = BinaryPolicyCost(crawl_rates_incompl_obs, crawl_probs_compl_obs, importances_incompl_obs, ch_rates_incompl_obs, \
                                                                                                            importances_compl_obs, ch_rates_compl_obs)
    print("\tHarmonic policy cost was: ", policy_cost_harmonic, ".\n\tBinary policy cost was: ", policy_cost_binary, ".", sep='')
    #===============================================================================================================================

    #===============================================================================================================================
    # LambdaCrawl ignoring complete change observations
    print("\n")
    print("*****Running LambdaCrawl (**all sources are treated as having incomplete change observations**).")
    crawl_rates_incompl_obs, crawl_probs_compl_obs = LambdaCrawl(importances, ch_rates, epsilon_incompl_obs, [], [], bandwidth) 
    policy_cost_harmonic = HarmonicPolicyCost(crawl_rates_incompl_obs, crawl_probs_compl_obs, importances, ch_rates, [], [])
    policy_cost_binary = BinaryPolicyCost(crawl_rates_incompl_obs, crawl_probs_compl_obs, importances, ch_rates, [], [])
    print("\tHarmonic policy cost was: ", policy_cost_harmonic, ".\n\tBinary policy cost was: ", policy_cost_binary, ".", sep='')


# Compute means and confidence intervals. Adapted from the Stack Overflow post at 
# https://stackoverflow.com/questions/15033511/compute-a-confidence-interval-from-sample-data
def mean_confidence_interval(data, confidence = 0.95):
    a = 1.0 * data
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h


def process_results(runs):
    ms = np.zeros_like(runs[0,:])
    hs = np.zeros_like(ms)
    
    for j in range(runs.shape[1]):
        m, h = mean_confidence_interval(runs[:,j])
        ms[j] = m
        hs[j] = h
        
    return ms, hs


# This method produces results as is Figure 1 of the NeurIPS-2019 paper. It computes the policy and its harmonic and binary costs for LambdaCrawl, 
# LambdaCrawlApprox, BinaryLambdaCrawl(0.0), BinaryLambdaCrawl(0.4), UniformCrawl, and ChangeRateProportionalCrawl, assuming known change rates,
# on the full dataset from the NeurIPS-2019 paper.
def Experiment1(importances_and_ch_rates_incompl_obs_file = "imps_and_chrates_incompl.txt", \
                                                                                importances_and_ch_rates_compl_obs_file="imps_and_chrates_compl.txt"):
    print("Reading the data...")
    importances_and_ch_rates_incompl_obs = np.genfromtxt(importances_and_ch_rates_incompl_obs_file, delimiter="\t")
    importances_and_ch_rates_compl_obs = np.genfromtxt(importances_and_ch_rates_compl_obs_file, delimiter="\t")
    ExperimentHelper(importances_and_ch_rates_incompl_obs, importances_and_ch_rates_compl_obs)
                    
                    
# This method produces results as is Figure 2 of the NeurIPS-2019 paper. It computes the policy and its harmonic and binary costs for LambdaCrawl, 
# LambdaCrawlApprox, BinaryLambdaCrawl(0.0), and BinaryLambdaCrawl(0.4) on the set of URLs with complete observations only. Out of the above 
# algorithms, only LambdaCrawl and LambdaCrawlApprox have special handling for these observations, the rest ignore them. For the sake of comparison, 
# we also run LambdaCrawl and LambdaCrawlApprox in "ignorance mode" as well as in normal mode.
def Experiment2(importances_and_ch_rates_compl_obs_file="imps_and_chrates_compl.txt"):
    print("Reading the data...")
    importances_and_ch_rates_incompl_obs = np.empty([0,2])
    importances_and_ch_rates_compl_obs = np.genfromtxt(importances_and_ch_rates_compl_obs_file, delimiter="\t")
    ExperimentHelper(importances_and_ch_rates_incompl_obs, importances_and_ch_rates_compl_obs)


# This method produces results as is Figure 3 of the NeurIPS-2019 paper. It simulates runs of the RL algorithms LambdaLearnAndCrawl, 
# LambdaLearnAndCrawlApprox, and BinaryLambdaLearnAndCrawl using the ground-truth Poisson change rates previously learned from the dataset,
# averages their results, and outputs the means and confidence intervals in a tikzpicture plot format.
def Experiment3(importances_and_ch_rates_incompl_obs_file = "imps_and_chrates_incompl.txt", \
                                                                                importances_and_ch_rates_compl_obs_file="imps_and_chrates_compl.txt"):
    print("Reading the data...")
    importances_and_ch_rates_incompl_obs = np.genfromtxt(importances_and_ch_rates_incompl_obs_file, delimiter="\t")
    importances_and_ch_rates_compl_obs = np.genfromtxt(importances_and_ch_rates_compl_obs_file, delimiter="\t")
    size = 100000
    # Determine the fraction of the overall dataset to sample
    sample_frac = size / (importances_and_ch_rates_incompl_obs.shape[0] + importances_and_ch_rates_compl_obs.shape[0])
    epoch_length = 1
    horizon = 21
    num_runs = 20
    epsilon_incompl_obs = 0.1
    LLC_runs = np.empty([num_runs,horizon])
    LLCA_runs = np.empty([num_runs,horizon])
    LLCB_runs = np.empty([num_runs,horizon])
    
    for run in range(num_runs):
        print("%%%%%%  EPISODE ", run + 1, " %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%") 
        
        # Sample this fraction of the subset of the sources with incomplete change observations
        importances_and_ch_rates_incompl_obs_partial = importances_and_ch_rates_incompl_obs[np.random.choice( \
            importances_and_ch_rates_incompl_obs.shape[0], int(sample_frac * importances_and_ch_rates_incompl_obs.shape[0]), replace=False),:]
        importances_incompl_obs = importances_and_ch_rates_incompl_obs_partial[:,0]
        ch_rates_incompl_obs_actual = importances_and_ch_rates_incompl_obs_partial[:,1]

        # Sample this fraction of the subset of the sources with complete change observations
        importances_and_ch_rates_compl_obs_partial = importances_and_ch_rates_compl_obs[np.random.choice( \
            importances_and_ch_rates_compl_obs.shape[0], int(sample_frac * importances_and_ch_rates_compl_obs.shape[0]), replace=False),:]
        importances_compl_obs = importances_and_ch_rates_compl_obs_partial[:,0]
        ch_rates_compl_obs_actual = importances_and_ch_rates_compl_obs_partial[:,1]

        # Set the bandwidth constraint to 20\% of the total number of sources
        bandwidth = 0.2 * (importances_and_ch_rates_incompl_obs_partial.shape[0] + importances_and_ch_rates_compl_obs_partial.shape[0])
        
        # use the entire available observation history for learning
        suffix_len = math.inf
        
        policy_cost, total_time, idealized_policy_costs_harmonic, idealized_policy_costs_binary = LambdaLearnAndCrawl(importances_incompl_obs, ch_rates_incompl_obs_actual , epsilon_incompl_obs, \
            importances_compl_obs, ch_rates_compl_obs_actual, bandwidth, epsilon_learn_global, epoch_length, suffix_len, horizon)
        LLC_runs[run,:] = idealized_policy_costs_harmonic[:]
        print("LambdaLearnAndCrawl's harmonic costs: ", idealized_policy_costs_harmonic)

        print("\n")
        policy_cost_llaca, total_time_llaca, idealized_policy_costs_llaca_harmonic, idealized_policy_costs_llaca_binary = LambdaLearnAndCrawlApprox(\
                                                                        importances_incompl_obs, ch_rates_incompl_obs_actual , epsilon_incompl_obs, \
                                importances_compl_obs, ch_rates_compl_obs_actual, bandwidth, epsilon_learn_global, epoch_length, suffix_len, horizon)
        LLCA_runs[run,:] = idealized_policy_costs_harmonic[:]
        print("LambdaLearnAndCrawlApprox's harmonic costs: ", idealized_policy_costs_llaca_harmonic)
        
        print("\n")
        policy_cost_llcb, total_time_llcb, idealized_policy_costs_llcb_harmonic, idealized_policy_costs_llcb_binary = LambdaLearnAndCrawlBinary(\
                                                                                        importances_incompl_obs, ch_rates_incompl_obs_actual , 0.4, \
                                importances_compl_obs, ch_rates_compl_obs_actual, bandwidth, epsilon_learn_global, epoch_length, suffix_len, horizon)
        
        LLCB_runs[run,:] = idealized_policy_costs_harmonic[:]
        print("BinaryLambdaLearnAndCrawl's harmonic costs: ", idealized_policy_costs_harmonic)
        
        print("\n")
    
    LLC_ms, LLC_hs = process_results(LLC_runs)
    LLC_plot_string = ""
    for i in range(LLC_ms.shape[0]):
        LLC_plot_string += "(" + str(i) + "," + str(LLC_ms[i]) + ")+=(0," + str(LLC_hs[i]) + ")-=(0," + str(LLC_hs[i]) +")\n"
    
    print("LLC:\n", LLC_plot_string, "\n\n", LLC_limit_plot_string)
    
    LLCA_ms, LLCA_hs = process_results(LLCA_runs)
    LLCA_plot_string = ""
    for i in range(LLCA_ms.shape[0]):
        LLCA_plot_string += "(" + str(i) + "," + str(LLCA_ms[i]) + ")+=(0," + str(LLCA_hs[i]) + ")-=(0," + str(LLCA_hs[i]) +")\n"
    
    print("LLCA:\n", LLCA_plot_string, "\n\n", LLCA_limit_plot_string)
    
    LLCB_ms, LLCB_hs = process_results(LLCB_runs)
    LLCB_plot_string = ""
    for i in range(LLCB_ms.shape[0]):
        LLCB_plot_string += "(" + str(i) + "," + str(LLCB_ms[i]) + ")+=(0," + str(LLCB_hs[i]) + ")-=(0," + str(LLCB_hs[i]) +")\n"
    
    print("BLLC:\n", LLCB_plot_string, "\n\n", LLCB_limit_plot_string)