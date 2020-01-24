import downsample_experiments
import loading_data
import numpy as np
import pickle

dataset_to_metric = {"sst": "acc", "mrpc": "acc_and_f1", "cola": "mcc", "rte": "acc"}
size_of_subsample = 20
num_samples = 2000
comparison_statistic = "max"
dataset = "sst"
random_seed = 1
np.random.seed(random_seed)
max_budget = 30
max_num_runs_started = 200
print_debug = False


def main():
    data = loading_data.load_all_data()
    
    print(data.keys())

    dataset_to_results = {}
    #for dataset in data:
    

    print(dataset)
    cur_data = extract_data(data, dataset)
        
    early_stop_by_dim(cur_data, dataset)

        
    #dataset_to_results[dataset] = cur_results

        
    #    print("")
    #dataset_to_results['settings'] = {'num_samples': num_samples,
    #                                         'max_budget': max_budget,
    #                                         'max_num_runs_started': max_num_runs_started,
    #                                         'seed': random_seed}
    #pickle_results(dataset_to_results)
    #import pdb; pdb.set_trace()

def pickle_results(unique_to_numeach_to_baseline,unique_to_numeach_to_results, budget_to_perf, dataset):
    f = open('results/early_stopping_by_dim/{}_{}_results_and_budget_to_perf'.format(dataset, comparison_statistic),
                                                                                                       'wb')
    pickle.dump({"unique_to_numeach_to_baseline":unique_to_numeach_to_baseline,
                 "unique_to_numeach_to_results":unique_to_numeach_to_results,
                 "budget_to_perf": budget_to_perf}, f)
    f.close()


def early_stop_by_dim(data, dataset):
    budget_to_perf = {}
    unique_to_numeach_to_results = {}
    unique_to_numeach_to_baseline = {}
    # num_of_each_seed and num_unique_seeds control the number we start
    #for num_unique_seeds in range(2, data.shape[0]):
    for num_unique_seeds in range(2, 10):
        unique_to_numeach_to_results[num_unique_seeds] = {}
        unique_to_numeach_to_baseline[num_unique_seeds] = {}
        for num_of_each_seed in range(2, num_unique_seeds+1):

            cur_result, cur_baseline = one_experiment(data, num_of_each_seed, num_unique_seeds, budget_to_perf)
            unique_to_numeach_to_results[num_unique_seeds][num_of_each_seed] = cur_result
            unique_to_numeach_to_baseline[num_unique_seeds][num_of_each_seed] = cur_baseline

        pickle_results(unique_to_numeach_to_baseline, unique_to_numeach_to_results, budget_to_perf, dataset)
    import pdb; pdb.set_trace()


def print_debug_info(unique_to_numeach_to_baseline,unique_to_numeach_to_results):
    for unique_seeds in unique_to_numeach_to_results:
        num_wi_stop, num_do_stop, percent_data = np.nonzero((unique_to_numeach_to_results[unique_seeds][2] -
                    unique_to_numeach_to_baseline[unique_seeds][2]) > 0)
        print("")
        print("unique_seeds: {}".format(unique_seeds))
        print("num_wi_stop, num_do_stop, num_dev_evals")
        for i in range(num_wi_stop.shape[0]):
            print(num_wi_stop[i], num_do_stop[i], percent_data[i])
        
    

def print_budgets_for_which_we_win(budget_to_perf):
    counter_we_win = 0
    counter_total = 0
    max_budget = max(budget_to_perf.keys())
    for b in range(max_budget + 1):
        if b in budget_to_perf:
            counter_total += 1
            baseline_average = budget_to_perf[b][:,6].mean()
            result_average = budget_to_perf[b][:,5].mean()
            if result_average > baseline_average:
                counter_we_win += 1
                print(b, baseline_average, result_average)

    print("fraction we win: {}, number we win: {}, total: {}".format(1.0*counter_we_win / counter_total,
                                                                     counter_we_win, counter_total))

def one_experiment(data, num_of_each_seed, num_unique_seeds, budget_to_perf):
    results_wi_do_data = np.zeros((num_samples, num_unique_seeds, num_unique_seeds, data.shape[2]))
    baseline_wi_do_data = np.zeros((num_samples, num_unique_seeds, num_unique_seeds, data.shape[2]))

    for sample_num in range(num_samples):
        if sample_num % 10 == 0:
            print(sample_num)
        points, wi_seeds, do_seeds = subsample(data, num_of_each_seed, num_unique_seeds)

        # num_do_stop and num_wi_stop control number stopped
        for num_wi_stop in range(num_unique_seeds):
            for num_do_stop in range(num_unique_seeds):
                
                # num row stop * mtx_size[0] + num cols to stop * mtx_size[1] - num nows to stop * num cols to stop
                # this calculation is only correct if num_unique_seeds == num_of_each_seed
                total_num_to_stop = (num_wi_stop * num_unique_seeds +
                                     num_do_stop * num_unique_seeds - num_wi_stop * num_do_stop)
                
                if total_num_to_stop == 0 or num_wi_stop + num_do_stop > num_unique_seeds:
                    continue

                for percent_data in range(1, data.shape[2]):

                    by_dim_max, total_stopped_early = early_stop_at_percent(data, points, wi_seeds, do_seeds,
                                                       num_wi_stop, num_do_stop, percent_data)
                    
                    results_wi_do_data[sample_num, num_wi_stop, num_do_stop, percent_data] = by_dim_max

                    independent_max = early_stop_individual_runs(data, points, total_stopped_early, percent_data)
                    baseline_wi_do_data[sample_num, num_wi_stop, num_do_stop, percent_data] = independent_max


                    total_trained_fully = num_unique_seeds * num_of_each_seed - total_stopped_early
                    cur_budget = total_trained_fully * data.shape[2] + total_stopped_early * percent_data
                    if cur_budget not in budget_to_perf:
                        budget_to_perf[cur_budget] = np.asarray([[num_of_each_seed, num_unique_seeds,
                                                                  num_wi_stop, num_do_stop, percent_data,
                                                                 by_dim_max, independent_max]])
                    else:

                        new_result = np.asarray([[num_of_each_seed, num_unique_seeds,
                                                  num_wi_stop, num_do_stop,
                                                  percent_data, by_dim_max, independent_max]])
                        budget_to_perf[cur_budget] = np.append(budget_to_perf[cur_budget], new_result, axis=0)


    return results_wi_do_data.mean(axis=0), baseline_wi_do_data.mean(axis=0)
                    

def subsample(data, num_of_each_seed, num_unique_seeds):

    # sample WI seeds and DO seeds
    wi_seeds = np.random.choice(data.shape[0], size=num_unique_seeds, replace=False)
    do_seeds = np.random.choice(data.shape[1], size=num_unique_seeds, replace=False)

    points = tuple()
    for i in range(num_of_each_seed):
        points += (tuple(zip(wi_seeds, np.roll(do_seeds, i))))

    points = np.asarray(points)
    return points, wi_seeds, do_seeds
    

def early_stop_at_percent(data, points, wi_seeds, do_seeds, num_wi_stop, num_do_stop, percent_data):
    
    wi_performances, do_performances = rank_seeds(data, points, percent_data, wi_seeds, do_seeds)

    wi_to_stop = wi_performances[:num_wi_stop,1]
    do_to_stop = do_performances[:num_do_stop,1]

    # get the wi points to stop early
    wi_points_stop_early = np.isin(points[:,0], wi_to_stop)
    do_points_stop_early = np.isin(points[:,1], do_to_stop)
    points_stop_early = np.logical_or(wi_points_stop_early, do_points_stop_early)
    points_train_fully = np.logical_not(points_stop_early)

    points_stop_early = points[points_stop_early]
    points_train_fully = points[points_train_fully]    

    debug = False
    if debug:

        tmp = data[points_train_fully[:,0], points_train_fully[:,1], :]
        #print(num_wi_stop, num_do_stop, percent_data, tmp.shape)
        if tmp.shape[0] == 0:
            import pdb; pdb.set_trace()
        
            
    early_stopped_max = data[points_stop_early[:,0], points_stop_early[:,1], :percent_data].max()
    train_fully_max = data[points_train_fully[:,0], points_train_fully[:,1], :].max()
    global_max = max(early_stopped_max, train_fully_max)

    return global_max, points_stop_early.shape[0]
    

def rank_seeds(data, points, percent_data, wi_seeds, do_seeds):

    wi_performances = []
    for wi_seed in wi_seeds:
        cur_wi_points = points[points[:,0] == wi_seed]
        wi_performances.append([compute_comparison_stat(comparison_statistic, data, cur_wi_points, percent_data),
                                wi_seed])

    do_performances = []
    for do_seed in do_seeds:
        cur_do_points = points[points[:,1] == do_seed]
        do_performances.append([compute_comparison_stat(comparison_statistic, data, cur_do_points, percent_data),
                                do_seed])

    wi_performances.sort()
    do_performances.sort()

    return np.asarray(wi_performances), np.asarray(do_performances)

def compute_comparison_stat(comparison_statistic, data, points, percent_data):
    cur_data = data[points[:,0], points[:,1], :percent_data]
    cur_data_maxes = cur_data.max(axis=1)

    if comparison_statistic == "average":
        return cur_data_maxes.mean()
    elif comparison_statistic == "max":
        return cur_data_maxes.max()
    else:
        raise NotImplementedError


def early_stop_individual_runs(data, points, num_stop, num_evals):
    cur_subsample = data[points[:,0], points[:,1], :]
    ranks = find_rank_after_num_evals(cur_subsample, num_evals)
    
    train_fully = cur_subsample[ranks[num_stop:]]
    train_fully_maxes = train_fully.max(axis=1)
    
    stop_train = cur_subsample[ranks[:num_stop]]
    stop_train_maxes = stop_train[:,:num_evals].max(axis=1)
    
    if train_fully_maxes.shape[0] == 0:
        train_fully_maxes = stop_train_maxes
    if stop_train_maxes.shape[0] == 0:
        stop_train_maxes = train_fully_maxes
            
    train_fully_max = max(train_fully_maxes)
    stop_train_max = max(stop_train_maxes)
    return max(train_fully_max, stop_train_max)
    
def find_rank_after_num_evals(cur_subsample, num_evals):
    partly_trained = cur_subsample[:,:num_evals]
    to_rank = partly_trained.max(axis=1)
    ranks = np.argsort(to_rank)
    
    return ranks



    

def one_experiment_DEPRICATED(cur_data, num_exp, num_stop, num_evals):
    subsample_maxes = []

    for sample_num in range(num_samples):
        
        cur_subsample = subsample(cur_data, num_exp)
        
        ranks = find_rank_after_num_evals(cur_subsample, num_evals)
        
        train_fully = cur_subsample[ranks[num_stop:]]
        train_fully_maxes = train_fully.max(axis=1)
        
        stop_train = cur_subsample[ranks[:num_stop]]
        stop_train_maxes = stop_train[:,:num_evals].max(axis=1)

        if train_fully_maxes.shape[0] == 0:
            train_fully_maxes = stop_train_maxes
        if stop_train_maxes.shape[0] == 0:
            stop_train_maxes = train_fully_maxes
        
        train_fully_max = max(train_fully_maxes)
        stop_train_max = max(stop_train_maxes)
        subsample_maxes.append(max(train_fully_max, stop_train_max))

    print(num_exp, num_stop, num_evals, np.mean(subsample_maxes))
    return np.mean(subsample_maxes)
            
            

def extract_data(data, dataset):
    cur_data = data[dataset]
    evals = []
    for init in range(len(cur_data)):
        evals.append([])
        for data_order in range(len(cur_data[init+1])):
            ds_exp = downsample_experiments.main(cur_data[init+1][data_order+1][dataset_to_metric[dataset]]['during'],
                                                 dataset, max_num_evals=30)
            evals[init].append(ds_exp)


    just_results = np.array(evals)[:,:,:,1]
    return just_results



if __name__ == "__main__":
    main()
