import downsample_experiments
import loading_data
import numpy as np
import pickle

dataset_to_metric = {"sst": "acc", "mrpc": "acc_and_f1", "cola": "mcc", "rte": "acc"}
size_of_subsample = 20
num_samples = 50000
random_seed = 1
np.random.seed(random_seed)
max_budget = 30
max_num_runs_started = 200
print_debug = False


def main():
    data = loading_data.load_all_data()
    
    print(data.keys())

    dataset_to_results = {}
    for dataset in data:
        print(dataset)
        cur_data = extract_data(data, dataset)
        
        
        cur_results = early_stop_by_dim(cur_data)
        
        dataset_to_results[dataset] = cur_results

        
        print("")
    dataset_to_results['settings'] = {'num_samples': num_samples,
                                             'max_budget': max_budget,
                                             'max_num_runs_started': max_num_runs_started,
                                             'seed': random_seed}
    pickle_results(dataset_to_results)
    import pdb; pdb.set_trace()

def pickle_results(dataset_to_results):
    f = open('results/early_stopping_by_dim', 'wb')
    pickle.dump(dataset_to_results, f)
    f.close()



def early_stop_by_dim(data):
    # num_of_each_seed and num_unique_seeds control the number we start
    for num_of_each_seed in range(2, data.shape[0]):
        for num_unique_seeds in range(2, data.shape[0]):
            one_experiment(data, num_of_each_seed, num_unique_seeds)



def one_experiment(data, num_of_each_seed, num_unique_seeds):
    for sample_num in range(num_samples):
        points, wi_seeds, do_seeds = subsample(data, num_of_each_seed, num_unique_seeds)

        # num_do_stop and num_wi_stop control number stopped
        for num_wi_stop in range(num_unique_seeds):
            for num_do_stop in range(num_unique_seeds):
                # num row stop * mtx_size[0] + num cols to stop * mtx_size[1] - num nows to stop * num cols to stop
                total_num_to_stop = (num_wi_stop * num_unique_seeds +
                                     num_do_stop * num_unique_seeds - num_wi_stop * num_do_stop)
                if total_num_to_stop == 0:
                    continue
                
                for percent_data in range(1, data.shape[2]):
                    early_stop_at_percent(data, points, wi_seeds, do_seeds, num_wi_stop, num_do_stop, percent_data)


def subsample(data, num_of_each_seed, num_unique_seeds):

    # sample WI seeds and DO seeds
    wi_seeds = np.random.randint(data.shape[0], size=num_unique_seeds)
    do_seeds = np.random.randint(data.shape[1], size=num_unique_seeds)

    points = tuple()
    for i in range(num_of_each_seed):
        points += (tuple(zip(wi_seeds, np.roll(do_seeds, i))))

    points = np.asarray(points)
    return points, wi_seeds, do_seeds
    

def early_stop_at_percent(data, points, wi_seeds, do_seeds, num_wi_stop, num_do_stop, percent_data):
    rank_seeds(data, points, percent_data, wi_seeds, do_seeds)
    

def rank_seeds(data, points, percent_data, wi_seeds, do_seeds):
    comparison_statistic = "average"

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

    import pdb; pdb.set_trace()


def compute_comparison_stat(comparison_statistic, data, points, percent_data):
    cur_data = data[points[:,0], points[:,1], :percent_data]
    cur_data_maxes = cur_data.max(axis=1)

    if comparison_statistic == "average":
        return cur_data_maxes.mean()
    elif comparison_statistic == "max":
        return cur_data_maxes.max()
    else:
        raise NotImplementedError

    

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


def find_rank_after_num_evals(cur_subsample, num_evals):
    partly_trained = cur_subsample[:,:num_evals]
    to_rank = partly_trained.max(axis=1)
    ranks = np.argsort(to_rank)
    
    return ranks
            
            

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
