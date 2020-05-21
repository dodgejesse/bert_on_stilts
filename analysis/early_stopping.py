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
    import pdb; pdb.set_trace()
    print(data.keys())

    dataset_to_results = {}
    for dataset in data:
        print(dataset)

        cur_data = extract_data(data, dataset)

        budget_to_assignments = find_valid_assignments(cur_data)
        budget_to_results = early_stop_with_fixed_budget(cur_data, budget_to_assignments)

        dataset_to_results[dataset] = budget_to_results

        
        print("")
    dataset_to_results['settings'] = {'num_samples': num_samples,
                                             'max_budget': max_budget,
                                             'max_num_runs_started': max_num_runs_started,
                                             'seed': random_seed}
    pickle_results(dataset_to_results)
    import pdb; pdb.set_trace()

def pickle_results(dataset_to_results):
    f = open('results/debug_early_stopping', 'wb')
    pickle.dump(dataset_to_results, f)
    f.close()

def find_valid_assignments(cur_data):

    budget_to_assignments = {}
    
    num_evals_per_exp = cur_data.shape[1]
    total_num_experiments = cur_data.shape[0]
    for budget_num_runs in range(1, total_num_experiments + 1):
        if budget_num_runs > max_budget:
            continue

        budget_to_assignments[budget_num_runs] = []
        
        # the budget in terms of the number of evals
        budget_total_evals = budget_num_runs * num_evals_per_exp


        # number of runs to start
        for num_exp in range(1, total_num_experiments + 1):
            if num_exp > max_num_runs_started:
                continue

            assignment_within_budget(num_exp, 0,
                                     num_evals_per_exp, num_evals_per_exp, budget_total_evals,
                                     budget_to_assignments[budget_num_runs])
            
            # number of runs to kill
            for num_stop in range(1, num_exp + 1):

                # amount of data to train on
                for num_evals in range(1, num_evals_per_exp):

                    assignment_within_budget(num_exp, num_stop,
                                             num_evals, num_evals_per_exp, budget_total_evals,
                                             budget_to_assignments[budget_num_runs])

    total_experiments = 0
    for budget in budget_to_assignments:
        total_experiments += len(budget_to_assignments[budget])
    return budget_to_assignments

                    
# to check if the assignment to num_exp, num_stop, and num_evals leads to a budget less than budget_total_evals
def assignment_within_budget(num_exp, num_stop, num_evals,
                             num_evals_per_exp, budget_total_evals, assignments):
    num_train_fully = num_exp - num_stop
    budget_for_fully_trained = num_train_fully * num_evals_per_exp
    budget_for_stopped = num_stop * num_evals
    budget_for_cur_assignments = budget_for_stopped + budget_for_fully_trained
    if budget_for_cur_assignments == budget_total_evals:
        frac_of_data = round(num_evals * 1.0 / num_evals_per_exp, 4)
        if print_debug:
            print("num_exp = {}, num_stop = {}, num_evals = {}, budget_total_evals = {}, within_budget = {}".format(
                num_exp, num_stop, frac_of_data, budget_for_cur_assignments,
                budget_for_cur_assignments <= budget_total_evals))
            import pdb; pdb.set_trace()
        assignments.append({"num_exp": num_exp, "num_stop": num_stop, "num_evals": num_evals,
                            "budget": budget_for_cur_assignments, "num_evals_per_exp":num_evals_per_exp})

                
def early_stop_with_fixed_budget(cur_data, budget_to_assignments):
    budget_to_results = {}
    counter = 0
    for budget in budget_to_assignments:
        print("")
        print(budget)
        print("")
        if budget >= 3 and print_debug:
            import pdb; pdb.set_trace()

        avg_performance = []
        for experiment in budget_to_assignments[budget]:
            exp_perf = one_experiment(cur_data, experiment["num_exp"], experiment["num_stop"],
                                      experiment["num_evals"])
            # a counter which breaks ties
            counter += 1
            avg_performance.append((exp_perf, counter, experiment))
        print("")
        try:
            avg_performance.sort()
        except:
            import pdb; pdb.set_trace()
        for item in avg_performance:
            print(item[0], {'num_exp': item[2]['num_exp'],
                            '%_stop': round(item[2]['num_stop'] * 1.0 / item[2]['num_exp'], 4),
                            '%_data': round(item[2]['num_evals'] * 1.0 / item[2]['num_evals_per_exp'], 4)})
        budget_to_results[budget] = avg_performance
    return budget_to_results


def one_experiment(cur_data, num_exp, num_stop, num_evals):
    # to remove the batch numbers, and just have the evaluation results:
    cur_data = cur_data[:,:,1]
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
            

# this method is depricated
def early_stop_with_varying_budgets(cur_data):
    frac_runs_to_data_to_avg = {}
    for frac_of_runs in range(10):
        frac_of_runs = (frac_of_runs + 1) / 10
        frac_runs_to_data_to_avg[frac_of_runs] = {}
        
        for frac_of_data in range(10):

            frac_of_data = (frac_of_data + 1) / 10
            max_batch_num = cur_data[0][-1][0]
            batch_num_to_beat = max_batch_num * frac_of_data
            run_num_cutoff = round(frac_of_runs * size_of_subsample)

            subsample_maxes = []
            for sample_num in range(num_samples):

                cur_subsample = subsample(cur_data, size_of_subsample)
                
                ranks = find_rank_after_frac_of_data(cur_subsample, batch_num_to_beat)
                
                train_fully = cur_subsample[ranks[:run_num_cutoff]]
                train_fully_maxes = find_maxes(train_fully, max_batch_num)
                
                stop_train = cur_subsample[ranks[run_num_cutoff:]]
                stop_train_maxes = find_maxes(stop_train, batch_num_to_beat)
                
                if True:
                    debug_train_fully_maxes = find_maxes(train_fully, batch_num_to_beat)


                train_fully_max = max(train_fully_maxes)
                stop_train_max = max(stop_train_maxes)
                subsample_maxes.append(max(train_fully_max, stop_train_max))
            frac_runs_to_data_to_avg[frac_of_runs][frac_of_data] = np.mean(subsample_maxes)
            print(frac_of_runs, frac_of_data, np.mean(subsample_maxes))

                

def find_maxes_DEPRECATED(cur_subsample, batch_num_to_beat):
    maxes = []
    for eval_index in range(len(cur_subsample[0])):
        if cur_subsample[0][eval_index][0] >= batch_num_to_beat:

            for one_run in cur_subsample:
                
                one_run_evals = [one_eval[1] for one_eval in one_run]
                one_run_evals = one_run_evals[:eval_index+1]
                maxes.append(max(one_run_evals))

            return maxes

# deprecated
def find_rank_after_frac_of_data(cur_subsample, batch_num_to_beat):
    for eval_index in range(len(cur_subsample[0])):
        if cur_subsample[0][eval_index][0] >= batch_num_to_beat:

            try:
                cur_evals = [one_eval[eval_index][1] for one_eval in cur_subsample]
            except:
                import pdb; pdb.set_trace()

            # to make it sort the other direction
            cur_evals = np.array(cur_evals) * -1
            
            ranks = np.argsort(cur_evals)
            
            return ranks
            
    

def extract_data(data, dataset):
    cur_data = data[dataset]
    evals = []
    for init in cur_data:
        for data_order in cur_data[init]:
            ds_exp = downsample_experiments.main(cur_data[init][data_order][dataset_to_metric[dataset]]['during'],
                                                 dataset, max_num_evals=30)
            evals.append(ds_exp)

    return np.array(evals)

def subsample(cur_data, size_of_subsample):
    idx = np.random.randint(len(cur_data), size=size_of_subsample)
    cur_subsample = cur_data[idx]
    return cur_subsample

    

if __name__ == "__main__":
    main()
