import pickle
import numpy as np
import collections

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d


dataset_to_metric = {"sst": "acc", "mrpc": "acc_and_f1", "cola": "mcc", "rte": "acc"}
print_debug = False
# budget_to_perf key:
# num_of_each_seed, num_unique_seeds, num_wi_stop, num_do_stop, percent_data, by_dim_max, independent_max

def main():
    for dataset in dataset_to_metric:
        for comparison_statistic in ["average", "max"]:
            f = open('results/early_stopping_by_dim/{}_{}_results_and_budget_to_perf'.format(
                dataset, comparison_statistic), 'rb')    

            results = pickle.load(f)
            f.close()
            process_results(results, dataset)
            

def process_results(results, dataset):
    unique_to_numeach_to_baseline = results["unique_to_numeach_to_baseline"]
    unique_to_numeach_to_results = results["unique_to_numeach_to_results"]
    budget_to_perf = results["budget_to_perf"]
    #max_by_bucket(budget_to_perf, dataset)
    #unique_assignments_per_budget(budget_to_perf)
    print_budgets_for_which_we_win(budget_to_perf)
    print_debug_info(unique_to_numeach_to_baseline, unique_to_numeach_to_results)


def unique_assignments_per_budget(budget_to_perf):
    
    for budget in range(max(budget_to_perf)):
        if budget not in budget_to_perf:
            continue

        cur_best = 0
        cur_baseline_loses = True
        best_assignment = None
        
        unique_assignments = np.unique(budget_to_perf[budget][:,:-2],axis=0)
        print(budget)

        for assignment in unique_assignments:
            cur_assignment_indices = np.all(budget_to_perf[budget][:,:-2] == assignment, axis=1)
            
            num_cur_assignment = cur_assignment_indices.sum()
            mean_perf = budget_to_perf[budget][cur_assignment_indices][:,5].mean()
            mean_base = budget_to_perf[budget][cur_assignment_indices][:,6].mean()

            if num_cur_assignment < 100:
                continue

            if print_debug:
                print(assignment, num_cur_assignment, mean_base, mean_perf, mean_base < mean_perf)

            if cur_best < max(mean_base, mean_perf):
                cur_best = max(mean_base, mean_perf)
                cur_baseline_loses = mean_base < mean_perf
                best_assignment = assignment
        print(best_assignment, cur_best, cur_baseline_loses, " <- best")
        import pdb; pdb.set_trace()
        print("")


    
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

#def max_by_bucket(budget_to_perfs, dataset):
#    best_hparams, best_perfs, baselines = max_by_bucket_helper(budget_to_perfs, dataset, bline = False)
#    best_hparams, bline, baselines = max_by_bucket_helper(budget_to_perfs, dataset, bline = True)
#    save_figure(dataset, best_perfs, baselines, bline)
    

def max_by_bucket(budget_to_perfs, dataset):
    buckets = {}
    indep_baseline = {}
    baselines = {}

    for budget, vals  in budget_to_perfs.items():
        bucket = (budget)//10
        if bucket not in buckets: buckets[bucket] = collections.defaultdict(list)
        if bucket not in baselines: baselines[bucket] = []
        if bucket not in indep_baseline: indep_baseline[bucket] = collections.defaultdict(list)
        for a, b, c, d, e, by_dim_max, independent_max in vals:
            hparams = a, b, c, d, e

            buckets[bucket][hparams].append(by_dim_max)
            indep_baseline[bucket][hparams].append(independent_max)
            baselines[bucket].append(independent_max)

    best_hparams = {}
    best_perfs = {}
    best_perfs_indep = {}
    for bucket in buckets:
        for hparam in list(buckets[bucket].keys()):
            if len(buckets[bucket][hparam]) < 100:
                del buckets[bucket][hparam]
            else:
                buckets[bucket][hparam] = np.mean(buckets[bucket][hparam])
                indep_baseline[bucket][hparam] = np.mean(indep_baseline[bucket][hparam])
        if buckets[bucket]:
            best_hparams[bucket] = max(buckets[bucket].keys(), key= lambda x: buckets[bucket][x])
            best_perfs[bucket] = max(buckets[bucket].values())
            best_perfs_indep[bucket] = indep_baseline[bucket][best_hparams[bucket]]
            baselines[bucket] = np.mean(baselines[bucket])

    save_figure(dataset, best_perfs, baselines, best_perfs_indep, best_hparams)

def save_figure(dataset, best_perfs, baselines, best_perfs_indep, best_hparams):
    plt.figure()
    plt.title(dataset)
    xs = sorted(list(best_hparams.keys()))
    sigma = 2
    plt.xscale('log')
    plt.plot(xs, gaussian_filter1d([best_perfs[x] for x in xs], sigma=sigma), label='perf')
    plt.plot(xs, gaussian_filter1d([baselines[x] for x in xs],sigma=sigma), label='baseline')
    plt.plot(xs, gaussian_filter1d([best_perfs_indep[x] for x in xs],sigma=sigma), label='independent')
    plt.legend(loc=7, bbox_to_anchor=(1.5,0.5))
    dirname = "/home/jessedd/data/results/bert_on_stilts/plot_drafts/"
    filename = "max_by_bucket_debug.pdf"
    plt.savefig(dirname + filename)


    
    import pdb; pdb.set_trace()


if __name__ == "__main__":
    main()
