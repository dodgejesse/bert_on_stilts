import pickle

dataset_to_metric = {"sst": "acc", "mrpc": "acc_and_f1", "cola": "mcc", "rte": "acc"}


def main():
    for dataset in dataset_to_metric:
        for comparison_statistic in ["average", "max"]:
            f = open('results/early_stopping_by_dim/{}_{}_results_and_budget_to_perf'.format(
                dataset, comparison_statistic), 'rb')    

            results = pickle.load(f)
            f.close()
            process_results(results)
            

def process_results(results):
    import pdb; pdb.set_trace()
    unique_to_numeach_to_baseline = results["unique_to_numeach_to_baseline"]
    unique_to_numeach_to_results = results["unique_to_numeach_to_results"]
    budget_to_perf = results["budget_to_perf"]
    print_budgets_for_which_we_win(budget_to_perf)
    print_debug_info(unique_to_numeach_to_baseline, unique_to_numeach_to_results)


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



if __name__ == "__main__":
    main()
