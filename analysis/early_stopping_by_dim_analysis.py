import pickle
import early_stopping_by_dim

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
    early_stopping_by_dim.print_budgets_for_which_we_win(budget_to_perf)
    early_stopping_by_dim.print_debug_info(unique_to_numeach_to_baseline, unique_to_numeach_to_results)


if __name__ == "__main__":
    main()
