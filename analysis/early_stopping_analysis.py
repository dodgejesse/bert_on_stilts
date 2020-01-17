import pickle


round_digits = 4
top_k = 1

def main():
    f = open('results/early_stopping_results', 'rb')
    dataset_to_budget_to_results = pickle.load(f)
    f.close()
    single_best(dataset_to_budget_to_results)

def single_best(dataset_to_budget_to_results):
    for dataset in dataset_to_budget_to_results:
        if dataset == "settings":
            continue
        print("")
        print(dataset)

        for budget in dataset_to_budget_to_results[dataset]:

            baseline = find_baseline(dataset_to_budget_to_results[dataset][budget])
            
            for k in range(top_k):
                kth_best_result = dataset_to_budget_to_results[dataset][budget][-(k + 1)]
                print_one_result(kth_best_result, budget, baseline)
            if k > 1:
                print("")


def find_baseline(experiments):
    for experiment in experiments:
        if experiment[2]['num_stop'] == 0:
            return experiment


def print_one_result(result, budget, baseline):
    to_print = "budget = {}".format(budget)

    to_print += ", performance = {}".format(round(result[0], round_digits))
    to_print += ", improvement = {}".format(round(result[0] - baseline[0], round_digits))
    to_print += ", num started = {}".format(result[2]['num_exp'])
    to_print += ", num fully train = {}".format(result[2]['num_exp'] - result[2]['num_stop'])
    to_print += ", percent stopped = {}".format(round(result[2]['num_stop'] * 1.0 / result[2]['num_exp']
                                                      , round_digits))
    to_print += ", percent data = {}".format(round(result[2]['num_evals'] * 1.0 / result[2]['num_evals_per_exp'],
                                                   round_digits))
    
    print(to_print)

                
            
    


if __name__ == "__main__":
    main()
