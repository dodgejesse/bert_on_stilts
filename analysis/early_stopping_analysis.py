import pickle
import numpy as np

print_debug = True
round_digits = 4
top_k = 1

def main():
    f = open('results/early_stopping_results', 'rb')
    dataset_to_budget_to_results = pickle.load(f)
    f.close()
    return single_best(dataset_to_budget_to_results)

def single_best(dataset_to_budget_to_results):
    dataset_to_plot_data = {}
    for dataset in dataset_to_budget_to_results:
        if dataset == "settings":
            continue
        print("")
        print(dataset)
        dataset_to_plot_data[dataset] = {}
        for budget in dataset_to_budget_to_results[dataset]:

            baseline = find_baseline(dataset_to_budget_to_results[dataset][budget])
            
            for k in range(top_k):
                kth_best_result = dataset_to_budget_to_results[dataset][budget][-(k + 1)]
                result = print_one_result(kth_best_result, budget, baseline)
            if k > 1:
                print("")
            else:
                dataset_to_plot_data[dataset][budget] = result


        # to print average improvement:
        improves = [dataset_to_plot_data[dataset][budget]["improvement"] for budget in dataset_to_plot_data[dataset]]
        #average = []
        #for budget in dataset_to_plot_data[dataset]:
        #    average += dataset_to_plot_data[dataset][budget]["improvement"]
        #average = average / len(dataset_to_plot_data[dataset])

        improves = np.asarray(improves)
        improves = improves * 100
        print("average improvement: {}".format(np.mean(improves)))
        print("std dev: {}".format(np.std(improves)))
        print("var: {}".format(np.var(improves)))
              
        
    return dataset_to_plot_data

def find_baseline(experiments):
    for experiment in experiments:
        if experiment[2]['num_stop'] == 0:
            return experiment


def print_one_result(result, budget, baseline):
    perf = result[0]
    improve = result[0] - baseline[0]
    num_started = result[2]['num_exp']
    num_fully_train = result[2]['num_exp'] - result[2]['num_stop']
    percent_stop = result[2]['num_stop'] * 1.0 / result[2]['num_exp']
    percent_data = result[2]['num_evals'] * 1.0 / result[2]['num_evals_per_exp']
    
    to_print = "budget = {}".format(budget)
    
    to_print += ", performance = {}".format(round(perf, round_digits))
    to_print += ", improvement = {}".format(round(improve, round_digits))
    to_print += ", num started = {}".format(num_started)
    to_print += ", num fully train = {}".format(num_fully_train)
    to_print += ", percent stopped = {}".format(round(percent_stop, round_digits))
    to_print += ", percent data = {}".format(round(percent_data, round_digits))

    if print_debug:
        print(to_print)

    return {'performance': perf, 'improvement': improve, 'num_started': num_started,
            'num_fully_train': num_fully_train, 'percent_stop': percent_stop, 'percent_data': percent_data}
            
    


if __name__ == "__main__":
    main()
