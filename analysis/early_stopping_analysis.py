import pickle


round_digits = 4
top_k = 3

def main():
    f = open('results/debug_early_stopping', 'rb')
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
            for k in range(top_k):
                kth_best_result = dataset_to_budget_to_results[dataset][budget][-(k + 1)]
                print_one_result(kth_best_result, budget)
            print("")




def print_one_result(result, budget):
    perf = round(result[0], round_digits)
    num_exp = result[2]['num_exp']
    percent_stop = round(result[2]['num_stop'] * 1.0 / num_exp, round_digits)
    percent_data = round(result[2]['num_evals'] * 1.0 / result[2]['num_evals_per_exp'],
                         round_digits)
    
    to_print = "budget = {}, performance = {}, num started = {}, percent stopped = {}, percent data = {}".format(
        budget, perf, num_exp, percent_stop, percent_data)
    print(to_print)

                
            
    


if __name__ == "__main__":
    main()
