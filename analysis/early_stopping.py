import downsample_experiments
import loading_data
import numpy as np

dataset_to_metric = {"sst": "acc", "mrpc": "acc_and_f1", "cola": "mcc", "rte": "acc"}
size_of_subsample = 20
num_samples = 10000
np.random.seed(1)

def main():
    data = loading_data.load_all_data()
    
    print(data.keys())

    for dataset in data:
        print(dataset)
        cur_data = extract_data(data, dataset)

        early_stop_with_varying_budgets(cur_data)
        print("")

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

                
                

def find_maxes(cur_subsample, batch_num_to_beat):
    maxes = []
    for eval_index in range(len(cur_subsample[0])):
        if cur_subsample[0][eval_index][0] >= batch_num_to_beat:

            for one_run in cur_subsample:
                
                one_run_evals = [one_eval[1] for one_eval in one_run]
                one_run_evals = one_run_evals[:eval_index+1]
                maxes.append(max(one_run_evals))

            return maxes
            
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
                                                 dataset)
            evals.append(ds_exp)

    return np.array(evals)

def subsample(cur_data, size_of_subsample):
    idx = np.random.randint(len(cur_data), size=size_of_subsample)
    cur_subsample = cur_data[idx]
    return cur_subsample

    

if __name__ == "__main__":
    main()
