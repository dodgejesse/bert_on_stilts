import pickle
import numpy as np
from init_vs_data_order import make_two_d_mtx

dataset_to_metric = {"sst": "acc", "mrpc": "acc_and_f1", "cola": "mcc"}

num_permutations = 10000
best_found_performance = True
global_stat = "mean"


def main():
    with open('results/all_data', 'rb') as f:
        data = pickle.load(f)




    for dataset in data:
        mtx = np.array(make_two_d_mtx(data[dataset], dataset, best=best_found_performance))
        init_stat_original, data_stat_original = calc_global_stat(mtx)
        shuff_results = []
        for i in range(num_permutations):
            flat_array = np.ndarray.flatten(mtx)
            flat_shuff_mtx = np.random.permutation(flat_array)
            shuff_mtx = np.reshape(flat_shuff_mtx, mtx.shape)
            init_stat_shuff, data_stat_shuff = calc_global_stat(shuff_mtx)
            shuff_results.append((init_stat_shuff, data_stat_shuff))

        # find fraction that are greater than the computed 

        shuff_results = np.array(shuff_results)
        init_p_val = 1.0 * sum(shuff_results[:,0]>init_stat_original) / num_permutations
        data_p_val = 1.0 * sum(shuff_results[:,1]>data_stat_original) / num_permutations
        print(dataset, "init p-value: {}, data p-value: {}".format(init_p_val, data_p_val))
        #import pdb; pdb.set_trace()    

    

def calc_global_stat(mtx):
    if global_stat == "max":
        return calc_maxes(mtx)
    elif global_stat == "mean":
        return calc_means(mtx)
    else:
        raise NotImplementedError

def calc_maxes(mtx):
    init_maxes = mtx.max(axis=1)
    init_global_stat = max(init_maxes) - min(init_maxes)
    
    data_maxes = mtx.max(axis=0)
    data_global_stat = max(data_maxes) - min(data_maxes)

    return init_global_stat, data_global_stat

    
def calc_means(mtx):
    init_means = mtx.mean(axis=1)
    init_global_stat = max(init_means) - min(init_means)
    
    data_means = mtx.mean(axis=0)
    data_global_stat = max(data_means) - min(data_means)

    return init_global_stat, data_global_stat
    


def debug(data):
    for dataset in data:
        for init in data[dataset]:
            for data_seed in data[dataset][init]:
                print(dataset, init, data_seed)



                
if __name__ == "__main__":
    main()
