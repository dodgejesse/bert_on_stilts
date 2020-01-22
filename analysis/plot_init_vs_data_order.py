import loading_data
import numpy as np
import scipy.stats
import itertools

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



dataset_to_metric = {"sst": "acc", "mrpc": "acc_and_f1", "cola": "mcc", "rte": "acc"}

dataset_to_failurenum = {"sst": 0, "mrpc": 0.75, "cola": 0.1, "rte": .5}

def main():
    data = loading_data.load_all_data()

    # use best-found instead of final performance
    best_found_performance = True
    type_of_sort = "max" #"mean" #"numfail"
    print_debug = False

    init_means = {}
    init_num_fails = {}
    all_points = {}
    unsorted_mtxs = {}

    print("first is init, then data")    
    for dataset in data:
    #for dataset in ['mrpc']:


        mtx = make_two_d_mtx(data[dataset], dataset, best=best_found_performance)

        unsorted_mtxs[dataset] = mtx
        #plot_mtx(mtx, dataset, sort_criteria="unsorted")


        if print_debug:
            print(dataset, "unsorted")
            print(np.array(mtx).mean(axis=1).tolist())
            print(np.array(mtx).mean(axis=0).tolist())
            print("")

        init_means[dataset] = np.array(mtx).mean(axis=1)
        init_num_fails[dataset] = (np.array(mtx) > dataset_to_failurenum[dataset]).sum(axis=1)
        all_points[dataset] = []
        for row in mtx:
            all_points[dataset] = all_points[dataset] + row
        
        sorted_mtx, indices = sort_rows_and_cols(mtx, dataset, type_of_sort=type_of_sort)


        if print_debug:
            print(dataset, "sorted")
            print(np.array(sorted_mtx).mean(axis=1).tolist())
            print(np.array(sorted_mtx).mean(axis=0).tolist())
            print("")
            print("")

            print(indices)

        
        plot_mtx(sorted_mtx, indices, dataset, sort_criteria=type_of_sort, best=best_found_performance)

    #import pdb; pdb.set_trace()
    #print_init_mean_corr_all_pairs(init_means)
    #print_init_mean_corr_all_pairs(all_points)
    #sorted_together = sort_together(unsorted_mtxs, ["cola", "mrpc"])

    


def sort_together(unsorted_mtxs, datasets):

    col_sorted = {}
    for dataset in datasets:
        cur = unsorted_mtxs[dataset]
        cur_mtx = np.array(cur)
        sorted_by_cols = cur_mtx[:, np.argsort(np.array(cur_mtx).mean(0))]
        col_sorted[dataset] = sorted_by_cols

    combined = []
    for row in range(10):
        combined.append([])
        for dataset in datasets:
            combined[row] = combined[row] + col_sorted[dataset][row].tolist()
        
        
    combined_t = np.array(combined).T
    sorted_rows_and_cols = combined_t[:, np.argsort(np.array(combined_t).mean(0))]
    combined = sorted_rows_and_cols.T


    plot_mtx(combined[:,0:10], "cola", "with_mrpc")
    plot_mtx(combined[:,10:], "mrpc", "with_cola")
    
    return combined

    

def plot_mtx(mtx, indices, dataset, sort_criteria="unsorted", best=False):
    plt.figure()
    plt.imshow(mtx)
    plt.colorbar()

    plt.yticks(range(max(indices[0])), indices[0])
    plt.xticks(range(max(indices[1])), indices[1])
    plt.ylabel("init seed")
    plt.xlabel("data order seed")


    mtx_size = len(mtx)
    dirname = "/home/jessedd/data/results/bert_on_stilts/plot_drafts/"
    filename = "{}_sort={}_numexp={}".format(dataset,sort_criteria,mtx_size)
    plt.title(filename)

    filename = dirname + filename
    if best:
        filename += "_bestfound"
    else:
        filename += "_final"
    filename += ".pdf"
    
    print("saving to {}".format(filename))
    plt.savefig(filename, bbox_inches='tight')

    
def print_init_mean_corr_all_pairs(init_means):

    for pair in itertools.combinations(dataset_to_metric.keys(), 2):
        
        first_dataset = pair[0]
        second_dataset = pair[1]
        if not len(first_dataset) == len(second_dataset):
            continue
        print("rank correlation between {} and {}".format(first_dataset, second_dataset))
        print(scipy.stats.spearmanr(init_means[first_dataset], init_means[second_dataset]))
        print("correlation between {} and {}".format(first_dataset, second_dataset))
        print(np.corrcoef(init_means[first_dataset], init_means[second_dataset]))
        print("")
        
    
    import pdb; pdb.set_trace()

def make_two_d_mtx(data, dataset, best=False):

    metric = dataset_to_metric[dataset]

    #to find the num experiments:
    max_seed = 0
    for seed in data:
        if seed > max_seed:
            max_seed = seed
    mtx = [[0 for i in range(max_seed)] for j in range(max_seed)] 

    for init_seed in range(max_seed):
        for data_seed in range(max_seed):
            if best:
                during_indices_and_evals = data[init_seed+1][data_seed+1][metric]['during']
                during_evals = [i_and_e[1] for i_and_e in during_indices_and_evals]
                mtx[init_seed][data_seed] = max(during_evals)
            else:
                mtx[init_seed][data_seed] = data[init_seed+1][data_seed+1][metric]['after']

    return mtx



def sort_rows_and_cols(mtx, dataset, type_of_sort="mean", print_debug = False):
    # plan:
    # add column which is the average
    # sort by that column
    # remove that column
    # transpose, do it again
    # transpose back

    mtx = np.array(mtx)    

    data_indices = sort_diff_ways(mtx, dataset, type_of_sort, print_debug=False)
    sorted_by_cols = mtx[:, data_indices]

    sorted_by_cols_t = sorted_by_cols.T
    
    init_indices = sort_diff_ways(sorted_by_cols_t, dataset, type_of_sort,print_debug=True)
    sorted_rows_and_cols = sorted_by_cols_t[:, init_indices]

    final = sorted_rows_and_cols.T

    if print_debug:
        for init_row in final:
            print(init_row.tolist())

    all_indices = [init_indices + 1, data_indices + 1]
    
    return final, all_indices


def sort_diff_ways(mtx, dataset, type_of_sort, print_debug=False):
    if type_of_sort == "mean":
        return np.argsort(mtx.mean(0))
    elif type_of_sort == "max":
        return np.argsort(mtx.max(0))
    elif type_of_sort == "numfail":
        fail_num = dataset_to_failurenum[dataset]
        number_of_successes = (mtx > fail_num).sum(0)
        if print_debug:
            print("dataset: {}, num successes: {}".format(dataset, number_of_successes.tolist()))
        return np.argsort(number_of_successes)
        
    else:
        raise NotImplementedError
        
    
if __name__ == "__main__":
    main()
