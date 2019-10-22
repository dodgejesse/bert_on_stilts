import loading_data
import numpy as np
import scipy.stats
import itertools

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



dataset_to_metric = {"sst": "acc", "mrpc": "acc_and_f1", "cola": "mcc"}

def main():
    data = loading_data.load_all_data()

    init_means = {}
    all_points = {}
    unsorted_mtxs = {}

    
    for dataset in data:
        
        mtx = make_two_d_mtx(data[dataset]['None'], dataset)

        unsorted_mtxs[dataset] = mtx
        plot_mtx(mtx, dataset, sort_criteria="unsorted")
        print(np.array(mtx).mean(axis=1))
        print(np.array(mtx).mean(axis=0))


        init_means[dataset] = np.array(mtx).mean(axis=1)
        all_points[dataset] = []
        for row in mtx:
            all_points[dataset] = all_points[dataset] + row
        
        sorted_mtx = sort_rows_and_cols(mtx)
        plot_mtx(sorted_mtx, dataset, sort_criteria="independent_sort")

    print_init_mean_corr_all_pairs(init_means)
    #print_init_mean_corr_all_pairs(all_points)
    sorted_together = sort_together(unsorted_mtxs, ["cola", "mrpc"])

    


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

    

def plot_mtx(mtx, dataset, sort_criteria="unsorted"):
    plt.figure()
    plt.imshow(mtx)
    plt.colorbar()
    filename = "/home/jessedd/data/results/bert_on_stilts/plot_drafts/{}_sort={}.pdf".format(dataset, sort_criteria)
    print("saving to {}".format(filename))
    plt.savefig(filename)

    
def print_init_mean_corr_all_pairs(init_means):

    for pair in itertools.combinations(dataset_to_metric.keys(), 2):
        first_dataset = pair[0]
        second_dataset = pair[1]
        print("rank correlation between {} and {}".format(first_dataset, second_dataset))
        print(scipy.stats.spearmanr(init_means[first_dataset], init_means[second_dataset]))
        print("correlation between {} and {}".format(first_dataset, second_dataset))
        print(np.corrcoef(init_means[first_dataset], init_means[second_dataset]))
        print("")
        
    
    import pdb; pdb.set_trace()

def make_two_d_mtx(data, dataset):

    metric = dataset_to_metric[dataset]
    mtx = [[0 for i in range(10)] for j in range(10)] 

    for init_seed in range(10):
        for data_seed in range(10):
            mtx[init_seed][data_seed] = data[init_seed+1][data_seed+1][metric]['after']

    return mtx



def sort_rows_and_cols(mtx):
    # plan:
    # add column which is the average
    # sort by that column
    # remove that column
    # transpose, do it again
    # transpose back

    mtx = np.array(mtx)    

    sorted_by_cols = mtx[:, np.argsort(np.array(mtx).mean(0))]

    sorted_by_cols_t = sorted_by_cols.T
    sorted_rows_and_cols = sorted_by_cols_t[:, np.argsort(np.array(sorted_by_cols_t).mean(0))]

    final = sorted_rows_and_cols.T

    for init_row in final:
        print(init_row.tolist())

    #print(final.mean(axis=0))
    #print(final.mean(axis=1))
    
        
    #import pdb; pdb.set_trace()                
    #col_mean = mtx.mean(axis=0)
    #with_col_mean = np.append([col_mean],  mtx, axis=0)
    
    return final

    
if __name__ == "__main__":
    main()
