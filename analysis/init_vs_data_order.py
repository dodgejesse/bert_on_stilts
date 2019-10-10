import loading_data
import numpy as np

def main():
    data = loading_data.load_all_data()
    import pdb; pdb.set_trace()
    #for dataset in data:
    make_two_d_mtx(data['mrpc']['None'])


def make_two_d_mtx(data):
    mtx = [[0 for i in range(10)] for j in range(10)] 

    for init_seed in range(10):
        for data_seed in range(10):
            mtx[init_seed][data_seed] = data[init_seed+1][data_seed+1]['acc_and_f1']['after']

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


    
    import pdb; pdb.set_trace()                
    #col_mean = mtx.mean(axis=0)
    #with_col_mean = np.append([col_mean],  mtx, axis=0)
    


    
if __name__ == "__main__":
    main()
