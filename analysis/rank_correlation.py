import scipy.stats
import numpy as np
import loading_data


dataset_to_metric = {"sst": "acc", "mrpc": "acc_and_f1", "cola": "mcc"}

def correlation_between_init_loss_and_val_perf(data):
    for dataset in data:
        metric = dataset_to_metric[dataset]
        for data_size in data[dataset]:


            cur_data = data[dataset][data_size]

            init_to_avg_val_perf = avg_val_perf_per_init(cur_data, metric)


            init_seed_to_init_loss = get_init_to_metric(cur_data, "loss")
            init_seed_to_init_perf = get_init_to_metric(cur_data, metric)
            #init_seed_to_final_perf = get_init_to_final_metric(cur_data, metric)
            
            corr_between_two_lists(init_seed_to_init_perf, init_seed_to_init_loss)

            corr_between_first_and_all(init_seed_to_init_loss, init_to_avg_val_perf)
            


def corr_between_two_lists(init_seed_to_init_perf, init_seed_to_init_loss):
    paired_data = []
    for init_seed in init_seed_to_init_loss:
        paired_data.append([init_seed_to_init_loss[init_seed], init_seed_to_init_perf[init_seed]])
    print("the rank correlation between the loss and valid perf before training:")
    print(scipy.stats.spearmanr(paired_data))
    print("the correlation between the loss and valid perf before training:")
    print(np.corrcoef(paired_data, rowvar=False))


def corr_between_first_and_all(init_seed_to_init_loss, init_to_avg_val_perf):
    num_inits = 10
    print("SHOULD FIX THIS MAGIC NUMBER")


    evals = []

    for i in range(num_inits):
        init_seed = i + 1
        #cur = [init_seed_to_init_loss[init_seed]] + init_to_avg_val_perf[init_seed].tolist()
        cur = init_to_avg_val_perf[init_seed].tolist()
        evals.append(cur)

    import pdb; pdb.set_trace()        
    print(scipy.stats.spearmanr(evals, axis=0))
    print(np.corrcoef(evals))


def avg_val_perf_per_init(cur_data, metric):
    init_to_avg = {}
    for init_seed in cur_data:
        if init_seed not in init_to_avg:
            init_to_avg[init_seed] = []
        for data_seed in cur_data[init_seed]:
            
            all_val_perf = cur_data[init_seed][data_seed][metric]["during"]

            val_perf = [one_eval[1] for one_eval in all_val_perf]

            # start average with initial points
            if len(init_to_avg[init_seed]) == 0:
                init_to_avg[init_seed] = val_perf
                continue
            else:
                new_sum = np.asarray(init_to_avg[init_seed]) + np.asarray(val_perf)
                init_to_avg[init_seed] = new_sum
            
        init_to_avg[init_seed] = init_to_avg[init_seed] / len(cur_data[init_seed])
    return init_to_avg


def get_init_to_metric(cur_data, metric):
    init_seed_to_init_metric = {}
    for init_seed in cur_data:
        for data_seed in cur_data[init_seed]:

            # if first data_seed
            if init_seed not in init_seed_to_init_metric:
                init_seed_to_init_metric[init_seed] = cur_data[init_seed][data_seed][metric]['before']
            # check that the init metric is the same for the same init seeds
            else:
                assert init_seed_to_init_metric[init_seed] == cur_data[init_seed][data_seed][metric]['before']
    return init_seed_to_init_metric


def check_intuitions():
    first = [1,2,3,4,5,6,7,8,9,10]
    for i in range(10):
        # second = random permutation of first
        second = np.random.permutation(first)
        print(scipy.stats.spearmanr(first, second))

    print("")
    for i in range(10):
        lower_half = [1,2,3,4,5]
        upper_half = [6,7,8,9,10]
        lower_half = np.random.permutation(lower_half)
        upper_half = np.random.permutation(upper_half)

        third = lower_half.tolist() + upper_half.tolist()
        print(scipy.stats.spearmanr(first, third))

    

def main():
    #check_intuitions()
    data = loading_data.load_all_data()
    correlation_between_init_loss_and_val_perf(data)


if __name__ == "__main__":
    main()
