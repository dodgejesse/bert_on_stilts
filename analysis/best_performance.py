import loading_data

dataset_to_metric = {"sst": "acc", "mrpc": "acc_and_f1", "cola": "mcc", "rte": "acc"}


def main():
    data = loading_data.load_all_data()

    for dataset in data:
        max_perf = -1
        cur_data = data[dataset]
        cur_metric = dataset_to_metric[dataset]
        #import pdb; pdb.set_trace()
        for init_seed in cur_data:
            for data_seed in cur_data[init_seed]:
                cur_evals = cur_data[init_seed][data_seed][cur_metric]['during']
                cur_evals_without_index = [eval_with_index[1] for eval_with_index in cur_evals]
                cur_exp_max = max(cur_evals_without_index)
                if max_perf < cur_exp_max:
                    max_perf = cur_exp_max

        print(dataset, max_perf)




if __name__ == "__main__":
    main()
