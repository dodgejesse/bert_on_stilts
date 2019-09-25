
import os
import numpy as np

dataset = "sst"
data_path = "/home/jessedd/projects/bert_on_stilts/beaker/output/{}/".format(dataset)


def sort_seeds_by_performance(data):
    tmp = [[data[seed]["iter_2"], seed] for seed in data]
    tmp.sort()
    print([s[1] for s in tmp])
    #print(tmp)
    

def find_correlation(data, first_field, second_field):
    tmp = [[data[seed][first_field], data[seed][second_field]] for seed in data]
    tmp.sort()
    print(tmp)
    print(np.corrcoef(tmp, rowvar=False))



def process_mrpc_lines(lines):
    performance = {}
    cur_seed = -1
    for line in lines:
        line = line.strip()
        if "seed: " in line:
            performance["seed"] = int(line.split(":")[-1])
        elif "unaltered init" in line:
            performance["logit_acc"] = float(line.split(":")[-1])
        elif "shifted logit accuracy" in line:
            performance["shifted_logit_acc"] = float(line.split(":")[-1])
        elif "scaled and shifted accuracy" in line:
            performance["scale_shift_logit_acc"] = float(line.split(":")[-1])
        elif "train_examples_number" in line:
            performance["num_train"] = line.split(":")[-1].strip()
        elif "iter:" in line:
            iter_num = line.split(":")[3].split(",")[0].strip()
            iter_acc = float(line.split(":")[4].split(",")[0].strip())
            performance["iter_" + iter_num] = iter_acc

    return performance
    
def process_cola_lines(lines):
    performance = {}
    cur_seed = -1
    for line in lines:
        line = line.strip()
        if "seed: " in line:
            performance["seed"] = int(line.split(":")[-1])
        elif "unaltered init" in line:
            performance["logit_acc"] = float(line.split(":")[-1])
        elif "shifted logit accuracy" in line:
            performance["shifted_logit_acc"] = float(line.split(":")[-1])
        elif "scaled and shifted accuracy" in line:
            performance["scale_shift_logit_acc"] = float(line.split(":")[-1])
        elif "train_examples_number" in line:
            performance["num_train"] = line.split(":")[-1].strip()
        elif "iter:" in line:
            iter_num = line.split(":")[3].split(",")[0].strip()
            iter_acc = float(line.split(":")[4].split(",")[0].strip())
            performance["iter_" + iter_num] = iter_acc

    return performance
    

def process_sst_lines(lines):
    performance = {}
    cur_seed = -1
    for line in lines:
        line = line.strip()
        if "seed: " in line:
            performance["seed"] = int(line.split(":")[-1])
        elif "unaltered logit accuracy" in line:
            performance["logit_acc"] = float(line.split(":")[-1])
        elif "shifted logit accuracy" in line:
            performance["shifted_logit_acc"] = float(line.split(":")[-1])
        elif "scaled and shifted accuracy" in line:
            performance["scale_shift_logit_acc"] = float(line.split(":")[-1])
        elif "train_examples_number" in line:
            performance["num_train"] = line.split(":")[-1].strip()
        elif "iter:" in line:
            iter_num = line.split(":")[3].split(",")[0].strip()
            iter_acc = float(line.split(":")[4].split(",")[0].strip())
            performance["iter_" + iter_num] = iter_acc

    return performance

def load_data():
    data = {}
    for filename in os.listdir(data_path):

        with open(data_path + filename) as f:
            lines = f.readlines()
            if dataset == "sst":
                cur_performance = process_sst_lines(lines)
            elif dataset == "cola":
                cur_performance = process_cola_lines(lines)
            elif dataset == "mrpc":
                cur_performance = process_mrpc_lines(lines)
            else:
                assert False, "not implemented"
            if "num_train" not in cur_performance:
                continue
            
            if cur_performance["num_train"] not in data:
                data[cur_performance["num_train"]] = {}
            data[cur_performance["num_train"]][cur_performance["seed"]] = cur_performance
    return data

def main():
    data = load_data()
    sort_seeds_by_performance(data['None'])
    #sort_seeds_by_performance(data['5000'])
    #find_correlation(data['5000'], "logit_acc", "iter_0")
    #find_correlation(data['5000'], "iter_0","logit_acc")
    #find_correlation(data['None'], "logit_acc", "iter_0")
    #find_correlation(data['None'], "iter_0","logit_acc")
    import pdb; pdb.set_trace()

if __name__ == "__main__":
    main()
