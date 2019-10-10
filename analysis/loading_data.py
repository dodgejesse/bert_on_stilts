import os
import ast
import re

#datasets = ["mrpc", "sst", "cola"]
datasets = ["sst", "mrpc"]

data_path = "/home/jessedd/projects/bert_on_stilts/beaker/output/saved_logs/"


def check_finished(lines):

    performance = {}
    
    for line in lines:
        if "init seed:" in line:
            performance["init_seed"] = int(line.split(":")[-1])
        elif "data order seed:" in line:
            performance["data_seed"] = int(line.split(":")[-1])
            return performance
        elif "train_examples_number:" in line:
            performance["num_train"] = line.split()[-1]

    return performance

def clean_line_data(line_data):
    if "Z" in line_data:
        # [\r\n]?[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9:.]+Z
        matches = re.findall('[\r\n]?[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9:.]+Z', line_data)
        for match in matches:
            if match in line_data:
                line_data = line_data.replace(match + " ", "")
        
    return line_data
            

def process_lines(lines):
    performance = {}

    cur_metric, eval_timing = "", ""

    for line in lines:
        line = line.strip()
        if cur_metric != "" and eval_timing != "":
            if cur_metric not in performance:
                performance[cur_metric] = {}


            line_data = line.split(" ", 1)[1]
            line_data = clean_line_data(line_data)

            performance[cur_metric][eval_timing] = ast.literal_eval(line_data)
            
            eval_timing = ""
            continue
        
        if "init seed:" in line:
            performance["init_seed"] = int(line.split(":")[-1])
        elif "data order seed:" in line:
            performance["data_seed"] = int(line.split(":")[-1])
        elif "train_examples_number:" in line:
            performance["num_train"] = line.split()[-1]

        
        elif "the losses at each train step:" in line:
            cur_metric, eval_timing = "train_loss", "during"
            
        # the metrics
        elif "metric:" in line:
            cur_metric = line.split()[-1]

        elif "validation {} of initialized model:".format(cur_metric) in line:
            eval_timing = "before"
        elif "validation {} throughout training:".format(cur_metric) in line:
            eval_timing = "during"
        elif "validation {} after training:".format(cur_metric) in line:
            eval_timing = "after"

    return performance        

def load_data(dir_path, check_which_finished=False):
    data = {}
    for filename in os.listdir(dir_path):

        with open(dir_path + filename) as f:
            lines = f.readlines()
            if check_which_finished:
                cur_performance = check_finished(lines)
            else:
                cur_performance = process_lines(lines)

            if "init_seed" not in cur_performance:
                continue
            
            num_train = cur_performance["num_train"]
            init_seed = cur_performance["init_seed"]
            data_seed = cur_performance["data_seed"]


            if num_train not in data:
                data[num_train] = {}
            if init_seed not in data[num_train]:
                data[num_train][init_seed] = {}

            data[num_train][init_seed][data_seed] = cur_performance

    return data


def load_all_data(check_which_finished=False):
    all_data = {}
    for dataset in datasets:
        dir_path = data_path + dataset + "/"
        data = load_data(dir_path, check_which_finished=check_which_finished)
        all_data[dataset] = data
        #import pdb; pdb.set_trace()

    return all_data

def print_finished(all_data):
    pairs_which_finished = []
    init_seed_to_data_seed = all_data['sst']['None']
    for init_seed in init_seed_to_data_seed:
        for data_seed in init_seed_to_data_seed[init_seed]:
            pairs_which_finished.append((init_seed, data_seed))
    print(pairs_which_finished)
    pairs_which_finished.sort()
    print(pairs_which_finished)
    print(len(pairs_which_finished))

    print("pairs which didn't finish:")
    for i in range(10):
        for j in range(10):
            if (i+1, j+1) not in pairs_which_finished:
                print('"{} {}"'.format(i+1, j+1), end=" ")

def main():
    import pdb; pdb.set_trace()
    all_data = load_all_data(check_which_finished=False)
    #print_finished(all_data)

if __name__ == "__main__":
    main()
