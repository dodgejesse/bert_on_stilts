import os
import ast

#datasets = ["mrpc", "sst", "cola"]
datasets = ["mrpc"]


data_path = "/home/jessedd/projects/bert_on_stilts/beaker/output/saved_logs/"


def process_lines(lines):
    performance = {}

    cur_metric, eval_timing = "", ""

    for line in lines:
        line = line.strip()
        if cur_metric != "" and eval_timing != "":
            if cur_metric not in performance:
                performance[cur_metric] = {}


            line_data = line.split(" ", 1)[1]
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

def load_data(dir_path):
    data = {}
    for filename in os.listdir(dir_path):

        with open(dir_path + filename) as f:
            lines = f.readlines()
            cur_performance = process_lines(lines)

            num_train = cur_performance["num_train"]
            init_seed = cur_performance["init_seed"]
            data_seed = cur_performance["data_seed"]
            
            if num_train not in data:
                data[num_train] = {}
            if init_seed not in data[num_train]:
                data[num_train][init_seed] = {}

            data[num_train][init_seed][data_seed] = cur_performance

    return data


def load_all_data():
    all_data = {}
    for dataset in datasets:
        dir_path = data_path + dataset + "/"
        data = load_data(dir_path)
        all_data[dataset] = data
        #import pdb; pdb.set_trace()

    return all_data

def main():
    all_data = load_all_data()
    import pdb; pdb.set_trace()

if __name__ == "__main__":
    main()
