import os
import ast
import re
import pickle

datasets = ["mrpc", "sst", "cola", "rte"]
#datasets = ["cola"]

# the data is split in two locations -- output/${DATASET} and output/saved_logs/${DATASET}

data_path = "/home/jessedd/projects/bert_on_stilts/beaker/output/"


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
        # the regex given by colin: [\r\n]?[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9:.]+Z
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

            #line_data = line.split(" ", 1)[1]
            line_data = line
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

def load_data(dir_path, data, check_which_finished=False):

    for filename in os.listdir(dir_path):

        with open(dir_path + filename) as f:
            lines = f.readlines()
            if check_which_finished:
                cur_performance = check_finished(lines)
            else:
                cur_performance = process_lines(lines)

            if "init_seed" not in cur_performance:
                continue

            if "num_train" in cur_performance:
                num_train = cur_performance["num_train"]
            else:
                num_train = "None"
            init_seed = cur_performance["init_seed"]
            data_seed = cur_performance["data_seed"]


            if num_train not in data:
                data[num_train] = {}
            if init_seed not in data[num_train]:
                data[num_train][init_seed] = {}

            if data_seed in data[num_train][init_seed]:
                import pdb; pdb.set_trace()
            
            #assert data_seed not in data[num_train][init_seed]
            data[num_train][init_seed][data_seed] = cur_performance




def load_all_data(check_which_finished=False, reload_from_disk=False):
    if not reload_from_disk:
        with open('results/all_data_to_release', 'rb') as f:
            data = pickle.load(f)
            return data

        
    all_data = {}
    for dataset in datasets:

        data = {}
        dir_path = data_path + dataset + "/"
        load_data(dir_path, data, check_which_finished=check_which_finished)
        dir_path = data_path + "saved_logs/" + dataset + "/"
        load_data(dir_path, data, check_which_finished=check_which_finished)

        if len(data) > 0:
            all_data[dataset] = data
        #import pdb; pdb.set_trace()

    return all_data

def load_all_data_for_release():
    with open('results/all_data', 'rb') as f:
        data = pickle.load(f)
        

        for dataset in data:
            for wi in data[dataset]:
                for do in data[dataset][wi]:
                    if 'num_train' in data[dataset][wi][do]:
                        del data[dataset][wi][do]['num_train']
                    if 'init_seed' in data[dataset][wi][do]:
                        del data[dataset][wi][do]['init_seed']
                    if 'data_seed' in data[dataset][wi][do]:
                        del data[dataset][wi][do]['data_seed']


        # to check that all the data is here
        for dataset in data:
            for wi in data[dataset]:
                for do in data[dataset][wi]:
                    check_one_run(data[dataset][wi][do], dataset)
        import pdb; pdb.set_trace()                    
        return data

def check_one_run(data, dataset):
    if not 'train_loss' in data or not 'loss' in data:
        print("PROBLEMS")
    if dataset == 'sst' or dataset == 'rte':
        if not 'acc' in data:
            print("PROBLEMS")
    if dataset == 'cola':
        if not 'mcc' in data:
            print("PROBLEMS")
    if dataset == 'mrpc':
        if not 'acc' in data or not 'f1' in data or not 'acc_and_f1' in data:
            print("PROBLEMS")

def print_finished(all_data, dataset="sst"):
    pairs_which_finished = []
    init_seed_to_data_seed = all_data[dataset]['None']
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
