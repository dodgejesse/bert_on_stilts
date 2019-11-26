import loading_data
import pickle

data = loading_data.load_all_data()



batch_nums = [one_eval[0] for one_eval in data['cola'][1][1]['mcc']['during']]
batch_nums_new = [one_eval[0] for one_eval in data['cola'][11][11]['mcc']['during']]
print(data['cola'][1][1].keys())

diffs_between_evals = []
for i in range(len(batch_nums_new) - 1):
    diffs_between_evals.append(batch_nums_new[i+1]- batch_nums_new[i])
print(diffs_between_evals)


in_batch_nums = True
for item in batch_nums_new:
    if not item in batch_nums:
        batch_nums_new = False
print(in_batch_nums)
import pdb; pdb.set_trace()

task = "mrpc"
print(task)
#import pdb; pdb.set_trace()
for init_seed in range(20):
    if init_seed + 1not in data[task]:
        print("init: ", init_seed + 1)
        continue    
    for data_seed in range(20):
        if data_seed + 1 not in data[task][init_seed+1]:
            print("init and data:", init_seed + 1, data_seed + 1)


task = "cola"
print(task)
#import pdb; pdb.set_trace()
for init_seed in range(15):
    if init_seed + 1not in data[task]:
        print("init: ", init_seed + 1)
        continue    
    for data_seed in range(15):
        if data_seed + 1 not in data[task][init_seed+1]:
            print("init and data:", init_seed + 1, data_seed + 1)
