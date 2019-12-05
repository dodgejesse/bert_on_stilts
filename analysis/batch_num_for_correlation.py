import loading_data
import numpy as np

dataset_to_metric = {"sst": "acc", "mrpc": "acc_and_f1", "cola": "mcc"}


def main():

    sst = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000, 3100, 3200, 3300, 3400, 3500, 3600, 3700, 3800, 3900, 4000, 4100, 4200, 4310, 4410, 4510, 4610, 4710, 4810, 4910, 5010, 5110, 5210, 5310, 5410, 5510, 5610, 5710, 5810, 5910, 6010, 6110, 6210, 6310, 6410, 6510, 6610, 6710, 6810, 6910, 7010, 7110, 7210, 7310, 7410, 7510, 7610, 7710, 7810, 7910, 8010, 8110, 8210, 8310, 8410, 8520, 8620, 8720, 8820, 8920, 9020, 9120, 9220, 9320, 9420, 9520, 9620, 9720, 9820, 9920, 10020, 10120, 10220, 10320, 10420, 10520, 10620, 10720, 10820, 10920, 11020, 11120, 11220, 11320, 11420, 11520, 11620, 11720, 11820, 11920, 12020, 12120, 12220, 12320, 12420, 12520, 12620]
    mrpc = [0,23,46,69,92,115,138,161,184,207,230,253,276,299,322,345,368,391,414,437,460,483,506,529,552,575,598,621,644,667]
    cola = [0,53,106,159,212,265,318,371,424,477,530,588,641,694,747,800,853,906,959,1012,1065,1123,1176,1229,1282,1335,1388,1441,1494,1547,1600]

    data = loading_data.load_all_data()
    check_batch_exists("mrpc", data["mrpc"], mrpc)
    check_batch_exists("cola", data["cola"], cola)
    check_batch_exists("sst", data["sst"], sst)
    
    #do_dumb_thing()

def check_batch_exists(dataset, data, batch_nums):
    metric = dataset_to_metric[dataset]
    batch_nums_to_check = [eval_result[0] for eval_result in data[1][1][metric]['during']]

    print(dataset)
    for batch_num in batch_nums:
        if batch_num not in batch_nums_to_check:
            print("PROBLEM!")
    print("")


def do_dumb_thing(data):

    for dataset in dataset_to_metric:
        cur_data = data[dataset]



        if dataset == "mrpc" or dataset == "cola" or True:
            import pdb; pdb.set_trace()
            metric = dataset_to_metric[dataset]
            batch_nums = [eval_result[0] for eval_result in cur_data[1][1][metric]['during']]
            print(batch_nums)


            continue





        
            diffs = []
            
            for i in range(len(batch_nums)-1):
                diffs.append(batch_nums[i+1] - batch_nums[i])


            to_downsample = [eval_result[0] for eval_result in cur_data[1][1][metric]['during']]
            for batch_num in batch_nums:
                if not batch_num in to_downsample:
                    print(batch_num)
            print(diffs)


            #print(np.array(diffs, batch_nums))
            
            print(np.array(diffs))
            print(np.array(batch_nums))
            
            max_diffs = max(diffs)
            max_batch_num = max(batch_nums)
            
            #for i in range(100):
                
            
    
if __name__ == "__main__":
    main()
