


round_digits = 10


def print_beginning(final_results):
    print("")
    print("=======================================================================================")
    print("final results")
    print("")
    print("task: " + str(final_results["task"]))
    print("init seed: " + str(final_results["init_seed"]))
    print("data seed: " + str(final_results["data_seed"]))
    print("")
        



def print_train_loss(final_results):
    train_results = final_results["train_results"]
    train_losses = []

    for epoch in train_results:
        cur_results = [train_results[epoch][0][iter_num][0].tr_loss for iter_num in train_results[epoch][0]]

        for i in range(1, len(cur_results)):
            cur_results[-i] = cur_results[-i] - cur_results[-(i+1)]

        train_losses += cur_results

    print("")
    print("the losses at each train step:")
    print(train_losses)
    print("")
    
            
def print_valid_loss(final_results):
    init_loss = final_results['init_results'][0]['loss']
    valid_loss_throughout_train = []

    for epoch in final_results["train_results"]:
        cur_train_results = final_results["train_results"][epoch]
        valid_loss_throughout_train += [cur_train_results[0][iter_num][1] for iter_num in cur_train_results[0]]

    computed_losses = []
    for i in range(len(valid_loss_throughout_train)):
        if valid_loss_throughout_train[i] is not None:
            computed_losses.append([i, valid_loss_throughout_train[i]['loss']])

    print("")
    print("validation loss of initialized model:")
    print(init_loss)
    print("validation losses throughout training:")
    print(computed_losses)
    print("validation loss after training:")
    print(final_results['val_results']['loss'])
    print("")


def print_valid_metrics(final_results):
    for metric in final_results['init_results'][0]['metrics']:

        init_perf = final_results['init_results'][0]['metrics'][metric]
        valid_perf_throughout_train = []
        
        for epoch in final_results["train_results"]:
            cur_train_results = final_results["train_results"][epoch]
            valid_perf_throughout_train += [cur_train_results[0][iter_num][1] for iter_num in cur_train_results[0]]


        computed_perfs = []
        for i in range(len(valid_perf_throughout_train)):
            if valid_perf_throughout_train[i] is not None:
                computed_perfs.append([i, valid_perf_throughout_train[i]['metrics'][metric]])


        print("")
        print("metric: {}".format(metric))
        print("validation {} of initialized model:".format(metric))
        print(init_perf)
        print("validation {} throughout training:".format(metric))
        print(computed_perfs)
        print("validation {} after training:".format(metric))
        print(final_results['val_results']['metrics'][metric])
        print("")
        
    
def print_all_results(final_results):
    #print_beginning(final_results)
    print_train_loss(final_results)
    print_valid_loss(final_results)
    print_valid_metrics(final_results)

    import pdb; pdb.set_trace()

    
def print_results_OLD(final_results):
    print_beginning(final_results)
    if "init_results" in final_results:
        print("performance of initialized, untrained model:")
        for perf in final_results["init_results"][0]:
            print("unaltered init " + perf + ": " + str(round(final_results["init_results"][0][perf], round_digits)))
        print("unaltered loss: " + str(round(final_results["init_results"][1], round_digits)))
        print("relabeled logit accuracy: " + str(round(final_results["init_results"][2], round_digits)))
        print("shifted logit accuracy: " + str(round(final_results["init_results"][3], round_digits)))
        print("scaled and shifted accuracy: " + str(round(final_results["init_results"][4], round_digits)))
        print("")
    if "train_results" in final_results:
        print("performance during training:")
        results = final_results["train_results"]
        for iter_num in results:
            perf_string = ""
            for performance in results[iter_num]['metrics']:
                perf_string += ", " + performance + ": " + str(round(
                    results[iter_num]['metrics'][performance],round_digits))
            print("iter: " + str(iter_num) + perf_string + 
                  ", loss: " + str(round(results[iter_num]['loss'],round_digits)))
        print("")
    if "val_results" in final_results:
        print("performance after training:")
        results = final_results["val_results"]

        perf_string = ""
        for performance in results['metrics']:
            perf_string += performance + ": " + str(round(
                    results['metrics'][performance],round_digits)) + ", "
        print("after training validation " + perf_string +
                  "loss: " + str(round(results['loss'],round_digits)))
        print("")
    if "test_results" in final_results:
        print(final_results["test_results"])
                          
