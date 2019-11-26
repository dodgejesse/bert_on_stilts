
round_digits = 10


def print_beginning(final_results, out_string):
    out_string += "\n"
    out_string += "=======================================================================================" + "\n"
    out_string += "final results" + "\n"
    out_string += "\n"
    out_string += "task: " + str(final_results["task"]) + "\n"
    out_string += "init seed: " + str(final_results["init_seed"]) + "\n"
    out_string += "data order seed: " + str(final_results["data_order_seed"]) + "\n"
    out_string += "\n"
    return out_string


def print_train_loss(final_results, out_string):
    train_results = final_results["train_results"]
    train_losses = []

    for epoch in train_results:
        cur_results = [train_results[epoch][0][iter_num][0].tr_loss for iter_num in train_results[epoch][0]]

        for i in range(1, len(cur_results)):
            cur_results[-i] = cur_results[-i] - cur_results[-(i+1)]

        train_losses += cur_results

    out_string += "\n"
    out_string += "the losses at each train step:" + "\n"
    out_string += str(train_losses) + "\n"
    out_string += "\n"

    return out_string
    
            
def print_valid_loss(final_results, out_string):
    init_loss = final_results['init_results'][0]['loss']
    valid_loss_throughout_train = []

    for epoch in final_results["train_results"]:
        cur_train_results = final_results["train_results"][epoch]
        valid_loss_throughout_train += [cur_train_results[0][iter_num][1] for iter_num in cur_train_results[0]]

    computed_losses = []
    for i in range(len(valid_loss_throughout_train)):
        if valid_loss_throughout_train[i] is not None:
            computed_losses.append([i, valid_loss_throughout_train[i]['loss']])

    out_string += "\n"
    out_string += "metric: loss" + "\n"
    out_string += "validation loss of initialized model:" + "\n"
    out_string += str(init_loss) + "\n"
    out_string += "validation loss throughout training:" + "\n"
    out_string += str(computed_losses) + "\n"
    out_string += "validation loss after training:" + "\n"
    out_string += str(final_results['val_results']['loss']) + "\n"
    out_string += "\n"

    return out_string


def print_valid_metrics(final_results, out_string):
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


        out_string += "\n"
        out_string += "metric: {}".format(metric) + "\n"
        out_string += "validation {} of initialized model:".format(metric) + "\n"
        out_string += str(init_perf) + "\n"
        out_string += "validation {} throughout training:".format(metric) + "\n"
        out_string += str(computed_perfs) + "\n"
        out_string += "validation {} after training:".format(metric) + "\n"
        out_string += str(final_results['val_results']['metrics'][metric]) + "\n"
        out_string += "\n"
    return out_string


def print_batch_indices(final_results, out_string):
    batch_indices = []
    for epoch in final_results["train_results"]:
        cur_train_results = final_results["train_results"][epoch]
        batch_indices += [[cur_train_results[0][iter_num][2].tolist() for iter_num in cur_train_results[0]]]

    out_string += "\n"
    out_string += "indices of examples in batches:"
    out_string += "\n"
    out_string += str(batch_indices)
    out_string += "\n"
    return out_string
        


def print_all_results(final_results, save_loc):
    out_string = ""
    out_string = print_beginning(final_results, out_string)
    out_string = print_train_loss(final_results, out_string)
    out_string = print_valid_loss(final_results, out_string)
    out_string = print_valid_metrics(final_results, out_string)
    out_string = print_batch_indices(final_results, out_string)
    if save_loc is None:
        print(out_string)
    else:
        with open(save_loc, 'w') as f:
            f.write(out_string)

    
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
                          
