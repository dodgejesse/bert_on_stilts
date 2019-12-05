import loading_data
import batch_nums_for_early_stopping

def main(data, dataset):
    batch_nums = batch_nums_for_early_stopping.get_batch_nums()

    to_return = []
    for one_eval in data:
        if one_eval[0] in batch_nums[dataset]:
            to_return.append(one_eval)

    return to_return



if __name__ == "__main__":
    main(loading_data.load_all_data())
