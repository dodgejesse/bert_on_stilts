
# results is a dict with four keys: 'logits', 'loss', 'metrics', 'labels'
def logits(results):

    shifted, scaled_and_shifted = norm_output(results['logits'])

    re_results = relabeled_accuracy(results['logits'], results['labels'], "logits")
    sh_results = relabeled_accuracy(shifted, results['labels'], "shifted logits")
    sc_results = relabeled_accuracy(scaled_and_shifted, results['labels'], "scaled and shifted logits")    

    #if "acc" in results['metrics']:
    #    performance = results['metrics']['acc']
    #elif "mcc" in results['metrics']:
    #    performance = results['metrics']['mcc']
    return results['metrics'], results['loss'], re_results, sh_results, sc_results

def norm_output(logits):
    mean = logits.mean(axis=0)
    stds = logits.std(axis=0)
    
    shifted = logits - mean
    scaled_and_shifted = shifted / stds

    return shifted, scaled_and_shifted

def relabeled_accuracy(output, target, name_of_exp):
    pred_labels = output.argmax(axis=1)
    cur_acc = (pred_labels == target).mean()
    relabeled_acc = max(cur_acc, 1-cur_acc)
    print(name_of_exp + " accuracy: " + str(round(relabeled_acc, 4)))
    return relabeled_acc
