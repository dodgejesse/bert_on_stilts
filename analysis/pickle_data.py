import loading_data
import pickle

data = loading_data.load_all_data(False, True)



for task in data:
    data[task] = data[task]['None']


import pdb; pdb.set_trace()


f = open('results/mrpc25_cola25_rte25_sst15', 'wb')
pickle.dump(data, f)

