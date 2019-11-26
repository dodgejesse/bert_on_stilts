import loading_data
import pickle

data = loading_data.load_all_data(False, True)



for task in data:
    data[task] = data[task]['None']


import pdb; pdb.set_trace()


f = open('results/mrpc20_cola20_sst10', 'wb')
pickle.dump(data, f)

