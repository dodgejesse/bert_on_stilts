import early_stopping_analysis
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np

dataset_orders = ['mrpc', 'rte', 'cola', 'sst']


def main():
    unformatted_data = early_stopping_analysis.main()
    data = format_data(unformatted_data)

    #plt.style.use('ggplot')
    plt.rcParams.update({'font.size': 15})    
    fig = plt.figure(figsize=(6,24))

    counter = 0
    for dataset in dataset_orders:
        counter += 1

        ax1 = fig.add_subplot(4,1,counter)


        
        make_one_plot(data[dataset], dataset, ax1)
        
    save_figure()

        
def format_data(unformatted_data):
    data = {}
    for dataset in unformatted_data:
        data[dataset] = {}
        data[dataset]['num_started'] = []
        data[dataset]['num_fully_train'] = []
        data[dataset]['percent_data'] = []
        for budget in unformatted_data[dataset]:
            data[dataset]['num_started'].append(unformatted_data[dataset][budget]['num_started'])
            data[dataset]['num_fully_train'].append(unformatted_data[dataset][budget]['num_fully_train'])
            data[dataset]['percent_data'].append(unformatted_data[dataset][budget]['percent_data'])
    return data


def make_one_plot(data, dataset, ax1):
    
    line1, = ax1.plot(range(1,31), data['num_started'], marker='s', fillstyle='none', color='#1f77b4')
    line2, = ax1.plot(range(1,31), data['num_fully_train'], marker='o', fillstyle='none', color='#ff7f0e')
    
    ax2 = ax1.twinx()
    line3, = ax2.plot(range(1,31), data['percent_data'], marker='x', fillstyle='none', color='#2ca02c')


    align_y_axes = True
    if align_y_axes:
        # to set the y-axes to have the same number of ticks, so we can use a grid
        #ax1.set_yticks(np.linspace(ax1.get_ybound()[0], ax1.get_ybound()[1], 6))
        #ax2.set_yticks(np.linspace(ax2.get_ybound()[0], ax2.get_ybound()[1], 6))

        ax2_y_ticks = np.linspace(0, 1, len(ax1.get_yticks())-1).tolist()
        ax2_y_ticks = [-ax2_y_ticks[1]] + ax2_y_ticks
        ax2.set_yticks(ax2_y_ticks)



        # to get the ylim
        lower_proportion = ax1.get_ylim()[0] / ax1.get_yticks()[0]
        ax2_lower_y_lim = ax2_y_ticks[0]*lower_proportion
        
        upper_dist = ax1.get_yticks()[-1] - ax1.get_yticks()[-2]
        upper_proportion = (ax1.get_ylim()[1] - ax1.get_yticks()[-2]) / upper_dist
        ax2_step_dist = ax2_y_ticks[-1] - ax2_y_ticks[-2]
        ax2_upper_y_lim = ax2.get_yticks()[-2] + ax2_step_dist * upper_proportion
        
        ax2.set_ylim((ax2_lower_y_lim, ax2_upper_y_lim))
        
        ax2_ylabels = [str(round(percent * 100)) + "%" for percent in ax2_y_ticks]
        #ax2_ylabels = ax2_ylabels[1,len(ax2_ylabels)]]
        ax2.set_yticklabels(ax2_ylabels)

        upper_ylim_1 = True
        if upper_ylim_1:
            ax1.set_ylim((ax1.get_ylim()[0], ax1.get_yticks()[-1]))
            ax2.set_ylim((ax2.get_ylim()[0], ax2.get_yticks()[-1]))
        
    else:
        ax2.set_ylim((0,1))

    if dataset == 'sst':
        ax1.set_xlabel('Budget sufficient to train X models on all data')
    ax1.set_ylabel('Number of experiments')
    ax2.set_ylabel('Percent of data trained on\nbefore early stopping')
    ax1.set_title("Optimal early stopping for {}".format(get_dataset_name(dataset)))
    
    
    #ax1.set_zorder(ax2.get_zorder()+1)
    #ax1.patch.set_visible(False)
    #ax1.yaxis.grid(color='gray', linestyle='dashed')
    ax2.grid(None)
    #ax2.set_yticks(np.linspace(ax2.get_yticks()[0], ax2.get_yticks()[-1], len(ax1.get_yticks())))

    if dataset == 'mrpc':
        ax2.legend((line1, line2, line3), ("Number started", "Number stopped early",
                                           "% of data before stopping"), loc=2)


def save_figure():
    dirname = "/home/jessedd/data/results/bert_on_stilts/plot_drafts/"
    filename = "numstarted_numfinished_percentdata.pdf"
    print("saving to {}".format(dirname + filename))
    plt.savefig(dirname + filename, bbox_inches='tight')
    
    
def get_dataset_name(dataset):
    correct_names = {'cola': 'CoLA', 'mrpc': 'MRPC', 'sst': 'SST', 'rte': 'RTE'}
    return correct_names[dataset]
    
    
if __name__ == "__main__":
    main()
