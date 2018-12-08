import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects

def stacked_bar(data, series_labels, category_labels=None,
                show_values=False, value_format="{}", y_label=None,
                grid=True, reverse=False, stack_colors = None):
    """Plots a stacked bar chart with the data and labels provided.

    Keyword arguments:
    data            -- 2-dimensional numpy array or nested list
                       containing data for each series in rows
    series_labels   -- list of series labels (these appear in
                       the legend)
    category_labels -- list of category labels (these appear
                       on the x-axis)
    show_values     -- If True then numeric value labels will
                       be shown on each bar
    value_format    -- Format string for numeric value labels
                       (default is "{}")
    y_label         -- Label for y-axis (str)
    grid            -- If True display grid
    reverse         -- If True reverse the order that the
                       series are displayed (left-to-right
                       or right-to-left)
    """

    ny = len(data[0])
    ind = list(range(ny))

    axes = []
    cum_size = np.zeros(ny)

    data = np.array(data)

    if reverse:
        data = np.flip(data, axis=1)
        category_labels = reversed(category_labels)

    for i, row_data in enumerate(data):
        axes.append(plt.bar(ind, row_data, width = .40,  bottom=cum_size,
                            label=series_labels[i], color = stack_colors[i]) )
        cum_size += row_data

    if category_labels:
        plt.xticks(ind, category_labels, fontsize=14)

    if y_label:
        plt.ylabel(y_label, fontsize=16)
        plt.xlabel('Number of Threads', fontsize=16)

    # handles, labels = axes.get_legend_handles_labels()
    plt.legend(handles=axes[::-1] , bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)
    # plt.legend(handles[::-1], labels[::-1], title='Line', loc='upper left')

    if grid:
        plt.grid()

    if show_values:
        for axis in axes:
            for bar in axis:
                w, h = bar.get_width(), bar.get_height()
                txt =plt.text(bar.get_x() + w/2, bar.get_y() + h/2,
                         value_format.format(h), ha="center",
                         va="center", fontsize = 11)
                txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='w')])

# https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.dtypes.html
# https://jakevdp.github.io/PythonDataScienceHandbook/04.08-multiple-subplots.html
data_array= np.loadtxt("/home/mohit/Dropbox/spring_2019_MILE_project/netbeans/fast-NetMF/CPU/6_run_set/averaged.csv", dtype=np.dtype([('dataset', 'U50'),
                                                           ('graph_size','U50'),
                                                           ('step','U100'),
                                                           ('num_threads','i'),
                                                           ('wall','f'),


                                                         ]),

                         delimiter = ',', skiprows= 1)



a = np.array([list(item) for item in data_array])

dataset = 'PPI'
graph_size = 'large'

dataset_info = {
  'PPI' : ' (|V|= 3,852, |E| = 37,841) ',
  'Blog': ' (|V|= 10,312, |E| = 333,983) ',
}

stack_colors = [ 'darkgrey',  'green', 'maroon' , 'mediumpurple', 'coral' , 'slateblue']

load_graph_values = a[ (a[:,0] == dataset) &  (a[:,1] == graph_size) & (a[:,2] == 'Graph Loaded from file')][:,4]
norm_adj = a[ (a[:,0] == dataset) &  (a[:,1] == graph_size) & (a[:,2] == 'Normalized Adjacency Matrix')][:,4]
eigen_large_values = a[ (a[:,0] == dataset) &  (a[:,1] == graph_size) & (a[:,2] == 'Eigen Decomposition')][:,4]
m_approx_values = a[ (a[:,0] == dataset) &  (a[:,1] == graph_size) & (a[:,2] == 'Approximated M')][:,4]
svd_large_values = a[ (a[:,0] == dataset) &  (a[:,1] == graph_size) & (a[:,2] == 'SVD')][:,4]
write_graph_values =a[ (a[:,0] == dataset) &  (a[:,1] == graph_size) & (a[:,2] == 'Embedding Written to file')][:,4]

np.array(load_graph_values, dtype = 'f')

series_labels = ['Load Graph', 'Normalized A', 'Eigen Decompse' , 'Approx M' , 'SVD' , 'Write Embb' ]

data = [
    np.array( load_graph_values, dtype = 'f'),
    np.array( norm_adj,  dtype = 'f'),
    np.array( eigen_large_values,  dtype = 'f'),
    np.array( m_approx_values, dtype = 'f'),
    np.array( svd_large_values, dtype = 'f'),
    np.array( write_graph_values, dtype = 'f')
]

category_labels = ['1', '2', '4', '8' , '16', '32']

stacked_bar(
    data,
    series_labels,
    category_labels=category_labels,
    show_values=True,
    value_format="{:.3f}",
    y_label="Execution Time (sec)",
    grid = False,
    stack_colors = stack_colors
)

plt.title( 'Dataset : '+ dataset + dataset_info[dataset]  + ', Algo : '+graph_size, fontsize = 16)
plt.show()
