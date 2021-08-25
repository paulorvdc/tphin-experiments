import pandas as pd
import glob
import os
from tqdm import tqdm

from tphin_utils import get_metric

path = '/media/pauloricardo/basement/commodities_usecase/pred_iterative/'
all_files = glob.glob(os.path.join(path, "*.csv"))

algorithms = ['regularization', 'deep_walk', 'node2vec', 'line', 'struc2vec', 'metapath2vec', 'gcn']
metrics = ['acc', 'precision', 'recall', 'f1']
edge_types = ['event_trend']
intervals = ['week', 'month']
commodities = ['corn', 'soybean']
time_windows = [3, 6, 12]

results_df = {'metric': [], 'algorithm': [], 'interval': [], 'commodity': [], 'time_window': [], 'type': [], 'iteration': [], 'value': []}
for interval in intervals:
        for commodity in commodities:
            for time_window in time_windows:
                true_path = '{}true_{}_{}_{}.csv'.format(path, interval, commodity, time_window)
                if true_path in all_files:
                    true = pd.read_csv(true_path)['0']
                for algorithm in tqdm(algorithms):
                    for i in range(10):
                        pred_path = '{}lstm_{}_{}_{}_{}_{}.csv'.format(path, algorithm, interval, commodity, time_window, i)
                        if pred_path in all_files:
                            pred = pd.read_csv(pred_path)['0']
                            for metric in metrics:
                                for edge_type in edge_types:
                                    results_df['metric'].append(metric)
                                    results_df['algorithm'].append(algorithm)
                                    results_df['interval'].append(interval)
                                    results_df['commodity'].append(commodity)
                                    results_df['time_window'].append(time_window)
                                    results_df['type'].append(edge_type)
                                    results_df['iteration'].append(i)
                                    results_df['value'].append(get_metric(metric, true, pred))
                        
results_df = pd.DataFrame(results_df)
results_df = results_df.groupby(by=['metric', 'algorithm', 'interval', 'commodity', 'time_window', 'type'], as_index=False).mean()
from datetime import datetime
results_df.to_csv('/media/pauloricardo/basement/commodities_usecase/new/results_{}.csv'.format(datetime.now()))

"""
import seaborn as sns
import matplotlib.pyplot as plt
from textwrap import wrap
from ephin_utils import decode_html_text

df = pd.read_csv('/media/pauloricardo/basement/projeto/df11-03.csv').drop(columns='Unnamed: 0').reset_index(drop=True)

results_df['algorithm'] = results_df['algorithm'].apply(lambda x: x if x != 'regularization' else 'ephin')
results_df['value'] = results_df['value'].apply(lambda x: x * 100)
results_df['split'] = results_df['split'].apply(lambda x: str(x * 100))

for idxt, target in enumerate(targets):
    results_filtered = results_df[results_df['target'] == target]
    for idxe, edge_type in enumerate(edge_types):
        types_filtered = results_filtered[results_filtered['type'] == edge_type]
        if types_filtered.shape[0] >= 1:
            plt.figure(idxt + idxe)
            ax = sns.lineplot(x="split", y="value", hue="algorithm", marker="o", data=types_filtered)
            ax.set_title("\n".join(wrap('event: ' + decode_html_text(df['text'].iloc[target]) + ' (event to ' + edge_type.split('_')[1] + ' link prediction)')), fontsize=18)
            ax.set_xlabel('removed (%)', fontsize=14)
            ax.set_ylabel('accuracy (%)', fontsize=14)
            ax.get_figure().set_size_inches(12,8)
            ax.get_figure().savefig('/media/pauloricardo/basement/projeto/line_graphs/line_' + str(target) + edge_type + '.pdf')"""
