import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import matplotlib

from math import sqrt
SPINE_COLOR = 'gray'

def latexify(fig_width=None, fig_height=None, columns=1):
    """Set up matplotlib's RC params for LaTeX plotting.
    Call this before plotting a figure.

    Parameters
    ----------
    fig_width : float, optional, inches
    fig_height : float,  optional, inches
    columns : {1, 2}
    """

    # code adapted from http://www.scipy.org/Cookbook/Matplotlib/LaTeX_Examples

    # Width and max height in inches for IEEE journals taken from
    # computer.org/cms/Computer.org/Journal%20templates/transactions_art_guide.pdf

    assert(columns in [1,2])

    if fig_width is None:
        fig_width = 3.39 if columns==1 else 6.9 # width in inches

    if fig_height is None:
        golden_mean = (sqrt(5)-1.0)/2.0    # Aesthetic ratio
        fig_height = fig_width*golden_mean # height in inches

    MAX_HEIGHT_INCHES = 8.0
    if fig_height > MAX_HEIGHT_INCHES:
        print("WARNING: fig_height too large:" + fig_height + 
              "so will reduce to" + MAX_HEIGHT_INCHES + "inches.")
        fig_height = MAX_HEIGHT_INCHES

    params = {'backend': 'ps',
              'text.latex.preamble': [r'\usepackage{gensymb}'],
              'axes.labelsize': 9, # fontsize for x and y labels (was 10)
              'axes.titlesize': 9,
              'font.size': 9, # was 10
              'legend.fontsize': 7, # was 10
              'xtick.labelsize': 9,
              'ytick.labelsize': 9,
              'text.usetex': True,
              'figure.figsize': [fig_width,fig_height],
              'font.family': 'serif'
    }

    matplotlib.rcParams.update(params)


def format_axes(ax):

    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color(SPINE_COLOR)
        ax.spines[spine].set_linewidth(0.5)

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_tick_params(direction='out', color=SPINE_COLOR)

    return ax

# cols = ['long', 'lat', 'W', 'T', 'H', 'WS', 'WD', 'time']
# lls = {0: 1.2278131, 1: 0.85081714, 2: 6.74952, 3: 3.2240736, 4: 2.5545661, 5: 3.8056831, 6: 7.780356, 7: 3.7107303}
# lls_no_ard = 0.7346
# latexify(fig_width=3.32,fig_height=1.5)
# ax = plt.figure().subplots()
# ax.bar(cols, list(lls.values()), width = 0.2,label='ARD')
# ax.axhline(y=lls_no_ard, color='r', linestyle='-',label='Non-ARD',lw=1)
# ax.set_ylabel("Learnt\nLength Scale")
# ax.set_xlabel("Features")
# # ax.set_xticklabels(cols, Rotation=90) 
# # ax.set_title("Best Stationary GP")
# plt.legend(loc='upper center')
# plt.tight_layout()
# format_axes(ax)
# plt.savefig("lls_comparison.pdf")

# FOR COMPLETE PLOT
fold = 0
test_input = pd.read_csv('../data/time_feature'+'/fold'+str(fold)+'/test_data_'+'mar_nsgp'+'.csv.gz')

test_input.time = pd.to_datetime(test_input.time)
# for station in test_input.station_id.unique():
station = test_input.station_id.unique()[-5]
rows = test_input[test_input['station_id']==station].index
Xs = test_input.time[rows]
latexify(fig_width=3.32, fig_height=2.4)
ax = plt.figure().subplots(3,1)

SGP = pd.read_csv('../local_p_with_matern12_fold_0.csv.gz')
a = ax[0].plot(Xs,np.array(SGP.pred_PM25)[rows],c='tab:orange',label="Predicted",linewidth=0.5)
    
b = ax[0].plot(Xs,np.array(test_input['PM25_Concentration'])[rows],c='tab:blue',label="Ground Truth",linewidth=0.5)
ax[0].set_title("a)")
# ax[0].set_xlabel("Timeline")
# ax[0].set_ylabel("PM2.5")
ax[0].set_ylim(0,350)
ax[0].set_xticks([])
# plt.savefig("sgp.png")

XGB = pd.read_csv('../ml_results/fold0/XGB_scaled.csv.gz')
# plt.figure(figsize=(30,7))
a = ax[1].plot(Xs,np.array(XGB.prediction)[rows],c='tab:orange',label="Predicted",linewidth=0.5)
b = ax[1].plot(Xs,np.array(test_input['PM25_Concentration'])[rows],c='tab:blue',label="Ground Truth",linewidth=0.5)
ax[1].legend(loc='lower left',bbox_to_anchor=(0.6,0.5))
ax[1].set_title("b)")
# ax[1].set_xlabel("Timeline")
ax[1].set_ylabel("PM2.5")
ax[1].set_ylim(0,350)
ax[1].set_xticks([])
# plt.savefig("xgb.png")

RF = pd.read_csv('../ml_results/fold0/RF_scaled.csv.gz')
# plt.figure(figsize=(30,7))
a = ax[2].plot(Xs,np.array(RF.prediction)[rows],c='tab:orange',label="Predicted",linewidth=0.5)
b = ax[2].plot(Xs,np.array(test_input['PM25_Concentration'])[rows],c='tab:blue',label="Ground Truth",linewidth=0.5)
# ax[2].legend()
ax[2].set_title("c)")
ax[2].set_xlabel("Timeline (March 2015)")
# ax[2].set_ylabel("PM2.5")
ax[2].set_ylim(0,350)
ax[2].set_xticks([])
plt.tight_layout()
format_axes(ax[0])
format_axes(ax[1])
format_axes(ax[2])

plt.savefig("comp_{}.pdf".format(station))

