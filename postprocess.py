import time as timelib
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
from log2traces import *
from markov import *
from information_theory import *
from log_functions import *

from plots import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pylab import *
from astropy.visualization import LogStretch
from astropy.visualization.mpl_normalize import ImageNormalize

from math import sqrt
from scipy import stats

##################################
##################################
##################################
####                          ####
#### Main                     ####
####                          ####
##################################
##################################

begin_time = timelib.time()

filename = "Outputs/Sessions.csv"

parameters = [
    "requests", "timespan", "requested_category_richness",
    "requested_topic_richness", "star_chain_like", "bifurcation", "entropy",
    "standard_deviation", "popularity_mean", "inter_req_mean_seconds",
    "TV_proportion", "Series_proportion", "News_proportion",
    "Celebrities_proportion", "VideoGames_proportion", "Music_proportion",
    "Movies_proportion", "Sport_proportion", "Comic_proportion",
    "Look_proportion", "Other_proportion", "Humor_proportion",
    "Student_proportion", "Events_proportion", "Wellbeing_proportion",
    "None_proportion", "Food_proportion", "Tech_proportion", "read_pages"
]

log_scale_parameters = ["requests", "timespan", "inter_req_mean_seconds", "standard_deviation", "read_pages"]

log_scale_parameters_y = ["entropy", "bifurcation", "TV_proportion","Celebrities_proportion","Series_proportion","Movies_proportion","Music_proportion","Unclassifiable_proportion","Comic_proportion","VideoGames_proportion","Other_proportion","Sport_proportion","News_proportion", "requested_category_richness", "requested_topic_richness", "popularity_mean"]

# for scatter plots
my_parameters = ["requests", "timespan", "standard_deviation", "star_chain_like", "bifurcation", "inter_req_mean_seconds"]

# for log normalization
lognorm = ["requests", "timespan", "inter_req_mean_seconds", "standard_deviation", "popularity_mean"]

# latex code generation output file
latex_output = open("Outputs/latex_dimensions.tex", "w")

# GENERATORS
histogen = False
scattergen = False
scatter_density = False
pearson = False

####################
# READING DATA FILES
start_time = timelib.time()
print("\n   * Loading "+filename+" ...", end="\r")
sessions = pd.read_csv(filename, sep=',')
print("   * "+filename+" loaded ({} rows) in {:.1f} seconds.".format(sessions.shape[0], timelib.time()-start_time))
sessions.fillna(0, inplace=True)

########
# FILTER
sessions = sessions[sessions.requests > 6]
sessions = sessions[sessions.timespan > 0]
sessions = sessions[sessions.inter_req_mean_seconds > 0]
print("\n   * Sessions filtered: {} rows".format(sessions.shape[0]))

# standard deviation
sessions["standard_deviation"] = sessions.variance.apply(lambda x: sqrt(x))

###############
# NORMALIZATION
start_time = timelib.time()
normalized_parameters = []
print("\n   * Normalizing parameters ...", end="\r")
for p in parameters:
    normalized_parameters.append("normalized_"+p)
    if p in lognorm:
        sessions["normalized_"+p] = log_normalize(sessions[p])
    else:
        sessions["normalized_"+p] = normalize(sessions[p])
print("   * Parameters normalized in %.1f seconds." %(timelib.time()-start_time))

#######################
# GENERATING HISTOGRAMS
print("\n   > Histograms: {}".format(histogen))
print("   > Scatter plots: {}".format(scattergen))
print("   > Density scatter plots: {}".format(scatter_density))
print("   > Pearson correlation: {}".format(pearson))

if histogen:
    start_time = timelib.time()
    print("\n   * Generating histograms ...", end="\r")
    for p in parameters:
        # regular histograms
        if p not in log_scale_parameters:
            plt.hist(sessions[p].values, align="left")
        plt.grid(alpha=0.5)
        plt.xlabel(p)
        plt.ylabel("Frequency")
        latex_output.write("\\begin{frame}\n\\frametitle{"+p.replace("_", "\_")+"}\n    \\begin{center}\n        \\begin{tabular}{>{\\centering\\arraybackslash}m{0.4\\linewidth}>{\\centering\\arraybackslash}m{0.15\\linewidth}>{\\centering\\arraybackslash}m{0.4\\linewidth}}\n            \\includegraphics[width=0.4\\textwidth, keepaspectratio]{plots/"+p+"} &")
        if p in lognorm:
            latex_output.write(" \\tiny{$\\xrightarrow[\\frac{\\log(x)-\\log(min)}{\\log(max)-\\log(min)}]{}$} &")
        else:
            latex_output.write(" $\\xrightarrow[\\frac{x-min}{max-min}]{}$ &")
        latex_output.write(" \\includegraphics[width=0.4\\textwidth, keepaspectratio]{plots/Normalized/normalized_"+p+"} \\\\\n")
        if p in log_scale_parameters:
            values = sessions[p].values
            min_value = values.min()
            max_value = values.max()
            if p == "read_pages":
                plt.hist(sessions[p].values, align="left", bins=np.logspace(0,np.log10(max_value), 10))
            else:
                plt.hist(sessions[p].values, align="left", bins=np.logspace(np.log10(min_value),np.log10(max_value), 20))
            plt.gca().set_xscale('log')
            plt.gca().set_yscale('log')
            latex_output.write("            log scale on x and y & & log scale on y\n")
        elif p in log_scale_parameters_y:
            plt.gca().set_yscale('log')
            latex_output.write("            log scale on y & & log scale on y\n")
        latex_output.write("        \\end{tabular}\n    \\end{center}\n\\end{frame}\n\n")
        plt.savefig("Matplot/"+p+".png",format='png')
        plt.clf()
        # normalized histograms
        bincuts=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
        plt.hist(sessions["normalized_"+p].values, bins=bincuts, align="left")
        plt.grid(alpha=0.5)
        plt.xlabel("Normalized "+p)
        plt.ylabel("Frequency")
        if p in log_scale_parameters or p in log_scale_parameters_y:
            plt.gca().set_yscale('log')
        plt.savefig("Matplot/Normalized/normalized_"+p+".png",format='png')
        plt.clf()
        plt.close()
    print("   * Histograms generated in %.1f seconds." %(timelib.time()-start_time))

##########################
# GENERATING SCATTER PLOTS
if scattergen:
    start_time = timelib.time()
    print("\n   * Generating scatter plots ...", end="\r")
    for i in range (0, len(my_parameters)):
        for j in range (i+1, len(my_parameters)):
            plt.scatter(sessions[my_parameters[i]].values, sessions[my_parameters[j]].values)
            plt.grid(alpha=0.5)
            plt.xlabel(my_parameters[i])
            plt.ylabel(my_parameters[j])
            if my_parameters[i] in log_scale_parameters:
                plt.gca().set_xscale('log')
            if my_parameters[j] in log_scale_parameters:
                plt.gca().set_yscale('log')
            plt.savefig("Matplot/Scatter/"+my_parameters[i]+"-VS-"+my_parameters[j]+".png",format='png')
            plt.clf()
            plt.scatter(sessions["normalized_"+my_parameters[i]].values, sessions["normalized_"+my_parameters[j]].values)
            plt.grid(alpha=0.5)
            plt.xlabel("Normalized "+my_parameters[i])
            plt.ylabel("Normalized "+my_parameters[j])
            plt.savefig("Matplot/Scatter/Normalized/normalized_"+my_parameters[i]+"-VS-normalized_"+my_parameters[j]+".png",format='png')
            plt.clf()
            plt.close()
    print("   * Scatter plots generated in %.1f seconds." %(timelib.time()-start_time))

##################################
# GENERATING DENSITY SCATTER PLOTS
if scatter_density:
    start_time = timelib.time()
    print("\n   * Generating scatter density ...", end="\r")
    latex_output.write("\n%%%%%%%%\n% SCATTER PLOTS\n%%%%%%%%\n\n")
    for i in range (0, len(my_parameters)):
        for j in range (i+1, len(my_parameters)):
            norm = ImageNormalize(vmin=0., vmax=1000, stretch=LogStretch())

            plt.hist2d(sessions["normalized_"+my_parameters[i]].values, sessions["normalized_"+my_parameters[j]].values, norm=norm, bins=(20,20))
            plt.colorbar(label="Number of points per pixel")
            plt.xlabel("normalized_"+my_parameters[i])
            plt.ylabel("normalized_"+my_parameters[j])
            plt.savefig("Matplot/Density/"+my_parameters[i]+"-VS-"+my_parameters[j]+".png",format='png')
            plt.clf()
            plt.close()

            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1, projection="scatter_density")
            density = ax.scatter_density(sessions["normalized_"+my_parameters[i]].values, sessions["normalized_"+my_parameters[j]].values, norm=norm)
            plt.xlabel("normalized_"+my_parameters[i])
            plt.ylabel("normalized_"+my_parameters[j])
            fig.colorbar(density, label="Number of points per pixel")
            plt.savefig("Matplot/Density/v2/"+my_parameters[i]+"-VS-"+my_parameters[j]+".png",format='png')
            plt.clf()

            latex_output.write("\\begin{frame}\n\\frametitle{"+my_parameters[i].replace("_", "\_")+" VS "+my_parameters[j].replace("_", "\_")+"}\n    \\begin{center}\n        \\includegraphics[scale=0.3]{plots/Scatter/"+my_parameters[i]+"-VS-"+my_parameters[j]+"}\n    \\end{center}\n    \\begin{tabular}{cc}\n        \\includegraphics[scale=0.3]{plots/Density/"+my_parameters[i]+"-VS-"+my_parameters[j]+"} & \\includegraphics[scale=0.3]{plots/Density/v2/"+my_parameters[i]+"-VS-"+my_parameters[j]+"}\n    \\end{tabular}\n\\end{frame}\n\n")
    print("   * Scatter density generated in %.1f seconds." %(timelib.time()-start_time))

#########
# PEARSON
if pearson:
    start_time = timelib.time()
    print("\n   * Computing Pearson correlation ...", end="\r")
    latex_output.write("\n%%%%%%%%\n% PEARSON\n%%%%%%%%\n\n\\begin{frame}{Pearson}\n    \\begin{center}\n        \\resizebox{\\textwidth}{!}{\n        \\begin{tabular}{|c|c|c|c|c|c|c|}\n            \\hline\n            correlation")
    for i in range (0, len(my_parameters)):
        latex_output.write(" & "+my_parameters[i].replace("_", "\_"))
    latex_output.write(" \\\\\n            \\hline")
    pearson_mat = np.zeros((len(my_parameters), len(my_parameters)))
    for i in range (0, len(my_parameters)):
        latex_output.write("\n"+my_parameters[i].replace("_", "\_"))
        for j in range (0, len(my_parameters)):
            val = stats.pearson_matr(sessions[my_parameters[i]].values, sessions[my_parameters[j]].values)[0]
            pearson_mat[i][j] = val
            latex_output.write(" & {:.3f}".format(val))
        latex_output.write(" \\\\\n            \\hline")
    latex_output.write("\n        \\end{tabular}}\n    \\end{center}\n\\end{frame}")
    print("   * Pearson correlation computed in {:.1f} seconds.".format(timelib.time()-begin_time))

###############
# END OF SCRIPT
print("\n   * Done in {:.1f} seconds.\n".format(timelib.time()-begin_time))
