import time as timelib
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
from collections import Counter

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from thema_mapper import *
from log2traces import *
from markov import *
from information_theory import *
from log_functions import *
from graph import *

from plots import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pylab import *
import seaborn as sns
import mpl_scatter_density
import matplotlib.colors as colors
from matplotlib.mlab import bivariate_normal
from astropy.visualization import LogStretch
from astropy.visualization.mpl_normalize import ImageNormalize


##################################
##################################
##################################
####                          ####
#### Main                     ####
####                          ####
##################################
##################################

begin_time = timelib.time()

log_filename = "Files/MyLog.csv"
url_data_filename = "Files/MyURLs.csv"
filename = "Outputs/Sessions.csv"

# dimensions = ["requests", "timespan", "inter_req_mean_seconds", "standard_deviation"]
dimensions = ["requests","timespan","requested_category_richness","requested_my_thema_richness","star_chain_like","bifurcation","entropy","standard_deviation","popularity_mean","inter_req_mean_seconds","TV_proportion","Celebrities_proportion","Series_proportion","Movies_proportion","Music_proportion","Unclassifiable_proportion","Comic_proportion","VideoGames_proportion","Other_proportion","Sport_proportion","News_proportion","read_pages"]
latex_output = open("Outputs/latex_clusters.tex", "w")
elbow = False
graph = True

####################
# READING DATA FILES
print("\n   * Loading files ...")
start_time = timelib.time()
print("        Loading "+log_filename+" ...", end="\r")
log = pd.read_csv(log_filename, sep=',', na_filter=False, low_memory=False)
print("        "+log_filename+" loaded ({} rows) in {:.1f} seconds.".format(log.shape[0], timelib.time()-start_time))
start_time = timelib.time()
print("        Loading "+filename+" ...", end="\r")
sessions = pd.read_csv(filename, sep=',')
print("        "+filename+" loaded ({} rows) in {:.1f} seconds.".format(sessions.shape[0], timelib.time()-start_time))
sessions.fillna(0, inplace=True)
start_time = timelib.time()
print("        Loading "+url_data_filename+" ...", end="\r")
urldata = pd.read_csv(url_data_filename, sep=',', na_filter=False)
print("        "+url_data_filename+" loaded ({} rows) in {:.1f} seconds.".format(urldata.shape[0], timelib.time()-start_time))


########
# FILTER
sessions = sessions[sessions.requests > 6]
sessions = sessions[sessions.variance > 0]
sessions = sessions[sessions.inter_req_mean_seconds > 0]
print("\n   * Sessions filtered: {} rows".format(sessions.shape[0]))

# standard deviation
sessions["standard_deviation"] = sessions.variance.apply(lambda x: sqrt(x))

print("\n   > Elbow analysis: {}".format(elbow))
print("   > Session graph generation: {}".format(graph))

############
# CLUSTERING
start_time = timelib.time()
print("\n   * Clustering ...")
kmeans = KMeans(n_clusters=10, random_state=0).fit(sessions[dimensions].values)
cluster_labels=kmeans.labels_
sessions["cluster_id"] = cluster_labels

num_cluster = sessions.cluster_id.unique()
num_cluster.sort()

# recap center
latex_output.write("% RECAP\n\\begin{tabular}{|c|")
for dim in dimensions:
    latex_output.write("c|")
latex_output.write("}\n    \\hline\n    cluster")
for dim in dimensions:
    latex_output.write(" & "+str(dim).replace("_", "\_"))
latex_output.write(" \\\\\n    \\hline\n")
for cluster_id in num_cluster:
    latex_output.write("    "+str(cluster_id))
    for i in range(0, kmeans.cluster_centers_.shape[1]):
        latex_output.write(" & {:.3f}".format(kmeans.cluster_centers_[cluster_id][i]))
    latex_output.write(" \\\\\n    \\hline\n")
latex_output.write("\\end{tabular}\n\n")

# display
for cluster_id in num_cluster:
    cluster_session = sessions[sessions.cluster_id == cluster_id].global_session_id.unique()
    print("        Producing display of sessions for cluster %d"%cluster_id) 
    cluster_sessions = sessions[sessions.cluster_id == cluster_id].global_session_id.unique()
    cluster_log = log[log.global_session_id.isin(cluster_sessions)]
    sessions_id = plot_sessions(cluster_log,'Clusters/cluster%d.png'%cluster_id, cluster_id, 
                  labels=list(log.requested_my_thema.unique()),
                  N_max_sessions=10,field="requested_my_thema",
                  max_time=None,time_resolution=None,mark_requests=False)
    if graph:
        print("          Generating session graphs ...", end="\r")
        session_draw(cluster_id, sessions_id, log)
        print("          Session graphs successfully generated")
        latex_output.write("% cluster "+str(cluster_id)+"\n\\begin{frame}{Cluster "+str(cluster_id)+"}\n    \\begin{columns}\n        \\begin{column}{.6\\textwidth}\n            \\includegraphics[width=\\textwidth, keepaspectratio]{clusters/cluster"+str(cluster_id)+"}\n        \\end{column}\n        \\begin{column}{.4\\textwidth}\n            \\begin{center}\n              \\scalebox{.5}{\\begin{tabular}{|c|c|}\n                  \\hline\n                  \\multicolumn{2}{|c|}{mean} \\\\\n                  \\hline\n")
        for i in range(0, kmeans.cluster_centers_.shape[1]):
            latex_output.write("                  "+dimensions[i].replace("_", "\_")+" & {:.3f} \\\\\n                  \\hline\n".format(kmeans.cluster_centers_[cluster_id][i]))
        latex_output.write("              \\end{tabular}}\n\n              \\includegraphics[width=\\textwidth, keepaspectratio]{clusters/palette}\n            \\end{center}\n        \\end{column}\n    \\end{columns}\n\\end{frame}\n\n")

        latex_output.write("\\begin{frame}{Cluster "+str(cluster_id)+" -- Graphs}\n    \\resizebox{\\textwidth}{!}{\n    \\begin{tabular}{c|c|c|c|c}\n        \\huge{"+str(sessions_id[0])+"} & \\huge{"+str(sessions_id[1])+"} & \\huge{"+str(sessions_id[2])+"} & \\huge{"+str(sessions_id[3])+"} & \\huge{"+str(sessions_id[4])+"} \\\\\n        \\includegraphics[width=\\textwidth, keepaspectratio]{graphs/"+str(cluster_id)+"_session"+str(sessions_id[0])+"} & \\includegraphics[width=\\textwidth, keepaspectratio]{graphs/"+str(cluster_id)+"_session"+str(sessions_id[1])+"} & \\includegraphics[width=\\textwidth, keepaspectratio]{graphs/"+str(cluster_id)+"_session"+str(sessions_id[2])+"} & \\includegraphics[width=\\textwidth, keepaspectratio]{graphs/"+str(cluster_id)+"_session"+str(sessions_id[3])+"} & \\includegraphics[width=\\textwidth, keepaspectratio]{graphs/"+str(cluster_id)+"_session"+str(sessions_id[4])+"} \\\\\n        \\hline\n        \\huge{"+str(sessions_id[5])+"} & \\huge{"+str(sessions_id[6])+"} & \\huge{"+str(sessions_id[7])+"} & \\huge{"+str(sessions_id[8])+"} & \\huge{"+str(sessions_id[9])+"} \\\\\n        \\includegraphics[width=\\textwidth, keepaspectratio]{graphs/"+str(cluster_id)+"_session"+str(sessions_id[5])+"} & \\includegraphics[width=\\textwidth, keepaspectratio]{graphs/"+str(cluster_id)+"_session"+str(sessions_id[6])+"} & \\includegraphics[width=\\textwidth, keepaspectratio]{graphs/"+str(cluster_id)+"_session"+str(sessions_id[7])+"} & \\includegraphics[width=\\textwidth, keepaspectratio]{graphs/"+str(cluster_id)+"_session"+str(sessions_id[8])+"} & \\includegraphics[width=\\textwidth, keepaspectratio]{graphs/"+str(cluster_id)+"_session"+str(sessions_id[9])+"}\n    \\end{tabular}}\n\\end{frame}\n\n")
plot_palette(labels=list(log.requested_my_thema.unique()), filename="Clusters/palette.png")
print("     Clustered in {:.1f} seconds.".format((timelib.time()-start_time)))

# elbow analysis
if elbow:
    start_time = timelib.time()
    print("\n   * Computing elbow ...", end="\r")
    distorsions = []
    explore_N_clusters=40
    for k in range(2, explore_N_clusters):
        kmeans = KMeans(n_clusters=k).fit(sessions[dimensions].values)
        distorsions.append(kmeans.inertia_)
    fig = plt.figure(figsize=(15, 5))
    plt.plot(range(2, explore_N_clusters), distorsions)
    plt.grid(True)
    plt.title('Elbow curve')
    plt.savefig('Clusters/elbow.png', format='png')
    plt.clf()
    plt.close()
    print("     Elbow computed in {:.1f} seconds.".format((timelib.time()-start_time)))

###############
# END OF SCRIPT
print("\n   * Done in {:.1f} seconds.\n".format(timelib.time()-begin_time)) 