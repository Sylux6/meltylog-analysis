import time as timelib
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as News_proportion
from sklearn.cluster import KMeans

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
import pathlib

##################################
##################################
##################################
####                          ####
#### Main                     ####
####                          ####
##################################
##################################

begin_time = timelib.time()

log_filename = "Outputs/MyLog.csv"
urls_filename = "Outputs/pages.csv"
session_filename = "Outputs/Sessions.csv"

pathlib.Path("Latex").mkdir(parents=True, exist_ok=True)
pathlib.Path("Graphs").mkdir(parents=True, exist_ok=True)
pathlib.Path("Clusters").mkdir(parents=True, exist_ok=True)
latex_output = open("Latex/latex_clusters.tex", "w")

###########
# VARIABLES
# dim1
# dimensions = ["requests", "timespan", "standard_deviation", "inter_req_mean_seconds", "read_pages"]
# dim2
# dimensions = ["star_chain_like", "bifurcation"]
# dim3
dimensions = ["popularity_mean", "entropy", "requested_category_richness", "requested_topic_richness", 'TV_proportion', 'Series_proportion', 'News_proportion', 'Celebrities_proportion', 'VideoGames_proportion', 'Music_proportion', 'Movies_proportion', 'Sport_proportion', 'Comic_proportion', 'Look_proportion', 'Other_proportion', 'Humor_proportion', 'Student_proportion', 'Events_proportion', 'Wellbeing_proportion', 'None_proportion', 'Food_proportion', 'Tech_proportion']
# dim1+dim2+dim3
# dimensions = ["requests", "timespan", "standard_deviation", "inter_req_mean_seconds", "read_pages", "star_chain_like", "bifurcation", "popularity_mean", "entropy", "requested_category_richness", "requested_topic_richness", 'TV_proportion', 'Series_proportion', 'News_proportion', 'Celebrities_proportion', 'VideoGames_proportion', 'Music_proportion', 'Movies_proportion', 'Sport_proportion', 'Comic_proportion', 'Look_proportion', 'Other_proportion', 'Humor_proportion', 'Student_proportion', 'Events_proportion', 'Wellbeing_proportion', 'None_proportion', 'Food_proportion', 'Tech_proportion']
NB_CLUSTERS = [4, 5, 6, 7, 8, 9, 10]
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
print("        Loading "+urls_filename+" ...", end="\r")
urls = pd.read_csv(urls_filename, sep=',', na_filter=False, low_memory=False)
print("        "+urls_filename+" loaded ({} rows) in {:.1f} seconds.".format(log.shape[0], timelib.time()-start_time))
start_time = timelib.time()
print("        Loading "+session_filename+" ...", end="\r")
sessions = pd.read_csv(session_filename, sep=',')
print("        "+session_filename+" loaded ({} rows) in {:.1f} seconds.".format(sessions.shape[0], timelib.time()-start_time))
sessions.fillna(0, inplace=True)

########
# FILTER
sessions = sessions[sessions.requests > 6]
print("\n   * Sessions filtered: {} rows".format(sessions.shape[0]))

normalized_dimensions = list(map(lambda x: "normalized_"+x, dimensions)) # normalized dimensions labels list

print("\n   > Elbow analysis: {}".format(elbow))
print("   > Session graph generation: {}".format(graph))
print("   > NB_CLUSTERS: {}".format(NB_CLUSTERS))

# # LaTeX init
# latex_output.write("\\documentclass[xcolor={dvipsnames}, handout]{beamer}\n")

latex_output.write("\\begin{frame}{Clustering}\n    Clustering on "+str(len(dimensions))+" dimensions:\n    \\begin{multicols}{2}\n        \\footnotesize{\n            \\begin{enumerate}\n")
for d in dimensions:
    latex_output.write("                \\item "+d.replace("_", "\_")+"\n")
latex_output.write("            \\end{enumerate}\n        }\n    \\end{multicols}\n\\end{frame}\n\n")

############
# CLUSTERING

topic_list = list(log.requested_topic.unique())
category_list = list(urls.category.unique())

for n in NB_CLUSTERS:
    start_time = timelib.time()
    print("\n   * Clustering ("+str(n)+" clusters) ...")
    pathlib.Path("Graphs/"+str(n)).mkdir(parents=True, exist_ok=True)
    pathlib.Path("Clusters/"+str(n)).mkdir(parents=True, exist_ok=True)
    kmeans = KMeans(n_clusters=n, random_state=0).fit(sessions[normalized_dimensions].values)
    cluster_labels=kmeans.labels_
    sessions["cluster_id"] = cluster_labels
    num_cluster = sessions.cluster_id.unique()
    num_cluster.sort()

    # compute centroids
    centroids = pd.DataFrame(columns=["cluster_id"] + dimensions)
    centroids["cluster_id"] = num_cluster
    for dim in dimensions:
        mean = []
        for cluster_id in num_cluster:
            mean.append(sessions[sessions.cluster_id==cluster_id][dim].mean())
        centroids[dim] = mean

    latex_output.write("\\begin{frame}{Clustering: "+str(n)+" clusters}\n    \\begin{center}\n        \\resizebox{\\textwidth}{!}{\n            \\begin{tabular}{ccccc}\n                \\includegraphics[width=\\textwidth, keepaspectratio]{Clusters/"+str(n)+"/cluster0}")
    for i in range(1, 10):
        if i >= n: # no clusters left
            break
        if i == 5: # second row
            latex_output.write(" \\\\\n                \\includegraphics[width=\\textwidth, keepaspectratio]{Clusters/"+str(n)+"/cluster5}")
            continue
        latex_output.write(" & \\includegraphics[width=\\textwidth, keepaspectratio]{Clusters/"+str(n)+"/cluster"+str(i)+"}")
    latex_output.write("\n            \\end{tabular}\n        }\n\n        \\begin{columns}\n            \\begin{column}{.65\\textwidth}\n                \\begin{center}\n                \\scalebox{0.25}{\n")

    # recap centroids
    latex_output.write("                    \\begin{tabular}{|c|")
    for cluster_id in num_cluster:
        latex_output.write("c|")
    latex_output.write("}\n                        \\hline\n                        ")
    for cluster_id in num_cluster:
        latex_output.write(" & "+str(cluster_id))
    latex_output.write(" \\\\\n                        \\hline\n                        size")
    for cluster_id in num_cluster:
        latex_output.write(" & "+str(sessions[sessions.cluster_id==cluster_id].shape[0]))
    latex_output.write(" \\\\\n                        \\hline\n")
    for dim in dimensions:
        latex_output.write("                        "+str(dim).replace("_", "\_"))
        for cluster_id in num_cluster:
            latex_output.write(" & {:.3f}".format(centroids[centroids.cluster_id==cluster_id][dim].values[0]))
        latex_output.write(" \\\\\n                        \\hline\n")
    latex_output.write("                    \\end{tabular}\n                }\n                \\end{center}\n            \\end{column}\n            \\begin{column}{.35\\textwidth}\n                \\includegraphics[width=\\textwidth, keepaspectratio]{Clusters/palette_topic}\n            \\end{column}\n        \\end{columns}\n    \\end{center}\n\\end{frame}\n\n")

    # display
    for cluster_id in num_cluster:
        cluster_session = sessions[sessions.cluster_id == cluster_id].global_session_id.unique()
        print("          Producing display of sessions for cluster %d"%cluster_id,end="\r") 
        cluster_sessions = sessions[sessions.cluster_id == cluster_id].global_session_id.unique()
        cluster_log = log[log.global_session_id.isin(cluster_sessions)]
        sessions_id = plot_sessions(cluster_log,'Clusters/'+str(n)+'/cluster%d.png'%cluster_id, cluster_id,
                    labels=list(log.requested_topic.unique()),
                    N_max_sessions=10,field="requested_topic",
                    max_time=None,time_resolution=None,mark_requests=False)
        if graph:
            session_draw(cluster_id, n, sessions_id, log, urls, category_list)
            latex_output.write("% cluster "+str(cluster_id)+"\n\\begin{frame}{Cluster "+str(cluster_id)+"}\n    \\begin{columns}\n        \\begin{column}{.6\\textwidth}\n            \\includegraphics[width=\\textwidth, keepaspectratio]{Clusters/"+str(n)+"/cluster"+str(cluster_id)+"}\n        \\end{column}\n        \\begin{column}{.4\\textwidth}\n            \\begin{center}\n              \\scalebox{.4}{\\begin{tabular}{|c|c|}\n                  \\hline\n                  \\multicolumn{2}{|c|}{mean} \\\\\n                  \\hline\n                  size & "+str(sessions[sessions.cluster_id==cluster_id].shape[0])+" \\\\\n                  \\hline\n")
            for dim in dimensions:
                latex_output.write("                  "+dim.replace("_", "\_")+" & {:.3f} \\\\\n                  \\hline\n".format(centroids[centroids.cluster_id==cluster_id][dim].values[0]))
            latex_output.write("              \\end{tabular}}\n\n              \\includegraphics[width=\\textwidth, keepaspectratio]{Clusters/palette_topic}\n            \\end{center}\n        \\end{column}\n    \\end{columns}\n\\end{frame}\n\n")

            latex_output.write("\\begin{frame}{Cluster "+str(cluster_id)+" -- Graphs}\n    \\resizebox{\\textwidth}{!}{\n    \\begin{tabular}{c|c|c|c|c}\n        \\huge{"+str(sessions_id[0])+"} & \\huge{"+str(sessions_id[1])+"} & \\huge{"+str(sessions_id[2])+"} & \\huge{"+str(sessions_id[3])+"} & \\huge{"+str(sessions_id[4])+"} \\\\\n        \\includegraphics[width=\\textwidth, keepaspectratio]{Graphs/"+str(n)+"/"+str(cluster_id)+"_session"+str(sessions_id[0])+"} & \\includegraphics[width=\\textwidth, keepaspectratio]{Graphs/"+str(n)+"/"+str(cluster_id)+"_session"+str(sessions_id[1])+"} & \\includegraphics[width=\\textwidth, keepaspectratio]{Graphs/"+str(n)+"/"+str(cluster_id)+"_session"+str(sessions_id[2])+"} & \\includegraphics[width=\\textwidth, keepaspectratio]{Graphs/"+str(n)+"/"+str(cluster_id)+"_session"+str(sessions_id[3])+"} & \\includegraphics[width=\\textwidth, keepaspectratio]{Graphs/"+str(n)+"/"+str(cluster_id)+"_session"+str(sessions_id[4])+"} \\\\\n        \\hline\n        \\huge{"+str(sessions_id[5])+"} & \\huge{"+str(sessions_id[6])+"} & \\huge{"+str(sessions_id[7])+"} & \\huge{"+str(sessions_id[8])+"} & \\huge{"+str(sessions_id[9])+"} \\\\\n        \\includegraphics[width=\\textwidth, keepaspectratio]{Graphs/"+str(n)+"/"+str(cluster_id)+"_session"+str(sessions_id[5])+"} & \\includegraphics[width=\\textwidth, keepaspectratio]{Graphs/"+str(n)+"/"+str(cluster_id)+"_session"+str(sessions_id[6])+"} & \\includegraphics[width=\\textwidth, keepaspectratio]{Graphs/"+str(n)+"/"+str(cluster_id)+"_session"+str(sessions_id[7])+"} & \\includegraphics[width=\\textwidth, keepaspectratio]{Graphs/"+str(n)+"/"+str(cluster_id)+"_session"+str(sessions_id[8])+"} & \\includegraphics[width=\\textwidth, keepaspectratio]{Graphs/"+str(n)+"/"+str(cluster_id)+"_session"+str(sessions_id[9])+"}\n    \\end{tabular}}\n\n")
             
            # recap centroids
            latex_output.write("    \\begin{columns}\n        \\begin{column}{.65\\textwidth}\n            \\begin{center}\n                \\scalebox{.25}{\n                    \\begin{tabular}{|c|")
            for cid in num_cluster:
                latex_output.write("c|")
            latex_output.write("}\n                        \\hline\n                        ")
            for cid in num_cluster:
                latex_output.write(" & "+str(cid))
            latex_output.write(" \\\\\n                        \\hline\n                        size")
            for cid in num_cluster:
                latex_output.write(" & "+str(sessions[sessions.cluster_id==cid].shape[0]))
            latex_output.write(" \\\\\n                        \\hline\n")
            for dim in dimensions:
                latex_output.write("                        "+str(dim).replace("_", "\_"))
                for cid in num_cluster:
                    latex_output.write(" & {:.3f}".format(centroids[centroids.cluster_id==cid][dim].values[0]))
                latex_output.write(" \\\\\n                        \\hline\n")
            latex_output.write("                    \\end{tabular}\n                }\n            \\end{center}\n        \\end{column}\n        \\begin{column}{.35\\textwidth}\n            \\includegraphics[width=\\textwidth, keepaspectratio]{Clusters/palette_category}\n        \\end{column}\n    \\end{columns}\n\\end{frame}\n\n")

        print("          Display of sessions succesfully produced for cluster %d"%cluster_id) 
    print("   * Clustered in {:.1f} seconds.".format((timelib.time()-start_time)))
plot_palette(labels=topic_list, filename="Clusters/palette_topic.png")
plot_palette(labels=category_list, filename="Clusters/palette_category.png")

# elbow analysis
if elbow:
    start_time = timelib.time()
    print("\n   * Computing elbow ...", end="\r")
    distorsions = []
    explore_N_clusters=40
    for k in range(2, explore_N_clusters):
        kmeans = KMeans(n_clusters=k).fit(sessions[normalized_dimensions].values)
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