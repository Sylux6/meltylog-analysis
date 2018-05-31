import time as timelib
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score

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
import shutil

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

shutil.rmtree("Latex", ignore_errors=True)
pathlib.Path("Latex").mkdir(parents=True, exist_ok=True)
pathlib.Path("Latex/Graphs").mkdir(parents=True, exist_ok=True)
pathlib.Path("Latex/Clusters").mkdir(parents=True, exist_ok=True)
pathlib.Path("Latex/pca").mkdir(parents=True, exist_ok=True)
pathlib.Path("Latex/pca/pairwise").mkdir(parents=True, exist_ok=True)
pathlib.Path("Latex/silhouette").mkdir(parents=True, exist_ok=True)
latex_output = open("Latex/latex_clusters.tex", "w")
print("\n   * 'Latex' directory created.")

###########
# VARIABLES
# dim1
dimensions_1 = ["requests", "timespan", "standard_deviation", "inter_req_mean_seconds"]
# dim2
dimensions_2 = ["star_chain_like"]

if len(list(set(dimensions_1) & set(dimensions_2))) > 0:
    print("Error: intersection dimensions not empty")
    quit()

lognorm = ["requests", "timespan", "inter_req_mean_seconds", "standard_deviation", "popularity_mean", "variance"]
n_clusters_1 = 3
n_clusters_2 = 3

###############################################################################
# READING DATA FILES
print("\n   * Loading files ...")
start_time = timelib.time()
print("        Loading "+log_filename+" ...", end="\r")
log = pd.read_csv(log_filename, sep=',', na_filter=False, low_memory=False)
print("        "+log_filename+" loaded ({} rows) in {:.1f} seconds.".format(log.shape[0], timelib.time()-start_time))
start_time = timelib.time()
print("        Loading "+urls_filename+" ...", end="\r")
urls = pd.read_csv(urls_filename, sep=',', na_filter=False, low_memory=False)
print("        "+urls_filename+" loaded ({} rows) in {:.1f} seconds.".format(urls.shape[0], timelib.time()-start_time))
start_time = timelib.time()
print("        Loading "+session_filename+" ...", end="\r")
sessions = pd.read_csv(session_filename, sep=',')
print("        "+session_filename+" loaded ({} rows) in {:.1f} seconds.".format(sessions.shape[0], timelib.time()-start_time))
sessions.fillna(0, inplace=True)

###############################################################################
# FILTER
sessions = sessions[sessions.requests >= 6]
for d in lognorm:
    sessions = sessions[sessions[d] > 0]
print("\n   * Sessions filtered: {} rows".format(sessions.shape[0]))

normalized_dimensions_1 = list(map(lambda x: "normalized_"+x, dimensions_1)) # normalized dimensions labels list
weighted_dimensions_1 = list(map(lambda x: "weighted_"+x, dimensions_1)) # weighted dimensions labels list
normalized_dimensions_2 = list(map(lambda x: "normalized_"+x, dimensions_2)) # normalized dimensions labels list
weighted_dimensions_2 = list(map(lambda x: "weighted_"+x, dimensions_2)) # weighted dimensions labels list

print("   > n_clusters_1: {}".format(n_clusters_1))
print("   > n_clusters_2: {}".format(n_clusters_2))

###############################################################################
# NORMALIZATION
start_time = timelib.time()
print("\n   * Normalizing dimensions ...", end="\r")
for d in dimensions_1:
    if d in lognorm:
        sessions["normalized_"+d] = sessions[d]+1 # here we need to shift our data for log normalization
        sessions["normalized_"+d] = sessions["normalized_"+d].apply(lambda x: math.log(x))
    else:
        sessions["normalized_"+d] = sessions[d]
scaler = StandardScaler()
sessions[normalized_dimensions_1] = scaler.fit_transform(sessions[normalized_dimensions_1])

for d in dimensions_2:
    if d in lognorm:
        sessions["normalized_"+d] = sessions[d]+1 # here we need to shift our data for log normalization
        sessions["normalized_"+d] = sessions["normalized_"+d].apply(lambda x: math.log(x))
    else:
        sessions["normalized_"+d] = sessions[d]
scaler = StandardScaler()
sessions[normalized_dimensions_2] = scaler.fit_transform(sessions[normalized_dimensions_2])
print("   * Dimensions normalized in %.1f seconds." %(timelib.time()-start_time))

###############################################################################
# WEIGHTS
start_time = timelib.time()
print("\n   * Weighting dimensions ...", end="\r")
cv = {}
cv_tot = 0
w = {}
for d in dimensions_1:
    cv[d] = sqrt(sessions[d].var()) / sessions[d].mean()
    cv_tot = cv_tot + cv[d]
for d in dimensions_1:
    w[d] = cv[d] / cv_tot
for d in dimensions_1:
    sessions["weighted_"+d] = sqrt(w[d]) * sessions["normalized_"+d]

cv_tot = 0
for d in dimensions_2:
    cv[d] = sqrt(sessions[d].var()) / sessions[d].mean()
    cv_tot = cv_tot + cv[d]
for d in dimensions_2:
    w[d] = cv[d] / cv_tot
for d in dimensions_2:
    sessions["weighted_"+d] = sqrt(w[d]) * sessions["normalized_"+d]
print("   * Dimensions weighted in {:.1f} seconds.".format((timelib.time()-start_time)))

latex_output.write("\\documentclass[xcolor={dvipsnames}, handout]{beamer}\n\n\\usetheme{Warsaw}\n\\usepackage[utf8]{inputenc}\n\\usepackage[T1]{fontenc}\n\\usepackage{graphicx}\n\\usepackage[english]{babel}\n\\usepackage{amsmath}\n\\usepackage{amssymb}\n\\usepackage{mathrsfs}\n\\usepackage{verbatim}\n\\usepackage{lmodern}\n\\usepackage{listings}\n\\usepackage{caption}\n\\usepackage{multicol}\n\\usepackage{epsfig}\n\\usepackage{array}\n\\usepackage{tikz}\n\\usepackage{collcell}\n\n\\definecolor{mygreen}{rgb}{0,0.6,0}\n\\setbeamertemplate{headline}{}{}\n\\addtobeamertemplate{footline}{\insertframenumber/\inserttotalframenumber}\n\n\\title{Melty Clusterization}\n\\author{Sylvain Ung}\n\\institute{Laboratoire d'informatique de Paris 6}\n\\date{\\today}\n\n\\begin{document}\n\\setbeamertemplate{section page}\n{\n  \\begin{centering}\n    \\vskip1em\\par\n    \\begin{beamercolorbox}[sep=4pt,center]{part title}\n      \\usebeamerfont{section title}\\insertsection\\par\n    \\end{beamercolorbox}\n  \\end{centering}\n}\n\n\\begin{frame}\n    \\titlepage\n\\end{frame}\n\n")

latex_output.write("\\begin{frame}{Clustering}\n    Clustering on "+str(len(dimensions_1))+" dimension(s) ("+str(n_clusters_1)+" clusters):\n    \\begin{multicols}{2}\n        \\footnotesize{\n            \\begin{enumerate}\n")
for d in dimensions_1:
    latex_output.write("                \\item "+d.replace("_", "\_")+"\n")
latex_output.write("            \\end{enumerate}\n        }\n    \\end{multicols}\n    Clustering on "+str(len(dimensions_2))+" dimension(s) ("+str(n_clusters_2)+" clusters):\n    \\begin{multicols}{2}\n        \\footnotesize{\n            \\begin{enumerate}")
for d in dimensions_2:
    latex_output.write("                \\item "+d.replace("_", "\_")+"\n")
latex_output.write("            \\end{enumerate}\n        }\n    \\end{multicols}\n\\end{frame}\n\n")

###############################################################################
# CLUSTERING

topic_list = list(log.requested_topic.unique())
category_list = list(urls.category.unique())

start_time = timelib.time()
print("\n   * Clustering ("+str(n_clusters_1)+"x"+str(n_clusters_2)+" clusters) ...")
pathlib.Path("Latex/Clusters/"+str(n_clusters_1)+"x"+str(n_clusters_2)).mkdir(parents=True, exist_ok=True)

kmeans1 = KMeans(n_clusters=n_clusters_1, random_state=0).fit(sessions[weighted_dimensions_1].values)
kmeans2 = KMeans(n_clusters=n_clusters_2, random_state=0).fit(sessions[weighted_dimensions_2].values)
sessions["cluster_id_1"] = kmeans1.labels_
sessions["cluster_id_2"] = kmeans2.labels_

cid_1 = sessions.cluster_id_1.unique()
cid_2 = sessions.cluster_id_2.unique()
cid_1.sort()
cid_2.sort()
gcid_dic = {}
gcid = 0
for i in cid_1:
    for j in cid_2:
        gcid_dic[str(i)+str(j)] = gcid
        gcid = gcid + 1
sessions["global_cluster_id"] = sessions["cluster_id_1"].map(str) + sessions["cluster_id_2"].map(str)
sessions["global_cluster_id"] = sessions["global_cluster_id"].map(gcid_dic)

num_cluster = sessions.global_cluster_id.unique()
num_cluster.sort()

# compute centroids for recap
centroids = pd.DataFrame(columns=["global_cluster_id"] + dimensions_1 + dimensions_2)
centroids["global_cluster_id"] = num_cluster
for dim in dimensions_1 + dimensions_2:
    mean = []
    for cluster_id in num_cluster:
        mean.append(sessions[sessions.global_cluster_id==cluster_id][dim].mean())
    centroids[dim] = mean

# generating cluster mosaic
ordered_list = list()
span = {}
tmp = centroids.sort_values("timespan")
ordered_list = ordered_list + list((tmp.iloc[:3,:].sort_values("star_chain_like").global_cluster_id.values))
ordered_list = ordered_list + list((tmp.iloc[3:6,:].sort_values("star_chain_like").global_cluster_id.values))
ordered_list = ordered_list + list((tmp.iloc[6:9,:].sort_values("star_chain_like").global_cluster_id.values))
for cluster_id in ordered_list:
    span[cluster_id] = sessions[sessions.global_cluster_id==cluster_id].timespan.max()
for i in range(0, len(ordered_list)):
    if i < 3:
        span[ordered_list[i]] = max(span[ordered_list[0]], span[ordered_list[1]], span[ordered_list[2]])
    elif i < 6:
        span[ordered_list[i]] = max(span[ordered_list[3]], span[ordered_list[4]], span[ordered_list[5]])
    else:
        span[ordered_list[i]] = max(span[ordered_list[6]], span[ordered_list[7]], span[ordered_list[8]])
for cluster_id in ordered_list:
    cluster_sessions = sessions[sessions.global_cluster_id == cluster_id].global_session_id.unique()
    cluster_log = log[log.global_session_id.isin(cluster_sessions)]
    plot_sessions_bis(cluster_log, 'Latex/Clusters/'+str(n_clusters_1)+"x"+str(n_clusters_2)+'/_cluster%d.png'%cluster_id, cluster_id, N_max_sessions=5, max_time=span[cluster_id], time_resolution=None, mark_requests=False)
latex_output.write("\\begin{frame}{Cluster mosaic}\n    \\begin{center}\n        \\resizebox{\\textwidth}{!}{\n            \\begin{tabular}{ccc}\n                \\includegraphics[width=\\textwidth, keepaspectratio]{Clusters/"+str(n_clusters_1)+"x"+str(n_clusters_2)+"/_cluster"+str(ordered_list[0])+".png} & \\includegraphics[width=\\textwidth, keepaspectratio]{Clusters/"+str(n_clusters_1)+"x"+str(n_clusters_2)+"/_cluster"+str(ordered_list[1])+".png} & \\includegraphics[width=\\textwidth, keepaspectratio]{Clusters/"+str(n_clusters_1)+"x"+str(n_clusters_2)+"/_cluster"+str(ordered_list[2])+".png} \\\\\n                \\includegraphics[width=\\textwidth, keepaspectratio]{Clusters/"+str(n_clusters_1)+"x"+str(n_clusters_2)+"/_cluster"+str(ordered_list[3])+".png} & \\includegraphics[width=\\textwidth, keepaspectratio]{Clusters/"+str(n_clusters_1)+"x"+str(n_clusters_2)+"/_cluster"+str(ordered_list[4])+".png} & \\includegraphics[width=\\textwidth, keepaspectratio]{Clusters/"+str(n_clusters_1)+"x"+str(n_clusters_2)+"/_cluster"+str(ordered_list[5])+".png} \\\\\n                \\includegraphics[width=\\textwidth, keepaspectratio]{Clusters/"+str(n_clusters_1)+"x"+str(n_clusters_2)+"/_cluster"+str(ordered_list[6])+".png} & \\includegraphics[width=\\textwidth, keepaspectratio]{Clusters/"+str(n_clusters_1)+"x"+str(n_clusters_2)+"/_cluster"+str(ordered_list[7])+".png} & \\includegraphics[width=\\textwidth, keepaspectratio]{Clusters/"+str(n_clusters_1)+"x"+str(n_clusters_2)+"/_cluster"+str(ordered_list[8])+".png}\n            \\end{tabular}\n        }\n    \\end{center}\n\\end{frame}\n\n")

if n_clusters_1*n_clusters_2 > 10:
    resizebox = ".8"
else:
    resizebox = ""
latex_output.write("\\begin{frame}{Clustering: "+str(n_clusters_1)+"x"+str(n_clusters_2)+" clusters}\n    \\begin{center}\n        \\resizebox{"+resizebox+"\\textwidth}{!}{\n            \\begin{tabular}{ccccc}\n                \\includegraphics[width=\\textwidth, keepaspectratio]{Clusters/"+str(n_clusters_1)+"x"+str(n_clusters_2)+"/cluster0}")
for i in range(1, 15):
    if i >= n_clusters_1*n_clusters_2: # no clusters left
        break
    if i == 5: # second row
        latex_output.write(" \\\\\n                \\includegraphics[width=\\textwidth, keepaspectratio]{Clusters/"+str(n_clusters_1)+"x"+str(n_clusters_2)+"/cluster5}")
        continue
    elif i == 10: # second row
        latex_output.write(" \\\\\n                \\includegraphics[width=\\textwidth, keepaspectratio]{Clusters/"+str(n_clusters_1)+"x"+str(n_clusters_2)+"/cluster10}")
        continue
    latex_output.write(" & \\includegraphics[width=\\textwidth, keepaspectratio]{Clusters/"+str(n_clusters_1)+"x"+str(n_clusters_2)+"/cluster"+str(i)+"}")
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
    latex_output.write(" & "+str(sessions[sessions.global_cluster_id==cluster_id].shape[0]))
latex_output.write(" \\\\\n                        \\hline\n")
for dim in dimensions_1 + dimensions_2:
    latex_output.write("                        "+str(dim).replace("_", "\_"))
    for cluster_id in num_cluster:
        latex_output.write(" & {:.3f}".format(centroids[centroids.global_cluster_id==cluster_id][dim].values[0]))
    latex_output.write(" \\\\\n                        \\hline\n")
latex_output.write("                    \\end{tabular}\n                }\n                \\end{center}\n            \\end{column}\n            \\begin{column}{.35\\textwidth}\n                \\includegraphics[width=\\textwidth, keepaspectratio]{Clusters/palette_topic}\n            \\end{column}\n        \\end{columns}\n    \\end{center}\n\\end{frame}\n\n")

# boxplot


# display
for cluster_id in num_cluster:
    print("          Producing display of sessions for cluster %d"%cluster_id,end="\r") 
    cluster_sessions = sessions[sessions.global_cluster_id == cluster_id].global_session_id.unique()
    cluster_log = log[log.global_session_id.isin(cluster_sessions)]
    sessions_id = plot_sessions(cluster_log,'Latex/Clusters/'+str(n_clusters_1)+"x"+str(n_clusters_2)+'/cluster%d.png'%cluster_id, cluster_id, labels=list(log.requested_topic.unique()), N_max_sessions=10,field="requested_topic", max_time=None,time_resolution=None,mark_requests=False)

    # graph
    session_draw(cluster_id, sessions_id, log, urls, category_list)
    latex_output.write("% cluster "+str(cluster_id)+"\n\\begin{frame}{Cluster "+str(cluster_id)+"}\n    \\begin{columns}\n        \\begin{column}{.6\\textwidth}\n            \\includegraphics[width=\\textwidth, keepaspectratio]{Clusters/"+str(n_clusters_1)+"x"+str(n_clusters_2)+"/cluster"+str(cluster_id)+"}\n        \\end{column}\n        \\begin{column}{.4\\textwidth}\n            \\begin{center}\n              \\scalebox{.4}{\\begin{tabular}{|c|c|}\n                  \\hline\n                  \\multicolumn{2}{|c|}{mean} \\\\\n                  \\hline\n                  size & "+str(sessions[sessions.global_cluster_id==cluster_id].shape[0])+" \\\\\n                  \\hline\n")
    for dim in dimensions_1 + dimensions_2:
        latex_output.write("                  "+dim.replace("_", "\_")+" & {:.3f} \\\\\n                  \\hline\n".format(centroids[centroids.global_cluster_id==cluster_id][dim].values[0]))
    latex_output.write("              \\end{tabular}}\n\n              \\includegraphics[width=\\textwidth, keepaspectratio]{Clusters/palette_topic}\n            \\end{center}\n        \\end{column}\n    \\end{columns}\n\\end{frame}\n\n")

    latex_output.write("\\begin{frame}{Cluster "+str(cluster_id)+" -- Graphs}\n    \\resizebox{\\textwidth}{!}{\n    \\begin{tabular}{c|c|c|c|c}\n        \\huge{"+str(sessions_id[0])+"} & \\huge{"+str(sessions_id[1])+"} & \\huge{"+str(sessions_id[2])+"} & \\huge{"+str(sessions_id[3])+"} & \\huge{"+str(sessions_id[4])+"} \\\\\n        \\includegraphics[width=\\textwidth, keepaspectratio]{Graphs/session"+str(sessions_id[0])+"} & \\includegraphics[width=\\textwidth, keepaspectratio]{Graphs/session"+str(sessions_id[1])+"} & \\includegraphics[width=\\textwidth, keepaspectratio]{Graphs/session"+str(sessions_id[2])+"} & \\includegraphics[width=\\textwidth, keepaspectratio]{Graphs/session"+str(sessions_id[3])+"} & \\includegraphics[width=\\textwidth, keepaspectratio]{Graphs/session"+str(sessions_id[4])+"} \\\\\n        \\hline\n        \\huge{"+str(sessions_id[5])+"} & \\huge{"+str(sessions_id[6])+"} & \\huge{"+str(sessions_id[7])+"} & \\huge{"+str(sessions_id[8])+"} & \\huge{"+str(sessions_id[9])+"} \\\\\n        \\includegraphics[width=\\textwidth, keepaspectratio]{Graphs/session"+str(sessions_id[5])+"} & \\includegraphics[width=\\textwidth, keepaspectratio]{Graphs/session"+str(sessions_id[6])+"} & \\includegraphics[width=\\textwidth, keepaspectratio]{Graphs/session"+str(sessions_id[7])+"} & \\includegraphics[width=\\textwidth, keepaspectratio]{Graphs/session"+str(sessions_id[8])+"} & \\includegraphics[width=\\textwidth, keepaspectratio]{Graphs/session"+str(sessions_id[9])+"}\n    \\end{tabular}}\n\n")
        
    # recap centroids
    latex_output.write("    \\begin{columns}\n        \\begin{column}{.65\\textwidth}\n            \\begin{center}\n                \\scalebox{.25}{\n                    \\begin{tabular}{|c|")
    for cid in num_cluster:
        latex_output.write("c|")
    latex_output.write("}\n                        \\hline\n                        ")
    for cid in num_cluster:
        latex_output.write(" & "+str(cid))
    latex_output.write(" \\\\\n                        \\hline\n                        size")
    for cid in num_cluster:
        latex_output.write(" & "+str(sessions[sessions.global_cluster_id==cid].shape[0]))
    latex_output.write(" \\\\\n                        \\hline\n")
    for dim in dimensions_1 + dimensions_2:
        latex_output.write("                        "+str(dim).replace("_", "\_"))
        for cid in num_cluster:
            latex_output.write(" & {:.3f}".format(centroids[centroids.global_cluster_id==cid][dim].values[0]))
        latex_output.write(" \\\\\n                        \\hline\n")
    latex_output.write("                    \\end{tabular}\n                }\n            \\end{center}\n        \\end{column}\n        \\begin{column}{.35\\textwidth}\n            \\includegraphics[width=\\textwidth, keepaspectratio]{Clusters/palette_category}\n        \\end{column}\n    \\end{columns}\n\\end{frame}\n\n")

    print("          Display of sessions succesfully produced for cluster %d"%cluster_id) 
print("   * Clustered in {:.1f} seconds.".format((timelib.time()-start_time)))
plot_palette(labels=topic_list, filename="Latex/Clusters/palette_topic.png")
plot_palette(labels=category_list, filename="Latex/Clusters/palette_category.png")

###############################################################################
# END OF SCRIPT
latex_output.write("\\end{document}")
print("\n   * Done in {:.1f} seconds.\n".format(timelib.time()-begin_time))