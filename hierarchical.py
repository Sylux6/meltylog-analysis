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
import matplotlib as mpl
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
pathlib.Path("Latex/boxplot").mkdir(parents=True, exist_ok=True)
pathlib.Path("shared").mkdir(parents=True, exist_ok=True)
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
n_clusters_1 = 2
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
    latex_output.write("                \\item "+features_map(d)+"\n")
latex_output.write("            \\end{enumerate}\n        }\n    \\end{multicols}\n    Clustering on "+str(len(dimensions_2))+" dimension(s) ("+str(n_clusters_2)+" clusters):\n    \\begin{multicols}{2}\n        \\footnotesize{\n            \\begin{enumerate}")
for d in dimensions_2:
    latex_output.write("                \\item "+features_map(d)+"\n")
latex_output.write("            \\end{enumerate}\n        }\n    \\end{multicols}\n\\end{frame}\n\n")

###############################################################################
# CLUSTERING

topic_list = list(np.unique(log[["requested_topic", "referrer_topic"]].values))
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

# random sessions selection
start_time = timelib.time()
print("\n   * Selecting a good sample of sessions ...", end="\r")
selected_sessions = {}
centroids["sum"] = centroids[dimensions_1+dimensions_2].sum(axis=1)
sessions["dist"] = sessions.global_cluster_id.map(pd.Series(data=centroids["sum"], index=centroids.global_cluster_id.values))
sessions["dist"] = sessions[dimensions_1+dimensions_2].sum(axis=1) - sessions["dist"]
sessions["dist"] = sqrt((sessions["dist"] * sessions["dist"]))
for cluster_id in num_cluster:
    selected_sessions[cluster_id] = list(sessions[sessions.global_cluster_id==cluster_id].sort_values(["dist"]).global_session_id.values)[:5]
print("   * Sample of sessions selected in {:.1f} seconds.".format((timelib.time()-start_time)))

# sort
sorted_clusters = list()
span = {}
tmp = centroids.sort_values("timespan")
for i in range(0, n_clusters_1):
    sorted_clusters = sorted_clusters + list((tmp.iloc[i*n_clusters_2:(i+1)*n_clusters_2,:].sort_values("star_chain_like").global_cluster_id.values))
for cluster_id in sorted_clusters:
    span[cluster_id] = sessions[sessions.global_session_id.isin(selected_sessions[cluster_id])].timespan.max()
for i in range(0, len(sorted_clusters)):
    comparelist = list()
    for j in range(i-(i%n_clusters_2), i-(i%n_clusters_2)+n_clusters_2):
        comparelist = comparelist + [span[sorted_clusters[j]]]
    span[sorted_clusters[i]] = max(comparelist)

# rename clusters
order_dic = {}
for i in range(0, len(sorted_clusters)):
    order_dic[sorted_clusters[i]] = str(i+1)

# PCA
start_time = timelib.time()
print("\n   * Computing PCA components ...", end="\r")
pca = PCA(n_components=2)
clustering_data=pca.fit_transform(sessions[weighted_dimensions_1+weighted_dimensions_2].values)

fig, ax = plt.subplots()
fig.set_size_inches([ 7, 3])
matrix = pca.components_[:2,:]
cax = ax.matshow(matrix, cmap="coolwarm", clim=[-1, 1])
for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
        c = matrix[i,j]
        ax.text(j,i, '%0.2f'%c, va='center', ha='center', color="w", size=10)
ax.set_yticks(range(2))
ax.set_xticks(range(len(dimensions_1+dimensions_2)))
ax.set_yticklabels(['PC-%d'%n for n in range(1,2+1)])
ax.set_xticklabels(list(map(lambda x: features_map(x), dimensions_1+dimensions_2)))
plt.setp(ax.get_xticklabels(), rotation=45, ha="left", rotation_mode="anchor")
fig.colorbar(cax, orientation="horizontal")
plt.savefig('Latex/pca/components.png', format='png', bbox_inches="tight", dpi=1000)
plt.clf()
plt.close()
del matrix

df = pd.DataFrame(clustering_data, columns=["pc1", "pc2"])
sessions["pc1"] = df.pc1.values
sessions["pc2"] = df.pc2.values
fig=plt.figure()
plt.axis('equal')
plt.xlabel('PC-1')
plt.ylabel('PC-2')
plt.grid(True)
# Labeling the clusters
for cluster_id in num_cluster:
    plt.scatter(sessions[sessions.global_cluster_id==cluster_id].pc1.mean(), sessions[sessions.global_cluster_id==cluster_id].pc2.mean(), marker='o', c="white", alpha=1, s=ceil(10000*(sessions[sessions.global_cluster_id==cluster_id].shape[0]/sessions.shape[0])), edgecolor='k')
    plt.scatter(sessions[sessions.global_cluster_id==cluster_id].pc1.mean(), sessions[sessions.global_cluster_id==cluster_id].pc2.mean(), marker='$%d$' % (int(order_dic[cluster_id])), alpha=1, s=50, edgecolor='k')
plt.savefig('Latex/pca/pca_scatterplot.png', dpi=1000)
plt.clf()
plt.close()
del df

latex_output.write("\\section{PCA}\n\\begin{frame}{PCA}\n    \\begin{center}\n        \\includegraphics[scale=.3]{pca/components}\n\n        \\includegraphics[scale=.3]{pca/pca_scatterplot}\n    \\end{center}\n\\end{frame}\n\n")
print("   * PCA components computed in {:.1f} seconds.".format((timelib.time()-start_time)))

# generating cluster mosaic
for cluster_id in sorted_clusters:
    cluster_sessions = sessions[sessions.global_cluster_id == cluster_id].global_session_id.unique()
    cluster_log = log[log.global_session_id.isin(cluster_sessions)]
    plot_sessions_min(order_dic[cluster_id], sessions, cluster_log, 'Latex/Clusters/'+str(n_clusters_1)+"x"+str(n_clusters_2)+'/_cluster'+order_dic[cluster_id]+'.png', cluster_id, N_max_sessions=5, max_time=span[cluster_id], time_resolution=None, mark_requests=False, sessions=selected_sessions[cluster_id])

latex_output.write("\\section{Mosaic}\n\\begin{frame}{Cluster mosaic}\n    \\begin{center}\n        \\scalebox{.25}{\n            \\begin{tabular}{ccc}\n")
for i in range(0, n_clusters_1*n_clusters_2):
    if i%n_clusters_2 == 0:
        latex_output.write("\\\\\n                \\includegraphics[width=\\textwidth, keepaspectratio]{Clusters/"+str(n_clusters_1)+"x"+str(n_clusters_2)+"/_cluster"+str(order_dic[sorted_clusters[i]])+".png}")
    elif i%n_clusters_2 == n_clusters_2-1:
        latex_output.write("\\includegraphics[width=\\textwidth, keepaspectratio]{Clusters/"+str(n_clusters_1)+"x"+str(n_clusters_2)+"/_cluster"+str(order_dic[sorted_clusters[i]])+".png}")
    else:
        latex_output.write("\\includegraphics[width=\\textwidth, keepaspectratio]{Clusters/"+str(n_clusters_1)+"x"+str(n_clusters_2)+"/_cluster"+str(order_dic[sorted_clusters[i]])+".png} & ")
latex_output.write("\n            \\end{tabular}\n        }\n    \\end{center}\n\\end{frame}\n\n")

# boxplot
for dim in dimensions_1+dimensions_2:
    box = pd.DataFrame()
    for cluster_id in sorted_clusters:
        newcol = pd.DataFrame({"Cluster "+order_dic[cluster_id]: sessions[sessions.global_cluster_id==cluster_id][dim]})
        box = pd.concat([box, newcol], axis=1)
    box.boxplot(showfliers=False)
    plt.title(features_map(dim))
    plt.xticks(fontsize=8)
    plt.savefig("Latex/boxplot/"+dim+".png", dpi=1000)
    plt.clf()
latex_output.write("\\section{Boxplots}\n\\begin{frame}{Boxplots}\n    \\begin{center}\n        \\resizebox{\\textwidth}{!}{\n            \\begin{tabular}{ccc}\n")
for i in range(0, len(dimensions_1+dimensions_2)):
    if i%3 == 0:
        latex_output.write(" \\\\\n                \\includegraphics[width=\\textwidth, keepaspectratio]{boxplot/"+str((dimensions_1+dimensions_2)[i])+"}")
    elif i%3 == 2:
        latex_output.write("\\includegraphics[width=\\textwidth, keepaspectratio]{boxplot/"+str((dimensions_1+dimensions_2)[i])+"}")
    else:
        latex_output.write("\\includegraphics[width=\\textwidth, keepaspectratio]{boxplot/"+str((dimensions_1+dimensions_2)[i])+"} & ")
latex_output.write("\n            \\end{tabular}\n        }\n    \\end{center}\n\\end{frame}\n\n")

# entropy
shannon = []
for cluster_id in sorted_clusters:
    shannon.append(sessions[sessions.global_cluster_id==cluster_id].entropy.mean())
shannon = np.reshape(shannon, (n_clusters_1, n_clusters_2)).transpose()
num_label = np.reshape(sorted_clusters, (n_clusters_1, n_clusters_2)).transpose()
fig, ax = plt.subplots()
cax = ax.matshow(shannon.transpose(), cmap="coolwarm")
for i in range(shannon.shape[0]):
    for j in range(shannon.shape[1]):
        ax.text(i, j, 'cluster '+order_dic[num_label[i][j]]+'\n%0.2f'%shannon[i][j], va='center', ha='center', color="w")
plt.xticks([])
plt.yticks([])
fig.colorbar(cax)
plt.title("Entropy")
plt.savefig("Latex/Clusters/entropy.png", dpi=1000)
plt.clf()
latex_output.write("\\section{Entropy}\n\\begin{frame}{Entropy}\n    \\begin{center}        \\includegraphics[width=\\textwidth,height=0.8\\textheight,keepaspectratio]{Clusters/entropy}\n    \\end{center}\n\\end{frame}\n\n")

# pie
start_time = timelib.time()
print("\n   * Plotting topic proportion ...", end="\r")
pie = pd.DataFrame()
pie["global_cluster_id"] = num_cluster
log["global_cluster_id"] = log.global_session_id.map(pd.Series(data=sessions.global_cluster_id, index=sessions.global_session_id))
cluster_log = log.dropna()

for topic in topic_list:
    pie[topic] = cluster_log[["global_cluster_id", "requested_topic"]][cluster_log.requested_topic==topic].groupby("global_cluster_id").count()
pie.fillna(0, inplace=True)
pie.set_index("global_cluster_id", inplace=True)

fig, axs = plt.subplots(n_clusters_1, n_clusters_2, figsize=(2*n_clusters_1, 2*n_clusters_2))
fig.subplots_adjust(hspace=0.1, wspace=0.0)
axs = axs.ravel()
for n in range(len(sorted_clusters)):
    color_vals = [n for n in range(len(topic_list))]
    my_norm = mpl.colors.Normalize(0, len(topic_list))
    my_cmap = mpl.cm.get_cmap("tab20", len(color_vals))
    patches, texts, autotexts = axs[n].pie(pie.ix[n], labels = None, autopct='%1.1f%%', startangle=90, colors=my_cmap(my_norm(color_vals)))
    axs[n].set_aspect('equal')
    axs[n].set_title("Cluster "+order_dic[sorted_clusters[n]])
    for item in autotexts:
        item.set_text("")
ax_cb = fig.add_axes([.9,.25,.03,.5])
cb = mpl.colorbar.ColorbarBase(ax_cb, cmap=my_cmap, norm=my_norm, ticks=color_vals)
cb.set_label("Topic")
cb.set_ticklabels(topic_list)
plt.savefig("Latex/Clusters/topic.png", bbox_inches="tight", dpi=1000)
plt.clf()
latex_output.write("\\section{Topic proportion}\n\\begin{frame}{Topic proportion}\n    \\begin{center}        \\includegraphics[width=\\textwidth,height=0.8\\textheight,keepaspectratio]{Clusters/topic}\n    \\end{center}\n\\end{frame}\n\n")
print("   * Topic proportion plotted in {:.1f} seconds.".format((timelib.time()-start_time)))

# markov
print("\n   * Computing markov matrix ...", end="\r")
markov = np.zeros((len(category_list), len(category_list)))
total_entries = log.shape[0]
for i in range(0, len(category_list)):
   for j in range(0, len(category_list)):
       markov[i][j] = log[(log.referrer_category==category_list[i]) & (log.requested_category==category_list[j])].shape[0] / total_entries
fig, ax = plt.subplots()
cax = ax.matshow(markov, cmap="coolwarm")
for i in range(markov.shape[0]):
   for j in range(markov.shape[1]):
       ax.text(j, i, '\n%.2f' % markov[i][j], va='center', ha='center', size=6, color="w")
fig.colorbar(cax)
ax.set_xticks(np.arange(len(category_list)))
ax.set_yticks(np.arange(len(category_list)))
ax.set_xticklabels(category_list)
ax.set_yticklabels(category_list)
ax.xaxis.set_label_position('top') 
plt.setp(ax.get_xticklabels(), rotation=45, ha="left", rotation_mode="anchor")
plt.xlabel("Requested category")
plt.ylabel("Referrer category")
plt.title("Markov matrix", y=1.3)
plt.savefig("Latex/pca/markov.png", bbox_inches="tight", dpi=1000)
plt.clf()
latex_output.write(
   "\\section{Markov}\n\\begin{frame}{Markov matrix}\n    \\begin{center}        \\includegraphics[width=\\textwidth,height=0.8\\textheight,keepaspectratio]{pca/markov}\n    \\end{center}\n\\end{frame}\n\n"
)
print("   * Markov matrix computed in {:.1f} seconds.".format((timelib.time() - start_time)))

# recap
if n_clusters_1*n_clusters_2 > 10:
    resizebox = ".8"
else:
    resizebox = ""
latex_output.write("\\section{Recap}\n\\begin{frame}{Clustering: "+str(n_clusters_1)+"x"+str(n_clusters_2)+" clusters}\n    \\begin{center}\n        \\resizebox{"+resizebox+"\\textwidth}{!}{\n            \\begin{tabular}{ccccc}\n")
for i in range(0, len(sorted_clusters)):
    if i%5 == 0:
        latex_output.write("\\\\\n                \\includegraphics[width=\\textwidth, keepaspectratio]{Clusters/"+str(n_clusters_1)+"x"+str(n_clusters_2)+"/cluster"+order_dic[sorted_clusters[i]]+"}")
    elif i%5 == 4:
        latex_output.write("\\includegraphics[width=\\textwidth, keepaspectratio]{Clusters/"+str(n_clusters_1)+"x"+str(n_clusters_2)+"/cluster"+order_dic[sorted_clusters[i]]+"}")
    else:
        latex_output.write("\\includegraphics[width=\\textwidth, keepaspectratio]{Clusters/"+str(n_clusters_1)+"x"+str(n_clusters_2)+"/cluster"+order_dic[sorted_clusters[i]]+"} & ")
latex_output.write("\n            \\end{tabular}\n        }\n\n        \\begin{columns}\n            \\begin{column}{.65\\textwidth}\n                \\begin{center}\n                \\scalebox{0.25}{\n")
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
    latex_output.write("                        "+features_map(dim))
    for cluster_id in num_cluster:
        latex_output.write(" & {:.3f}".format(centroids[centroids.global_cluster_id==cluster_id][dim].values[0]))
    latex_output.write(" \\\\\n                        \\hline\n")
latex_output.write("                    \\end{tabular}\n                }\n                \\end{center}\n            \\end{column}\n            \\begin{column}{.35\\textwidth}\n                \\includegraphics[width=\\textwidth, keepaspectratio]{Clusters/palette_topic}\n            \\end{column}\n        \\end{columns}\n    \\end{center}\n\\end{frame}\n\n")

# display
for cluster_id in sorted_clusters:
    print("          Producing display of sessions for cluster %d"%cluster_id,end="\r")
    cluster_sessions = sessions[sessions.global_cluster_id == cluster_id].global_session_id.unique()
    cluster_log = log[log.global_session_id.isin(cluster_sessions)]
    sessions_id = plot_sessions(order_dic[cluster_id], cluster_log,'Latex/Clusters/'+str(n_clusters_1)+"x"+str(n_clusters_2)+'/cluster'+order_dic[cluster_id]+'.png', cluster_id, labels=list(log.requested_topic.unique()), N_max_sessions=10,field="requested_topic", max_time=span[cluster_id],time_resolution=None,mark_requests=False, sessions=selected_sessions[cluster_id])

    # graph
    session_draw(cluster_id, sessions_id, log, urls, category_list)
    latex_output.write("% cluster "+order_dic[cluster_id]+"\n\\section{Cluster "+order_dic[cluster_id]+"}\n\\begin{frame}{Cluster "+order_dic[cluster_id]+"}\n    \\begin{columns}\n        \\begin{column}{.6\\textwidth}\n            \\includegraphics[width=\\textwidth, keepaspectratio]{Clusters/"+str(n_clusters_1)+"x"+str(n_clusters_2)+"/cluster"+order_dic[cluster_id]+"}\n        \\end{column}\n        \\begin{column}{.4\\textwidth}\n            \\begin{center}\n              \\scalebox{.4}{\\begin{tabular}{|c|c|}\n                  \\hline\n                  \\multicolumn{2}{|c|}{mean} \\\\\n                  \\hline\n                  size & "+str(sessions[sessions.global_cluster_id==cluster_id].shape[0])+" \\\\\n                  \\hline\n")
    for dim in dimensions_1+dimensions_2:
        latex_output.write("                  "+features_map(dim)+" & {:.3f} \\\\\n                  \\hline\n".format(centroids[centroids.global_cluster_id==cluster_id][dim].values[0]))
    latex_output.write("              \\end{tabular}}\n\n              \\includegraphics[width=\\textwidth, keepaspectratio]{Clusters/palette_topic}\n            \\end{center}\n        \\end{column}\n    \\end{columns}\n\\end{frame}\n\n")

    latex_output.write("\\begin{frame}{Cluster "+order_dic[cluster_id]+" -- Graphs}\n    \\resizebox{\\textwidth}{!}{\n    \\begin{tabular}{c|c|c|c|c}\n        \\huge{"+str(sessions_id[0])+"} & \\huge{"+str(sessions_id[1])+"} & \\huge{"+str(sessions_id[2])+"} & \\huge{"+str(sessions_id[3])+"} & \\huge{"+str(sessions_id[4])+"} \\\\\n        \\includegraphics[width=\\textwidth, keepaspectratio]{Graphs/session"+str(sessions_id[0])+"} & \\includegraphics[width=\\textwidth, keepaspectratio]{Graphs/session"+str(sessions_id[1])+"} & \\includegraphics[width=\\textwidth, keepaspectratio]{Graphs/session"+str(sessions_id[2])+"} & \\includegraphics[width=\\textwidth, keepaspectratio]{Graphs/session"+str(sessions_id[3])+"} & \\includegraphics[width=\\textwidth, keepaspectratio]{Graphs/session"+str(sessions_id[4])+"}\n    \\end{tabular}}\n\n")

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
        latex_output.write("                        "+features_map(dim))
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
