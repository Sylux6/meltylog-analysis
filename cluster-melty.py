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
# dimensions = ["requests", "timespan", "standard_deviation", "inter_req_mean_seconds", "read_pages"]
# dim2
# dimensions = ["star_chain_like"]
# dim3
# dimensions = ["popularity_mean", "entropy", "requested_category_richness", "requested_topic_richness", 'TV_proportion', 'Series_proportion', 'News_proportion', 'Celebrities_proportion', 'VideoGames_proportion', 'Music_proportion', 'Movies_proportion', 'Sport_proportion', 'Comic_proportion', 'Look_proportion', 'Other_proportion', 'Humor_proportion', 'Student_proportion', 'Events_proportion', 'Wellbeing_proportion', 'None_proportion', 'Food_proportion', 'Tech_proportion']
# dim1+dim2
dimensions = ["requests", "timespan", "standard_deviation", "inter_req_mean_seconds", "read_pages", "star_chain_like"]
# dim1+dim2+dim3
# dimensions = ["requests", "timespan", "standard_deviation", "inter_req_mean_seconds", "read_pages", "star_chain_like", "bifurcation", "popularity_mean", "entropy", "requested_category_richness", "requested_topic_richness", 'TV_proportion', 'Series_proportion', 'News_proportion', 'Celebrities_proportion', 'VideoGames_proportion', 'Music_proportion', 'Movies_proportion', 'Sport_proportion', 'Comic_proportion', 'Look_proportion', 'Other_proportion', 'Humor_proportion', 'Student_proportion', 'Events_proportion', 'Wellbeing_proportion', 'None_proportion', 'Food_proportion', 'Tech_proportion']
lognorm = ["requests", "timespan", "inter_req_mean_seconds", "standard_deviation", "popularity_mean", "variance"]
range_n_clusters = [2, 3, 4, 5]
max_components = len(dimensions)
threshold_explained_variance = 0.99
include_pairwises = True

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

normalized_dimensions = list(map(lambda x: "normalized_"+x, dimensions)) # normalized dimensions labels list
weighted_dimensions = list(map(lambda x: "weighted_"+x, dimensions)) # weighted dimensions labels list

print("   > range_n_clusters: {}".format(range_n_clusters))
print("   > include pairwises: {}".format(include_pairwises))

###############################################################################
# NORMALIZATION
start_time = timelib.time()
print("\n   * Normalizing dimensions ...", end="\r")
for d in dimensions:
    if d in lognorm:
        sessions["normalized_"+d] = sessions[d]+1 # here we need to shift our data for log normalization
        sessions["normalized_"+d] = sessions["normalized_"+d].apply(lambda x: math.log(x))
    else:
        sessions["normalized_"+d] = sessions[d]
scaler = StandardScaler()
sessions[normalized_dimensions] = scaler.fit_transform(sessions[normalized_dimensions])
print("   * Dimensions normalized in %.1f seconds." %(timelib.time()-start_time))

###############################################################################
# WEIGHTS
start_time = timelib.time()
print("\n   * Weighting dimensions ...", end="\r")
cv = {}
cv_tot = 0
w = {}
for d in dimensions:
    cv[d] = sessions[d].var() / sessions[d].mean()
    cv_tot = cv_tot + cv[d]
for d in dimensions:
    w[d] = cv[d] / cv_tot
for d in dimensions:
    sessions["weighted_"+d] = sqrt(w[d]) * sessions["normalized_"+d]
print("   * Dimensions weighted in {:.1f} seconds.".format((timelib.time()-start_time)))

latex_output.write("\\documentclass[xcolor={dvipsnames}, handout]{beamer}\n\n\\usetheme{Warsaw}\n\\usepackage[utf8]{inputenc}\n\\usepackage[T1]{fontenc}\n\\usepackage{graphicx}\n\\usepackage[english]{babel}\n\\usepackage{amsmath}\n\\usepackage{amssymb}\n\\usepackage{mathrsfs}\n\\usepackage{verbatim}\n\\usepackage{lmodern}\n\\usepackage{listings}\n\\usepackage{caption}\n\\usepackage{multicol}\n\\usepackage{epsfig}\n\\usepackage{array}\n\\usepackage{tikz}\n\\usepackage{collcell}\n\n\\definecolor{mygreen}{rgb}{0,0.6,0}\n\\setbeamertemplate{headline}{}{}\n\\addtobeamertemplate{footline}{\insertframenumber/\inserttotalframenumber}\n\n\\title{Melty Clusterization}\n\\author{Sylvain Ung}\n\\institute{Laboratoire d'informatique de Paris 6}\n\\date{\\today}\n\n\\begin{document}\n\\setbeamertemplate{section page}\n{\n  \\begin{centering}\n    \\vskip1em\\par\n    \\begin{beamercolorbox}[sep=4pt,center]{part title}\n      \\usebeamerfont{section title}\\insertsection\\par\n    \\end{beamercolorbox}\n  \\end{centering}\n}\n\n\\begin{frame}\n    \\titlepage\n\\end{frame}\n\n")

latex_output.write("\\begin{frame}{Clustering}\n    Clustering on "+str(len(dimensions))+" dimension(s):\n    \\begin{multicols}{2}\n        \\footnotesize{\n            \\begin{enumerate}\n")
for d in dimensions:
    latex_output.write("                \\item "+d.replace("_", "\_")+"\n")
latex_output.write("            \\end{enumerate}\n        }\n    \\end{multicols}\n\\end{frame}\n\n")

if max_components > 1:
    ###############################################################################
    # CORRELATION ANALYSIS

    start_time = timelib.time()
    print("\n   * Computing correlation ...", end="\r")
    corr=sessions[weighted_dimensions].corr()
    fig, ax = plt.subplots()
    fig.set_size_inches([ 14, 14])
    matrix = corr.values
    ax.matshow(matrix, cmap=plt.cm.coolwarm)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[0]):
            c = matrix[j,i]
            ax.text(i, j, '%0.2f'%c, va='center', ha='center')
    ax.set_xticks(range(len(dimensions)))
    ax.set_yticks(range(len(dimensions)))
    ax.set_xticklabels(dimensions)
    ax.set_yticklabels(dimensions)
    plt.tick_params(axis='both', which='both', labelsize=8)
    plt.savefig('Latex/pca/corr_before_pca.png', format='png')
    plt.clf()
    plt.close()
    latex_output.write("\\begin{frame}{Correlation analysis}\n    \\begin{center}\n        \\includegraphics[width=\\textwidth, height=\\textheight, keepaspectratio]{pca/corr_before_pca}\n    \\end{center}\n\\end{frame}\n\n")
    print("   * Correlation computed in {:.1f} seconds.".format((timelib.time()-start_time)))

    ###############################################################################
    # PCA
    start_time = timelib.time()
    print("\n   * Computing PCA explained variance ...", end="\r")
    pca = PCA(n_components=max_components)

    # Data in PCA coordinates: n_samples x n_components
    weighted_pca_data=pca.fit_transform(sessions[weighted_dimensions].values)

    # selecting components that explain variance
    n_components_threshold=len(pca.explained_variance_ratio_[pca.explained_variance_ratio_.cumsum()<threshold_explained_variance])+1

    plt.figure()
    plt.plot(range(1,max_components+1),100.0*pca.explained_variance_ratio_, 'r+')
    plt.axis([0, max_components+1, 0, 100])
    plt.gca().axvline(x=n_components_threshold,c='b',alpha=0.25)
    plt.text(n_components_threshold+0.5,75,
            '%0.2f%% explained variancce.'%(100*pca.explained_variance_ratio_.cumsum()[n_components_threshold-1]))
    plt.xlabel('Component')
    plt.ylabel('% Explained Variance')
    plt.grid()
    plt.savefig('Latex/pca/explained_variance_ratio.png')
    plt.clf()
    plt.close()
    latex_output.write("\\begin{frame}{PCA explained variance}\n    \\begin{center}\n        \\includegraphics[width=\\textwidth, height=\\textheight, keepaspectratio]{pca/explained_variance_ratio}\n    \\end{center}\n\\end{frame}\n\n")
    print("   * PCA explained variance computed in {:.1f} seconds.".format((timelib.time()-start_time)))

    pca = PCA(n_components=n_components_threshold)
    clustering_data=pca.fit_transform(sessions[weighted_dimensions].values)

    ###############################################################################
    # EXPLAINING PCA COMPONENTS

    fig, ax = plt.subplots()
    fig.set_size_inches([ 14, 14])
    matrix = pca.components_[:n_components_threshold,:].T
    ax.matshow(matrix, cmap=plt.cm.coolwarm)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            c = matrix[i,j]
            ax.text(j,i, '%0.2f'%c, va='center', ha='center')
    ax.set_xticks(range(n_components_threshold))
    ax.set_yticks(range(len(dimensions)))
    ax.set_xticklabels(['PC-%d'%n for n in range(1,n_components_threshold+1)])
    ax.set_yticklabels(dimensions)
    plt.savefig('Latex/pca/components.png', format='png')
    plt.clf()
    plt.close()
    del matrix
    latex_output.write("\\begin{frame}{PCA components}\n    \\begin{center}\n        \\includegraphics[width=\\textwidth, height=\\textheight, keepaspectratio]{pca/components}\n    \\end{center}\n\\end{frame}\n\n")

###############################################################################
# CLUSTERING

topic_list = list(log.requested_topic.unique())
category_list = list(urls.category.unique())

for n in range_n_clusters:
    start_time = timelib.time()
    print("\n   * Clustering ("+str(n)+" clusters) ...")
    pathlib.Path("Latex/Graphs/"+str(n)).mkdir(parents=True, exist_ok=True)
    pathlib.Path("Latex/Clusters/"+str(n)).mkdir(parents=True, exist_ok=True)

    if max_components > 1:
        ###############################################################################
        # Scatterplot
        kmeans=KMeans(n_clusters=n)
        cluster_labels=kmeans.fit_predict(clustering_data)
        fig=plt.figure()
        plt.scatter(clustering_data[:,0],clustering_data[:,1], c=kmeans.labels_, alpha=0.1)
        plt.axis('equal')
        plt.xlabel('PC-1')
        plt.ylabel('PC-2')
        plt.grid(True)
        # Labeling the clusters
        plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='o',
                    c="white", alpha=1, s=200, edgecolor='k')
        for i, c in enumerate(kmeans.cluster_centers_):
            plt.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                        s=50, edgecolor='k')
        plt.savefig('Latex/pca/pca_scatterplot_cluster_%d.png'%n)
        plt.clf()
        plt.close()
        latex_output.write("\\begin{frame}{PCA clustering -- "+str(n)+" clusters}\n    \\begin{center}\n        \\includegraphics[width=\\textwidth, height=0.8\\textheight, keepaspectratio]{pca/pca_scatterplot_cluster_"+str(n)+"}\n    \\end{center}\n\\end{frame}\n\n")

        ########################################################
        # Scatterplot in pairwise original feature space pairs #
        ########################################################
        if include_pairwises:
            start_time2 = timelib.time()
            print("          Plotting pairwises...", end="\r")
            # weighted space
            count = 0 # pairwises counter
            latex_output.write("\\begin{frame}{Scatterplots in pairwise weighted features space pairs}\n    \\begin{center}\n        \\resizebox{\\textwidth}{!}{\n            \\begin{tabular}{ccccc}")
            centroids_inverse_pca = pca.inverse_transform(kmeans.cluster_centers_)

            # compute original space centroids
            centroids_inverse_normalized = centroids_inverse_pca.copy()
            cluster_labels=kmeans.labels_
            sessions["cluster_id"] = cluster_labels
            num_cluster = sessions.cluster_id.unique()
            for cluster_id in num_cluster:
                for i in range(0, len(dimensions)):
                    centroids_inverse_normalized[cluster_id][i] = sessions[sessions.cluster_id==cluster_id][dimensions[i]].mean()

            for ftr1 in range(len(dimensions)):
                for ftr2 in range(ftr1+1,len(dimensions)):
                    fig=plt.figure()
                    plt.scatter(sessions[weighted_dimensions].values[:,ftr1],sessions[weighted_dimensions].values[:,ftr2], c=kmeans.labels_, alpha=0.1)
                    plt.axis('equal')
                    plt.xlabel(dimensions[ftr1])
                    plt.ylabel(dimensions[ftr2])
                    plt.grid(True)
                    # Labeling the clusters
                    plt.scatter(centroids_inverse_pca[:,ftr1], centroids_inverse_pca[:, ftr2], marker='o',
                                c="white", alpha=1, s=200, edgecolor='k')
                    for i, c in enumerate(centroids_inverse_pca):
                        plt.scatter(c[ftr1], c[ftr2], marker='$%d$' % i, alpha=1,
                                    s=50, edgecolor='k')
                    plt.savefig('Latex/pca/pairwise/weighted_pca_scatterplot_%d_clusters_ftr1_%d_ftr2_%d.png'%(n,ftr1,ftr2))
                    plt.clf()
                    plt.close()
                    if count%5 == 4:
                        latex_output.write("\\includegraphics[width=\\textwidth, height=0.8\\textheight, keepaspectratio]{pca/pairwise/weighted_pca_scatterplot_"+str(n)+"_clusters_ftr1_"+str(ftr1)+"_ftr2_"+str(ftr2)+"} \\\\\n")
                    else:
                        latex_output.write("\\includegraphics[width=\\textwidth, height=0.8\\textheight, keepaspectratio]{pca/pairwise/weighted_pca_scatterplot_"+str(n)+"_clusters_ftr1_"+str(ftr1)+"_ftr2_"+str(ftr2)+"} & ")
                    count = count + 1
            latex_output.write("\n            \\end{tabular}\n        }\n    \\end{center}\n\\end{frame}\n\n")

            # normalized space
            count = 0 # pairwises counter
            latex_output.write("\\begin{frame}{Scatterplots in pairwise normalized features space pairs}\n    \\begin{center}\n        \\resizebox{\\textwidth}{!}{\n            \\begin{tabular}{ccccc}")
            centroids_inverse_weighted = centroids_inverse_pca
            for i in range(0, len(centroids_inverse_weighted)):
                for j in range(0, len(centroids_inverse_weighted[i])):
                    centroids_inverse_weighted[i][j] = centroids_inverse_weighted[i][j] / sqrt(w[dimensions[j]])
            for ftr1 in range(len(dimensions)):
                for ftr2 in range(ftr1+1,len(dimensions)):
                    fig=plt.figure()
                    plt.scatter(sessions[normalized_dimensions].values[:,ftr1],sessions[normalized_dimensions].values[:,ftr2], c=kmeans.labels_, alpha=0.1)
                    plt.axis('equal')
                    plt.xlabel(dimensions[ftr1])
                    plt.ylabel(dimensions[ftr2])
                    plt.grid(True)
                    # Labeling the clusters
                    plt.scatter(centroids_inverse_weighted[:,ftr1], centroids_inverse_weighted[:, ftr2], marker='o',
                                c="white", alpha=1, s=200, edgecolor='k')
                    for i, c in enumerate(centroids_inverse_weighted):
                        plt.scatter(c[ftr1], c[ftr2], marker='$%d$' % i, alpha=1,
                                    s=50, edgecolor='k')
                    plt.savefig('Latex/pca/pairwise/normalized_pca_scatterplot_%d_clusters_ftr1_%d_ftr2_%d.png'%(n,ftr1,ftr2))
                    plt.clf()
                    plt.close()
                    if count%5 == 4:
                        latex_output.write("\\includegraphics[width=\\textwidth, height=0.8\\textheight, keepaspectratio]{pca/pairwise/normalized_pca_scatterplot_"+str(n)+"_clusters_ftr1_"+str(ftr1)+"_ftr2_"+str(ftr2)+"} \\\\\n")
                    else:
                        latex_output.write("\\includegraphics[width=\\textwidth, height=0.8\\textheight, keepaspectratio]{pca/pairwise/normalized_pca_scatterplot_"+str(n)+"_clusters_ftr1_"+str(ftr1)+"_ftr2_"+str(ftr2)+"} & ")
                    count = count + 1
            latex_output.write("\n            \\end{tabular}\n        }\n    \\end{center}\n\\end{frame}\n\n")

            # original space
            count = 0 # pairwises counter
            latex_output.write("\\begin{frame}{Scatterplots in pairwise normalized features space pairs}\n    \\begin{center}\n        \\resizebox{\\textwidth}{!}{\n            \\begin{tabular}{ccccc}")
            for ftr1 in range(len(dimensions)):
                for ftr2 in range(ftr1+1,len(dimensions)):
                    fig=plt.figure()
                    plt.scatter(sessions[dimensions].values[:,ftr1],sessions[dimensions].values[:,ftr2], c=kmeans.labels_, alpha=0.1)
                    plt.xlabel(dimensions[ftr1])
                    plt.ylabel(dimensions[ftr2])
                    plt.grid(True)
                    # Labeling the clusters
                    plt.scatter(centroids_inverse_normalized[:,ftr1], centroids_inverse_normalized[:, ftr2], marker='o', c="white", alpha=1, s=200, edgecolor='k')
                    for i, c in enumerate(centroids_inverse_normalized):
                        plt.scatter(c[ftr1], c[ftr2], marker='$%d$' % i, alpha=1,s=50, edgecolor='k')
                    # logscale
                    if dimensions[ftr1] in lognorm:
                        plt.gca().set_xscale('log')
                    if dimensions[ftr2] in lognorm:
                        plt.gca().set_yscale('log')
                    plt.savefig('Latex/pca/pairwise/original_pca_scatterplot_%d_clusters_ftr1_%d_ftr2_%d.png'%(n,ftr1,ftr2))
                    plt.clf()
                    plt.close()
                    if count%5 == 4:
                        latex_output.write("\\includegraphics[width=\\textwidth, height=0.8\\textheight, keepaspectratio]{pca/pairwise/original_pca_scatterplot_"+str(n)+"_clusters_ftr1_"+str(ftr1)+"_ftr2_"+str(ftr2)+"} \\\\\n")
                    else:
                        latex_output.write("\\includegraphics[width=\\textwidth, height=0.8\\textheight, keepaspectratio]{pca/pairwise/original_pca_scatterplot_"+str(n)+"_clusters_ftr1_"+str(ftr1)+"_ftr2_"+str(ftr2)+"} & ")
                    count = count + 1
            latex_output.write("\n            \\end{tabular}\n        }\n    \\end{center}\n\\end{frame}\n\n")
            print("          Pairwises plotted in {:.1f} seconds.".format((timelib.time()-start_time2)))

    kmeans = KMeans(n_clusters=n, random_state=0).fit(sessions[weighted_dimensions].values)
    cluster_labels=kmeans.labels_
    sessions["cluster_id"] = cluster_labels
    num_cluster = sessions.cluster_id.unique()
    num_cluster.sort()

    # compute centroids for recap
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
        sessions_id = plot_sessions(cluster_log,'Latex/Clusters/'+str(n)+'/cluster%d.png'%cluster_id, cluster_id,
                    labels=list(log.requested_topic.unique()),
                    N_max_sessions=10,field="requested_topic",
                    max_time=None,time_resolution=None,mark_requests=False)

        # graph
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
plot_palette(labels=topic_list, filename="Latex/Clusters/palette_topic.png")
plot_palette(labels=category_list, filename="Latex/Clusters/palette_category.png")

###############################################################################
# END OF SCRIPT
latex_output.write("\\end{document}")
print("\n   * Done in {:.1f} seconds.\n".format(timelib.time()-begin_time))