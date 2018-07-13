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

from plots import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pylab import *
import pathlib
import shutil

from graph_tool.all import *

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
session_filename = "Outputs/depth_session.csv"
pages_filemname = "Outputs/pages_sessions.csv"

shutil.rmtree("Matplot/pages", ignore_errors=True)
pathlib.Path("Matplot/pages").mkdir(parents=True, exist_ok=True)

shutil.rmtree("Latex", ignore_errors=True)
pathlib.Path("Latex").mkdir(parents=True, exist_ok=True)
pathlib.Path("Latex/images").mkdir(parents=True, exist_ok=True)
pathlib.Path("shared/graph_properties").mkdir(parents=True, exist_ok=True)
latex_output = open("Latex/main.tex", "w")

latex_output.write("\\documentclass[xcolor={dvipsnames}, handout]{beamer}\n\n\\usetheme{Warsaw}\n\\usepackage[utf8]{inputenc}\n\\usepackage[T1]{fontenc}\n\\usepackage{graphicx}\n\\usepackage[english]{babel}\n\\usepackage{amsmath}\n\\usepackage{amssymb}\n\\usepackage{mathrsfs}\n\\usepackage{verbatim}\n\\usepackage{lmodern}\n\\usepackage{listings}\n\\usepackage{caption}\n\\usepackage{multicol}\n\\usepackage{epsfig}\n\\usepackage{array}\n\\usepackage{tikz}\n\\usepackage{collcell}\n\n\\definecolor{mygreen}{rgb}{0,0.6,0}\n\\setbeamertemplate{headline}{}{}\n\\addtobeamertemplate{footline}{\insertframenumber/\inserttotalframenumber}\n\n\\title{Graph properties}\n\\author{Sylvain Ung}\n\\institute{Laboratoire d'informatique de Paris 6}\n\\date{\\today}\n\n\\begin{document}\n\\setbeamertemplate{section page}\n{\n  \\begin{centering}\n    \\vskip1em\\par\n    \\begin{beamercolorbox}[sep=4pt,center]{part title}\n      \\usebeamerfont{section title}\\insertsection\\par\n    \\end{beamercolorbox}\n  \\end{centering}\n}\n\n\\begin{frame}\n    \\titlepage\n\\end{frame}\n\n")

###############################################################################
# READING DATA FILES
print("\n   * Loading files ...")
start_time = timelib.time()
print("        Loading "+urls_filename+" ...", end="\r")
urls = pd.read_csv(urls_filename, sep=',', na_filter=False, low_memory=False)
print("        "+urls_filename+" loaded ({} rows) in {:.1f} seconds.".format(urls.shape[0], timelib.time()-start_time))
start_time = timelib.time()
print("        Loading "+session_filename+" ...", end="\r")
sessions = pd.read_csv(session_filename, sep=',')
print("        "+session_filename+" loaded ({} rows) in {:.1f} seconds.".format(sessions.shape[0], timelib.time()-start_time))
start_time = timelib.time() 
print("        Loading "+pages_filemname+" ...", end="\r")
pages_sessions = pd.read_csv(pages_filemname, sep=',', na_filter=False, low_memory=False)
print("        "+pages_filemname+" loaded ({} rows) in {:.1f} seconds.".format(urls.shape[0], timelib.time()-start_time))

dimensions = ["betweenness", "in_degree", "out_degree", "depth"]
pages_data = urls[urls.url.isin(pages_sessions.url)]
category_list = pages_data[["category", "url"]].groupby("category").count().sort_values(by="url", ascending=False).index.values.tolist()

print("\n   * Computing features ...")

# betweenness
start_time= timelib.time()
print("        Computing betweenness ...", end='\r')
pages_betweenness = pages_sessions[["url", "betweenness"]].groupby("url").aggregate(lambda x: mean(x))
pages_data["betweenness"] = pages_data.url.map(pd.Series(data=pages_betweenness.betweenness.values, index=pages_betweenness.index))
print("        Betweenness computed in %.1f seconds." %(timelib.time()-start_time))

# in_degree
start_time= timelib.time()
print("        Computing in_degree ...", end='\r')
pages_in_degree = pages_sessions[["url", "in_degree"]].groupby("url").aggregate(lambda x: mean(x))
pages_data["in_degree"] = pages_data.url.map(pd.Series(data=pages_in_degree.in_degree.values, index=pages_in_degree.index))
print("        In_degree computed in %.1f seconds." %(timelib.time()-start_time))

# out_degree
start_time= timelib.time()
print("        Computing out_degree ...", end='\r')
pages_out_degree = pages_sessions[["url", "out_degree"]].groupby("url").aggregate(lambda x: mean(x))
pages_data["out_degree"] = pages_data.url.map(pd.Series(data=pages_out_degree.out_degree.values, index=pages_out_degree.index))
print("        Out_degree computed in %.1f seconds." %(timelib.time()-start_time))

# depth
start_time= timelib.time()
print("        Computing depth ...", end='\r')
pages_depth = pages_sessions[["url", "depth"]].groupby("url").aggregate(lambda x: mean(x))
pages_data["depth"] = pages_data.url.map(pd.Series(data=pages_depth.depth.values, index=pages_depth.index))
print("        Depth computed in %.1f seconds." %(timelib.time()-start_time))

start_time= timelib.time()
print("\n   * Plotting depth session ...", end="\r")

plt.hist(sessions.depth.values, align="left")
plt.grid(alpha=0.5)
plt.xlabel("Depth")
plt.ylabel("Frequency")
ax = plt.gca()
ax.set_xticks([n for n in range(sessions.depth.max())])
plt.gca().set_yscale('log')
plt.savefig("Latex/images/depth.png", format='png', bbox_inches="tight")
plt.savefig("shared/graph_properties/depth.svg", format='svg', bbox_inches="tight")
plt.clf()
latex_output.write("\\begin{frame}{Diameter}\n    \\begin{center}        \\includegraphics[width=\\textwidth,height=0.8\\textheight,keepaspectratio]{images/depth.png"+"}\n    \\end{center}\n\\end{frame}\n\n")

print("   * Depth plotted in %.1f seconds." %(timelib.time()-start_time))

print("\n   * Scattering ...")

for f1 in range(0, len(dimensions)):
    for f2 in range(f1+1, len(dimensions)):
        start_time= timelib.time()
        print("        "+dimensions[f1]+"-VS-"+dimensions[f2]+" ...", end="\r")
        fig, ax = plt.subplots()
        count=0
        for n in range(len(category_list)):
            tmp = pages_data[pages_data.category==category_list[n]]
            if count<10:
                plt.scatter(tmp[dimensions[f1]].values, tmp[dimensions[f2]].values, c=cm.tab10(n%10), label=category_list[n], alpha=0.5)
            else:
                plt.scatter(tmp[dimensions[f1]].values, tmp[dimensions[f2]].values, c=cm.tab10(n%10), label=category_list[n], alpha=0.5, marker="x")
            count+=1
        plt.grid(alpha=0.5)
        plt.xlabel(dimensions[f1])
        plt.ylabel(dimensions[f2])
        plt.gca().set_xscale('log')
        plt.gca().set_yscale('log')
        ax.legend(loc="upper left", bbox_to_anchor=(1,1), title="category")
        plt.savefig("Latex/images/"+dimensions[f1]+"-VS-"+dimensions[f2]+".png", format='png', bbox_inches="tight")
        plt.savefig("shared/graph_properties/"+dimensions[f1]+"-VS-"+dimensions[f2]+".svg", format='svg', bbox_inches="tight")
        latex_output.write("\\begin{frame}{"+dimensions[f1].replace("_", "\_")+" VS "+dimensions[f2].replace("_", "\_")+"}\n    \\begin{center}        \\includegraphics[width=\\textwidth,height=0.8\\textheight,keepaspectratio]{images/"+dimensions[f1]+"-VS-"+dimensions[f2]+".png"+"}\n    \\end{center}\n\\end{frame}\n\n")
        plt.clf()
        print("        "+dimensions[f1]+"-VS-"+dimensions[f2]+" in %.1f seconds." %(timelib.time()-start_time))

###############################################################################
# END OF SCRIPT
latex_output.write("\\end{document}")
print("\n   * Done in {:.1f} seconds.\n".format(timelib.time()-begin_time))
