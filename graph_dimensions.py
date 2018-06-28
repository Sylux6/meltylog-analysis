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
session_filename = "Outputs/Sessions.csv"
pages_filemname = "Outputs/pages_sessions.csv"

shutil.rmtree("Matplot/pages", ignore_errors=True)
pathlib.Path("Matplot/pages").mkdir(parents=True, exist_ok=True)

###############################################################################
# READING DATA FILES
print("\n   * Loading files ...")
# start_time = timelib.time()
# print("        Loading "+log_filename+" ...", end="\r")
# log = pd.read_csv(log_filename, sep=',', na_filter=False, low_memory=False)
# print("        "+log_filename+" loaded ({} rows) in {:.1f} seconds.".format(log.shape[0], timelib.time()-start_time))
start_time = timelib.time()
print("        Loading "+urls_filename+" ...", end="\r")
urls = pd.read_csv(urls_filename, sep=',', na_filter=False, low_memory=False)
print("        "+urls_filename+" loaded ({} rows) in {:.1f} seconds.".format(urls.shape[0], timelib.time()-start_time))
# start_time = timelib.time()
# print("        Loading "+session_filename+" ...", end="\r")
# sessions = pd.read_csv(session_filename, sep=',')
# print("        "+session_filename+" loaded ({} rows) in {:.1f} seconds.".format(sessions.shape[0], timelib.time()-start_time))
# sessions.fillna(0, inplace=True)
start_time = timelib.time() 
print("        Loading "+pages_filemname+" ...", end="\r")
pages_sessions = pd.read_csv(pages_filemname, sep=',', na_filter=False, low_memory=False)
print("        "+pages_filemname+" loaded ({} rows) in {:.1f} seconds.".format(urls.shape[0], timelib.time()-start_time))

dimensions = ["betweenness", "in_degree", "out_degree", "excentricity"]
pages_data = urls[urls.url.isin(pages_sessions.url)]

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

# excentricity
start_time= timelib.time()
print("        Computing excentricity ...", end='\r')
pages_excentricity = pages_sessions[["url", "excentricity"]].groupby("url").aggregate(lambda x: mean(x))
pages_data["excentricity"] = pages_data.url.map(pd.Series(data=pages_excentricity.excentricity.values, index=pages_excentricity.index))
print("        Excentricity computed in %.1f seconds." %(timelib.time()-start_time))

print("\n   * Scattering ...")

for f1 in range(0, len(dimensions)):
    for f2 in range(f1+1, len(dimensions)):
        print("        "+dimensions[f1]+"-VS-"+dimensions[f2])
        plt.scatter(pages_data[dimensions[f1]].values, pages_data[dimensions[f2]].values)
        plt.grid(alpha=0.5)
        plt.xlabel(dimensions[f1])
        plt.ylabel(dimensions[f2])
        plt.savefig("Matplot/pages/"+dimensions[f1]+"-VS-"+dimensions[f2]+".png", format='png', bbox_inches="tight", dpi=1000)
        plt.clf()

###############################################################################
# END OF SCRIPT
print("\n   * Done in {:.1f} seconds.\n".format(timelib.time()-begin_time))
