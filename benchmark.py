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

pathlib.Path("Matplot")

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

t = np.zeros((6, 1000))
for n in range(1000):
    print("   * Computing graph session features {}/1000...".format(n), end='\r')
    sample = sessions.sample(n)
    graphs = {}
    total_session = sample.shape[0]
    
    # graph-tools
    start_time = timelib.time()
    for gsid in sample.global_session_id.values:
        session = log[log.global_session_id==gsid]
        s_urls = session.requested_url
        s_urls = s_urls.append(session.referrer_url)
        s_urls.drop_duplicates(inplace=True)
        s_list = list(s_urls)
        g = Graph()
        v = {}
        for u in s_urls:
            v[u] = g.add_vertex()
        session.apply(lambda x: g.add_edge(v[x.referrer_url], v[x.requested_url]), axis=1)
        graphs[gsid] = g
    t[0][n] = timelib.time()-start_time

    # beetweenness
    start_time = timelib.time()
    for gsid in sample.global_session_id.values:
        vp, ep = betweenness(graphs[gsid])
        betweenness_val = vp.a
    t[1][n] = timelib.time()-start_time

    # depth
    start_time = timelib.time()
    for gsid in sample.global_session_id.values:
        origins = []
        for vertex in graphs[gsid].vertices():
            if vertex.in_degree() == 0:
                origins.append(vertex)
        depth=0
        for vertex in origins:
            dist = shortest_distance(graphs[gsid], source=vertex, directed=True).a
            dist[dist==2147483647]=-1
            depth=max(depth, dist.max())
    t[2][n] = timelib.time()-start_time

    # shortest distance
    start_time = timelib.time()
    for gsid in sample.global_session_id.values:
        dist = shortest_distance(graphs[gsid], directed=True).a
        dist[dist==2147483647]=-1
    t[3][n] = timelib.time()-start_time

    # in_degree
    start_time = timelib.time()
    for gsid in sample.global_session_id.values:
        for vertex in graphs[gsid].vertices():
            vertex.in_degree()
    t[4][n] = timelib.time()-start_time

    # out_degree
    start_time = timelib.time()
    for gsid in sample.global_session_id.values:
        for vertex in graphs[gsid].vertices():
            vertex.out_degree()
    t[5][n] = timelib.time()-start_time

print("   * Graph session features sessions computed.")

fig, ax = plt.subplots()
plt.plot(t[0], label="graph")
plt.plot(t[1], label="beetweenness")
plt.plot(t[2], label="depth")
plt.plot(t[3], label="shortest distance")
plt.plot(t[4], label="in degree")
plt.plot(t[5], label="out degree")
plt.grid(alpha=0.5)
plt.title("Benchmark")
plt.xlabel("Number of sessions")
plt.ylabel("Execution time (seconds)")
plt.gca().set_xscale('log')
plt.gca().set_yscale('log')
ax.legend(loc="upper left", bbox_to_anchor=(1,1))
plt.savefig("Matplot/benchmark.png", format='png', bbox_inches="tight", dpi=1000)
plt.savefig("shared/benchmark.svg", format='svg', bbox_inches="tight")

###############################################################################
# END OF SCRIPT
print("\n   * Done in {:.1f} seconds.\n".format(timelib.time()-begin_time))
