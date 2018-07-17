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

pathlib.Path("Matplot").mkdir(parents=True, exist_ok=True)
pathlib.Path("shared").mkdir(parents=True, exist_ok=True)

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

nb = 20
range_req = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 800, 1000]
r = np.zeros((len(range_req)-1, nb))
t = np.zeros((6, len(range_req)-1, nb))

for i in range(1, len(range_req)):

    sample = sessions[(sessions.requests>=range_req[i-1])&(sessions.requests<range_req[i])].sample(nb)
    gsid = sample.global_session_id.values
    print("   * Computing graph session features [{}, {}[ ...".format(range_req[i-1], range_req[i]), end='\r')

    # graph-tools
    graphs = {}
    for n in range(len(sample.global_session_id.values)):
        session = log[log.global_session_id==gsid[n]]
        session = session[["requested_url", "referrer_url"]]
        start_time = timelib.time()
        s_urls = session.requested_url
        s_urls = s_urls.append(session.referrer_url)
        s_urls.drop_duplicates(inplace=True)
        s_list = list(s_urls)
        g = Graph()
        v = {}
        for u in s_urls:
            v[u] = g.add_vertex()
        session.apply(lambda x: g.add_edge(v[x.referrer_url], v[x.requested_url]), axis=1)
        graphs[gsid[n]] = g 
        r[i-1][n] = session.shape[0]
        t[0][i-1][n] = timelib.time()-start_time

    # beetweenness
    for n in range(len(sample.global_session_id.values)):
        start_time = timelib.time()
        vp, ep = betweenness(graphs[gsid[n]])
        betweenness_val = vp.a
        t[1][i-1][n] = timelib.time()-start_time

    # depth
    for n in range(len(sample.global_session_id.values)):
        start_time = timelib.time()
        origins = []
        for vertex in graphs[gsid[n]].vertices():
            if vertex.in_degree() == 0:
                origins.append(vertex)
        depth=0
        for vertex in origins:
            dist = shortest_distance(graphs[gsid[n]], source=vertex, directed=True).a
            dist[dist==2147483647]=-1
            depth=max(depth, dist.max())
        t[2][i-1][n] = timelib.time()-start_time

    # shortest distance
    for n in range(len(sample.global_session_id.values)):
        start_time = timelib.time()
        dist = shortest_distance(graphs[gsid[n]], directed=True).get_2d_array(range(g.num_vertices()))
        dist[dist==2147483647]=-1
        t[3][i-1][n] = timelib.time()-start_time

    # in_degree
    for n in range(len(sample.global_session_id.values)):
        start_time = timelib.time()
        for vertex in graphs[gsid[n]].vertices():
            vertex.in_degree()
        t[4][i-1][n] = timelib.time()-start_time

    # out_degree
    for n in range(len(sample.global_session_id.values)):
        start_time = timelib.time()
        for vertex in graphs[gsid[n]].vertices():
            vertex.out_degree()
        t[5][i-1][n] = timelib.time()-start_time

print("   * Graph session features sessions computed.")

fig, ax = plt.subplots()
plt.scatter(r.reshape((len(range_req)-1)*nb), t[0].reshape((len(range_req)-1)*nb), label="graph")
plt.scatter(r.reshape((len(range_req)-1)*nb), t[1].reshape((len(range_req)-1)*nb), label="beetweenness")
plt.scatter(r.reshape((len(range_req)-1)*nb), t[2].reshape((len(range_req)-1)*nb), label="depth")
plt.scatter(r.reshape((len(range_req)-1)*nb), t[3].reshape((len(range_req)-1)*nb), label="shortest distance")
plt.scatter(r.reshape((len(range_req)-1)*nb), t[4].reshape((len(range_req)-1)*nb), label="in degree")
plt.scatter(r.reshape((len(range_req)-1)*nb), t[5].reshape((len(range_req)-1)*nb), label="out degree")
plt.grid(alpha=0.5)
plt.title("Benchmark")
plt.xlabel("Number of requests")
plt.ylabel("Execution time (seconds)")
plt.gca().set_xscale('log')
plt.gca().set_yscale('log')
ax.legend(loc="upper left", bbox_to_anchor=(1,1))
plt.savefig("Matplot/benchmark.png", format='png', bbox_inches="tight", dpi=1000)
plt.savefig("shared/benchmark.svg", format='svg', bbox_inches="tight")

###############################################################################
# END OF SCRIPT
print("\n   * Done in {:.1f} seconds.\n".format(timelib.time()-begin_time))
