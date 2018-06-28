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

<<<<<<< HEAD
sample = sessions
# sample = sessions[sessions.requests > 6]
# sample = sample[sample.requests > 1000]
# sample = sample.sample(n=10000)

pages_sessions = pd.DataFrame(columns=["global_session_id", "url", "betweenness", "in_degree", "out_degree", "excentricity"])
=======
sample = sessions[sessions.requests > 6]
sample = sample[sample.requests < 1000]
# sample = sessions.sample(n=int(sessions.shape[0]*0.05))
pages_sessions = pd.DataFrame(columns=["global_session_id", "url", "betweeness", "in_degree", "out_degree", "excentricity"])
>>>>>>> c8c0f406b72844e1c6280c071d4d04709a1a24b7
session_data = pd.DataFrame(columns=["global_session_id", "diameter"])

total_session = sample.shape[0]
count = 0

start_time = timelib.time()
<<<<<<< HEAD
for gsid in sample.global_session_id.values:
    count = count + 1
    print("   * Computing graph session features {}/{}...".format(count, total_session), end='\r')
    
=======
print("\n   * Computing sessions diameter {}/{}...".format(count, total_session), end='\r')
for gsid in sample.global_session_id.values:
    count = count + 1
    print("   * Computing sessions diameter {}/{}...".format(count, total_session), end='\r')
>>>>>>> c8c0f406b72844e1c6280c071d4d04709a1a24b7
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
    g.set_directed(False)

<<<<<<< HEAD
    vp, ep = betweenness(g)
    betweenness_val = vp.a
    dist = np.zeros((len(s_list), len(s_list)))
    for i in range(0, len(s_list)):
        dist[i] = shortest_distance(g, source=v[s_list[i]]).a


    for i in range(0, len(s_list)):
=======
    # # diameter
    # for i in range(0, len(s_list)):
    #     diameter = 0
    #     for j in range(i+1, len(s_list)):
    #         vlist, elist = shortest_path(g, v[s_list[i]], v[s_list[j]])
    #         if len(vlist) > diameter:
    #             diameter = len(vlist)
    #     session_data = session_data.append({"global_session_id": gsid, "diameter": diameter-1}, ignore_index=True)

    # pages sessions
    vp, ep = betweenness(g)
    betweenness_val = vp.a
    for i in range(0, len(s_list)):
        excentricity = 0
        for j in range(0, len(s_list)):
            dist = shortest_distance(g, source=v[s_list[i]], target=v[s_list[j]])
            if dist > excentricity:
                excentricity = dist
>>>>>>> c8c0f406b72844e1c6280c071d4d04709a1a24b7
        pages_sessions = pages_sessions.append(
            {
                "global_session_id": gsid,
                "url": s_list[i],
                "in_degree": v[s_list[i]].in_degree(),
                "out_degree": v[s_list[i]].out_degree(),
<<<<<<< HEAD
                "betweenness": betweenness_val[i],
                "excentricity": dist[i].max()
            },
            ignore_index=True)
    
    session_data = session_data.append({"global_session_id": gsid, "diameter": dist.max()}, ignore_index=True)

sessions["diameter"] = sessions.global_session_id.map(pd.Series(data=session_data.diameter, index=session_data.global_session_id))
sessions.to_csv(r"Outputs/sessions_new.csv", index=None)
pages_sessions.to_csv(r"Outputs/pages_sessions_new.csv", index=None)
=======
                "betweeness": betweenness_val[i],
                "excentricity": excentricity
            },
            ignore_index=True)

# sessions["diameter"] = sessions.global_session_id.map(pd.Series(data=session_data.diameter, index=session_data.global_session_id))
pages_sessions.to_csv(r"Outputs/pages_sessions.csv", index=None)
>>>>>>> c8c0f406b72844e1c6280c071d4d04709a1a24b7
print("   * Diameter sessions computed in %.1f seconds." %(timelib.time()-start_time))

###############################################################################
# END OF SCRIPT
print("\n   * Done in {:.1f} seconds.\n".format(timelib.time()-begin_time))
