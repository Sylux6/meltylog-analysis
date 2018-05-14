import time as timelib
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
from collections import Counter

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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
from math import *

##################################
##################################
##################################
####                          ####
#### Main                     ####
####                          ####
##################################
##################################

pathlib.Path("Outputs").mkdir(parents=True, exist_ok=True)
begin_time = timelib.time()

####################
## READING THE FILES
print("\n   * Loading files ...")
log_filename = 'Melty/log.csv'
url_data_filename = 'Melty/pages.csv'
start_time = timelib.time()
print("        Loading "+log_filename+" ...", end="\r")
log = pd.read_csv(log_filename, sep=',', dtype='object', na_filter=False)
# robot filter
log = log[~log.agent.str.contains("BTWebClient")]
log = log[~log.agent.str.contains("Genieo")]
log = log[~log.agent.str.contains("urllib")]
print("        "+log_filename+" loaded ({} rows) in {:.1f} seconds.".format(log.shape[0], timelib.time()-start_time))
start_time = timelib.time()
print("        Loading "+url_data_filename+" ...", end="\r")
urldata = pd.read_csv(url_data_filename, sep=',',dtype='object', na_filter=False)
urldata.drop_duplicates(subset=['url'], inplace=True)
print("        "+url_data_filename+" loaded ({} rows) in {:.1f} seconds.".format(urldata.shape[0], timelib.time()-start_time))

#######################################
## ASSIGNING URL TO URLs IN LOG ENTRIES
start_time = timelib.time()
print("\n   * Asigning URL data to log entries ...")
log = log_classification(log, urldata, fields=['category', 'topic'])
## ASSIGNING VALUES TO NA VALUES
log.requested_category.fillna('other', inplace=True)
log.referrer_category.fillna('other', inplace=True)
log.requested_topic.fillna('Unclassifiable', inplace=True)
log.referrer_topic.fillna('Unclassifiable', inplace=True)
print("     URL data assigned to log entries in %.1f seconds." %(timelib.time()-start_time))

####################################
# DATAFRAME WITH SESSION INFORMATION
session_data = pd.DataFrame(columns=['global_session_id', 'user', 'requests', 'req_from_unknown', 'pc_from_unknown','timespan', 'start', 'end','requested_category_richness', 'requested_topic_richness','root_social', 'root_search','star_chain_like', 'bifurcation','cluster_id','requested_topic_proportion','entropy', "variance", "popularity_mean"])

###############################
# COMPUTING SESSION INFORMATION

print("\n   * Computing session information ...")

# Computing global session ids
start_time = timelib.time()
print("        Computing session IDs ...", end='\r')
log = log_sessions(log, max_inactive_minutes=30)
print("        Session IDs computed in %.1f seconds." %(timelib.time()-start_time))

# remove parallel edges
log = log.drop_duplicates(subset=["global_session_id", "referrer_url", "requested_url"])
log.to_csv(r"Outputs/_MyLog.csv", index=None)
print("        Parallel requests removed.")

# Counting requests per session
start_time = timelib.time()
print("        Computing session IDs ...", end='\r')
log['requests'] = pd.Series(np.ones(log.shape[0])).values
session_requests = log[['global_session_id', 'requests']].groupby('global_session_id').count().sort_values(by='requests', ascending=False)
log.drop(labels=['requests'], inplace=True, axis=1)
session_data['global_session_id'] = session_requests.index
session_data['requests'] = session_requests.values
print("        Requests per session computed in %.1f seconds." %(timelib.time()-start_time))

# Computing how many requests come from unknown URLs
start_time = timelib.time()
print("        Computing percentage of requests come from unknown URLs ...", end='\r')
from_unknown = log[['referrer_url', 'global_session_id']].groupby('global_session_id').aggregate(lambda x: sum(x == 'Unknown URL'))
session_data['req_from_unknown'] = session_data.global_session_id.map(from_unknown.referrer_url)
session_data['pc_from_unknown'] = session_data.apply(lambda row: 100.0*row.req_from_unknown/row.requests, axis=1)
print("        Percentage of requests coming from unknown URLs computed in %.1f seconds." %(timelib.time()-start_time))

# Determining the user of the sessions
start_time = timelib.time()
print("        Computing the user of each session ...", end='\r')
lookup_table = pd.Series(data=log[['user', 'global_session_id']].drop_duplicates().user.values, index=log[['user', 'global_session_id']].drop_duplicates().global_session_id)
session_data['user'] = session_data.global_session_id.map(lookup_table)
print("        Sessions' users computed in %.1f seconds." %(timelib.time()-start_time))

# Determining session timespan
start_time = timelib.time()
print("        Computing the time span of each session ...", end='\r')
session_start = log[['timestamp', 'global_session_id']].groupby('global_session_id').min()
session_end = log[['timestamp', 'global_session_id']].groupby('global_session_id').max()
session_data['start'] = session_data.global_session_id.map(pd.Series(data=session_start.timestamp.values, index=session_start.index))
session_data['end'] = session_data.global_session_id.map(pd.Series(data=session_end.timestamp.values, index=session_end.index))
session_data['timespan'] = session_data.apply(lambda row: pd.Timedelta(pd.Timestamp(row.end)-pd.Timestamp(row.start)), axis=1)
print("        Sessions' time spans computed in %.1f seconds." %(timelib.time()-start_time))

# Determining session richness
start_time = timelib.time()
print("        Computing the richness of each session ...", end='\r')
session_category_richness = log[['requested_category', 'global_session_id']].groupby('global_session_id').aggregate(lambda x: len(set(x)))
session_topic_richness = log[['requested_topic', 'global_session_id']].groupby('global_session_id').aggregate(lambda x: len(set(x)))
session_data['requested_category_richness'] = session_data.global_session_id.map(pd.Series(data=session_category_richness.requested_category.values, index=session_category_richness.index))
session_data['requested_topic_richness'] = session_data.global_session_id.map(pd.Series(data=session_topic_richness.requested_topic.values, index=session_topic_richness.index))
print("        Sessions' richnesses computed in %.1f seconds." %(timelib.time()-start_time))

# Determining the origin of sessions
start_time = timelib.time()
print("        Determining the origin of each session ...", end='\r')
origin_social = log[['referrer_category', 'global_session_id']].groupby('global_session_id').aggregate(lambda x: any(x == 'social'))
origin_search = log[['referrer_category', 'global_session_id']].groupby('global_session_id').aggregate(lambda x: any(x == 'search'))
session_data['root_social'] = session_data.global_session_id.map(origin_social.referrer_category)
session_data['root_search'] = session_data.global_session_id.map(origin_search.referrer_category)
print("        Sessions' origins computed in %.1f seconds." %(timelib.time()-start_time))

# Determining mean seconds between requests
start_time = timelib.time()
print("        Computing mean inter-request time ...", end='\r')
inter_request_mean_seconds = log[['timestamp', 'global_session_id']].groupby('global_session_id').aggregate(lambda x: mean_interval_time(x))
session_data['inter_req_mean_seconds'] = session_data.global_session_id.map(inter_request_mean_seconds.timestamp)
print("        Mean inter-request computed in %.1f seconds." %(timelib.time()-start_time))

# Determining shape parameters
start_time = timelib.time()
print("        Computing sessions' shape parameters ...", end='\r')
star_chain_like = log[['referrer_url', 'requested_url', 'global_session_id']].groupby('global_session_id').aggregate(lambda x: star_chain_like(x))
session_data['star_chain_like'] = session_data.global_session_id.map(star_chain_like.referrer_url)
bifurcation = log[['referrer_url', 'requested_url', 'global_session_id']].groupby('global_session_id').aggregate(lambda x: bifurcation(x))
session_data['bifurcation'] = session_data.global_session_id.map(bifurcation.referrer_url)
print("        Shape parameters computed in %.1f seconds." %(timelib.time()-start_time))

# Computing topic proportion
start_time = timelib.time()
print("        Computing requested topic proportion ...", end="\r")
topic_list = [t for t in urldata[["topic"]].drop_duplicates("topic")["topic"].tolist() if "(ext)" not in t]
for topic in topic_list:
    proportion_topic = log[["global_session_id", "requested_topic"]].groupby("global_session_id").apply(lambda x: x[x.requested_topic == topic].shape[0])
    session_data[topic+"_proportion"] = session_data.global_session_id.map(pd.Series(data=proportion_topic.values, index=proportion_topic.index))
    session_data[topic+"_proportion"] = session_data[topic +"_proportion"] / session_data["requests"]
print("        Requested topic proportion computed in %.1f seconds." %(timelib.time()-start_time))

# Computing Shannon entropy
start_time= timelib.time()
print("        Computing shannon entropy ...", end='\r')
entropy = log[["global_session_id", "requested_topic"]].groupby("global_session_id").apply(lambda x: x.groupby("requested_topic").aggregate(lambda y: y.count()/x.shape[0]))
shannon = entropy.global_session_id.groupby("global_session_id").apply(lambda x: ShannonEntropy(x.values))
session_data["entropy"] = session_data.global_session_id.map(pd.Series(data=shannon.values, index=shannon.index))
print("        Shannon entropy computed in %.1f seconds." %(timelib.time()-start_time))

# Determining variance/standard_deviation between requests
start_time = timelib.time()
print("        Computing variance inter-request time ...", end='\r')
inter_request_variance_seconds = log[['timestamp', 'global_session_id']].groupby('global_session_id').aggregate(lambda x: variance_interval_time(x))
session_data['variance'] = session_data.global_session_id.map(inter_request_variance_seconds.timestamp)
session_data["standard_deviation"] = session_data.variance.apply(lambda x: sqrt(x))
print("        Variance inter-request computed in %.1f seconds." %(timelib.time()-start_time))

# Determining number of "read" pages
start_time = timelib.time()
print("        Determining number of read pages ...", end='\r')
read_pages = log[log.referrer_url.str.startswith("www.melty.fr")]
read_pages = read_pages[["referrer_url", "global_session_id"]].groupby("global_session_id").aggregate(lambda x: x.drop_duplicates(subset="referrer_url").count())
session_data["read_pages"] = session_data.global_session_id.map(read_pages.referrer_url)
session_data.read_pages.fillna(0, inplace=True)
print("        Number of read pages determined in %.1f seconds." %(timelib.time()-start_time))

# Computing popularity requested_url
start_time = timelib.time()
print("        Computing requested_url popularity ...", end='\r')
log["popularity_requested_url"] = 1
popularity = log[["requested_url", "popularity_requested_url"]].groupby("requested_url").count()
popularity_lookup_table = pd.Series(data=popularity.popularity_requested_url, index=popularity.index)
popularity_lookup_table = popularity_lookup_table.apply(lambda x: x / log.shape[0])
log["popularity_requested_url"] = log.requested_url.map(popularity_lookup_table)
popularity_mean = log[["global_session_id", "popularity_requested_url"]].groupby("global_session_id").aggregate(lambda x : x.mean())
session_data["popularity_mean"] = session_data.global_session_id.map(popularity_mean.popularity_requested_url)
print("        Requested_url popularity computed in %.1f seconds." %(timelib.time()-start_time))

##################################
##################################
##################################
####                          ####
#### RESULTS                  ####
####                          ####
##################################
##################################

result = session_data.sort_values(by="global_session_id", ascending=True)
result["timespan"] = result.timespan.apply(lambda x: x.seconds)

start_time = timelib.time()
print("\n   * Generating 'Outputs/Sessions.csv' ...", end="\r")
result.to_csv(r"Outputs/_Sessions.csv", index=None)
print("   * 'Outputs/Sessions.csv' generated in {:.1f} seconds.".format(timelib.time()-start_time))

###############
# END OF SCRIPT
print("\n   * Done in {:.1f} seconds.\n".format(timelib.time()-begin_time))