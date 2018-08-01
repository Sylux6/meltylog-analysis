#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 17:02:01 2018

@author: pedroramaciotti
"""

import pandas as pd
import time as timelib
import numpy as np
from math import log

def zf(s):
    s=str(s)
    if len(s)==1:
        return '0'+s
    else:
        return s

# shape parameters

def star_chain_like(log):
    return log[~log.requested_url.isin(log.referrer_url)].shape[0]/log.shape[0]

def bifurcation(log):
    return (log.referrer_url.value_counts()>1).sum()/len(log.referrer_url.unique())

# time intervals 

def mean_interval_time(log,field='timestamp'):
    if log.shape[0]==0:
        return 0.0
    return log[field].apply(lambda x: pd.Timestamp(x)).diff().apply(lambda x: x.seconds).mean()

def variance_interval_time(log,field='timestamp'):
    if log.shape[0]==0:
        return 0.0
    return log[field].apply(lambda x: pd.Timestamp(x)).diff().apply(lambda x: x.seconds).var()
    
# Assign data tu log entries
def log_classification(log,urldata,fields):
    new_log=log#.copy(deep=True)
    for field in fields:
        new_log['requested_'+field]=''
        new_log['referrer_'+field]=''
        lookup_table=pd.Series(data=urldata[field].values, index=urldata.url)
    
        req_urls=new_log.requested_url
        ref_urls=new_log.referrer_url
    
        new_log['requested_'+field]=req_urls.map(lookup_table)
        new_log['referrer_'+field]=ref_urls.map(lookup_table)
    return new_log

# Identify sessions

def log_sessions(log,max_inactive_minutes):
    new_log=log#.copy(deep=True)
    # sort by timestamp
    new_log.sort_values(by=['user','timestamp'],ascending=[True,True],inplace=True)
    # Detect time jumps greater than threshold
    gt_xmin = new_log.timestamp.apply(lambda x: pd.Timestamp(x)).diff() > pd.Timedelta(minutes=max_inactive_minutes)
    # Detect changes of user
    diff_user = new_log.user != new_log.user.shift()
    new_log['global_session_id'] = (diff_user | gt_xmin).cumsum()
    return new_log

# proportional abundance

def proportional_abundance(log,field,path='IT'):
    if log.shape[0]==0:
        raise AssertionError('Unexpected case.')
    new_log=log.copy(deep=True)
    if path=='IT':
        new_log.drop_duplicates(subset=['requested_url'],inplace=True)
    new_log['requests']=pd.Series(np.ones(new_log.shape[0])).values
    histogram=new_log[[field,'requests']].groupby(field).count()
    pa_df=histogram/histogram.values.sum()
    if abs(1.0-pa_df.values.sum())>1e-8:
        raise AssertionError("ERROR: Proportional abundance distribution does not sum up to one.")
    return pa_df.values[:,0],list(pa_df.index)
    
def normalize(serie):
    max_value = serie.max()
    min_value = serie.min()
    if max_value == min_value:
        return serie.apply(lambda x: x / serie.shape[0])
    else:
        return serie.apply(lambda x: (x - min_value)/(max_value- min_value))

def log_normalize(serie):
    if serie.min() == 0:
        serie = serie + 1
    max_value = log(serie.max())
    min_value = log(serie.min())
    if max_value == min_value:
        return serie.apply(lambda x: x / serie.shape[0])
    else:
        return serie.apply(lambda x: (log(x) - min_value)/(max_value- min_value))

def features_map(feature):
    if feature == "requests":
        return "Number of requests"
    elif feature == "timespan":
        return "Session Timespan"
    elif feature == "standard_deviation":
        return "Inter Request Stand. Deviation"
    elif feature == "inter_req_mean_seconds":
        return "Inter Request Mean Time"
    elif feature == "star_chain_like":
        return "Star-like Chain-like Index"
    else:
        return feature