#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 11:54:39 2018

@author: pedro
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from math import * 
import random

cmap = plt.cm.gist_ncar

def plot_palette(labels,filename):
    cmap = plt.cm.tab20
    colormap = cmap(np.linspace(0., 1., len(labels)))
    image=np.zeros((len(labels),1,4))
    image[:,0,:]=colormap
    plt.imshow(image)
    plt.yticks(range(len(labels)), labels)
    plt.xticks([0], [''], fontsize=4)
    plt.savefig(filename, format='png')
    plt.clf()
    plt.close()    
    return

def alter_plot_palette(labels,filename): 
    cmap = plt.cm.tab20
    colormap = cmap(np.linspace(0., 1., len(labels)))
    image=np.zeros((1,len(labels),4))
    image[0,:,:]=colormap
    plt.imshow(image)
    plt.xticks(range(len(labels)), labels, fontsize=4)
    plt.yticks([0], [''], fontsize=4)
    plt.savefig(filename, format='png')
    plt.clf()
    plt.close()    
    return
    
 
def plot_sessions(cluster_log, filename, cluster_id, labels,
                  N_max_sessions=10, field="category", width=0,
                  max_time=None, time_resolution=None, mark_requests=False):
    
    plot_log=cluster_log.copy(deep=True)
    
    # list of sessions
    sessions = list(plot_log.global_session_id.unique())
    if plot_log.shape[0]>N_max_sessions:
        # random.shuffle(sessions)
        sessions=sessions[:N_max_sessions]
        # updating log
        plot_log=plot_log[plot_log.global_session_id.isin(sessions)]
    
    # Session data
    session_data=pd.DataFrame(columns=['id','start','end','timespan','span_sec'])
    session_data['id']=sessions
    
    # start and ending times
    session_start=plot_log[['timestamp','global_session_id']].groupby('global_session_id').min()
    session_end=plot_log[['timestamp','global_session_id']].groupby('global_session_id').max()
    session_data['start']=session_data.id.map(pd.Series(data=session_start.timestamp.values
                ,index=session_start.index))
    session_data['end']=session_data.id.map(pd.Series(data=session_end.timestamp.values
                ,index=session_end.index))
    session_data['timespan']=session_data.apply(lambda row: pd.Timedelta(pd.Timestamp(row.end)-pd.Timestamp(row.start)) , axis=1)
    session_data['span_sec']=session_data.timespan.apply(lambda x: x.seconds)
    
    # Time Frame
    if max_time is None:
        padding_seconds=ceil(session_data.span_sec.max()/9.0)
        time_window_seconds=session_data.span_sec.max()+padding_seconds+1
    else:
        time_window_seconds=max_time+1
    
    # Values and colors
    # black + colormap
    # req   + difference in field
    cmap = plt.cm.tab20
    colormap=cmap(np.linspace(0., 1., len(labels)))
    colormap=np.vstack((np.array([0,0,0,1]),colormap))
    
    # Filling the matrix
    if width == 0:
        width = int(0.005*time_window_seconds)
    image=np.zeros((len(sessions),int(time_window_seconds),4))
    session_counter=0
    for session in sessions:
        # selecting requests for the session
        session_log=plot_log[plot_log.global_session_id==session].sort_values(by='timestamp')
        session_log['relative_seconds']=0
        session_start=pd.Timestamp(session_log.timestamp.min())#pd.Timestamp(session_data[session_data.id==session].start.iloc[0])
        session_log['relative_seconds']=session_log['timestamp'].apply(lambda x: (pd.Timestamp(x)-session_start).seconds )
        for r in range(0,session_log.shape[0]):
            if r==session_log.shape[0]-1:
                # if it is the last (or only request) we fill with blank pixels
                image[session_counter,session_log.iloc[r].relative_seconds:,:]=np.zeros((int(time_window_seconds)-session_log.iloc[r].relative_seconds,4))
            else:
                # color patch
                pix_start=session_log.iloc[r].relative_seconds
                pix_end=session_log.iloc[r+1].relative_seconds
                idx=labels.index(session_log.iloc[r][field])
                paint_color=colormap[idx+1,:]
                paint_color_patch=np.reshape(np.tile(paint_color,pix_end-pix_start),newshape=(pix_end-pix_start,4))
                image[session_counter,pix_start:pix_end,:]=paint_color_patch
                # black patch
                pix_start=session_log.iloc[r].relative_seconds
                pix_end=pix_start+width
                paint_color=np.array([0,0,0,1])
                paint_black_patch=np.reshape(np.tile(paint_color,pix_end-pix_start),newshape=(pix_end-pix_start,4))
                image[session_counter,pix_start:pix_end,:]=paint_black_patch

        session_counter+=1    
            
    # Plotting the matrix
    plt.imshow(image)
    plt.gcf().set_size_inches([ 4, 4])
    plt.gca().axis('auto')
    plt.xlabel('Seconds')
    plt.ylabel('Sessions')
    plt.title("Cluster "+str(cluster_id))
    plt.grid(alpha=0.5)
    plt.tick_params(axis="both", which="major", labelsize=6)
    # Lines and axes
    ax = plt.gca();
    #   Major ticks
    ax.set_xticks([n*floor(0.1*time_window_seconds) for n in range(0,10)]);
    ax.set_yticks(np.arange(0, len(sessions), 1));
    ax.set_yticklabels(sessions);    
    #   Minor ticks
    ax.set_yticks(np.arange(-.5, len(sessions), 1), minor=True);
    # Gridlines based on minor ticks
    ax.grid(which='minor', color='w', linestyle='-', linewidth=2)
    # Saving and closing
    plt.tight_layout()
    plt.savefig(filename, format='png')
    plt.clf()
    plt.close()    
    return sessions;


def plot_pie(labels,distribution,filename,threshold=0.0):
#    distribution=distribution[distribution>=threshold]
#    labels=list((np.array(labels))[distribution>=threshold])
    
    
    colormap=cmap(np.linspace(0., 1., len(labels)))
    patches, texts = plt.pie(distribution,labels=labels,#autopct='%1.1f%%',
        shadow=True, startangle=90,colors=colormap)
    for i in range(0,len(labels)):
        if distribution[i]<threshold:
            texts[i].set_fontsize(0)
            patches[i].set_label('')
    plt.gca().axis('equal')
    plt.savefig(filename, format='pdf')
    plt.clf()
    plt.close()
    return;
    
def plot_markov_matrix(matrix,labels,filename):
    plt.imshow(matrix, interpolation='none')
    plt.xticks(range(len(labels)), labels, fontsize=4)
    plt.yticks(range(len(labels)), labels, fontsize=8)
    plt.colorbar()
    plt.gcf().set_size_inches([ 7, 7])
    plt.savefig(filename)
    plt.clf()
    plt.close()
    return;

def plot_session_distributions(distributions,labels,filename):
    ylabels=['Session %d'%(n+1) for n in range(0,distributions.shape[0])]
    plt.imshow(distributions, interpolation='none',aspect='equal')
    plt.xticks(range(len(labels)), labels, fontsize=4)
    plt.yticks(range(len(ylabels)), ylabels, fontsize=4)#
    plt.colorbar()
    plt.gcf().set_size_inches([ 7, 2])
    plt.savefig(filename)
    plt.clf()
    plt.close()
    return;
