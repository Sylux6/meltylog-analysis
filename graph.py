import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from graph_tool.all import *

def session_draw(cluster_id, n_id, sessions_id, log, pages, labels):
    cmap = plt.cm.tab20
    colormap=cmap(np.linspace(0., 1., len(labels)))
    colormap=np.vstack((np.array([0,0,0,1]),colormap))
    for id in sessions_id:
        session = log[log.global_session_id==id]
        urls = session.requested_url
        urls = urls.append(session.referrer_url)
        urls.drop_duplicates(inplace=True)
        g = Graph()
        v = {}
        color = g.new_vertex_property("vector<float>")
        halo = g.new_vertex_property("bool")
        for u in urls:
            v[u] = g.add_vertex()
            if not u.replace('"', "").startswith("www.melty.fr"):
                halo[v[u]] = True
            else:
                halo[v[u]] = False
            idx = labels.index(pages[pages.url==u].category.values[0])
            color[v[u]] = colormap[idx+1].tolist()
        session.apply(lambda x: g.add_edge(v[x.referrer_url], v[x.requested_url]), axis=1)
        graph_draw(g, vertex_halo=halo, vertex_fill_color=color, output="Graphs/"+str(n_id)+"/"+str(cluster_id)+"_session"+str(id)+".png")
    return