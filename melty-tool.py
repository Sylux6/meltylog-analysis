import time as timelib
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as News_proportion
from sklearn.cluster import KMeans

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
import sys, os

#############################################################
#############################################################
####                 _ _               _              _  ####
####  _ __ ___   ___| | |_ _   _      | |_ ___   ___ | | ####
#### | '_ ` _ \ / _ \ | __| | | |_____| __/ _ \ / _ \| | ####
#### | | | | | |  __/ | |_| |_| |_____| || (_) | (_) | | ####
#### |_| |_| |_|\___|_|\__|\__, |      \__\___/ \___/|_| ####
####                       |___/                         ####
####                                                     ####
#############################################################
#############################################################

# ================= #
#     VARIABLES     #
# ================= #

logo = "\n                _ _               _              _ \n _ __ ___   ___| | |_ _   _      | |_ ___   ___ | |\n| '_ ` _ \ / _ \ | __| | | |_____| __/ _ \ / _ \| |\n| | | | | |  __/ | |_| |_| |_____| || (_) | (_) | |\n|_| |_| |_|\___|_|\__|\__, |      \__\___/ \___/|_|\n                      |___/                        \n\n"
log_filename = "Outputs/MyLog.csv"
session_filename = "Outputs/Sessions.csv"
dimensions = ['requests', 'timespan', 'requested_category_richness', 'requested_topic_richness', 'star_chain_like', 'bifurcation', 'requested_topic_proportion', 'entropy', 'variance', 'popularity_mean', 'inter_req_mean_seconds', 'TV_proportion', 'Series_proportion', 'News_proportion', 'Celebrities_proportion', 'VideoGames_proportion', 'Music_proportion', 'Movies_proportion', 'Sport_proportion', 'Comic_proportion', 'Look_proportion', 'Other_proportion', 'Humor_proportion', 'Student_proportion', 'Events_proportion', 'Wellbeing_proportion', 'None_proportion', 'Food_proportion', 'Tech_proportion', 'standard_deviation', 'read_pages']
normalized_dimensions = map(lambda x: "normalized_"+x, dimensions) # normalized dimensions labels list

# ====================== #
#     MENU FUNCTIONS     #
# ====================== #

menu_actions = {}

def main_menu():
    global menu_actions
    menu_actions = {
        "main_menu": main_menu,
        "1": clustering_menu,
        "2": session_log_menu,
        "q": exit,
    }
    os.system("clear")
    print(logo)
    print("# ==================== #\n#     MAIN PROGRAM     #\n# ==================== #\n")
    print("1. Clustering")
    print("2. Session log")
    print("\nq. Quit")
    choice = input(" >>  ")
    exec_menu(choice)
    return

def exec_menu(choice):
    ch = choice.lower()
    if ch == "":
        menu_actions["main_menu"]()
    else:
        try:
            menu_actions[ch]()
        except KeyError:
            print("Invalid selection, please try again.\n")
            exec_menu(input(" >>  "))
    return

def clustering_menu():
    global menu_actions
    menu_actions = {
        "main_menu": main_menu,
        "0": exit,
    }
    os.system("clear")
    print(logo)
    print("# ================== #\n#     CLUSTERING     #\n# ================== #\n")
    print("Enter the number of clusters")
    n_clusters = input(" >>  ")
    if n_clusters == "q":
        main_menu()
        return
    dim = select_dimensions()
    if not n_clusters.isdigit() or int(n_clusters) <= 0 or len(dim) == 0:
        clustering_menu()
    clustering(int(n_clusters), dim)
    return

def select_dimensions():
    dim = []
    choice = "start"
    while choice != "":
        os.system("clear")
        print(logo)
        print("# ================== #\n#     CLUSTERING     #\n# ================== #\n")
        print("Type a dimension to add/remove (press ENTER to confirm)")
        for d in dimensions:
            if d in dim:
                print("   [X] {}".format(d))
            else:
                print("   [ ] {}".format(d))
        choice = input(" >>  ")
        if choice == "q":
            main_menu()
        elif choice in dim:
            dim.remove(choice)
        elif choice in dimensions:
            dim.append(choice)
    return dim

def session_log_menu():
    os.system("clear")
    print(logo)
    print("# ======================== #\n#     SESSION LOG MENU     #\n# ======================== #\n")
    print("Enter a <global_session_id> to print it")
    choice = input(" >>  ")
    print_session(choice)
    return

def exit():
    os.system("clear")
    sys.exit()

# ======================= #
#     MELTY FUNCTIONS     #
# ======================= #

def print_session(session_id):
    if session_id == "q":
        menu_actions["main_menu"]()
    elif session_id.isdigit() and int(session_id) > 0:
        session = log[log.global_session_id == int(session_id)]
        if session.shape[0] == 0:
            print("Session {} not found.".format(session_id))
        else:
            print(session[["timestamp", "referrer_url", "requested_url", "agent"]])
        input("Press ENTER to continue ...")
    else:
        print("Invalid input, please try again.\n")
        print_session(input(" >>  "))
    main_menu()
    return

def clustering(n_clusters, dim):
    print(n_clusters)
    print(dim)
    input("Press ENTER to continue ...")
    main_menu()
    return

# ==================== #
#     MAIN PROGRAM     #
# ==================== #

os.system("clear")
print(logo)
# loading files
print("\n   * Loading files ...")
start_time = timelib.time()
print("\n        Loading "+log_filename+" ...", end="\r")
log = pd.read_csv(log_filename, sep=',', na_filter=False, low_memory=False)
print("        "+log_filename+" loaded ({} rows) in {:.1f} seconds.".format(log.shape[0], timelib.time()-start_time))
start_time = timelib.time()
print("        Loading "+session_filename+" ...", end="\r")
sessions = pd.read_csv(session_filename, sep=',')
print("        "+session_filename+" loaded ({} rows) in {:.1f} seconds.".format(sessions.shape[0], timelib.time()-start_time))
sessions.fillna(0, inplace=True)

pathlib.Path("Latex").mkdir(parents=True, exist_ok=True)
pathlib.Path("Graphs").mkdir(parents=True, exist_ok=True)
pathlib.Path("Clusters").mkdir(parents=True, exist_ok=True)
pathlib.Path("Sessions").mkdir(parents=True, exist_ok=True)

main_menu()
