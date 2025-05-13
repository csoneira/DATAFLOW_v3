#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# a = 1/0

stratos_save = True

fast_mode = False # Do not iterate TimTrack, neither save figures, etc.
debug_mode = False # Only 10000 rows with all detail
last_file_test = False

alternative_fitting = True

"""
Created on Thu Jun 20 09:15:33 2024

@author: csoneira@ucm.es
"""

# print("""
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠟⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⠆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣭⡆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣹⠄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⡁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⠄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⣀⣤⠤⢤⣀⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣠⠴⠒⢋⣉⣀⣠⣄⣀⣈⡇
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣸⡆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⣴⣾⣯⠴⠚⠉⠉⠀⠀⠀⠀⣤⠏⣿
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡿⡇⠁⠀⠀⠀⠀⡄⠀⠀⠀⠀⠀⠀⠀⠀⣠⣴⡿⠿⢛⠁⠁⣸⠀⠀⠀⠀⠀⣤⣾⠵⠚⠁
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠰⢦⡀⠀⣠⠀⡇⢧⠀⠀⢀⣠⡾⡇⠀⠀⠀⠀⠀⣠⣴⠿⠋⠁⠀⠀⠀⠀⠘⣿⠀⣀⡠⠞⠛⠁⠂⠁⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡈⣻⡦⣞⡿⣷⠸⣄⣡⢾⡿⠁⠀⠀⠀⣀⣴⠟⠋⠁⠀⠀⠀⠀⠐⠠⡤⣾⣙⣶⡶⠃⠀⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣂⡷⠰⣔⣾⣖⣾⡷⢿⣐⣀⣀⣤⢾⣋⠁⠀⠀⠀⣀⢀⣀⣀⣀⣀⠀⢀⢿⠑⠃⠀⠀⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠠⡦⠴⠴⠤⠦⠤⠤⠤⠤⠤⠴⠶⢾⣽⣙⠒⢺⣿⣿⣿⣿⢾⠶⣧⡼⢏⠑⠚⠋⠉⠉⡉⡉⠉⠉⠹⠈⠁⠉⠀⠨⢾⡂⠀⠀⠀⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠂⠀⠀⠀⠂⠐⠀⠀⠀⠈⣇⡿⢯⢻⣟⣇⣷⣞⡛⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⣠⣆⠀⠀⠀⠀⢠⡷⡛⣛⣼⣿⠟⠙⣧⠅⡄⠀⠀⠀⠀⠀⠀⠰⡆⠀⠀⠀⠀⢠⣾⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⣴⢶⠏⠉⠀⠀⠀⠀⠀⠿⢠⣴⡟⡗⡾⡒⠖⠉⠏⠁⠀⠀⠀⠀⣀⢀⣠⣧⣀⣀⠀⠀⠀⠚⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⣠⢴⣿⠟⠁⠀⠀⠀⠀⠀⠀⠀⣠⣷⢿⠋⠁⣿⡏⠅⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠙⣿⢭⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⢀⡴⢏⡵⠛⠀⠀⠀⠀⠀⠀⠀⣀⣴⠞⠛⠀⠀⠀⠀⢿⠀⠂⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠂⢿⠘⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⣀⣼⠛⣲⡏⠁⠀⠀⠀⠀⠀⢀⣠⡾⠋⠉⠀⠀⠀⠀⠀⠀⢾⡅⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
# ⠀⠀⡴⠟⠀⢰⡯⠄⠀⠀⠀⠀⣠⢴⠟⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⣹⠆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
# ⠀⡾⠁⠁⠀⠘⠧⠤⢤⣤⠶⠏⠙⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢾⡃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
# ⠘⣇⠂⢀⣀⣀⠤⠞⠋⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣼⠇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
# ⠀⠈⠉⠉⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠾⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢼⡆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢰⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠛⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
# """)

print("\n                     . .:.:.:.:. .:\\     /:. .:.:.:.:. ,")
print("               .-._  `..:.:. . .:.:`- -':.:. . .:.:.,'  _.-.")
print("              .:.:.`-._`-._..-''_...---..._``-.._.-'_.-'.:.:.")
print("           .:.:. . .:_.`' _..-''._________,``-.._ `.._:. . .:.:.")
print("        .:.:. . . ,-'_.-''      ||_-(O)-_||      ``-._`-. . . .:.:.")
print("       .:. . . .,'_.'           '---------'           `._`.. . . .:.")
print("     :.:. . . ,','               _________               `.`. . . .:.:")
print("    `.:.:. .,','            _.-''_________``-._            `._.     _.'")
print("  -._  `._./ /            ,'_.-'' ,       ``-._`.          ,' '`:..'  _.-")
print(" .:.:`-.._' /           ,','                   `.`.       /'  '  \\\\.-':.:.")
print(" :.:. . ./ /          ,','               ,       `.`.    / '  '  '\\\\ .:. :")
print(":.:. . ./ /          / /    ,                      \\ \\  :  '  '  ' \\\\. .:.:")
print(".:. . ./ /          / /            ,          ,     \\ \\ :  '  '  ' '::. .:.")
print(":. . .: :    o     / /                               \\ ;'  '  '  ' ':: . .:")
print(".:. . | |   /_\\   : :     ,                      ,    : '  '  '  ' ' :: .:.")
print(":. . .| |  ((<))  | |,          ,       ,             |\\'__',-._.' ' ||. .:")
print(".:.:. | |   `-'   | |---....____                      | ,---\\/--/  ' ||:.:.")
print("------| |         : :    ,.     ```--..._   ,         |''  '  '  ' ' ||----")
print("_...--. |  ,       \\ \\             ,.    `-._     ,  /: '  '  '  ' ' ;;..._")
print(":.:. .| | -O-       \\ \\    ,.                `._    / /:'  '  '  ' ':: .:.:")
print(".:. . | |_(`__       \\ \\                        `. / / :'  '  '  ' ';;. .:.")
print(":. . .<' (_)  `>      `.`.          ,.    ,.     ,','   \\  '  '  ' ;;. . .:")
print(".:. . |):-.--'(         `.`-._  ,.           _,-','      \\ '  '  '//| . .:.")
print(":. . .;)()(__)(___________`-._`-.._______..-'_.-'_________\\'  '  //_:. . .:")
print(".:.:,' \\/\\/--\\/--------------------------------------------`._',;'`. `.:.:.")
print(":.,' ,' ,'  ,'  /   /   /   ,-------------------.   \\   \\   \\  `. `.`. `..:")
print(",' ,'  '   /   /   /   /   //                   \\\\   \\   \\   \\   \\  ` `.SSt\n")


print("----------------------------------------------------------------------")
print("-------------------- RAW TO LIST SCRIPT IS STARTING ------------------")
print("----------------------------------------------------------------------")

# globals().clear()

# Standard library
import os
import re
import sys
import csv
import math
import random
import shutil
import builtins
from datetime import datetime, timedelta
from collections import defaultdict
from itertools import combinations
from functools import reduce

# Scientific computing
from math import sqrt
import numpy as np
import pandas as pd
import scipy.linalg as linalg
from scipy.constants import c
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit
from scipy.stats import (norm, poisson, linregress, median_abs_deviation, skew)

# Machine learning
from sklearn.linear_model import LinearRegression

# Plotting
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D  # for 3D plotting

# Image processing
from PIL import Image

# Progress bar
from tqdm import tqdm
    
# Store the current time at the start. To time the execution
start_execution_time_counting = datetime.now()

# -----------------------------------------------------------------------------
# Stuff that could change between mingos --------------------------------------
# -----------------------------------------------------------------------------

# Check if the script has an argument
if len(sys.argv) < 2:
    print("Error: No station provided.")
    print("Usage: python3 script.py <station>")
    sys.exit(1)

# Get the station argument
station = sys.argv[1]
# print(f"Station: {station}")

# -----------------------------------------------------------------------------

print("Creating the necessary directories...")

date_execution = datetime.now().strftime("%y-%m-%d_%H.%M.%S")

# Define base working directory
home_directory = os.path.expanduser(f"~")
station_directory = os.path.expanduser(f"~/DATAFLOW_v3/STATIONS/MINGO0{station}")
base_directory = os.path.expanduser(f"~/DATAFLOW_v3/STATIONS/MINGO0{station}/FIRST_STAGE/EVENT_DATA")
raw_working_directory = os.path.join(base_directory, "RAW")
raw_to_list_working_directory = os.path.join(base_directory, "RAW_TO_LIST")

# Define directory paths relative to base_directory
base_directories = {
    "stratos_list_events_directory": os.path.join(home_directory, "STRATOS_XY_DIRECTORY"),
    
    "base_plots_directory": os.path.join(raw_to_list_working_directory, "PLOTS"),
    
    "pdf_directory": os.path.join(raw_to_list_working_directory, "PLOTS/PDF_DIRECTORY"),
    "base_figure_directory": os.path.join(raw_to_list_working_directory, "PLOTS/FIGURE_DIRECTORY"),
    "figure_directory": os.path.join(raw_to_list_working_directory, f"PLOTS/FIGURE_DIRECTORY/FIGURES_EXEC_ON_{date_execution}"),
    
    "list_events_directory": os.path.join(base_directory, "LIST_EVENTS_DIRECTORY"),
    "full_list_events_directory": os.path.join(base_directory, "FULL_LIST_EVENTS_DIRECTORY"),
    
    "ancillary_directory": os.path.join(raw_to_list_working_directory, "ANCILLARY"),
    
    "empty_files_directory": os.path.join(raw_to_list_working_directory, "ANCILLARY/EMPTY_FILES"),
    "rejected_files_directory": os.path.join(raw_to_list_working_directory, "ANCILLARY/REJECTED_FILES"),
    "temp_files_directory": os.path.join(raw_to_list_working_directory, "ANCILLARY/TEMP_FILES"),
    
    "unprocessed_directory": os.path.join(raw_to_list_working_directory, "RAW_TO_LIST_FILES/UNPROCESSED_DIRECTORY"),
    "error_directory": os.path.join(raw_to_list_working_directory, "RAW_TO_LIST_FILES/ERROR_DIRECTORY"),
    "processing_directory": os.path.join(raw_to_list_working_directory, "RAW_TO_LIST_FILES/PROCESSING_DIRECTORY"),
    "completed_directory": os.path.join(raw_to_list_working_directory, "RAW_TO_LIST_FILES/COMPLETED_DIRECTORY"),
    
    "raw_directory": os.path.join(raw_working_directory, "."),
}

# Create ALL directories if they don't already exist
for directory in base_directories.values():
    os.makedirs(directory, exist_ok=True)

csv_path = os.path.join(base_directory, "calibrations.csv")

# Move files from RAW to RAW_TO_LIST/RAW_TO_LIST_FILES/UNPROCESSED,
# ensuring that only files not already in UNPROCESSED, PROCESSING,
# or COMPLETED are moved:

raw_directory = base_directories["raw_directory"]
unprocessed_directory = base_directories["unprocessed_directory"]
error_directory = base_directories["error_directory"]
stratos_list_events_directory = base_directories["stratos_list_events_directory"]
processing_directory = base_directories["processing_directory"]
completed_directory = base_directories["completed_directory"]

empty_files_directory = base_directories["empty_files_directory"]
rejected_files_directory = base_directories["rejected_files_directory"]
temp_files_directory = base_directories["temp_files_directory"]

raw_files = set(os.listdir(raw_directory))
unprocessed_files = set(os.listdir(unprocessed_directory))
processing_files = set(os.listdir(processing_directory))
completed_files = set(os.listdir(completed_directory))

# Search in all this directories for empty files and move them to the empty_files_directory
for directory in [raw_directory, unprocessed_directory, processing_directory, completed_directory]:
    files = os.listdir(directory)
    for file in files:
        file_empty = os.path.join(directory, file)
        if os.path.getsize(file_empty) == 0:
            # Ensure the empty files directory exists
            os.makedirs(empty_files_directory, exist_ok=True)
            
            # Define the destination path for the file
            empty_destination_path = os.path.join(empty_files_directory, file)
            
            # Remove the destination file if it already exists
            if os.path.exists(empty_destination_path):
                os.remove(empty_destination_path)
            
            print("Moving empty file:", file)
            shutil.move(file_empty, empty_destination_path)

# Files to move: in RAW but not in UNPROCESSED, PROCESSING, or COMPLETED
files_to_move = raw_files - unprocessed_files - processing_files - completed_files

# Copy files to UNPROCESSED
for file_name in files_to_move:
    src_path = os.path.join(raw_directory, file_name)
    dest_path = os.path.join(unprocessed_directory, file_name)
    try:
        shutil.move(src_path, dest_path)
        print(f"Move {file_name} to UNPROCESSED directory.")
    except Exception as e:
        print(f"Failed to copy {file_name}: {e}")


# Erase all files in the figure_directory
figure_directory = base_directories["figure_directory"]
files = os.listdir(figure_directory)

if files:  # Check if the directory contains any files
    print("Removing all files in the figure_directory...")
    for file in files:
        os.remove(os.path.join(figure_directory, file))


# Define input file path -----------------------------------------------------
input_file_config_path = os.path.join(station_directory, f"input_file_mingo0{station}.csv")

if os.path.exists(input_file_config_path):
    print("Searching input configuration file:", input_file_config_path)
    
    # It is a csv
    input_file = pd.read_csv(input_file_config_path, skiprows=1)
    
    if not input_file.empty:
        print("Input configuration file found and is not empty.")
        exists_input_file = True
    else:
        print("Input configuration file is empty.")
        exists_input_file = False
    
    # Print the head
    # print(input_file.head())
    
else:
    exists_input_file = False
    print("Input configuration file does not exist.")
    z_1 = 0
    z_2 = 150
    z_3 = 300
    z_4 = 450


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Header ----------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Execution options -----------------------------------------------------------
# -----------------------------------------------------------------------------

# Plots and savings -------------------------
crontab_execution = True
create_plots = False
create_essential_plots = True
save_plots = True
show_plots = False
create_pdf = True
limit = False
limit_number = 10000
number_of_time_cal_figures = 3
save_calibrations = True
save_full_data = True
presentation = False
presentation_plots = False
force_replacement = True # Creates a new datafile even if there is already one that looks complete
article_format = False

# Charge calibration to fC -------------------------
calibrate_charge = True

# Charge front-back --------------------------------
charge_front_back = True

# Slewing correction -------------------------------
slewing_correction = True

# Time calibration ---------------------------------
time_calibration = True

# Time window determination ------------------------


# Y position ---------------------------------------
y_position_complex_method = False
uniform_y_method = True
uniform_weighted_method = False

# RPC variables ------------------------------------
weighted = False

# TimTrack -----------------------------------------
fixed_speed = False
res_ana_removing_planes = False
timtrack_iteration = False
number_of_TT_executions = 2
residual_plots = False

if fast_mode:
    print('Working in fast mode.')
    residual_plots = False
    timtrack_iteration = False
    time_calibration = False
    charge_front_back = False
    create_plots = False
    # save_full_data = False
    limit = False
    limit_number = 10000
    
if debug_mode:
    print('Working in debug mode.')
    residual_plots = True
    timtrack_iteration = False
    time_calibration = False
    charge_front_back = False
    create_plots = True
    # save_full_data = False
    limit = True
    limit_number = 10000

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Filters ---------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

# General ---------------------------------------------------------------------

# Cross-talk limit
crosstalk_threshold_ns = 3

# -----------------------------------------------------------------------------
# Pre-cal Front & Back --------------------------------------------------------
# -----------------------------------------------------------------------------
if debug_mode:
    T_F_left_pre_cal = -500
    T_F_right_pre_cal = 500

    T_B_left_pre_cal = -500
    T_B_right_pre_cal = 500

    Q_F_left_pre_cal = -500
    Q_F_right_pre_cal = 500

    Q_B_left_pre_cal = -500
    Q_B_right_pre_cal = 500
else:
    T_F_left_pre_cal = -150
    T_F_right_pre_cal = -90

    T_B_left_pre_cal = T_F_left_pre_cal
    T_B_right_pre_cal = T_F_right_pre_cal

    Q_F_left_pre_cal = 70
    Q_F_right_pre_cal = 300

    Q_B_left_pre_cal = Q_F_left_pre_cal
    Q_B_right_pre_cal = Q_F_right_pre_cal

T_left_side = T_F_left_pre_cal
T_right_side = T_F_right_pre_cal

Q_left_side = Q_F_left_pre_cal
Q_right_side = 150

# -----------------------------------------------------------------------------
# Pre-cal Sum & Diff ----------------------------------------------------------
# -----------------------------------------------------------------------------
# Qsum
Q_left_pre_cal = 75
Q_right_pre_cal = 500
# Qdif
Q_diff_pre_cal_threshold = 20
# Tsum
T_sum_left_pre_cal = -150 # was -130 for mingo01 but for mingo03 is different
T_sum_right_pre_cal = -90

# Tdif
T_diff_pre_cal_threshold = 20

# -----------------------------------------------------------------------------
# Post-calibration ------------------------------------------------------------
# -----------------------------------------------------------------------------
# Qsum
Q_sum_left_cal = -20
Q_sum_right_cal = 300
# Qdif
Q_diff_cal_threshold = 10
Q_diff_cal_threshold_FB = 1.25
# Tsum
# ...
# Tdif
T_diff_cal_threshold = 1

# -----------------------------------------------------------------------------
# Once calculated the RPC variables -------------------------------------------
# -----------------------------------------------------------------------------
# Tsum
T_sum_RPC_left = -140
T_sum_RPC_right = -100
# Tdiff
T_diff_RPC_left = -0.8
T_diff_RPC_right = 0.8
# Qsum
Q_RPC_left = 0
Q_RPC_right = 500
# Qdiff
Q_dif_RPC_left = -1
Q_dif_RPC_right = 1
# Y pos
Y_RPC_left = -170 # -150
Y_RPC_right = 170 # 150

# -----------------------------------------------------------------------------
# Alternative fitter filter ---------------------------------------------------
# -----------------------------------------------------------------------------
alt_pos_filter = 600
alt_theta_left_filter = 0
alt_theta_right_filter = np.pi
alt_phi_left_filter = -1*np.pi
alt_phi_right_filter = np.pi
alt_slowness_filter_left = -0.02
alt_slowness_filter_right = 0.03 # 0.025

# -----------------------------------------------------------------------------
# TimTrack filter -------------------------------------------------------------
# -----------------------------------------------------------------------------
pos_filter = 600
proj_filter = 2
t0_left_filter = T_sum_RPC_left
t0_right_filter = T_sum_RPC_right
slowness_filter_left = -0.02 # -0.01
slowness_filter_right = 0.03 # 0.025
charge_event_left_filter = 0
charge_event_right_filter = 1e6

res_ystr_filter = 100
res_tsum_filter = 1.5
res_tdif_filter = 0.4

ext_res_ystr_filter = 120
ext_res_tsum_filter = 2
ext_res_tdif_filter = 1

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Calibrations ----------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

# General
calibrate_strip_T_percentile = 5
calibrate_strip_Q_percentile = 5
calibrate_strip_Q_FB_percentile = 5

# Time sum
CRT_gaussian_fit_quantile = 0.03
strip_time_diff_bound = 10
# time_coincidence_window = 7

# Front-back charge
distance_sum_charges_left_fit = -5
distance_sum_charges_right_fit = 200
distance_diff_charges_up_fit = 10
distance_diff_charges_low_fit = -10
distance_sum_charges_plot = 800
front_back_fit_threshold = 4 # It was 1.4

# Time dif calibration (time_dif_reference)
time_dif_distance = 30
time_dif_reference = np.array([
    [-0.0573, 0.031275, 1.033875, 0.761475],
    [-0.914, -0.873975, -0.19815, 0.452025],
    [0.8769, 1.2008, 1.014, 2.43915],
    [1.508825, 2.086375, 1.6876, 3.023575]
])

# Charge sum pedestal (charge_sum_reference)
charge_sum_distance = 30
charge_sum_reference = np.array([
    [89.4319, 98.19605, 95.99055, 91.83875],
    [96.55775, 94.50385, 94.9254, 91.0775],
    [92.12985, 92.23395, 90.60545, 95.5214],
    [93.75635, 93.57425, 93.07055, 89.27305]
])

# Charge dif calibration (charge_dif_reference)
charge_dif_distance = 30
charge_dif_reference = np.array([
    [4.512, 0.58715, 1.3204, -1.3918],
    [-4.50885, 0.918, -3.39445, -0.12325],
    [-3.8931, -3.28515, 3.27295, 1.0554],
    [-2.29505, 0.012, 2.49045, -2.14565]
])

# Time sum calibration (time_sum_reference)
time_sum_distance = 30
time_sum_reference = np.array([
    [0.0, -0.3886208, -0.53020947, 0.33711737],
    [-0.80494094, -0.68836069, -2.01289387, -1.13481931],
    [-0.23899338, -0.51373738, 0.50845317, 0.11685095],
    [0.33586285, 1.08329847, 0.91410244, 0.58815813]
])

# -----------------------------------------------------------------------------
# Variables to modify ---------------------------------------------------------
# -----------------------------------------------------------------------------

beta = 1

# -----------------------------------------------------------------------------
# Variables to not touch unless necessary -------------------------------------
# -----------------------------------------------------------------------------

Q_sum_color = 'orange'
Q_diff_color = 'red'
T_sum_color = 'blue'
T_diff_color = 'green'

fig_idx = 1
plot_list = []

# Front-back charge
output_order = 0
degree_of_polynomial = 4

# X ----------------------------
strip_length = 300

# Y ----------------------------
def y_pos(y_width):
    return np.cumsum(y_width) - (np.sum(y_width) + y_width) / 2

y_widths = [np.array([63, 63, 63, 98]), np.array([98, 63, 63, 63])]  # P1-P3 and P2-P4 widths
y_pos_T = [y_pos(y_widths[0]), y_pos(y_widths[1])]
y_width_P1_and_P3 = y_widths[0]
y_width_P2_and_P4 = y_widths[1]
y_pos_P1_and_P3 = y_pos(y_width_P1_and_P3)
y_pos_P2_and_P4 = y_pos(y_width_P2_and_P4)

# Miscelanous ----------------------------
c_mm_ns = c/1000000
muon_speed = beta * c_mm_ns
strip_speed = 2/3 * c_mm_ns # 200 mm/ns
tdiff_to_x = strip_speed # Factor to transform t_diff to X

# Timtrack parameters --------------------
vc    = beta * c_mm_ns #mm/ns
sc    = 1/vc
ss    = 1/strip_speed # slowness of the signal in the strip
cocut = 1  # convergence cut
d0    = 10 # initial value of the convergence parameter 
nplan = 4
lenx  = strip_length

anc_sy = 25 # 2.5 cm
anc_sts = 0.4 # 400ps
anc_std = 0.1 # 2 cm
anc_sx = tdiff_to_x * anc_std # 2 cm
anc_sz = 10 # 5 cm


# -----------------------------------------------------------------------------
# Some variables that define the analysis, define a dictionary with the variables:
# 'discarded_by_time_window', 'one_side_events', 'purity_of_data'
# -----------------------------------------------------------------------------
global_variables = {
    'CRT_avg': 0,
    'discarded_by_time_window_percentage': 0,
    'sigmoid_width': 0,
    'background_slope': 0,
    'one_side_events': 0,
    'purity_of_data_percentage': 0,
    'unc_y': anc_sy,
    'unc_tsum': anc_sts,
    'unc_tdif': anc_std
}

# Modify discarded_by_time_window entry
global_variables['discarded_by_time_window'] = 1

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Function definition ---------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

# Calibration functions
# def calibrate_strip_T(column):
#     q = calibrate_strip_T_percentile
#     mask = (abs(column) < T_diff_pre_cal_threshold)
#     column = column[mask]
#     column = column[column != 0]
#     column = column[(np.percentile(column, q) < column) & (column < np.percentile(column, 100 - q))]
#     column = column[(np.percentile(column, q) < column) & (column < np.percentile(column, 100 - q))]
#     column = column[(np.percentile(column, q) < column) & (column < np.percentile(column, 100 - q))]
#     offset = np.median([np.min(column), np.max(column)])
#     return offset


def calibrate_strip_T(column, num_bins=100):
    """
    Calibrates a given column of T values by filtering and determining an offset.

    Parameters:
        column (numpy.ndarray): Input array of T values.
        num_bins (int): Number of bins to use in the histogram.

    Returns:
        float: Calculated offset.
    """
    
    T_rel_th = 0.9
    
    # Apply mask to filter values within the threshold
    mask = (np.abs(column) < T_diff_pre_cal_threshold)
    column = column[mask]
    
    # Remove zero values
    column = column[column != 0]
    
    # Calculate histogram
    counts, bin_edges = np.histogram(column, bins=num_bins)
    
    # Find the maximum number of counts in any bin
    max_counts = np.max(counts)
    
    # Identify bins with counts above the relative threshold
    valid_bins = (counts > T_rel_th * max_counts)
    
    # Filter the original column values based on the valid bins
    column_filt = []
    for i, valid in enumerate(valid_bins):
        if valid:
            # Include values within the range of this bin
            bin_min = bin_edges[i]
            bin_max = bin_edges[i + 1]
            column_filt.extend(column[(column >= bin_min) & (column < bin_max)])
    column_filt = np.array(column_filt)
    
    # Calculate the offset using the mean of the filtered values
    offset = np.mean([np.min(column_filt), np.max(column_filt)])
    
    return offset


def calibrate_strip_T_diff(T_F, T_B):
    """
    Calibrates a given column of T values by filtering and determining an offset.

    Parameters:
        column (numpy.ndarray): Input array of T values.
        num_bins (int): Number of bins to use in the histogram.

    Returns:
        float: Calculated offset.
    """
    
    cond = (T_F != 0) & (T_F > T_left_side) & (T_F < T_right_side) & (T_B != 0) & (T_B > T_left_side) & (T_B < T_right_side)
    
    # Front
    T_F = T_F[cond]
    counts, bin_edges = np.histogram(T_F, bins='auto')
    max_counts = np.max(counts)
    min_counts = np.min(counts[counts > 0])
    threshold = max_counts / 10**1.5
    indices_above_threshold = np.where(counts > threshold)[0]
    if indices_above_threshold.size > 0:
        min_bin_edge_F = bin_edges[indices_above_threshold[0]]
        max_bin_edge_F = bin_edges[indices_above_threshold[-1] + 1]  # +1 to get the upper edge of the last bin
        # print(f"Minimum bin edge: {min_bin_edge_F}")
        # print(f"Maximum bin edge: {max_bin_edge_F}")
    else:
        print("No bins have counts above the threshold, Front.")
        threshold = (min_counts + max_counts) / 2
        indices_above_threshold = np.where(counts > threshold)[0]
        min_bin_edge_F = bin_edges[indices_above_threshold[0]]
        max_bin_edge_F = bin_edges[indices_above_threshold[-1] + 1]
    
    # Back
    T_B = T_B[cond]
    counts, bin_edges = np.histogram(T_B, bins='auto')
    max_counts = np.max(counts)
    min_counts = np.min(counts[counts > 0])
    threshold = max_counts / 10**1.5
    indices_above_threshold = np.where(counts > threshold)[0]
    if indices_above_threshold.size > 0:
        min_bin_edge_B = bin_edges[indices_above_threshold[0]]
        max_bin_edge_B = bin_edges[indices_above_threshold[-1] + 1]  # +1 to get the upper edge of the last bin
        # print(f"Minimum bin edge: {min_bin_edge_B}")
        # print(f"Maximum bin edge: {max_bin_edge_B}")
    else:
        print("No bins have counts above the threshold, Back.")
        threshold = (min_counts + max_counts) / 2
        indices_above_threshold = np.where(counts > threshold)[0]
        min_bin_edge_B = bin_edges[indices_above_threshold[0]]
        max_bin_edge_B = bin_edges[indices_above_threshold[-1] + 1]
    
    cond = (T_F > min_bin_edge_F) & (T_F < max_bin_edge_F) & (T_B > min_bin_edge_B) & (T_B < max_bin_edge_B)
            
    T_F = T_F[cond]
    T_B = T_B[cond]
    
    T_diff = ( T_F - T_B ) / 2
    
    # print("Zeroes:")
    # print(len(T_diff[T_diff == 0]))
    
    # ------------------------------------------------------------------------------
    
    T_rel_th = 0.1
    abs_th = 1
    
    # Apply mask to filter values within the threshold
    mask = (np.abs(T_diff) < T_diff_pre_cal_threshold)
    T_diff = T_diff[mask]
    
    # Remove zero values
    T_diff = T_diff[T_diff != 0]
    
    # Calculate histogram
    counts, bin_edges = np.histogram(T_diff, bins='auto')
    
    # Calculate the nunber of counts of the bin that has the most counts
    max_counts = np.max(counts)
    
    # Find bins with at least one count
    th = T_rel_th * max_counts
    if th < abs_th:
        th = abs_th
    non_empty_bins = counts >= th

    # Find the longest contiguous subset of non-empty bins
    max_length = 0
    current_length = 0
    start_index = 0
    temp_start = 0

    for i, is_non_empty in enumerate(non_empty_bins):
        if is_non_empty:
            if current_length == 0:
                temp_start = i
            current_length += 1
            if current_length > max_length:
                max_length = current_length
                start_index = temp_start
                end_index = i
        else:
            current_length = 0
    
    plateau_left = bin_edges[start_index]
    plateau_right = bin_edges[end_index + 1]
    
    # print(plateau_left)
    # print(plateau_right)
    
    # Calculate the offset using the mean of the filtered values
    offset = ( plateau_left + plateau_right ) / 2
    
    return offset


def calibrate_strip_Q_pedestal(Q_ch, T_ch, Q_other):
    """
    Calibrate the pedestal offset for the charge distribution (Q_ch) by finding
    the first bin of the longest subset of bins with at least one count.

    Parameters:
        Q_ch (numpy.ndarray): Array of charge values for the channel.
        num_bins (int): Number of bins to use for the histogram.

    Returns:
        float: Offset value to bring the distribution to zero.
    """
    
    # First let's tale good values of Time, we want to avoid outliers that might confuse the charge pedestal calibration
    cond = (T_ch != 0) & (T_ch > T_left_side) & (T_ch < T_right_side)
    T_ch = T_ch[cond]
    Q_ch = Q_ch[cond]
    Q_other = Q_other[cond]
    
    # Condition based on the charge difference: it cannot be too high
    Q_dif = Q_ch - Q_other
    percentile = 5
    cond = ( Q_dif > np.percentile(Q_dif, percentile) ) & ( Q_dif < np.percentile(Q_dif, 100 - percentile ) )
    T_ch = T_ch[cond]
    Q_ch = Q_ch[cond]
    
    counts, bin_edges = np.histogram(T_ch, bins='auto')
    max_counts = np.max(counts)
    min_counts = np.min(counts[counts > 0])
    threshold = max_counts / 10**1.5
    indices_above_threshold = np.where(counts > threshold)[0]

    if indices_above_threshold.size > 0:
        min_bin_edge = bin_edges[indices_above_threshold[0]]
        max_bin_edge = bin_edges[indices_above_threshold[-1] + 1]  # +1 to get the upper edge of the last bin
        # print(f"Minimum bin edge: {min_bin_edge}")
        # print(f"Maximum bin edge: {max_bin_edge}")
    else:
        print("No bins have counts above the threshold; Q pedestal calibration.")
        threshold = (min_counts + max_counts) / 2
        indices_above_threshold = np.where(counts > threshold)[0]
        min_bin_edge = bin_edges[indices_above_threshold[0]]
        max_bin_edge = bin_edges[indices_above_threshold[-1] + 1]
    
    Q_ch = Q_ch[(T_ch > min_bin_edge) & (T_ch < max_bin_edge)]
    
    # 5% of the maximum count
    rel_th = 0.015
    rel_th_cal = 0.3
    abs_th = 3
    q_quantile = 0.4 # percentile
    
    # First take the values that are not zero
    Q_ch = Q_ch[Q_ch != 0]
    
    # Remove the values that are not in (50,500)
    Q_ch = Q_ch[(Q_ch > Q_left_side) & (Q_ch < Q_right_side)]
    
    # Quantile filtering
    Q_ch = Q_ch[Q_ch > np.percentile(Q_ch, q_quantile)]
    
    # Calculate histogram
    counts, bin_edges = np.histogram(Q_ch, bins='auto')
    
    # Calculate the nunber of counts of the bin that has the most counts
    max_counts = np.max(counts)
    counts = counts[counts < max_counts]
    max_counts = np.max(counts)
    
    # Find bins with at least one count
    th = rel_th * max_counts
    if th < abs_th:
        th = abs_th
    non_empty_bins = counts >= th

    # Find the longest contiguous subset of non-empty bins
    max_length = 0
    current_length = 0
    start_index = 0
    temp_start = 0

    for i, is_non_empty in enumerate(non_empty_bins):
        if is_non_empty:
            if current_length == 0:
                temp_start = i
            current_length += 1
            if current_length > max_length:
                max_length = current_length
                start_index = temp_start
        else:
            current_length = 0

    # Get the first bin edge of the longest subset
    offset = bin_edges[start_index]
    
    # Second part --------------------------------------------------------------
    Q_ch_cal = Q_ch - offset
    
    # Remove values outside the range (-2, 2)
    Q_ch_cal = Q_ch_cal[(Q_ch_cal > -1) & (Q_ch_cal < 2)]
    
    # Calculate histogram
    counts, bin_edges = np.histogram(Q_ch_cal, bins='auto')
    
    # Find the bin with the most counts
    max_counts = np.max(counts)
    max_bin_index = np.argmax(counts)
    
    # Calculate the threshold
    threshold = rel_th_cal * max_counts
    
    # Start from the bin with the most counts and move left
    offset_bin_index = max_bin_index
    while offset_bin_index > 0 and counts[offset_bin_index] >= threshold:
        offset_bin_index -= 1
    
    # Determine the X value (left edge) of the bin where the threshold is crossed
    offset_cal = bin_edges[offset_bin_index]
    
    pedestal = offset + offset_cal
    pedestal = offset
    
    translate_charge_cal = True
    if translate_charge_cal:
        pedestal = pedestal - 0.25
        
    return pedestal


# def calibrate_strip_Q(Q_sum):
#     q = calibrate_strip_Q_percentile
#     mask_Q = (Q_sum != 0)
#     Q_sum = Q_sum[mask_Q]
#     mask_Q = (Q_sum > Q_left_pre_cal) & (Q_sum < Q_right_pre_cal)
#     Q_sum = Q_sum[mask_Q]
#     Q_sum = Q_sum[Q_sum > np.percentile(Q_sum, q)]
#     mean = np.mean(Q_sum)
#     std = np.std(Q_sum)
#     Q_sum = Q_sum[ abs(Q_sum - mean) < std ]
#     offset = np.min(Q_sum)
#     return offset

def calibrate_strip_Q_FB(Q_F, Q_B):
    q = calibrate_strip_Q_FB_percentile
    
    mask_Q = (Q_F != 0)
    Q_F = Q_F[mask_Q]
    mask_Q = (Q_F > Q_left_pre_cal) & (Q_F < Q_right_pre_cal)
    Q_F = Q_F[mask_Q]
    Q_F = Q_F[Q_F > np.percentile(Q_F, q)]
    mean = np.mean(Q_F)
    std = np.std(Q_F)
    Q_F = Q_F[ abs(Q_F - mean) < std ]
    offset_F = np.min(Q_F)
    
    mask_Q = (Q_B != 0)
    Q_B = Q_B[mask_Q]
    mask_Q = (Q_B > Q_left_pre_cal) & (Q_B < Q_right_pre_cal)
    Q_B = Q_B[mask_Q]
    Q_B = Q_B[Q_B > np.percentile(Q_B, q)]
    mean = np.mean(Q_B)
    std = np.std(Q_B)
    Q_B = Q_B[ abs(Q_B - mean) < std ]
    offset_B = np.min(Q_B)
    
    return (offset_F - offset_B) / 2

enumerate = builtins.enumerate

def polynomial(x, *coeffs):
    return sum(c * x**i for i, c in enumerate(coeffs))

# def scatter_2d_and_fit(xdat, ydat, title, x_label, y_label, name_of_file):
#     global fig_idx
    
#     ydat_translated = ydat

#     xdat_plot = xdat[(xdat < distance_sum_charges_plot) & (xdat > -distance_sum_charges_plot) & (ydat_translated < distance_sum_charges_plot) & (ydat_translated > -distance_sum_charges_plot)]
#     ydat_plot = ydat_translated[(xdat < distance_sum_charges_plot) & (xdat > -distance_sum_charges_plot) & (ydat_translated < distance_sum_charges_plot) & (ydat_translated > -distance_sum_charges_plot)]
#     xdat_pre_fit = xdat[(xdat < distance_sum_charges_right_fit) & (xdat > distance_sum_charges_left_fit) & (ydat_translated < distance_diff_charges_up_fit) & (ydat_translated > distance_diff_charges_low_fit)]
#     ydat_pre_fit = ydat_translated[(xdat < distance_sum_charges_right_fit) & (xdat > distance_sum_charges_left_fit) & (ydat_translated < distance_diff_charges_up_fit) & (ydat_translated > distance_diff_charges_low_fit)]
    
#     # Fit a polynomial of specified degree using curve_fit
#     initial_guess = [1] * (degree_of_polynomial + 1)
#     coeffs, _ = curve_fit(polynomial, xdat_pre_fit, ydat_pre_fit, p0=initial_guess)
#     y_pre_fit = polynomial(xdat_pre_fit, *coeffs)
    
#     # Filter data for fitting based on residues
#     threshold = front_back_fit_threshold  # Set your desired threshold here
#     residues = np.abs(ydat_pre_fit - y_pre_fit)  # Calculate residues
#     xdat_fit = xdat_pre_fit[residues < threshold]
#     ydat_fit = ydat_pre_fit[residues < threshold]
    
#     # Perform fit on filtered data
#     coeffs, _ = curve_fit(polynomial, xdat_fit, ydat_fit, p0=initial_guess)
    
#     y_mean = np.mean(ydat_fit)
#     y_check = polynomial(xdat_fit, *coeffs)
#     ss_res = np.sum((ydat_fit - y_check)**2)
#     ss_tot = np.sum((ydat_fit - y_mean)**2)
#     r_squared = 1 - (ss_res / ss_tot)
#     if r_squared < 0.5:
#         print(f"---> R**2 in {name_of_file[0:4]}: {r_squared:.2g}")
    
#     if create_plots:
#         x_fit = np.linspace(min(xdat_fit), max(xdat_fit), 100)
#         y_fit = polynomial(x_fit, *coeffs)
        
#         x_final = xdat_plot
#         y_final = ydat_plot - polynomial(xdat_plot, *coeffs)
        
#         plt.close()
        
#         # (16,6) was very nice
#         if article_format:
#             ww = (10.84, 4)
#         else:
#             ww = (13.33, 5)
            
#         plt.figure(figsize=ww)  # Use plt.subplots() to create figure and axis    
#         plt.scatter(xdat_plot, ydat_plot, s=1, label="Original data points")
#         # plt.scatter(xdat_pre_fit, ydat_pre_fit, s=1, color="magenta", label="Points for prefitting")
#         plt.scatter(xdat_fit, ydat_fit, s=1, color="orange", label="Points for fitting")
#         plt.scatter(x_final, y_final, s=1, color="green", label="Calibrated points")
#         plt.plot(x_fit, y_fit, 'r-', label='Polynomial Fit: ' + ' '.join([f'a{i}={coeff:.2g}' for i, coeff in enumerate(coeffs[::-1])]))
        
#         if not article_format:
#             plt.title(f"Fig. {output_order}, {title}")
#         plt.xlabel(x_label)
#         plt.ylabel(y_label)
#         plt.xlim([-5, 400])
#         plt.ylim([-11, 11])
        
#         plt.grid()
#         plt.legend(markerscale=5)  # Increase marker scale by 5 times
        
#         plt.tight_layout()
#         # plt.savefig(f"{output_order}_{name_of_file}.png", format="png")
        
#         if save_plots:
#             name_of_file = 'charge_diff_vs_charge_sum_cal'
#             final_filename = f'{fig_idx}_{name_of_file}.png'
#             fig_idx += 1
            
#             save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
#             plot_list.append(save_fig_path)
#             plt.savefig(save_fig_path, format='png')
            
#         if show_plots: plt.show()
#         plt.close()
#     return coeffs


def scatter_2d_and_fit_new(xdat, ydat, title, x_label, y_label, name_of_file):
    global fig_idx
    
    ydat_translated = ydat

    xdat_plot = xdat[(xdat < distance_sum_charges_plot) & (xdat > -distance_sum_charges_plot) & (ydat_translated < distance_sum_charges_plot) & (ydat_translated > -distance_sum_charges_plot)]
    ydat_plot = ydat_translated[(xdat < distance_sum_charges_plot) & (xdat > -distance_sum_charges_plot) & (ydat_translated < distance_sum_charges_plot) & (ydat_translated > -distance_sum_charges_plot)]
    xdat_pre_fit = xdat[(xdat < distance_sum_charges_right_fit) & (xdat > distance_sum_charges_left_fit) & (ydat_translated < distance_diff_charges_up_fit) & (ydat_translated > distance_diff_charges_low_fit)]
    ydat_pre_fit = ydat_translated[(xdat < distance_sum_charges_right_fit) & (xdat > distance_sum_charges_left_fit) & (ydat_translated < distance_diff_charges_up_fit) & (ydat_translated > distance_diff_charges_low_fit)]
    
    # Fit a polynomial of specified degree using curve_fit
    initial_guess = [1] * (degree_of_polynomial + 1)
    coeffs, _ = curve_fit(polynomial, xdat_pre_fit, ydat_pre_fit, p0=initial_guess)
    y_pre_fit = polynomial(xdat_pre_fit, *coeffs)
    
    # Filter data for fitting based on residues
    threshold = front_back_fit_threshold  # Set your desired threshold here
    residues = np.abs(ydat_pre_fit - y_pre_fit)  # Calculate residues
    xdat_fit = xdat_pre_fit[residues < threshold]
    ydat_fit = ydat_pre_fit[residues < threshold]
    
    # Perform fit on filtered data
    coeffs, _ = curve_fit(polynomial, xdat_fit, ydat_fit, p0=initial_guess)
    
    y_mean = np.mean(ydat_fit)
    y_check = polynomial(xdat_fit, *coeffs)
    ss_res = np.sum((ydat_fit - y_check)**2)
    ss_tot = np.sum((ydat_fit - y_mean)**2)
    r_squared = 1 - (ss_res / ss_tot)
    if r_squared < 0.5:
        print(f"---> R**2 in {name_of_file[0:4]}: {r_squared:.2g}")
    
    if create_plots:
        x_fit = np.linspace(min(xdat_fit), max(xdat_fit), 100)
        y_fit = polynomial(x_fit, *coeffs)
        
        x_final = xdat_plot
        y_final = ydat_plot - polynomial(xdat_plot, *coeffs)
        
        plt.close()
        
        # (16,6) was very nice
        if article_format:
            ww = (10.84, 4)
        else:
            ww = (13.33, 5)
            
        plt.figure(figsize=ww)  # Use plt.subplots() to create figure and axis    
        plt.scatter(xdat_plot, ydat_plot, s=1, label="Original data points")
        # plt.scatter(xdat_pre_fit, ydat_pre_fit, s=1, color="magenta", label="Points for prefitting")
        plt.scatter(xdat_fit, ydat_fit, s=1, color="orange", label="Points for fitting")
        plt.scatter(x_final, y_final, s=1, color="green", label="Calibrated points")
        plt.plot(x_fit, y_fit, 'r-', label='Polynomial Fit: ' + ' '.join([f'a{i}={coeff:.2g}' for i, coeff in enumerate(coeffs[::-1])]))
        
        if not article_format:
            plt.title(f"Fig. {output_order}, {title}")
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.xlim([-5, 200])
        plt.ylim([-11, 11])
        
        plt.grid()
        plt.legend(markerscale=5)  # Increase marker scale by 5 times
        
        plt.tight_layout()
        # plt.savefig(f"{output_order}_{name_of_file}.png", format="png")
        
        if save_plots:
            name_of_file = 'charge_diff_vs_charge_sum_cal'
            final_filename = f'{fig_idx}_{name_of_file}.png'
            fig_idx += 1
            
            save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
            plot_list.append(save_fig_path)
            plt.savefig(save_fig_path, format='png')
            
        if show_plots: plt.show()
        plt.close()
        
    return coeffs

def summary_skew(vdat):
    # Calculate the 5th and 95th percentiles
    try:
        percentile_left = np.percentile(vdat, 20)
        percentile_right = np.percentile(vdat, 80)
    except IndexError:
        print("Problem with indices")
        # print(vector)
        
    # Filter values inside the 5th and 95th percentiles
    vdat = [x for x in vdat if percentile_left <= x <= percentile_right]
    mean = np.mean(vdat)
    std = np.std(vdat)
    skewness = skew(vdat)
    return f"mean = {mean:.2g}, std = {std:.2g}, skewness = {skewness:.2g}"

def summary(vector):
    quantile_left = CRT_gaussian_fit_quantile * 100
    quantile_right = 100 - CRT_gaussian_fit_quantile * 100
    
    vector = np.array(vector)  # Convert list to NumPy array
    strip_time_diff_bound = 10
    cond = (vector > -strip_time_diff_bound) & (vector < strip_time_diff_bound)  # This should result in a boolean array
    vector = vector[cond]
    
    if len(vector) < 100:
        return np.nan
    try:
        percentile_left = np.percentile(vector, quantile_left)
        percentile_right = np.percentile(vector, quantile_right)
    except IndexError:
        print("Gave issue with:")
        print(vector)
        return np.nan
    vector = [x for x in vector if percentile_left <= x <= percentile_right]
    if len(vector) == 0:
        return np.nan
    mu, std = norm.fit(vector)
    return mu


def hist_1d(vdat, bin_number, title, axis_label, name_of_file):
    global fig_idx

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(1, 1, 1)
    
    # Create histogram without plotting it
    # counts, bins, _ = ax.hist(vdat, bins=bin_number, alpha=0.5, color="red",
    #                           label=f"All hits, {len(vdat)} events, {summary_skew(vdat)}", density=False)
    
    vdat = np.array(vdat)  # Convert list to NumPy array
    strip_time_diff_bound = 10
    cond = (vdat > -strip_time_diff_bound) & (vdat < strip_time_diff_bound)  # This should result in a boolean array
    vdat = vdat[cond]
    
    counts, bins, _ = ax.hist(vdat, bins=bin_number, alpha=0.5, color="red",
                              label=f"All hits, {len(vdat)} events", density=False)
    
    # Calculate bin centers for fitting the Gaussian
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Fit a Gaussian
    h1_q = CRT_gaussian_fit_quantile
    lower_bound = np.quantile(vdat, h1_q)
    upper_bound = np.quantile(vdat, 1 - h1_q)
    
    cond = (vdat > lower_bound) & (vdat < upper_bound)  # This should result in a boolean array
    vdat = vdat[cond]
    
    mu, std = norm.fit(vdat)

    # Plot the Gaussian fit
    p = norm.pdf(bin_centers, mu, std) * len(vdat) * (bins[1] - bins[0])  # Scale to match histogram
    label_plot = f'Gaussian fit:\n    $\\mu={mu:.2g}$,\n    $\\sigma={std:.2g}$\n    CRT$={std/np.sqrt(2)*1000:.3g}$ ps'
    ax.plot(bin_centers, p, 'k', linewidth=2, label=label_plot)

    ax.legend()
    ax.set_title(title)
    plt.xlabel(axis_label)
    plt.ylabel("Counts")
    plt.tight_layout()

    if save_plots:
        name_of_file = 'timing'
        final_filename = f'{fig_idx}_{name_of_file}.png'
        fig_idx += 1
        
        save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
        plot_list.append(save_fig_path)
        plt.savefig(save_fig_path, format='png')
        
    if show_plots: plt.show()
    plt.close()


def plot_histograms_and_gaussian(df, columns, title, figure_number, quantile=0.99, fit_gaussian=False):
    global fig_idx
    nrows, ncols = (2, 3) if figure_number == 1 else (3, 4)
    
    fig, axs = plt.subplots(nrows, ncols, figsize=(20, 5 * nrows), constrained_layout=True)
    axs = axs.flatten()

    # Define Gaussian function
    def gaussian(x, mu, sigma, amplitude):
        return amplitude * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

    # Precompute quantiles for faster filtering
    if fit_gaussian:
        quantile_bounds = {}
        for col in columns:
            data = df[col].values
            data = data[data != 0]
            if len(data) > 0:
                quantile_bounds[col] = np.quantile(data, [(1 - quantile), quantile])

    # Plot histograms and fit Gaussian if needed
    for i, col in enumerate(columns):
        data = df[col].values
        data = data[data != 0]  # Filter out zero values

        if len(data) == 0:  # Skip if no data
            axs[i].text(0.5, 0.5, "No data", transform=axs[i].transAxes, ha='center', va='center', color='gray')
            continue

        # Plot histogram
        hist_data, bin_edges, _ = axs[i].hist(data, bins='auto', alpha=0.75, label='Data')
        axs[i].set_title(col)
        axs[i].set_xlabel('Value')
        axs[i].set_ylabel('Frequency')

        # Fit Gaussian if enabled and data is sufficient
        if fit_gaussian and len(data) >= 10:
            try:
                # Use precomputed quantile bounds
                if col in quantile_bounds:
                    lower_bound, upper_bound = quantile_bounds[col]
                    filt_data = data[(data >= lower_bound) & (data <= upper_bound)]

                if len(filt_data) < 2:
                    axs[i].text(0.5, 0.5, "Not enough data to fit", transform=axs[i].transAxes, ha='center', va='center', color='gray')
                    continue

                # Fit Gaussian to the histogram data
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                popt, _ = curve_fit(gaussian, bin_centers, hist_data, p0=[np.mean(filt_data), np.std(filt_data), max(hist_data)])
                mu, sigma, amplitude = popt

                # Plot Gaussian fit
                x = np.linspace(lower_bound, upper_bound, 1000)
                axs[i].plot(x, gaussian(x, mu, sigma, amplitude), 'r-', label=f'Gaussian Fit\nμ={mu:.2g}, σ={sigma:.2g}')
                axs[i].legend()
            except (RuntimeError, ValueError):
                axs[i].text(0.5, 0.5, "Fit failed", transform=axs[i].transAxes, ha='center', va='center', color='red')

    # Remove unused subplots
    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])

    plt.suptitle(title, fontsize=16)

    if save_plots:
        final_filename = f'{fig_idx}_{title.replace(" ", "_")}.png'
        fig_idx += 1

        save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
        plot_list.append(save_fig_path)
        plt.savefig(save_fig_path, format='png')
    
    if show_plots:
        plt.show()
    plt.close()


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Body ------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

print("----------------------------------------------------------------------")
print("----------------------------------------------------------------------")
print("----------------- Data reading and preprocessing ---------------------")
print("----------------------------------------------------------------------")
print("----------------------------------------------------------------------")

# Determine the file path input

# Get lists of files in the directories
unprocessed_files = sorted(os.listdir(base_directories["unprocessed_directory"]))
processing_files = sorted(os.listdir(base_directories["processing_directory"]))
completed_files = sorted(os.listdir(base_directories["completed_directory"]))

def process_file(source_path, dest_path):
    print("Source path:", source_path)
    print("Destination path:", dest_path)
    
    if source_path == dest_path:
        return True
    
    if os.path.exists(dest_path):
        print(f"File already exists at destination (removing...)")
        os.remove(dest_path)
        # return False
    
    print("**********************************************************************")
    print(f"Moving\n'{source_path}'\nto\n'{dest_path}'...")
    print("**********************************************************************")
    
    shutil.move(source_path, dest_path)
    return True

def get_file_path(directory, file_name):
    return os.path.join(directory, file_name)


# Create ALL directories if they don't already exist
for directory in base_directories.values():
    os.makedirs(directory, exist_ok=True)

unprocessed_files = os.listdir(base_directories["unprocessed_directory"])
processing_files = os.listdir(base_directories["processing_directory"])
completed_files = os.listdir(base_directories["completed_directory"])

if last_file_test:
    if unprocessed_files:
        unprocessed_files = sorted(unprocessed_files)
        # file_name = unprocessed_files[-1]
        file_name = unprocessed_files[0]
        
        unprocessed_file_path = os.path.join(base_directories["unprocessed_directory"], file_name)
        processing_file_path = os.path.join(base_directories["processing_directory"], file_name)
        completed_file_path = os.path.join(base_directories["completed_directory"], file_name)
        
        print(f"Processing the last file in UNPROCESSED: {unprocessed_file_path}")
        print(f"Moving '{file_name}' to PROCESSING...")
        shutil.move(unprocessed_file_path, processing_file_path)
        print(f"File moved to PROCESSING: {processing_file_path}")

    elif processing_files:
        processing_files = sorted(processing_files)
        file_name = processing_files[-1]
        
        # unprocessed_file_path = os.path.join(base_directories["unprocessed_directory"], file_name)
        processing_file_path = os.path.join(base_directories["processing_directory"], file_name)
        completed_file_path = os.path.join(base_directories["completed_directory"], file_name)
        
        print(f"Processing the last file in PROCESSING: {processing_file_path}")
        error_file_path = os.path.join(base_directories["error_directory"], file_name)
        print(f"File '{processing_file_path}' is already in PROCESSING. Moving it temporarily to ERROR for analysis...")
        shutil.move(processing_file_path, error_file_path)
        processing_file_path = error_file_path
        print(f"File moved to ERROR: {processing_file_path}")

    elif completed_files:
        completed_files = sorted(completed_files)
        file_name = completed_files[-1]
        
        # unprocessed_file_path = os.path.join(base_directories["unprocessed_directory"], file_name)
        processing_file_path = os.path.join(base_directories["processing_directory"], file_name)
        completed_file_path = os.path.join(base_directories["completed_directory"], file_name)
        
        print(f"Reprocessing the last file in COMPLETED: {completed_file_path}")
        print(f"Moving '{completed_file_path}' to PROCESSING...")
        shutil.move(completed_file_path, processing_file_path)
        print(f"File moved to PROCESSING: {processing_file_path}")

    else:
        sys.exit("No files to process in UNPROCESSED, PROCESSING, or COMPLETED.")

else:
    if unprocessed_files:
        print("Shuffling the files in UNPROCESSED...")
        random.shuffle(unprocessed_files)
        for file_name in unprocessed_files:
            unprocessed_file_path = os.path.join(base_directories["unprocessed_directory"], file_name)
            processing_file_path = os.path.join(base_directories["processing_directory"], file_name)
            completed_file_path = os.path.join(base_directories["completed_directory"], file_name)

            print(f"Moving '{file_name}' to PROCESSING...")
            shutil.move(unprocessed_file_path, processing_file_path)
            print(f"File moved to PROCESSING: {processing_file_path}")
            break

    elif processing_files:
        print("Shuffling the files in PROCESSING...")
        random.shuffle(processing_files)
        for file_name in processing_files:
            # unprocessed_file_path = os.path.join(base_directories["unprocessed_directory"], file_name)
            processing_file_path = os.path.join(base_directories["processing_directory"], file_name)
            completed_file_path = os.path.join(base_directories["completed_directory"], file_name)

            print(f"Processing the last file in PROCESSING: {processing_file_path}")
            error_file_path = os.path.join(base_directories["error_directory"], file_name)
            print(f"File '{processing_file_path}' is already in PROCESSING. Moving it temporarily to ERROR for analysis...")
            shutil.move(processing_file_path, error_file_path)
            processing_file_path = error_file_path
            print(f"File moved to ERROR: {processing_file_path}")
            break

    elif completed_files:
        print("Shuffling the files in COMPLETED...")
        random.shuffle(completed_files)
        for file_name in completed_files:
            # unprocessed_file_path = os.path.join(base_directories["unprocessed_directory"], file_name)
            completed_file_path = os.path.join(base_directories["completed_directory"], file_name)
            processing_file_path = os.path.join(base_directories["processing_directory"], file_name)

            print(f"Moving '{file_name}' to PROCESSING...")
            shutil.move(completed_file_path, processing_file_path)
            print(f"File moved to PROCESSING: {processing_file_path}")
            break

    else:
        sys.exit("No files to process in UNPROCESSED, PROCESSING, or COMPLETED.")

# This is for all cases
file_path = processing_file_path

# Check the station number in the datafile
try:
    file_station_number = int(file_name[3])  # 4th character (index 3)
    if file_station_number != int(station):
        print(f'File station number is: {file_station_number}, it does not match.')
        sys.exit(f"File '{file_name}' does not belong to station {station}. Exiting.")
except ValueError:
    sys.exit(f"Invalid station number in file '{file_name}'. Exiting.")


left_limit_time = pd.to_datetime("1-1-2000", format='%d-%m-%Y')
right_limit_time = pd.to_datetime("1-1-2100", format='%d-%m-%Y')

if limit:
    print(f'Taking the first {limit_number} rows.')

# ------------------------------------------------------------------------------------------------------

# Move rejected_file to the rejected file folder
temp_file = os.path.join(base_directories["temp_files_directory"], f"temp_file_{date_execution}.csv")
rejected_file = os.path.join(base_directories["rejected_files_directory"], f"temp_file_{date_execution}.csv")

print(f"Temporal file is {temp_file}")
EXPECTED_COLUMNS = 71  # Expected number of columns

# Function to process each line
def process_line(line):
    line = re.sub(r'0000\.0000', '0', line)  # Replace '0000.0000' with '0'
    line = re.sub(r'\b0+([0-9]+)', r'\1', line)  # Remove leading zeros
    line = re.sub(r' +', ',', line.strip())  # Replace multiple spaces with a comma
    line = re.sub(r'X(202\d)', r'X\n\1', line)  # Replace X2024, X2025 with X\n202Y
    line = re.sub(r'(\w)-(\d)', r'\1 -\2', line)  # Ensure X-Y is properly spaced
    return line

# Function to check for malformed numbers (e.g., '-120.144.0')
def contains_malformed_numbers(line):
    return bool(re.search(r'-?\d+\.\d+\.\d+', line))  # Detects multiple decimal points

# Function to validate year, month, and day
def is_valid_date(values):
    try:
        year, month, day = int(values[0]), int(values[1]), int(values[2])
        if year not in {2023, 2024, 2025, 2026, 2027}:  # Check valid years
            return False
        if not (1 <= month <= 12):  # Check valid month
            return False
        if not (1 <= day <= 31):  # Check valid day
            return False
        return True
    except ValueError:  # In case of non-numeric values
        return False

# Process the file
read_lines = 0
written_lines = 0
with open(file_path, 'r') as infile, open(temp_file, 'w') as outfile, open(rejected_file, 'w') as rejectfile:
    for i, line in enumerate(infile, start=1):
        read_lines += 1
        
        cleaned_line = process_line(line)
        cleaned_values = cleaned_line.split(',')  # Split into columns

        # Validate line structure before further processing
        if len(cleaned_values) < 3 or not is_valid_date(cleaned_values[:3]):
            rejectfile.write(f"Line {i} (Invalid date): {line.strip()}\n")
            continue  # Skip this row

        if contains_malformed_numbers(line):
            rejectfile.write(f"Line {i} (Malformed number): {line.strip()}\n")  # Save rejected row
            continue  # Skip this row

        # Ensure correct column count
        if len(cleaned_values) == EXPECTED_COLUMNS:
            written_lines += 1
            outfile.write(cleaned_line + '\n')  # Save valid row
        else:
            rejectfile.write(f"Line {i} (Wrong column count): {line.strip()}\n")  # Save rejected row

read_df = pd.read_csv(temp_file, header=None, low_memory=False, nrows=limit_number if limit else None)
read_df = read_df.apply(pd.to_numeric, errors='coerce')

# Print the number of rows in input
print(f"\nOriginal file has {read_lines} lines.")
print(f"Processed file has {written_lines} lines.")
print(f"--> A {written_lines/read_lines*100:.2f}% of the lines were valid.\n")

# Assign name to the columns
read_df.columns = ['year', 'month', 'day', 'hour', 'minute', 'second'] + [f'column_{i}' for i in range(6, 71)]
read_df['datetime'] = pd.to_datetime(read_df[['year', 'month', 'day', 'hour', 'minute', 'second']])
# data = data.drop(columns=['year', 'month', 'day', 'hour', 'minute', 'second'])


# ------------------------------------------------------------------------------------------------------
# Filter 1: by date ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------

print("----------------------------------------------------------------------")
print("-------------------------- Filter 1: by date -------------------------")
print("----------------------------------------------------------------------")

selected_df = read_df[(read_df['datetime'] >= left_limit_time) & (read_df['datetime'] <= right_limit_time)]
if not isinstance(selected_df.set_index('datetime').index, pd.DatetimeIndex):
    raise ValueError("The index is not a DatetimeIndex. Check 'datetime' column formatting.")

# Print the count frequency of the values in column_6
print(selected_df['column_6'].value_counts())
# Take only the rows in which column_6 is equal to 1
selected_df = selected_df[selected_df['column_6'] == 1]

raw_data_len = len(selected_df)

if raw_data_len == 0:
    print(selected_df['column_6'].head())
    print("No coincidence events.")
    sys.exit()

# Note that the middle between start and end time could also be taken. This is for calibration storage.
datetime_value = selected_df['datetime'][0]
# Take the last datetime value
end_datetime_value = selected_df['datetime'].iloc[-1]
start_time = datetime_value
end_time = end_datetime_value
datetime_str = str(datetime_value)
save_filename_suffix = datetime_str.replace(' ', "_").replace(':', ".").replace('-', ".")


# -------------------------------------------------------------------------------
# ------------ Input file and data managing to select configuration -------------
# -------------------------------------------------------------------------------

if exists_input_file:
    # Ensure `start` and `end` columns are in datetime format
    input_file["start"] = pd.to_datetime(input_file["start"], dayfirst=True)
    input_file["end"] = pd.to_datetime(input_file["end"], dayfirst=True)
    input_file["end"].fillna(pd.to_datetime('now'), inplace=True)
    matching_confs = input_file[ (input_file["start"] <= start_time) & (input_file["end"] >= end_time) ]
    if not matching_confs.empty:
        if len(matching_confs) > 1:
            print(f"Warning:\nMultiple configurations match the date range\n{start_time} to {end_time}.\nTaking the first one.")
        selected_conf = matching_confs.iloc[0]
        print(f"Selected configuration: {selected_conf['conf']}")
        z_positions = np.array([selected_conf.get(f"P{i}", np.nan) for i in range(1, 5)])
    else:
        print("Error: No matching configuration found for the given date range. Using default z_positions.")
        z_positions = np.array([0, 150, 300, 450])  # In mm
else:
    print("Error: No input file. Using default z_positions.")
    z_positions = np.array([0, 150, 300, 450])  # In mm

# Print the resulting z_positions
z_positions = z_positions - z_positions[0]
print(f"Z positions: {z_positions}")


print("--------------------------------------------------------------------------")
print("--------------------------------------------------------------------------")
print(f"--------------- Starting date is {save_filename_suffix} ---------------------") # This is longer so it displays nicely
print("--------------------------------------------------------------------------")
print("--------------------------------------------------------------------------")

# Defining the directories that will store the data
save_full_filename = f"full_list_events_{save_filename_suffix}.txt"
save_filename = f"list_events_{save_filename_suffix}.txt"
save_pdf_filename = f"pdf_{save_filename_suffix}.pdf"

save_list_path = os.path.join(base_directories["list_events_directory"], save_filename)
save_full_path = os.path.join(base_directories["full_list_events_directory"], save_full_filename)
save_pdf_path = os.path.join(base_directories["pdf_directory"], save_pdf_filename)

# Check if the file exists and its size
if os.path.exists(save_filename):
    if os.path.getsize(save_filename) >= 1 * 1024 * 1024: # Bigger than 1MB
        if force_replacement == False:
            print("Datafile found and it looks completed. Exiting...")
            sys.exit()  # Exit the script
        else:
            print("Datafile found and it is not empty, but 'force_replacement' is True, so it creates new datafiles anyway.")
    else:
        print("Datafile found, but empty.")

column_indices = {
    'T1_F': range(55, 59), 'T1_B': range(59, 63), 'Q1_F': range(63, 67), 'Q1_B': range(67, 71),
    'T2_F': range(39, 43), 'T2_B': range(43, 47), 'Q2_F': range(47, 51), 'Q2_B': range(51, 55),
    'T3_F': range(23, 27), 'T3_B': range(27, 31), 'Q3_F': range(31, 35), 'Q3_B': range(35, 39),
    'T4_F': range(7, 11), 'T4_B': range(11, 15), 'Q4_F': range(15, 19), 'Q4_B': range(19, 23)
}

# Extract and assign appropriate column names
columns_data = {'datetime': selected_df['datetime'].values}
for key, idx_range in column_indices.items():
    for i, col_idx in enumerate(idx_range):
        column_name = f'{key}_{i+1}'
        columns_data[column_name] = selected_df.iloc[:, col_idx].values

# Create a DataFrame from the columns data
working_df = pd.DataFrame(columns_data)
working_df["datetime"] = selected_df['datetime']

print(working_df.columns.to_list())


# ----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------
# Original trigger type ------------------------------------------------------------
# ----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------

# Now obtain the trigger type
def create_original_tt(df):
    def get_original_tt(row):
        planes_with_charge = []
        for plane in range(1, 5):
            charge_columns = [f'Q{plane}_F_1', f'Q{plane}_F_2', f'Q{plane}_F_3', f'Q{plane}_F_4',
                              f'Q{plane}_B_1', f'Q{plane}_B_2', f'Q{plane}_B_3', f'Q{plane}_B_4']
            if any(row[col] != 0 for col in charge_columns):
                planes_with_charge.append(str(plane))
        return ''.join(planes_with_charge)
    
    df['original_tt'] = df.apply(get_original_tt, axis=1)
    return df

# Apply the function to the DataFrame
working_df = create_original_tt(working_df)
working_df['original_tt'] = working_df['original_tt'].apply(builtins.int)

if create_essential_plots or create_plots:
# if create_plots:
    event_counts = working_df['original_tt'].value_counts()

    # Plot the histogram of event counts
    plt.figure(figsize=(10, 6))
    event_counts.plot(kind='bar', alpha=0.7)
    plt.title(f'Number of Events per Original TT Label, {start_time}')
    plt.xlabel('Original TT Label')
    plt.ylabel('Number of Events')
    plt.xticks(rotation=45)
    plt.tight_layout()
    if save_plots:
        final_filename = f'{fig_idx}_original_TT.png'
        fig_idx += 1

        save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
        plot_list.append(save_fig_path)
        plt.savefig(save_fig_path, format='png')

    if show_plots: plt.show()
    plt.close()


# # Add 'event_id' and 'event_label' columns ----------------------------------------------
# ...['event_id'] = np.arange(len(...))  # Sequential event identifiers
# ...['event_label'] = 'date_filtered'  # Label for the events

# # Reorder columns to place 'event_id' and 'event_label' as the first columns
# columns_to_move = ['event_id', 'event_label']
# remaining_columns = [col for col in ....columns if col not in columns_to_move]
# ... = ...[columns_to_move + remaining_columns]

# # Save the DataFrame to a CSV file
# if debug_mode:
#     ...to_csv('hey.csv', sep=' ', index=False)


# New channel-wise plot -------------------------------------------------------
log_scale = True
if debug_mode:
    T_clip_min = -500
    T_clip_max = 500
    Q_clip_min = -500
    Q_clip_max = 500
    num_bins = 100  # Parameter for the number of bins
else:
    T_clip_min = -300
    T_clip_max = 100
    Q_clip_min = 0
    Q_clip_max = 500
    num_bins = 100  # Parameter for the number of bins


# if create_plots or create_essential_plots:
if create_plots:
    # Create the grand figure for T values
    fig_T, axes_T = plt.subplots(4, 4, figsize=(20, 10))  # Adjust the layout as necessary
    axes_T = axes_T.flatten()
    
    for i, key in enumerate(['T1', 'T2', 'T3', 'T4']):
        for j in range(4):
            col_F = f'{key}_F_{j+1}'
            col_B = f'{key}_B_{j+1}'
            y_F = working_df[col_F]
            y_B = working_df[col_B]
            
            # Plot histograms with T-specific clipping and bins
            axes_T[i*4 + j].hist(y_F[(y_F != 0) & (y_F > T_clip_min) & (y_F < T_clip_max)], 
                                 bins=num_bins, alpha=0.5, label=f'{col_F} (F)')
            axes_T[i*4 + j].hist(y_B[(y_B != 0) & (y_B > T_clip_min) & (y_B < T_clip_max)], 
                                 bins=num_bins, alpha=0.5, label=f'{col_B} (B)')
            axes_T[i*4 + j].axvline(x=T_F_left_pre_cal, color='red', linestyle='--', label='T_left_pre_cal')
            axes_T[i*4 + j].axvline(x=T_F_right_pre_cal, color='blue', linestyle='--', label='T_right_pre_cal')
            axes_T[i*4 + j].set_title(f'{col_F} vs {col_B}')
            axes_T[i*4 + j].legend()
            
            if log_scale:
                axes_T[i*4 + j].set_yscale('log')  # For T values

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.suptitle(f"Grand Figure for T values, mingo0{station}\n{start_time}", fontsize=16)
    
    if save_plots:
        final_filename = f'{fig_idx}_grand_figure_T.png'
        fig_idx += 1

        save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
        plot_list.append(save_fig_path)
        plt.savefig(save_fig_path, format='png')

    if show_plots: plt.show()
    plt.close(fig_T)

    # Create the grand figure for Q values
    fig_Q, axes_Q = plt.subplots(4, 4, figsize=(20, 10))  # Adjust the layout as necessary
    axes_Q = axes_Q.flatten()
    
    for i, key in enumerate(['T1', 'T2', 'T3', 'T4']):
        for j in range(4):
            col_F = f'{key.replace("T", "Q")}_F_{j+1}'
            col_B = f'{key.replace("T", "Q")}_B_{j+1}'
            y_F = working_df[col_F]
            y_B = working_df[col_B]
            
            # Plot histograms with Q-specific clipping and bins
            axes_Q[i*4 + j].hist(y_F[(y_F != 0) & (y_F > Q_clip_min) & (y_F < Q_clip_max)], 
                                 bins=num_bins, alpha=0.5, label=f'{col_F} (F)')
            axes_Q[i*4 + j].hist(y_B[(y_B != 0) & (y_B > Q_clip_min) & (y_B < Q_clip_max)], 
                                 bins=num_bins, alpha=0.5, label=f'{col_B} (B)')
            axes_Q[i*4 + j].axvline(x=Q_F_left_pre_cal, color='red', linestyle='--', label='Q_left_pre_cal')
            axes_Q[i*4 + j].axvline(x=Q_F_right_pre_cal, color='blue', linestyle='--', label='Q_right_pre_cal')
            axes_Q[i*4 + j].set_title(f'{col_F} vs {col_B}')
            axes_Q[i*4 + j].legend()
            
            if log_scale:
                axes_Q[i*4 + j].set_yscale('log')  # For Q values

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.suptitle(f"Grand Figure for Q values, mingo0{station}\n{start_time}", fontsize=16)
    
    if save_plots:
        final_filename = f'{fig_idx}_grand_figure_Q.png'
        fig_idx += 1
        
        save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
        plot_list.append(save_fig_path)
        plt.savefig(save_fig_path, format='png')

    if show_plots: plt.show()
    plt.close(fig_Q)

# -----------------------------------------------------------------------------------------------


if create_plots:
    # Initialize figure and axes for scatter plot of Time vs Charge
    fig_TQ, axes_TQ = plt.subplots(4, 4, figsize=(20, 10))  # Adjust the layout as necessary
    axes_TQ = axes_TQ.flatten()

    # Iterate over each module (T1, T2, T3, T4)
    for i, key in enumerate(['T1', 'T2', 'T3', 'T4']):
        for j in range(4):
            col_F = f'{key}_F_{j+1}'  # Time F column
            col_B = f'{key}_B_{j+1}'  # Time B column
            
            y_F = working_df[col_F]  # Time values for front
            y_B = working_df[col_B]  # Time values for back
            
            charge_col_F = f'{key.replace("T", "Q")}_F_{j+1}'  # Corresponding charge column for front
            charge_col_B = f'{key.replace("T", "Q")}_B_{j+1}'  # Corresponding charge column for back
            
            charge_F = working_df[charge_col_F]  # Charge values for front
            charge_B = working_df[charge_col_B]  # Charge values for back
            
            # Apply clipping ranges to the data
            mask_F = (y_F != 0) & (y_F > T_clip_min) & (y_F < T_clip_max) & (charge_F > Q_clip_min) & (charge_F < Q_clip_max)
            mask_B = (y_B != 0) & (y_B > T_clip_min) & (y_B < T_clip_max) & (charge_B > Q_clip_min) & (charge_B < Q_clip_max)
            
            # Plot scatter plots for Time F vs Charge F and Time B vs Charge B
            axes_TQ[i*4 + j].scatter(charge_F[mask_F], y_F[mask_F], alpha=0.5, label=f'{col_F} (F)', color='green', s=1)
            axes_TQ[i*4 + j].scatter(charge_B[mask_B], y_B[mask_B], alpha=0.5, label=f'{col_B} (B)', color='orange', s=1)
            
            # Plot threshold lines for time and charge
            axes_TQ[i*4 + j].axhline(y=T_F_left_pre_cal, color='red', linestyle='--', label='T_left_pre_cal')
            axes_TQ[i*4 + j].axhline(y=T_F_right_pre_cal, color='blue', linestyle='--', label='T_right_pre_cal')
            axes_TQ[i*4 + j].axvline(x=Q_F_left_pre_cal, color='red', linestyle='--', label='Q_left_pre_cal')
            axes_TQ[i*4 + j].axvline(x=Q_F_right_pre_cal, color='blue', linestyle='--', label='Q_right_pre_cal')
            
            axes_TQ[i*4 + j].set_title(f'{col_F} vs {col_B}')
            axes_TQ[i*4 + j].legend()

    # Adjust the layout and title
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.suptitle(f"Scatter Plot for T vs Q values, mingo0{station}\n{start_time}", fontsize=16)

    # Save the plot
    if save_plots:
        final_filename = f'{fig_idx}_scatter_plot_TQ.png'
        fig_idx += 1
        save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
        plot_list.append(save_fig_path)
        plt.savefig(save_fig_path, format='png')

    # Show the plot if requested
    if show_plots:
        plt.show()

    # Close the plot to avoid excessive memory usage
    plt.close(fig_TQ)

# ---------------------------------------------------------------------------------

print("--------------------------------------------------------------------------")
print("-------------------- Filter 1.1.1: uncalibrated data ---------------------")
print("--------------------------------------------------------------------------")
# FILTER 2: TF, TB, QF, QB PRECALIBRATED THRESHOLDS --> 0 if out ------------------

for col in working_df.columns:
    if working_df[col].isna().any():
        working_df[col].fillna(0, inplace=True)

# Loop through all relevant columns and apply the filtering
for col in working_df.columns:
    if col.startswith('T') or col.startswith('Q'):  # Check for T and Q columns
        if '_F_' in col:  # Check if '_F_' is in the column name
            # Apply the T_F filter for time columns (T)
            if col.startswith('T'):
                working_df[col] = np.where((working_df[col] > T_F_right_pre_cal) | (working_df[col] < T_F_left_pre_cal), 0, working_df[col])
            # Apply the Q_F filter for charge columns (Q)
            if col.startswith('Q'):
                working_df[col] = np.where((working_df[col] > Q_F_right_pre_cal) | (working_df[col] < Q_F_left_pre_cal), 0, working_df[col])
        elif '_B_' in col:  # Check if '_B_' is in the column name
            # Apply the T_B filter for time columns (T)
            if col.startswith('T'):
                working_df[col] = np.where((working_df[col] > T_B_right_pre_cal) | (working_df[col] < T_B_left_pre_cal), 0, working_df[col])
            # Apply the Q_B filter for charge columns (Q)
            if col.startswith('Q'):
                working_df[col] = np.where((working_df[col] > Q_B_right_pre_cal) | (working_df[col] < Q_B_left_pre_cal), 0, working_df[col])


# New channel-wise plot ----------------------------------------------------------
log_scale = True
if debug_mode:
    T_clip_min = -500
    T_clip_max = 500
    Q_clip_min = -500
    Q_clip_max = 500
    num_bins = 100  # Parameter for the number of bins
else:
    T_clip_min = -300
    T_clip_max = 100
    Q_clip_min = 0
    Q_clip_max = 500
    num_bins = 100  # Parameter for the number of bins


# if create_plots or create_essential_plots:
if create_plots:
    # Create the grand figure for T values
    fig_T, axes_T = plt.subplots(4, 4, figsize=(20, 10))  # Adjust the layout as necessary
    axes_T = axes_T.flatten()
    
    for i, key in enumerate(['T1', 'T2', 'T3', 'T4']):
        for j in range(4):
            col_F = f'{key}_F_{j+1}'
            col_B = f'{key}_B_{j+1}'
            y_F = working_df[col_F]
            y_B = working_df[col_B]
            
            # Plot histograms with T-specific clipping and bins
            axes_T[i*4 + j].hist(y_F[(y_F != 0) & (y_F > T_clip_min) & (y_F < T_clip_max)], 
                                 bins=num_bins, alpha=0.5, label=f'{col_F} (F)')
            axes_T[i*4 + j].hist(y_B[(y_B != 0) & (y_B > T_clip_min) & (y_B < T_clip_max)], 
                                 bins=num_bins, alpha=0.5, label=f'{col_B} (B)')
            axes_T[i*4 + j].set_title(f'{col_F} vs {col_B}')
            axes_T[i*4 + j].legend()
            
            if log_scale:
                axes_T[i*4 + j].set_yscale('log')  # For T values

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.suptitle(f"Grand Figure for T values, mingo0{station}\n{start_time}", fontsize=16)
    
    if save_plots:
        final_filename = f'{fig_idx}_grand_figure_T.png'
        fig_idx += 1

        save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
        plot_list.append(save_fig_path)
        plt.savefig(save_fig_path, format='png')

    if show_plots: plt.show()
    plt.close(fig_T)

    # Create the grand figure for Q values
    fig_Q, axes_Q = plt.subplots(4, 4, figsize=(20, 10))  # Adjust the layout as necessary
    axes_Q = axes_Q.flatten()
    
    for i, key in enumerate(['T1', 'T2', 'T3', 'T4']):
        for j in range(4):
            col_F = f'{key.replace("T", "Q")}_F_{j+1}'
            col_B = f'{key.replace("T", "Q")}_B_{j+1}'
            y_F = working_df[col_F]
            y_B = working_df[col_B]
            
            # Plot histograms with Q-specific clipping and bins
            axes_Q[i*4 + j].hist(y_F[(y_F != 0) & (y_F > Q_clip_min) & (y_F < Q_clip_max)], 
                                 bins=num_bins, alpha=0.5, label=f'{col_F} (F)')
            axes_Q[i*4 + j].hist(y_B[(y_B != 0) & (y_B > Q_clip_min) & (y_B < Q_clip_max)], 
                                 bins=num_bins, alpha=0.5, label=f'{col_B} (B)')
            axes_Q[i*4 + j].set_title(f'{col_F} vs {col_B}')
            axes_Q[i*4 + j].legend()
            
            if log_scale:
                axes_Q[i*4 + j].set_yscale('log')  # For Q values

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.suptitle(f"Grand Figure for Q values, mingo0{station}\n{start_time}", fontsize=16)
    
    if save_plots:
        final_filename = f'{fig_idx}_grand_figure_Q.png'
        fig_idx += 1
        
        save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
        plot_list.append(save_fig_path)
        plt.savefig(save_fig_path, format='png')

    if show_plots: plt.show()
    plt.close(fig_Q)


if create_plots:
    # Initialize figure and axes for scatter plot of Time vs Charge
    fig_TQ, axes_TQ = plt.subplots(4, 4, figsize=(20, 10))  # Adjust the layout as necessary
    axes_TQ = axes_TQ.flatten()

    # Iterate over each module (T1, T2, T3, T4)
    for i, key in enumerate(['T1', 'T2', 'T3', 'T4']):
        for j in range(4):
            col_F = f'{key}_F_{j+1}'  # Time F column
            col_B = f'{key}_B_{j+1}'  # Time B column
            
            y_F = working_df[col_F]  # Time values for front
            y_B = working_df[col_B]  # Time values for back
            
            charge_col_F = f'{key.replace("T", "Q")}_F_{j+1}'  # Corresponding charge column for front
            charge_col_B = f'{key.replace("T", "Q")}_B_{j+1}'  # Corresponding charge column for back
            
            charge_F = working_df[charge_col_F]  # Charge values for front
            charge_B = working_df[charge_col_B]  # Charge values for back
            
            # Apply clipping ranges to the data
            mask_F = (y_F != 0) & (y_F > T_clip_min) & (y_F < T_clip_max) & (charge_F > Q_clip_min) & (charge_F < Q_clip_max)
            mask_B = (y_B != 0) & (y_B > T_clip_min) & (y_B < T_clip_max) & (charge_B > Q_clip_min) & (charge_B < Q_clip_max)
            
            # Plot scatter plots for Time F vs Charge F and Time B vs Charge B
            axes_TQ[i*4 + j].scatter(charge_F[mask_F], y_F[mask_F], alpha=0.5, label=f'{col_F} (F)', color='green', s=1)
            axes_TQ[i*4 + j].scatter(charge_B[mask_B], y_B[mask_B], alpha=0.5, label=f'{col_B} (B)', color='orange', s=1)
            
            # Plot threshold lines for time and charge
            axes_TQ[i*4 + j].axhline(y=T_F_left_pre_cal, color='red', linestyle='--', label='T_left_pre_cal')
            axes_TQ[i*4 + j].axhline(y=T_F_right_pre_cal, color='blue', linestyle='--', label='T_right_pre_cal')
            axes_TQ[i*4 + j].axvline(x=Q_F_left_pre_cal, color='red', linestyle='--', label='Q_left_pre_cal')
            axes_TQ[i*4 + j].axvline(x=Q_F_right_pre_cal, color='blue', linestyle='--', label='Q_right_pre_cal')
            
            axes_TQ[i*4 + j].set_title(f'{col_F} vs {col_B}')
            axes_TQ[i*4 + j].legend()

    # Adjust the layout and title
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.suptitle(f"Scatter Plot for T vs Q values, mingo0{station}\n{start_time}", fontsize=16)

    # Save the plot
    if save_plots:
        final_filename = f'{fig_idx}_scatter_plot_TQ_filtered.png'
        fig_idx += 1
        save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
        plot_list.append(save_fig_path)
        plt.savefig(save_fig_path, format='png')

    # Show the plot if requested
    if show_plots:
        plt.show()

    # Close the plot to avoid excessive memory usage
    plt.close(fig_TQ)


# -----------------------------------------------------------------------------
# Comprobation of emptiness of the columns
# -----------------------------------------------------------------------------

# Count the number of nonzero values in each column
nonzero_counts = (working_df != 0).sum()

# Identify columns with fewer than 100 nonzero values
low_value_cols = nonzero_counts[nonzero_counts < 100].index.tolist()

if low_value_cols:
    print(f"Warning: The following columns contain fewer than 100 nonzero values and may require review: {low_value_cols}")
    print("Rejecting file due to insufficient data.")

    # Move the file to the error directory
    final_path = os.path.join(base_directories["error_directory"], file_name)
    print(f"Moving {file_path} to the error directory {final_path}...")
    shutil.move(file_path, final_path)
    
    sys.exit(1)


print("--------------------------------------------------------------------------")
print("-------------------- Charge pedestal calibration -------------------------")
print("--------------------------------------------------------------------------")

charge_test = working_df.copy()
charge_test_copy = charge_test.copy()

# New pedestal calibration for charges ------------------------------------------------
QF_pedestal = []
for key in ['1', '2', '3', '4']:
    Q_F_cols = [f'Q{key}_F_{i+1}' for i in range(4)]
    Q_F = working_df[Q_F_cols].values
    
    Q_B_cols = [f'Q{key}_B_{i+1}' for i in range(4)]
    Q_B = working_df[Q_B_cols].values
    
    T_F_cols = [f'T{key}_F_{i+1}' for i in range(4)]
    T_F = working_df[T_F_cols].values
    
    QF_pedestal_component = [calibrate_strip_Q_pedestal(Q_F[:,i], T_F[:,i], Q_B[:,i]) for i in range(4)]
    QF_pedestal.append(QF_pedestal_component)
QF_pedestal = np.array(QF_pedestal)

QB_pedestal = []
for key in ['1', '2', '3', '4']:
    Q_F_cols = [f'Q{key}_F_{i+1}' for i in range(4)]
    Q_F = working_df[Q_F_cols].values
    
    Q_B_cols = [f'Q{key}_B_{i+1}' for i in range(4)]
    Q_B = working_df[Q_B_cols].values
    
    T_B_cols = [f'T{key}_B_{i+1}' for i in range(4)]
    T_B = working_df[T_B_cols].values
    
    QB_pedestal_component = [calibrate_strip_Q_pedestal(Q_B[:,i], T_B[:,i], Q_F[:,i]) for i in range(4)]
    QB_pedestal.append(QB_pedestal_component)
QB_pedestal = np.array(QB_pedestal)

print("\nFront Charge Pedestal:")
print(QF_pedestal)
print("\nBack Charge Pedestal:")
print(QB_pedestal,"\n")

for i, key in enumerate(['Q1', 'Q2', 'Q3', 'Q4']):
    for j in range(4):
        mask = charge_test_copy[f'{key}_F_{j+1}'] != 0
        charge_test.loc[mask, f'{key}_F_{j+1}'] -= QF_pedestal[i][j]

for i, key in enumerate(['Q1', 'Q2', 'Q3', 'Q4']):
    for j in range(4):
        mask = charge_test_copy[f'{key}_B_{j+1}'] != 0
        charge_test.loc[mask, f'{key}_B_{j+1}'] -= QB_pedestal[i][j]


# Plot histograms of all the pedestal substractions
validate_charge_pedestal_calibration = True
if validate_charge_pedestal_calibration:
    # if create_plots or create_essential_plots:
    if create_plots:
        # Create the grand figure for Q values
        fig_Q, axes_Q = plt.subplots(4, 4, figsize=(20, 10))  # Adjust the layout as necessary
        axes_Q = axes_Q.flatten()
        
        for i, key in enumerate(['Q1', 'Q2', 'Q3', 'Q4']):
            for j in range(4):
                col_F = f'{key}_F_{j+1}'
                col_B = f'{key}_B_{j+1}'
                y_F = charge_test[col_F]
                y_B = charge_test[col_B]
                
                # Plot histograms with Q-specific clipping and bins
                axes_Q[i*4 + j].hist(y_F[(y_F != 0) & (y_F > Q_clip_min) & (y_F < Q_clip_max)], 
                                    bins=num_bins, alpha=0.5, label=f'{col_F} (F)')
                axes_Q[i*4 + j].hist(y_B[(y_B != 0) & (y_B > Q_clip_min) & (y_B < Q_clip_max)], 
                                    bins=num_bins, alpha=0.5, label=f'{col_B} (B)')
                axes_Q[i*4 + j].set_title(f'{col_F} vs {col_B}')
                axes_Q[i*4 + j].legend()
                
                if log_scale:
                    axes_Q[i*4 + j].set_yscale('log')  # For Q values

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.suptitle(f"Grand Figure for pedestal substracted values, mingo0{station}\n{start_time}", fontsize=16)
        
        if save_plots:
            final_filename = f'{fig_idx}_grand_figure_Q_pedestal.png'
            fig_idx += 1
            
            save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
            plot_list.append(save_fig_path)
            plt.savefig(save_fig_path, format='png')
        
        if show_plots: plt.show()
        plt.close(fig_Q)
        
        # ZOOOOOOOOOOOOOOOOOOOM ------------------------------------------------
        # Create the grand figure for Q values
        fig_Q, axes_Q = plt.subplots(4, 4, figsize=(20, 10))  # Adjust the layout as necessary
        axes_Q = axes_Q.flatten()
        
        for i, key in enumerate(['Q1', 'Q2', 'Q3', 'Q4']):
            for j in range(4):
                col_F = f'{key}_F_{j+1}'
                col_B = f'{key}_B_{j+1}'
                y_F = charge_test[col_F]
                y_B = charge_test[col_B]
                
                Q_clip_min = -5
                Q_clip_max = 5
                
                # Plot histograms with Q-specific clipping and bins
                axes_Q[i*4 + j].hist(y_F[(y_F != 0) & (y_F > Q_clip_min) & (y_F < Q_clip_max)], 
                                    bins=num_bins, alpha=0.5, label=f'{col_F} (F)')
                axes_Q[i*4 + j].hist(y_B[(y_B != 0) & (y_B > Q_clip_min) & (y_B < Q_clip_max)], 
                                    bins=num_bins, alpha=0.5, label=f'{col_B} (B)')
                axes_Q[i*4 + j].set_title(f'{col_F} vs {col_B}')
                axes_Q[i*4 + j].legend()
                # Show between -5 and 5
                axes_Q[i*4 + j].set_xlim([-5, 5])
        # Display a vertical green dashed, alpha = 0.5 line at 0
        for ax in axes_Q:
            ax.axvline(0, color='green', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.suptitle(f"Grand Figure for pedestal substracted values (zoom), mingo0{station}\n{start_time}", fontsize=16)
        
        if save_plots:
            final_filename = f'{fig_idx}_grand_figure_Q_pedestal_zoom.png'
            fig_idx += 1
            
            save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
            plot_list.append(save_fig_path)
            plt.savefig(save_fig_path, format='png')
        
        if show_plots: plt.show()
        plt.close(fig_Q)


# ----------------------------------------------------------------------------------
# ----------------------- Charge calibration from ns to fC -------------------------
# ----------------------------------------------------------------------------------

# --- Define FEE Calibration ---
FEE_calibration = {
    "Width": [
        0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150,
        160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290,
        300, 310, 320, 330, 340, 350, 360, 370, 380, 390
    ],
    "Fast Charge": [
        4.0530E+01, 2.6457E+02, 4.5081E+02, 6.0573E+02, 7.3499E+02, 8.4353E+02,
        9.3562E+02, 1.0149E+03, 1.0845E+03, 1.1471E+03, 1.2047E+03, 1.2592E+03,
        1.3118E+03, 1.3638E+03, 1.4159E+03, 1.4688E+03, 1.5227E+03, 1.5779E+03,
        1.6345E+03, 1.6926E+03, 1.7519E+03, 1.8125E+03, 1.8742E+03, 1.9368E+03,
        2.0001E+03, 2.0642E+03, 2.1288E+03, 2.1940E+03, 2.2599E+03, 2.3264E+03,
        2.3939E+03, 2.4625E+03, 2.5325E+03, 2.6044E+03, 2.6786E+03, 2.7555E+03,
        2.8356E+03, 2.9196E+03, 3.0079E+03, 3.1012E+03
    ]
}
FEE_calibration = pd.DataFrame(FEE_calibration)
cs = CubicSpline(FEE_calibration['Width'].to_numpy(),
                 FEE_calibration['Fast Charge'].to_numpy(),
                 bc_type='natural')

def interpolate_fast_charge(width_array):
    """ Interpolates fast charge for array-like width values using cubic spline. """
    width_array = np.asarray(width_array)
    return np.where(width_array == 0, 0, cs(width_array))

# --- Calibrate and store new columns in working_df ---
for key in ['Q1', 'Q2', 'Q3', 'Q4']:
    for j in range(1, 5):
        for suffix in ['F', 'B']:
            col = f"{key}_{suffix}_{j}"
            if col in charge_test.columns:
                col_fC = f"{col}_fC"
                raw = charge_test[col]
                mask = (raw != 0) & np.isfinite(raw)
                charge_test[col_fC] = 0.0  # initialize
                charge_test.loc[mask, col_fC] = interpolate_fast_charge(raw[mask])




if create_plots:
    Q_clip_min = 0
    Q_clip_max = 1750
    num_bins = 100
    log_scale = True
    
    fig_Q, axes_Q = plt.subplots(4, 4, figsize=(20, 10))
    axes_Q = axes_Q.flatten()

    for i, key in enumerate(['Q1', 'Q2', 'Q3', 'Q4']):
        for j in range(4):
            col_F = f'{key}_F_{j+1}_fC'
            col_B = f'{key}_B_{j+1}_fC'
            ax = axes_Q[i*4 + j]

            if col_F in charge_test.columns:
                y_F = charge_test[col_F]
                y_F = y_F[(y_F > Q_clip_min) & (y_F < Q_clip_max) & np.isfinite(y_F)]
                ax.hist(y_F, bins=num_bins, alpha=0.5, label=f'{col_F}')

            if col_B in charge_test.columns:
                y_B = charge_test[col_B]
                y_B = y_B[(y_B > Q_clip_min) & (y_B < Q_clip_max) & np.isfinite(y_B)]
                ax.hist(y_B, bins=num_bins, alpha=0.5, label=f'{col_B}')

            ax.set_title(f"{col_F} vs {col_B}")
            ax.set_xlabel('Charge [fC]')
            ax.legend()

            if log_scale:
                ax.set_yscale('log')

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.suptitle(f"Grand Figure for calibrated charge (fC), mingo0{station}\n{start_time}", fontsize=16)

    if save_plots:
        final_filename = f'{fig_idx}_grand_figure_Q_fC.png'
        fig_idx += 1
        save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
        plot_list.append(save_fig_path)
        plt.savefig(save_fig_path, format='png')

    if show_plots:
        plt.show()
    plt.close(fig_Q)


# ----------------------------------------------------------------------------------
# -------------------------- Position offset calibration ---------------------------
# ----------------------------------------------------------------------------------

print("--------------------------------------------------------------------------")
print("-------------------- Position offset calibration -------------------------")
print("--------------------------------------------------------------------------")

pos_test = working_df.copy()
for i, key in enumerate(['T1', 'T2', 'T3', 'T4']):
    for j in range(4):
        pos_test[f'{key}_diff_{j+1}'] = ( pos_test[f'{key}_F_{j+1}'] - pos_test[f'{key}_B_{j+1}'] ) / 2

pos_test_copy = pos_test.copy()
Tdiff_cal = []
for key in ['1', '2', '3', '4']:
    T_F_cols = [f'T{key}_F_{i+1}' for i in range(4)]
    T_F = working_df[T_F_cols].values
    
    T_B_cols = [f'T{key}_B_{i+1}' for i in range(4)]
    T_B = working_df[T_B_cols].values
    
    Tdiff_cal_component = [calibrate_strip_T_diff(T_F[:,i], T_B[:,i]) for i in range(4)]
    Tdiff_cal.append(Tdiff_cal_component)
Tdiff_cal = np.array(Tdiff_cal)

print("\nTime diff. offset:")
print(Tdiff_cal, "\n")

validate_pos_cal = False
if validate_pos_cal:

    for i, key in enumerate(['T1', 'T2', 'T3', 'T4']):
        for j in range(4):
            mask = pos_test_copy[f'{key}_diff_{j+1}'] != 0
            pos_test.loc[mask, f'{key}_diff_{j+1}'] -= Tdiff_cal[i][j]

    if create_plots:
        # Create the grand figure for Q values
        fig_Q, axes_Q = plt.subplots(4, 4, figsize=(20, 10))  # Adjust the layout as necessary
        axes_Q = axes_Q.flatten()
        
        for i, key in enumerate(['T1', 'T2', 'T3', 'T4']):
            for j in range(4):
                col_F = f'{key}_diff_{j+1}'
                y_F = pos_test[col_F]
                
                Q_clip_min = -5
                Q_clip_max = 5
                
                # Plot histograms with Q-specific clipping and bins
                axes_Q[i*4 + j].hist(y_F[(y_F != 0) & (y_F > Q_clip_min) & (y_F < Q_clip_max)], 
                                     bins=num_bins, alpha=0.5, label=f'{col_F}')
                axes_Q[i*4 + j].set_title(f'{col_F}')
                axes_Q[i*4 + j].legend()
                axes_Q[i*4 + j].set_xlabel('T_diff / ns')
                axes_Q[i*4 + j].set_xlim([Q_clip_min, Q_clip_max])
                
                # if log_scale:
                #     axes_Q[i*4 + j].set_yscale('log')  # For Q values
                
            for ax in axes_Q:
                ax.axvline(0, color='green', linestyle='--', alpha=0.5)
            
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.suptitle(f"Grand Figure for position calibration, new method, mingo0{station}\n{start_time}", fontsize=16)
        
        if save_plots:
            final_filename = f'{fig_idx}_grand_figure_T_diff_cal.png'
            fig_idx += 1
            save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
            plot_list.append(save_fig_path)
            plt.savefig(save_fig_path, format='png')
        if show_plots: plt.show()
        plt.close(fig_Q)


# ----------------------------------------------------------------------------------
# -------------------------- Semisums and semidifferences --------------------------
# ----------------------------------------------------------------------------------

for key in ['T1', 'T2', 'T3', 'T4']:
    T_F_cols = [f'{key}_F_{i+1}' for i in range(4)]
    T_B_cols = [f'{key}_B_{i+1}' for i in range(4)]
    Q_F_cols = [f'{key.replace("T", "Q")}_F_{i+1}' for i in range(4)]
    Q_B_cols = [f'{key.replace("T", "Q")}_B_{i+1}' for i in range(4)]

    T_F = working_df[T_F_cols].values
    T_B = working_df[T_B_cols].values
    Q_F = working_df[Q_F_cols].values
    Q_B = working_df[Q_B_cols].values

    new_cols = {}
    for i in range(4):
        new_cols[f'{key}_T_sum_{i+1}'] = (T_F[:, i] + T_B[:, i]) / 2
        new_cols[f'{key}_T_diff_{i+1}'] = (T_F[:, i] - T_B[:, i]) / 2
        new_cols[f'{key.replace("T", "Q")}_Q_sum_{i+1}'] = (Q_F[:, i] + Q_B[:, i]) / 2
        new_cols[f'{key.replace("T", "Q")}_Q_diff_{i+1}'] = (Q_F[:, i] - Q_B[:, i]) / 2

    working_df = pd.concat([working_df, pd.DataFrame(new_cols, index=working_df.index)], axis=1)


# if create_essential_plots or create_plots:
if create_plots:
    num_columns = len(working_df.columns) - 1  # Exclude 'datetime'
    num_rows = (num_columns + 7) // 8  # Adjust as necessary for better layout
    fig, axes = plt.subplots(num_rows, 8, figsize=(20, num_rows * 2))
    axes = axes.flatten()

    for i, col in enumerate([col for col in working_df.columns if col != 'datetime']):
        y = working_df[col]
        
        if 'Q_sum' in col:
            color = Q_sum_color
        if 'Q_diff' in col:
            color = Q_diff_color
        if 'T_sum' in col:
            color = T_sum_color
        if 'T_diff' in col:
            color = T_diff_color
        axes[i].hist(y[y != 0], bins=100, alpha=0.5, label=col, color=color)
        
        axes[i].set_title(col)
        axes[i].legend()
        if 'Q_sum' in col:
            axes[i].set_yscale('log')
    
    # Remove any unused axes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.suptitle("Uncalibrated data")
    
    if save_plots:
        name_of_file = 'uncalibrated'
        final_filename = f'{fig_idx}_{name_of_file}.png'
        fig_idx += 1

        save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
        plot_list.append(save_fig_path)
        plt.savefig(save_fig_path, format='png')
    
    if show_plots: 
        plt.show()
    plt.close()


print("----------------------------------------------------------------------")
print("----------------------------------------------------------------------")
print("--------------------- Filters and calibrations -----------------------")
print("----------------------------------------------------------------------")
print("----------------------------------------------------------------------")

print("-------------------- Filter 2: uncalibrated data ---------------------")

# FILTER 2: TSUM, TDIF, QSUM, QDIF PRECALIBRATED THRESHOLDS --> 0 if out ------------------------------
for col in working_df.columns:
    if 'T_sum' in col:
        working_df[col] = np.where((working_df[col] > T_sum_right_pre_cal) | (working_df[col] < T_sum_left_pre_cal), 0, working_df[col])
    if 'T_diff' in col:
        working_df[col] = np.where((working_df[col] > T_diff_pre_cal_threshold) | (working_df[col] < -T_diff_pre_cal_threshold), 0, working_df[col])
    if 'Q_sum' in col:
        working_df[col] = np.where((working_df[col] > Q_right_pre_cal) | (working_df[col] < Q_left_pre_cal), 0, working_df[col])
    if 'Q_diff' in col:
        working_df[col] = np.where((working_df[col] > Q_diff_pre_cal_threshold) | (working_df[col] < -Q_diff_pre_cal_threshold), 0, working_df[col])


# if create_essential_plots or create_plots:
if create_plots:
    num_columns = len(working_df.columns) - 1  # Exclude 'datetime'
    num_rows = (num_columns + 7) // 8  # Adjust as necessary for better layout
    fig, axes = plt.subplots(num_rows, 8, figsize=(20, num_rows * 2))
    axes = axes.flatten()

    for i, col in enumerate([col for col in working_df.columns if col != 'datetime']):
        y = working_df[col]
        
        if 'Q_sum' in col:
            color = Q_sum_color
        if 'Q_diff' in col:
            color = Q_diff_color
        if 'T_sum' in col:
            color = T_sum_color
        if 'T_diff' in col:
            color = T_diff_color
        axes[i].hist(y[y != 0], bins=100, alpha=0.5, label=col, color=color)
        
        axes[i].set_title(col)
        axes[i].legend()
        if 'Q_sum' in col:
            axes[i].set_yscale('log')
    
    # Remove any unused axes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.suptitle("Uncalibrated data")
    
    if save_plots:
        name_of_file = 'uncalibrated_filtered'
        final_filename = f'{fig_idx}_{name_of_file}.png'
        fig_idx += 1

        save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
        plot_list.append(save_fig_path)
        plt.savefig(save_fig_path, format='png')
    
    if show_plots: 
        plt.show()
    plt.close()


print("----------------------------------------------------------------------")
print("----------- Charge sum pedestal, calibration and filtering -----------")
print("----------------------------------------------------------------------")

for i, key in enumerate(['Q1', 'Q2', 'Q3', 'Q4']):
    for j in range(4):
        mask = working_df[f'{key}_Q_sum_{j+1}'] != 0
        # working_df.loc[mask, f'{key}_Q_sum_{j+1}'] -= calibration_Q[i][j]
        working_df.loc[mask, f'{key}_Q_sum_{j+1}'] -= ( QF_pedestal[i][j] + QB_pedestal[i][j] ) / 2


print("------------------ Filter 3: charge sum filtering --------------------")
for col in working_df.columns:
    if 'Q_sum' in col:
        working_df[col] = np.where((working_df[col] > Q_sum_right_cal) | (working_df[col] < Q_sum_left_cal), 0, working_df[col])


print("----------------------------------------------------------------------")
print("----------------- Time diff calibration and filtering ----------------")
print("----------------------------------------------------------------------")

for i, key in enumerate(['T1', 'T2', 'T3', 'T4']):
    for j in range(4):
        mask = working_df[f'{key}_T_diff_{j+1}'] != 0
        working_df.loc[mask, f'{key}_T_diff_{j+1}'] -= Tdiff_cal[i][j]

print("--------------------- Filter 3.2: time diff filtering ----------------")
for col in working_df.columns:
    if 'T_diff' in col:
        working_df[col] = np.where((working_df[col] > T_diff_cal_threshold) | (working_df[col] < -T_diff_cal_threshold), 0, working_df[col])


print("----------------------------------------------------------------------")
print("---------------- Charge diff calibration and filtering ---------------")
print("----------------------------------------------------------------------")

for i, key in enumerate(['Q1', 'Q2', 'Q3', 'Q4']):
    for j in range(4):
        mask = working_df[f'{key}_Q_diff_{j+1}'] != 0
        # working_df.loc[mask, f'{key}_Q_diff_{j+1}'] -= calibration_Q_FB[i][j]
        working_df.loc[mask, f'{key}_Q_diff_{j+1}'] -= ( QF_pedestal[i][j] - QB_pedestal[i][j] ) / 2


print("------------------ Filter 4: charge diff filtering -------------------")
for col in working_df.columns:
    if 'Q_diff' in col:
        working_df[col] = np.where((working_df[col] > Q_diff_cal_threshold) | (working_df[col] < -Q_diff_cal_threshold), 0, working_df[col])


# For articles and presentations
# if presentation_plots:
#     plane = 2
#     strip = 2
#     data = [f'P{plane}_T_sum_{strip}', f'P{plane}_T_diff_{strip}', f'Q{plane}_Q_sum_{strip}', f'Q{plane}_Q_diff_{strip}']
#     fig_idx = 0  # Assuming fig_idx is defined earlier
#     plot_list = []  # Assuming plot_list is defined earlier

#     for i, col in enumerate([col for col in working_df.columns if col != 'datetime'][:len(data)]):
#         y = working_df[col]
#         if 'Q_sum' in col:
#             color = 'green'
#         elif 'Q_diff' in col:
#             color = 'blue'
#         elif 'T_sum' in col:
#             color = T_sum_color
#         elif 'T_diff' in col:
#             color = 'red'
#         y_p = y[y != 0]
#         fig, ax = plt.subplots(figsize=(5, 3.5))
#         ax.hist(y_p, bins=500, label=f"{len(y_p)} entries", alpha=0.5, color=color)
#         if 'T_diff' in col:
#             ax.set_xlim([-1, 1])
#         if 'Q_diff' in col:
#             ax.set_xlim([-2, 2])
#         if 'Q_sum' in col:
#             ax.set_yscale('log')
#             ax.set_xlim([-5, 50])
#         if 'Q' in col:
#             ax.set_xlabel('QtW / ns')
#         elif 'T' in col:
#             ax.set_xlabel('T / ns')
#         ax.set_ylabel('Counts')
#         ax.legend(frameon=False, handletextpad=0, handlelength=0)
#         plt.tight_layout()
#         if save_plots:
#             name_of_file = data[i].replace(' ', '_').replace('/', '_')  # Sanitize file name
#             final_filename = f'{fig_idx}_{name_of_file}.png'
#             fig_idx += 1
#             save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
#             plot_list.append(save_fig_path)
#             plt.savefig(save_fig_path, format='png')
#         if show_plots: plt.show()
#         plt.close()


# if create_essential_plots or create_plots:
if create_plots:
    num_columns = len(working_df.columns) - 1  # Exclude 'datetime'
    num_rows = (num_columns + 7) // 8  # Adjust as necessary for better layout
    fig, axes = plt.subplots(num_rows, 8, figsize=(20, num_rows * 2))
    axes = axes.flatten()

    for i, col in enumerate([col for col in working_df.columns if col != 'datetime']):
        y = working_df[col]
        
        if 'Q_sum' in col:
            color = Q_sum_color
        if 'Q_diff' in col:
            color = Q_diff_color
        if 'T_sum' in col:
            color = T_sum_color
        if 'T_diff' in col:
            color = T_diff_color
        axes[i].hist(y[y != 0], bins=100, alpha=0.5, label=col, color=color)
        
        axes[i].set_title(col)
        axes[i].legend()
        if 'Q_sum' in col:
            axes[i].set_yscale('log')
    
    # Remove any unused axes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.suptitle("Calibrated filtered data before FB correction")
    
    if save_plots:
        name_of_file = 'calibrated_filtered_before_FB_corr'
        final_filename = f'{fig_idx}_{name_of_file}.png'
        fig_idx += 1

        save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
        plot_list.append(save_fig_path)
        plt.savefig(save_fig_path, format='png')
    
    if show_plots: 
        plt.show()
    plt.close()


print("----------------------------------------------------------------------")
print("------------------- Charge front-back correction ---------------------")
print("----------------------------------------------------------------------")

if charge_front_back:
    for key in [1, 2, 3, 4]:
        for i in range(4):
            # Extract data from the DataFrame
            Q_sum = working_df[f'Q{key}_Q_sum_{i+1}'].values
            Q_diff = working_df[f'Q{key}_Q_diff_{i+1}'].values

            # Apply condition to filter non-zero Q_sum and Q_diff
            cond = (Q_sum != 0) & (Q_diff != 0)
            Q_sum_adjusted = Q_sum[cond]
            Q_diff_adjusted = Q_diff[cond]
            
            # Skip correction if no data is left after filtering
            if np.sum(Q_sum_adjusted) == 0:
                continue

            # Perform scatter plot and fit
            title = f"Q{key}_{i+1}. Charge diff. vs. charge sum."
            x_label = "Charge sum"
            y_label = "Charge diff"
            name_of_file = f"Q{key}_{i+1}_charge_analysis_scatter_diff_vs_sum"
            coeffs = scatter_2d_and_fit_new(Q_sum_adjusted, Q_diff_adjusted, title, x_label, y_label, name_of_file)
            working_df.loc[cond, f'Q{key}_Q_diff_{i+1}'] = Q_diff_adjusted - polynomial(Q_sum_adjusted, *coeffs)
            
    print('\nCharge front-back correction performed.')
    
else:
    print('Charge front-back correction was selected to not be performed.')
    Q_diff_cal_threshold_FB = 10


print("----------------------------------------------------------------------")
print("------------- Filter 5: charge difference FB filtering ---------------")
for col in working_df.columns:
    if 'Q_diff' in col:
        working_df[col] = np.where(np.abs(working_df[col]) < Q_diff_cal_threshold_FB, working_df[col], 0)

# ------------------------------------------------------------

# if create_essential_plots or create_plots:
if create_plots:
    num_columns = len(working_df.columns) - 1  # Exclude 'datetime'
    num_rows = (num_columns + 7) // 8  # Adjust as necessary for better layout
    fig, axes = plt.subplots(num_rows, 8, figsize=(20, num_rows * 2))
    axes = axes.flatten()

    for i, col in enumerate([col for col in working_df.columns if col != 'datetime']):
        y = working_df[col]
        
        if 'Q_sum' in col:
            color = Q_sum_color
        if 'Q_diff' in col:
            color = Q_diff_color
        if 'T_sum' in col:
            color = T_sum_color
        if 'T_diff' in col:
            color = T_diff_color
        axes[i].hist(y[y != 0], bins=100, alpha=0.5, label=col, color=color)
        
        axes[i].set_title(col)
        axes[i].legend()
        if 'Q_sum' in col:
            axes[i].set_yscale('log')
    
    # Remove any unused axes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.suptitle("Calibrated filtered data including FB correction")
    
    if save_plots:
        name_of_file = 'calibrated_filtered'
        final_filename = f'{fig_idx}_{name_of_file}.png'
        fig_idx += 1

        save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
        plot_list.append(save_fig_path)
        plt.savefig(save_fig_path, format='png')
    
    if show_plots: 
        plt.show()
    plt.close()


print("----------------------------------------------------------------------")
print("------------- Filter if any variable in the strip is 0 ---------------")
print("----------------------------------------------------------------------")

# Now go throuhg every plane and strip and if any of the T_sum, T_diff, Q_sum, Q_diff == 0,
# put the four variables in that plane, strip and event to 0

total_events = len(working_df)

for plane in range(1, 5):
    for strip in range(1, 5):
        q_sum  = f'Q{plane}_Q_sum_{strip}'
        q_diff = f'Q{plane}_Q_diff_{strip}'
        t_sum  = f'T{plane}_T_sum_{strip}'
        t_diff = f'T{plane}_T_diff_{strip}'
        
        # Build mask
        mask = (
            (working_df[q_sum]  == 0) |
            (working_df[q_diff] == 0) |
            (working_df[t_sum]  == 0) |
            (working_df[t_diff] == 0)
        )
        
        # Count affected events
        num_affected_events = mask.sum()
        print(f"Plane {plane}, Strip {strip}: {num_affected_events} out of {total_events} events affected ({(num_affected_events / total_events) * 100:.2f}%)")

        # Zero the affected values
        working_df.loc[mask, [q_sum, q_diff, t_sum, t_diff]] = 0


# if create_essential_plots or create_plots:
if create_plots:
    num_columns = len(working_df.columns) - 1  # Exclude 'datetime'
    num_rows = (num_columns + 7) // 8  # Adjust as necessary for better layout
    fig, axes = plt.subplots(num_rows, 8, figsize=(20, num_rows * 2))
    axes = axes.flatten()

    for i, col in enumerate([col for col in working_df.columns if col != 'datetime']):
        y = working_df[col]
        
        if 'Q_sum' in col:
            color = Q_sum_color
        if 'Q_diff' in col:
            color = Q_diff_color
        if 'T_sum' in col:
            color = T_sum_color
        if 'T_diff' in col:
            color = T_diff_color
        axes[i].hist(y[y != 0], bins=100, alpha=0.5, label=col, color=color)
        
        axes[i].set_title(col)
        axes[i].legend()
        if 'Q_sum' in col:
            axes[i].set_yscale('log')
    
    # Remove any unused axes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.suptitle("Calibrated filtered data including FB correction removing zeroes in any variable")
    
    if save_plots:
        name_of_file = 'calibrated_filtered_removed_zeroes'
        final_filename = f'{fig_idx}_{name_of_file}.png'
        fig_idx += 1

        save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
        plot_list.append(save_fig_path)
        plt.savefig(save_fig_path, format='png')
    
    if show_plots: 
        plt.show()
    plt.close()



print("----------------------------------------------------------------------")
print("---------------------- Slewing correction 1/2 ------------------------")
print("----------------------------------------------------------------------")

if slewing_correction:
    
    # Select desired columns
    cols = working_df.columns
    t_sum_cols   = [c for c in cols if 'T_sum' in c and '_final' not in c]
    q_sum_cols   = [c for c in cols if 'Q_sum' in c and '_final' not in c]
    t_diff_cols  = [c for c in cols if 'T_diff' in c and '_final' not in c]
    type_col     = ['type'] if 'type' in cols else []

    data_df_times   = working_df[t_sum_cols]
    data_df_charges = working_df[q_sum_cols]
    data_df_tdiff   = working_df[t_diff_cols]
    type_series     = working_df[type_col] if type_col else None

    # Concatenate all relevant data with 'type' column
    data_df_filt = pd.concat([data_df_charges, data_df_times, data_df_tdiff], axis=1)
    data_slew = data_df_filt
    
    # Select y_pos for each plane
    y_lookup = {
        1: y_pos_T[0],
        2: y_pos_T[1],
        3: y_pos_T[0],
        4: y_pos_T[1],
    }
    
    results = []
    
    # Loop through all combinations of planes and strips
    for (p1, s1), (p2, s2) in combinations([(p, s) for p in range(1, 5) for s in range(1, 5)], 2):
        Q1 = data_slew[f"Q{p1}_Q_sum_{s1}"]
        Q2 = data_slew[f"Q{p2}_Q_sum_{s2}"]
        T1 = data_slew[f"T{p1}_T_sum_{s1}"]
        T2 = data_slew[f"T{p2}_T_sum_{s2}"]
        TD1 = data_slew[f"T{p1}_T_diff_{s1}"]
        TD2 = data_slew[f"T{p2}_T_diff_{s2}"]
        
        valid_mask = (
            (Q1 != 0) & (Q2 != 0) &
            (T1 != 0) & (T2 != 0) &
            (TD1 != 0) & (TD2 != 0)
        )

        # Apply mask to compute only valid values
        Q1 = Q1[valid_mask]
        Q2 = Q2[valid_mask]
        T1 = T1[valid_mask]
        T2 = T2[valid_mask]
        TD1 = TD1[valid_mask]
        TD2 = TD2[valid_mask]
        
        x1 = TD1 * tdiff_to_x  # mm
        x2 = TD2 * tdiff_to_x
        y1 = y_lookup[p1][s1 - 1]
        y2 = y_lookup[p2][s2 - 1]
        z1 = z_positions[p1 - 1]
        z2 = z_positions[p2 - 1]

        dx = x1 - x2
        dy = y1 - y2
        dz = z1 - z2

        travel_time = np.sqrt(dx**2 + dy**2 + dz**2) / c_mm_ns
        tsum_diff = (T1 - T2)
        corrected_tsum_diff = tsum_diff + travel_time

        results.append(pd.DataFrame({
            'plane1': p1, 'strip1': s1,
            'plane2': p2, 'strip2': s2,
            'Q_sum_semidiff': 0.5 * (Q1 - Q2),
            'Q_sum_semisum':  0.5 * (Q1 + Q2),
            'T_sum_corrected_diff': corrected_tsum_diff,
            'T_sum_diff': tsum_diff,
            'x_diff': dx,
            'travel_time': travel_time
        }))

    # Concatenate all results
    slew_df = pd.concat(results, ignore_index=True)

    # if create_essential_plots or create_plots:
    if create_plots:

        pair_labels = [
            (p1, s1, p2, s2)
            for p1 in range(1, 5)
            for s1 in range(1, 5)
            for p2 in range(p1 + 1, 5)
            for s2 in range(1, 5)
        ]

        # Parameters
        batch_size = 4  # number of pairs per batch
        num_batches = int(np.ceil(len(pair_labels) / batch_size))

        for batch in range(num_batches):
            start_idx = batch * batch_size
            end_idx = min((batch + 1) * batch_size, len(pair_labels))
            current_pairs = pair_labels[start_idx:end_idx]

            fig, axes = plt.subplots(1, batch_size, figsize=(6 * batch_size, 5), constrained_layout=True)
            axes = np.atleast_2d(axes)

            for col_idx, (p1, s1, p2, s2) in enumerate(current_pairs):
                mask = (
                    (slew_df['plane1'] == p1) & (slew_df['strip1'] == s1) &
                    (slew_df['plane2'] == p2) & (slew_df['strip2'] == s2)
                )
                data = slew_df[mask]

                # Only use rows with valid x_diff and T values
                valid_mask = (
                    (data['x_diff'] != 0) &
                    (data['T_sum_corrected_diff'] != 0) &
                    (data['T_sum_diff'] != 0)
                )
                data = data[valid_mask]

                x = data['x_diff']
                t_uncorrected = data['T_sum_diff']
                t_corrected = data['T_sum_corrected_diff']

                # dx vs T_sum_diff (uncorrected)
                ax1 = axes[0, col_idx]
                ax1.scatter(x, t_uncorrected, s=5, alpha=0.6, color='tab:red')
                ax1.set_title(f"P{p1}S{s1} vs P{p2}S{s2} — Uncorrected")
                ax1.set_xlabel("dx (mm)")
                ax1.set_ylabel("T_sum_diff (ns)")

                ax1.scatter(x, t_corrected, s=5, alpha=0.6, color='tab:blue')
                ax1.set_title(f"P{p1}S{s1} vs P{p2}S{s2} — Corrected")

            # Hide unused subplots
            for row in range(1):
                for col in range(len(current_pairs), batch_size):
                    axes[row, col].set_visible(False)

            plt.suptitle(f"Batch {batch + 1}/{num_batches} — dx vs Time Differences", fontsize=18, y=1.01)

            if save_plots:
                name_of_file = 'dx_vs_tsum'
                final_filename = f'{fig_idx}_{name_of_file}.png'
                fig_idx += 1

                save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
                plot_list.append(save_fig_path)
                plt.savefig(save_fig_path, format='png')

            if show_plots:
                plt.show()
            plt.close()

    
    # if create_essential_plots or create_plots:
    if create_plots:   
        pair_labels = [
            (p1, s1, p2, s2)
            for p1 in range(1, 5)
            for s1 in range(1, 5)
            for p2 in range(p1 + 1, 5)
            for s2 in range(1, 5)
        ]

        # Parameters
        batch_size = 4  # number of pairs per batch
        num_batches = int(np.ceil(len(pair_labels) / batch_size))
        
        for batch in range(num_batches):
            start_idx = batch * batch_size
            end_idx = min((batch + 1) * batch_size, len(pair_labels))
            current_pairs = pair_labels[start_idx:end_idx]

            fig, axes = plt.subplots(1, batch_size, figsize=(6 * batch_size, 5), constrained_layout=True)
            axes = np.atleast_1d(axes)

            for col_idx, (p1, s1, p2, s2) in enumerate(current_pairs):
                mask = (
                    (slew_df['plane1'] == p1) & (slew_df['strip1'] == s1) &
                    (slew_df['plane2'] == p2) & (slew_df['strip2'] == s2)
                )
                data = slew_df[mask]

                # Valid entries only
                valid_mask = (
                    (data['x_diff'] != 0) &
                    (data['travel_time'] != 0)
                )
                data = data[valid_mask]

                x = data['x_diff']
                t = data['travel_time']

                ax = axes[col_idx]
                ax.scatter(x, t, s=5, alpha=0.6, color='tab:green')
                ax.set_title(f"P{p1}S{s1} vs P{p2}S{s2}")
                ax.set_xlabel("dx (mm)")
                ax.set_ylabel("travel_time (ns)")

            # Hide unused subplots
            for col in range(len(current_pairs), batch_size):
                axes[col].set_visible(False)

            plt.suptitle(f"Batch {batch + 1}/{num_batches} — dx vs Travel Time", fontsize=18, y=1.01)

            if save_plots:
                name_of_file = 'dx_vs_travel_time'
                final_filename = f'{fig_idx}_{name_of_file}.png'
                fig_idx += 1

                save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
                plot_list.append(save_fig_path)
                plt.savefig(save_fig_path, format='png')

            if show_plots:
                plt.show()
            plt.close()


    
    # if create_essential_plots or create_plots:
    if create_plots:
        
        pair_labels = [
            (p1, s1, p2, s2)
            for p1 in range(1, 5)
            for s1 in range(1, 5)
            for p2 in range(p1 + 1, 5)
            for s2 in range(1, 5)
        ]

        # Parameters
        batch_size = 4  # number of pairs per batch
        num_batches = int(np.ceil(len(pair_labels) / batch_size))

        # ---- Loop through batches of pairs and plot all three histograms per pair ----
        for batch in range(num_batches):
            start_idx = batch * batch_size
            end_idx = min((batch + 1) * batch_size, len(pair_labels))
            current_pairs = pair_labels[start_idx:end_idx]

            fig, axes = plt.subplots(3, batch_size, figsize=(6 * batch_size, 12), constrained_layout=True)
            axes = np.atleast_2d(axes)

            for col_idx, (p1, s1, p2, s2) in enumerate(current_pairs):
                mask = (
                    (slew_df['plane1'] == p1) & (slew_df['strip1'] == s1) &
                    (slew_df['plane2'] == p2) & (slew_df['strip2'] == s2)
                )
                data = slew_df[mask]

                # Plot Q_sum_semidiff
                sns.histplot(data['Q_sum_semidiff'], bins=30, kde=True, ax=axes[0, col_idx])
                axes[0, col_idx].set_title(f"P{p1}S{s1} vs P{p2}S{s2} — Q_diff")
                axes[0, col_idx].set_xlabel("Q_sum_semidiff")
                axes[0, col_idx].set_ylabel("Counts")

                # Plot Q_sum_semisum
                sns.histplot(data['Q_sum_semisum'], bins=30, kde=True, ax=axes[1, col_idx])
                axes[1, col_idx].set_title(f"P{p1}S{s1} vs P{p2}S{s2} — Q_sum")
                axes[1, col_idx].set_xlabel("Q_sum_semisum")
                axes[1, col_idx].set_ylabel("Counts")

                # Plot T_sum_corrected_diff
                sns.histplot(data['T_sum_corrected_diff'], bins=30, kde=True, ax=axes[2, col_idx])
                axes[2, col_idx].set_title(f"P{p1}S{s1} vs P{p2}S{s2} — ΔT corrected")
                axes[2, col_idx].set_xlabel("T_sum_corrected_diff")
                axes[2, col_idx].set_ylabel("Counts")

            # Hide unused subplots
            for row in range(3):
                for col in range(len(current_pairs), batch_size):
                    axes[row, col].set_visible(False)

            plt.suptitle(f"Batch {batch + 1}/{num_batches} — Histograms per Plane/Strip Pair", fontsize=18, y=1.01)

            if save_plots:
                name_of_file = 'slewing'
                final_filename = f'{fig_idx}_{name_of_file}.png'
                fig_idx += 1

                save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
                plot_list.append(save_fig_path)
                plt.savefig(save_fig_path, format='png')
        
            if show_plots: 
                plt.show()
            plt.close()
            
            
    # if create_essential_plots or create_plots:
    if create_plots:
        
        pair_labels = [
            (p1, s1, p2, s2)
            for p1 in range(1, 5)
            for s1 in range(1, 5)
            for p2 in range(p1 + 1, 5)
            for s2 in range(1, 5)
        ]

        # Parameters
        batch_size = 4  # number of pairs per batch
        num_batches = int(np.ceil(len(pair_labels) / batch_size))
        
        for batch in range(num_batches):
            start_idx = batch * batch_size
            end_idx = min((batch + 1) * batch_size, len(pair_labels))
            current_pairs = pair_labels[start_idx:end_idx]

            fig = plt.figure(figsize=(6 * batch_size, 16), constrained_layout=True)
            spec = gridspec.GridSpec(nrows=4, ncols=batch_size, figure=fig)

            for col_idx, (p1, s1, p2, s2) in enumerate(current_pairs):
                mask = (
                    (slew_df['plane1'] == p1) & (slew_df['strip1'] == s1) &
                    (slew_df['plane2'] == p2) & (slew_df['strip2'] == s2)
                )
                data = slew_df[mask]

                # Drop rows with any invalid values for this pair
                valid_mask = (
                    (data['Q_sum_semidiff'] != 0) &
                    (data['Q_sum_semisum'] != 0) &
                    (data['T_sum_corrected_diff'] != 0)
                )
                data = data[valid_mask]

                x = data['T_sum_corrected_diff']
                y = data['Q_sum_semisum']
                z = data['Q_sum_semidiff']

                # 3D plot
                ax3d = fig.add_subplot(spec[0:2, col_idx], projection='3d')
                ax3d.scatter(x, y, z, s=5, alpha=0.6)
                ax3d.set_title(f"P{p1}S{s1} vs P{p2}S{s2}")
                ax3d.set_xlabel('ΔT corrected (ns)')
                ax3d.set_ylabel('Q_sum_semisum')
                ax3d.set_zlabel('Q_sum_semidiff')

                # XY projection
                ax_xy = fig.add_subplot(spec[2, col_idx])
                ax_xy.scatter(x, y, s=5, alpha=0.5)
                ax_xy.set_xlabel('ΔT corrected')
                ax_xy.set_ylabel('Q_sum_semisum')
                ax_xy.set_title('XY projection')

                # XZ projection
                ax_xz = fig.add_subplot(spec[3, col_idx])
                ax_xz.scatter(x, z, s=5, alpha=0.5, c='tab:red')
                ax_xz.set_xlabel('ΔT corrected')
                ax_xz.set_ylabel('Q_sum_semidiff')
                ax_xz.set_title('XZ projection')

            plt.suptitle(f"Batch {batch + 1}/{num_batches} — 3D Slewing Observables", fontsize=18, y=1.01)

            if save_plots:
                name_of_file = 'slewing_3d'
                final_filename = f'{fig_idx}_{name_of_file}.png'
                fig_idx += 1

                save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
                plot_list.append(save_fig_path)
                plt.savefig(save_fig_path, format='png')

            if show_plots:
                plt.show()
            plt.close()


    # THE FIT ----------------------------------------------------------------
    
    pair_labels = [
            (p1, s1, p2, s2)
            for p1 in range(1, 5)
            for s1 in range(1, 5)
            for p2 in range(p1 + 1, 5)
            for s2 in range(1, 5)
        ]
    
    # Store fitted model parameters
    fit_results = []

    def robust_z_filter(df, cols, threshold=3.5):

        mask = np.ones(len(df), dtype=bool)
        for col in cols:
            median = np.median(df[col])
            mad = median_abs_deviation(df[col], scale='normal')  # consistent with std if normal
            if mad == 0:
                continue  # skip flat distributions
            z_mod = 0.6745 * (df[col] - median) / mad
            mask &= np.abs(z_mod) < threshold
        return df[mask]
    
    for (p1, s1, p2, s2) in pair_labels:
        mask = (
            (slew_df['plane1'] == p1) & (slew_df['strip1'] == s1) &
            (slew_df['plane2'] == p2) & (slew_df['strip2'] == s2)
        )
        data = slew_df[mask]

        valid_mask = (
            (data['Q_sum_semidiff'] != 0) &
            (data['Q_sum_semisum'] != 0) &
            (data['T_sum_corrected_diff'] != 0)
        )
        data = data[valid_mask]

        if len(data) < 10:
            continue  # not enough data to fit
        
        # Apply some filtering on the values of Q_sum_semidiff and Q_sum_semisum
        data = data[
            (data['Q_sum_semidiff'] > -20) & (data['Q_sum_semidiff'] < 20) &
            (data['Q_sum_semisum'] > 10) & (data['Q_sum_semisum'] < 50) &
            (data['T_sum_corrected_diff'] > -4) & (data['T_sum_corrected_diff'] < 4)
        ]
        
        
        # Apply it to your DataFrame:
        data = robust_z_filter(data, ['Q_sum_semidiff', 'Q_sum_semisum', 'T_sum_corrected_diff'])
        
        not_use_q = False
        if not_use_q:
            X = data[['Q_sum_semidiff']].values
            y = data['T_sum_corrected_diff'].values

            model = LinearRegression()
            model.fit(X, y)
            
            b_semidiff = model.coef_[0]
            a_semisum = 0
        else:
            X = data[['Q_sum_semisum', 'Q_sum_semidiff']].values
            y = data['T_sum_corrected_diff'].values

            model = LinearRegression()
            model.fit(X, y)
            
            a_semisum = model.coef_[0]
            b_semidiff = model.coef_[1]

        # Store results
        fit_results.append({
            'plane1': p1, 'strip1': s1,
            'plane2': p2, 'strip2': s2,
            'a_semisum': a_semisum,
            'b_semidiff': b_semidiff,
            'c_offset': model.intercept_,
            'n_points': len(data)
        })

    # Create dataframe with all model parameters
    slewing_fit_df = pd.DataFrame(fit_results)
    
    print("Fitting results:")
    print(slewing_fit_df)
    
    
    # if create_essential_plots or create_plots:
    if create_plots:

        pair_labels = [
            (p1, s1, p2, s2)
            for p1 in range(1, 5)
            for s1 in range(1, 5)
            for p2 in range(p1 + 1, 5)
            for s2 in range(1, 5)
        ]

        batch_size = 4
        num_batches = int(np.ceil(len(pair_labels) / batch_size))

        for batch in range(num_batches):
            start_idx = batch * batch_size
            end_idx = min((batch + 1) * batch_size, len(pair_labels))
            current_pairs = pair_labels[start_idx:end_idx]

            fig = plt.figure(figsize=(6 * batch_size, 16), constrained_layout=True)
            spec = gridspec.GridSpec(nrows=4, ncols=batch_size, figure=fig)

            for col_idx, (p1, s1, p2, s2) in enumerate(current_pairs):
                mask = (
                    (slew_df['plane1'] == p1) & (slew_df['strip1'] == s1) &
                    (slew_df['plane2'] == p2) & (slew_df['strip2'] == s2)
                )
                data = slew_df[mask]

                valid_mask = (
                    (data['Q_sum_semidiff'] != 0) &
                    (data['Q_sum_semisum'] != 0) &
                    (data['T_sum_corrected_diff'] != 0)
                )
                data = data[valid_mask]

                if len(data) < 10:
                    continue

                # Retrieve fitted model coefficients from slewing_fit_df
                fit_row = slewing_fit_df[
                    (slewing_fit_df['plane1'] == p1) &
                    (slewing_fit_df['strip1'] == s1) &
                    (slewing_fit_df['plane2'] == p2) &
                    (slewing_fit_df['strip2'] == s2)
                ]
                if fit_row.empty:
                    continue

                a = fit_row['a_semisum'].values[0]
                b = fit_row['b_semidiff'].values[0]
                c = fit_row['c_offset'].values[0]

                x = data['T_sum_corrected_diff'].values
                y = data['Q_sum_semisum'].values
                z = data['Q_sum_semidiff'].values

                # 3D plot
                ax3d = fig.add_subplot(spec[0:2, col_idx], projection='3d')
                ax3d.scatter(x, y, z, s=5, alpha=0.6)
                ax3d.set_title(f"P{p1}S{s1} vs P{p2}S{s2}")
                ax3d.set_xlabel('ΔT corrected (ns)')
                ax3d.set_ylabel('Q_sum_semisum')
                ax3d.set_zlabel('Q_sum_semidiff')

                # XY projection
                ax_xy = fig.add_subplot(spec[2, col_idx])
                ax_xy.scatter(x, y, s=5, alpha=0.5)
                z_fixed = np.mean(z)
                y_line = np.linspace(np.min(y), np.max(y), 100)
                x_line = a * y_line + b * z_fixed + c
                ax_xy.plot(x_line, y_line, color='black', lw=1, label='Fitted projection')
                ax_xy.set_xlabel('ΔT corrected')
                ax_xy.set_ylabel('Q_sum_semisum')
                ax_xy.set_title('XY projection')
                ax_xy.legend(fontsize='x-small')

                # XZ projection
                ax_xz = fig.add_subplot(spec[3, col_idx])
                ax_xz.scatter(x, z, s=5, alpha=0.5, c='tab:red')
                y_fixed = np.mean(y)
                z_line = np.linspace(np.min(z), np.max(z), 100)
                x_line2 = a * y_fixed + b * z_line + c
                ax_xz.plot(x_line2, z_line, color='black', lw=1, label='Fitted projection')
                ax_xz.set_xlabel('ΔT corrected')
                ax_xz.set_ylabel('Q_sum_semidiff')
                ax_xz.set_title('XZ projection')
                ax_xz.legend(fontsize='x-small')

            plt.suptitle(f"Batch {batch + 1}/{num_batches} — 3D Slewing + Fitted Projections", fontsize=18, y=1.01)

            if save_plots:
                name_of_file = 'slewing_3d_fitproj'
                final_filename = f'{fig_idx}_{name_of_file}.png'
                fig_idx += 1

                save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
                plot_list.append(save_fig_path)
                plt.savefig(save_fig_path, format='png')

            if show_plots:
                plt.show()
            plt.close()


    # FIT VALIDATION (reduced to essential plots)
    # if create_essential_plots or create_plots:
    if create_plots:

        pair_labels = [
            (p1, s1, p2, s2)
            for p1 in range(1, 5)
            for s1 in range(1, 5)
            for p2 in range(p1 + 1, 5)
            for s2 in range(1, 5)
        ]

        batch_size = 4
        num_batches = int(np.ceil(len(pair_labels) / batch_size))
        
        for batch in range(num_batches):
            start_idx = batch * batch_size
            end_idx = min((batch + 1) * batch_size, len(pair_labels))
            current_pairs = pair_labels[start_idx:end_idx]

            fig, axes = plt.subplots(2, batch_size, figsize=(6 * batch_size, 8), constrained_layout=True)
            axes = np.atleast_2d(axes)

            for col_idx, (p1, s1, p2, s2) in enumerate(current_pairs):
                mask = (
                    (slew_df['plane1'] == p1) & (slew_df['strip1'] == s1) &
                    (slew_df['plane2'] == p2) & (slew_df['strip2'] == s2)
                )
                data = slew_df[mask]

                valid_mask = (
                    (data['Q_sum_semidiff'] != 0) &
                    (data['Q_sum_semisum'] != 0) &
                    (data['T_sum_corrected_diff'] != 0)
                )
                data = data[valid_mask]

                if len(data) < 10:
                    for row in range(2):
                        axes[row, col_idx].set_visible(False)
                    continue

                fit_row = slewing_fit_df[
                    (slewing_fit_df['plane1'] == p1) &
                    (slewing_fit_df['strip1'] == s1) &
                    (slewing_fit_df['plane2'] == p2) &
                    (slewing_fit_df['strip2'] == s2)
                ]
                if fit_row.empty:
                    for row in range(2):
                        axes[row, col_idx].set_visible(False)
                    continue

                a = fit_row['a_semisum'].values[0]
                b = fit_row['b_semidiff'].values[0]
                c = fit_row['c_offset'].values[0]

                qsum = data['Q_sum_semisum'].values
                qdiff = data['Q_sum_semidiff'].values
                t_true = data['T_sum_corrected_diff'].values
                t_pred = a * qsum + b * qdiff + c
                residual = t_true - t_pred
                
                # Filter residuals, remove if out of the range
                residual_range = 5
                cond = (residual > -1*residual_range) & (residual < residual_range)
                residual = residual[cond]
                t_true = t_true[cond]
                t_pred = t_pred[cond]
                
                # Plot predicted vs true with y=x line
                ax0 = axes[0, col_idx]
                ax0.scatter(t_true, t_pred, s=5, alpha=0.6)
                min_val = min(t_true.min(), t_pred.min())
                max_val = max(t_true.max(), t_pred.max())
                ax0.plot([min_val, max_val], [min_val, max_val], 'k--', lw=1)
                ax0.set_title(f"P{p1}S{s1} vs P{p2}S{s2}")
                ax0.set_xlabel("T_true")
                ax0.set_ylabel("T_predicted")

                # Residuals histogram
                ax1 = axes[1, col_idx]
                ax1.hist(t_true, bins=100, alpha=0.7, color='tab:gray', label = "Uncorrected")
                ax1.hist(residual, bins=100, alpha=0.7, color='green', label = "Residuals, same as corrected")
                ax1.set_xlabel("Residuals (ns)")
                ax1.set_ylabel("Counts")
                ax1.set_title("Residual Distribution")

            # Hide unused subplots
            for row in range(2):
                for col in range(len(current_pairs), batch_size):
                    axes[row, col].set_visible(False)

            plt.suptitle(f"Batch {batch + 1}/{num_batches} — Fit Check (Predicted vs Real, Residuals)", fontsize=18, y=1.01)

            if save_plots:
                name_of_file = 'model_validation_simple'
                final_filename = f'{fig_idx}_{name_of_file}.png'
                fig_idx += 1

                save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
                plot_list.append(save_fig_path)
                plt.savefig(save_fig_path, format='png')

            if show_plots:
                plt.show()
            plt.close()


print("----------------------------------------------------------------------")
print("----------------------- Time sum calibration -------------------------")
print("----------------------------------------------------------------------")

if time_calibration:
    old_timing_method = False
    if old_timing_method:
        # Initialize an empty list to store the resulting matrices for each event
        event_matrices = []
        
        # Iterate over each event (row) in the DataFrame
        for _, row in working_df.iterrows():
            event_matrix = []
            for module in ['T1', 'T2', 'T3', 'T4']:
                # Find the index of the strip with the maximum Q_sum for this module
                Q_sum_cols = [f'{module.replace("T", "Q")}_Q_sum_{i+1}' for i in range(4)]
                Q_sum_values = row[Q_sum_cols].values
                
                if sum(Q_sum_values) == 0:
                    event_matrix.append([0, 0, 0])
                    continue
                
                max_index = np.argmax(Q_sum_values) + 1
                    
                # Get the corresponding T_sum and T_diff for the module and strip
                T_sum_col = f'{module}_T_sum_{max_index}'
                T_diff_col = f'{module}_T_diff_{max_index}'
                T_sum_value = row[T_sum_col]
                T_diff_value = row[T_diff_col]
                
                # Append the row to the event matrix
                event_matrix.append([max_index, T_sum_value, T_diff_value])
            
            # Convert the event matrix to a numpy array and append it to the list of event matrices
            event_matrices.append(np.array(event_matrix))
        
        # Convert the list of event matrices to a 3D numpy array (events x modules x features)
        event_matrices = np.array(event_matrices)
        
        # The old code to do this -----------------------------
        
        yz_big = np.array([[[y, z] for y in y_pos_T[i % 2]] for i, z in enumerate(z_positions)])
        
        def calculate_diff(P_a, s_a, P_b, s_b, ps):
            
            # First position
            x_1 = ps[P_a-1, 1]
            yz_1 = yz_big[P_a-1, s_a-1]
            xyz_1 = np.append(x_1, yz_1)
            
            # Second position
            x_2 = ps[P_b-1, 1]
            yz_2 = yz_big[P_b-1, s_b-1]
            xyz_2 = np.append(x_2, yz_2)
            
            pos_x.append(x_1)
            pos_x.append(x_2)
            
            t_0_1 = ps[P_a-1, 2]
            t_0_2 = ps[P_b-1, 2]
            t_0.append(t_0_1)
            t_0.append(t_0_2)
            
            # Length
            dist = np.sqrt(np.sum((xyz_2 - xyz_1)**2))
            travel_time = dist / muon_speed
            
            v_travel_time.append(travel_time)
            
            # diff = travel_time
            diff = ps[P_b-1, 2] - ps[P_a-1, 2] - travel_time
            # diff = ps[P_b-1, 2] - ps[P_a-1, 2]
            return diff
        
        # Three layers spaced
        P1s1_P4s1 = []; P1s1_P4s2 = []; P1s2_P4s1 = []; P1s2_P4s2 = []; P1s2_P4s3 = []; P1s3_P4s2 = []; P1s3_P4s3 = []; P1s3_P4s4 = []; P1s4_P4s3 = []; P1s4_P4s4 = []; P1s1_P4s3 = []; P1s3_P4s1 = []; P1s2_P4s4 = []; P1s4_P4s2 = []; P1s1_P4s4 = [];

        # Two layers spaced
        P1s1_P3s1 = []; P1s1_P3s2 = []; P1s2_P3s1 = []; P1s2_P3s2 = []; P1s2_P3s3 = []; P1s3_P3s2 = []; P1s3_P3s3 = []; P1s3_P3s4 = []; P1s4_P3s3 = []; P1s4_P3s4 = []; P1s1_P3s3 = []; P1s3_P3s1 = []; P1s2_P3s4 = []; P1s4_P3s2 = []; P1s1_P3s4 = [];
        P2s1_P4s1 = []; P2s1_P4s2 = []; P2s2_P4s1 = []; P2s2_P4s2 = []; P2s2_P4s3 = []; P2s3_P4s2 = []; P2s3_P4s3 = []; P2s3_P4s4 = []; P2s4_P4s3 = []; P2s4_P4s4 = []; P2s1_P4s3 = []; P2s3_P4s1 = []; P2s2_P4s4 = []; P2s4_P4s2 = []; P2s1_P4s4 = [];

        # One layer spaced
        P1s1_P2s1 = []; P1s1_P2s2 = []; P1s2_P2s1 = []; P1s2_P2s2 = []; P1s2_P2s3 = []; P1s3_P2s2 = []; P1s3_P2s3 = []; P1s3_P2s4 = []; P1s4_P2s3 = []; P1s4_P2s4 = []; P1s1_P2s3 = []; P1s3_P2s1 = []; P1s2_P2s4 = []; P1s4_P2s2 = []; P1s1_P2s4 = [];
        P2s1_P3s1 = []; P2s1_P3s2 = []; P2s2_P3s1 = []; P2s2_P3s2 = []; P2s2_P3s3 = []; P2s3_P3s2 = []; P2s3_P3s3 = []; P2s3_P3s4 = []; P2s4_P3s3 = []; P2s4_P3s4 = []; P2s1_P3s3 = []; P2s3_P3s1 = []; P2s2_P3s4 = []; P2s4_P3s2 = []; P2s1_P3s4 = [];
        P3s1_P4s1 = []; P3s1_P4s2 = []; P3s2_P4s1 = []; P3s2_P4s2 = []; P3s2_P4s3 = []; P3s3_P4s2 = []; P3s3_P4s3 = []; P3s3_P4s4 = []; P3s4_P4s3 = []; P3s4_P4s4 = []; P3s1_P4s3 = []; P3s3_P4s1 = []; P3s2_P4s4 = []; P3s4_P4s2 = []; P3s1_P4s4 = [];
        
        pos_x = []
        v_travel_time = []
        t_0 = []
        
        # -----------------------------------------------------------------------------
        # Perform the calculation of a strip vs. the any other one --------------------
        # -----------------------------------------------------------------------------
        
        i = 0
        for event in event_matrices:
            if limit and i >= limit_number:
                break
            if np.all(event[:,0] == 0):
                continue
            
            istrip = event[:, 0]
            t0 = event[:,1] - strip_length / 2 / strip_speed
            x = event[:,2] * strip_speed
            
            ps = np.column_stack(( istrip, x,  t0 ))
            ps[:,2] = ps[:,2] - ps[0,2]
            
            # ---------------------------------------------------------------------
            # Fill the time differences vectors -----------------------------------
            # ---------------------------------------------------------------------
            
            # Three layers spacing ------------------------------------------------
            # P1-P4 ---------------------------------------------------------------
            P_a = 1; P_b = 4
            # Same strips
            s_a = 1; s_b = 1
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P1s1_P4s1.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 2; s_b = 2
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P1s2_P4s2.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 3; s_b = 3
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P1s3_P4s3.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 4; s_b = 4
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P1s4_P4s4.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            # Adjacent strips
            s_a = 1; s_b = 2
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P1s1_P4s2.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 2; s_b = 1
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P1s2_P4s1.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 2; s_b = 3
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P1s2_P4s3.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 3; s_b = 2
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P1s3_P4s2.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 3; s_b = 4
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P1s3_P4s4.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 4; s_b = 3
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P1s4_P4s3.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            # Two separated strips
            s_a = 1; s_b = 3
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P1s1_P4s3.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 3; s_b = 1
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P1s3_P4s1.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 2; s_b = 4
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P1s2_P4s4.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 4; s_b = 2
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P1s4_P4s2.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            # Three separated strips
            s_a = 1; s_b = 4
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P1s1_P4s4.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            
            # Two layers spacing --------------------------------------------------
            # P1-P3 ---------------------------------------------------------------
            P_a = 1; P_b = 3
            # Same strips
            s_a = 1; s_b = 1
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P1s1_P3s1.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 2; s_b = 2
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P1s2_P3s2.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 3; s_b = 3
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P1s3_P3s3.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 4; s_b = 4
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P1s4_P3s4.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            # Adjacent strips
            s_a = 1; s_b = 2
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P1s1_P3s2.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 2; s_b = 1
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P1s2_P3s1.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 2; s_b = 3
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P1s2_P3s3.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 3; s_b = 2
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P1s3_P3s2.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 3; s_b = 4
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P1s3_P3s4.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 4; s_b = 3
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P1s4_P3s3.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            # Two separated strips
            s_a = 1; s_b = 3
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P1s1_P3s3.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 3; s_b = 1
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P1s3_P3s1.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 2; s_b = 4
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P1s2_P3s4.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 4; s_b = 2
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P1s4_P3s2.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            # Three separated strips
            s_a = 1; s_b = 4
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P1s1_P3s4.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            
            # P2-P4 ---------------------------------------------------------------
            P_a = 2; P_b = 4
            # Same strips
            s_a = 1; s_b = 1
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P2s1_P4s1.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 2; s_b = 2
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P2s2_P4s2.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 3; s_b = 3
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P2s3_P4s3.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 4; s_b = 4
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P2s4_P4s4.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            # Adjacent strips
            s_a = 1; s_b = 2
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P2s1_P4s2.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 2; s_b = 1
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P2s2_P4s1.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 2; s_b = 3
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P2s2_P4s3.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 3; s_b = 2
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P2s3_P4s2.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 3; s_b = 4
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P2s3_P4s4.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 4; s_b = 3
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P2s4_P4s3.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            # Two separated strips
            s_a = 1; s_b = 3
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P2s1_P4s3.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 3; s_b = 1
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P2s3_P4s1.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 2; s_b = 4
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P2s2_P4s4.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 4; s_b = 2
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P2s4_P4s2.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            # Three separated strips
            s_a = 1; s_b = 4
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P2s1_P4s4.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            
            # One layer spacing ---------------------------------------------------
            # P3-P4 ---------------------------------------------------------------
            P_a = 3; P_b = 4
            # Same strips
            s_a = 1; s_b = 1
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P3s1_P4s1.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 2; s_b = 2
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P3s2_P4s2.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 3; s_b = 3
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P3s3_P4s3.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 4; s_b = 4
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P3s4_P4s4.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            # Adjacent strips
            s_a = 1; s_b = 2
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P3s1_P4s2.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 2; s_b = 1
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P3s2_P4s1.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 2; s_b = 3
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P3s2_P4s3.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 3; s_b = 2
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P3s3_P4s2.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 3; s_b = 4
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P3s3_P4s4.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 4; s_b = 3
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P3s4_P4s3.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            # Two separated strips
            s_a = 1; s_b = 3
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P3s1_P4s3.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 3; s_b = 1
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P3s3_P4s1.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 2; s_b = 4
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P3s2_P4s4.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 4; s_b = 2
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P3s4_P4s2.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            # Three separated strips
            s_a = 1; s_b = 4
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P3s1_P4s4.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            
            # P1-P2 ---------------------------------------------------------------
            P_a = 1; P_b = 2
            # Same strips
            s_a = 1; s_b = 1
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P1s1_P2s1.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 2; s_b = 2
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P1s2_P2s2.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 3; s_b = 3
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P1s3_P2s3.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 4; s_b = 4
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P1s4_P2s4.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            # Adjacent strips
            s_a = 1; s_b = 2
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P1s1_P2s2.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 2; s_b = 1
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P1s2_P2s1.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 2; s_b = 3
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P1s2_P2s3.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 3; s_b = 2
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P1s3_P2s2.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 3; s_b = 4
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P1s3_P2s4.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 4; s_b = 3
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P1s4_P2s3.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            # Two separated strips
            s_a = 1; s_b = 3
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P1s1_P2s3.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 3; s_b = 1
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P1s3_P2s1.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 2; s_b = 4
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P1s2_P2s4.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 4; s_b = 2
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P1s4_P2s2.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            # Three separated strips
            s_a = 1; s_b = 4
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P1s1_P2s4.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            
            # P2-P3 ---------------------------------------------------------------
            P_a = 2; P_b = 3
            # Same strips
            s_a = 1; s_b = 1
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P2s1_P3s1.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 2; s_b = 2
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P2s2_P3s2.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 3; s_b = 3
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P2s3_P3s3.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 4; s_b = 4
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P2s4_P3s4.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            # Adjacent strips
            s_a = 1; s_b = 2
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P2s1_P3s2.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 2; s_b = 1
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P2s2_P3s1.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 2; s_b = 3
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P2s2_P3s3.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 3; s_b = 2
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P2s3_P3s2.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 3; s_b = 4
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P2s3_P3s4.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 4; s_b = 3
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P2s4_P3s3.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            # Two separated strips
            s_a = 1; s_b = 3
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P2s1_P3s3.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 3; s_b = 1
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P2s3_P3s1.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 2; s_b = 4
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P2s2_P3s4.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 4; s_b = 2
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P2s4_P3s2.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            # Three separated strips
            s_a = 1; s_b = 4
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P2s1_P3s4.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
                
            i += 1
        
        vectors = [
            P1s1_P3s1, P1s1_P3s2, P1s2_P3s1, P1s2_P3s2, P1s2_P3s3,
            P1s3_P3s2, P1s3_P3s3, P1s3_P3s4, P1s4_P3s3, P1s4_P3s4,
            P1s1_P3s3, P1s3_P3s1, P1s2_P3s4, P1s4_P3s2, P1s1_P3s4,\
                
            P1s1_P4s1, P1s1_P4s2, P1s2_P4s1, P1s2_P4s2, P1s2_P4s3,
            P1s3_P4s2, P1s3_P4s3, P1s3_P4s4, P1s4_P4s3, P1s4_P4s4,
            P1s1_P4s3, P1s3_P4s1, P1s2_P4s4, P1s4_P4s2, P1s1_P4s4,\
                
            P2s1_P4s1, P2s1_P4s2, P2s2_P4s1, P2s2_P4s2, P2s2_P4s3,
            P2s3_P4s2, P2s3_P4s3, P2s3_P4s4, P2s4_P4s3, P2s4_P4s4,
            P2s1_P4s3, P2s3_P4s1, P2s2_P4s4, P2s4_P4s2, P2s1_P4s4,\
                
            P3s1_P4s1, P3s1_P4s2, P3s2_P4s1, P3s2_P4s2, P3s2_P4s3,
            P3s3_P4s2, P3s3_P4s3, P3s3_P4s4, P3s4_P4s3, P3s4_P4s4,
            P3s1_P4s3, P3s3_P4s1, P3s2_P4s4, P3s4_P4s2, P3s1_P4s4,\
                
            P1s1_P2s1, P1s1_P2s2, P1s2_P2s1, P1s2_P2s2, P1s2_P2s3,
            P1s3_P2s2, P1s3_P2s3, P1s3_P2s4, P1s4_P2s3, P1s4_P2s4,
            P1s1_P2s3, P1s3_P2s1, P1s2_P2s4, P1s4_P2s2, P1s1_P2s4,\
                
            P2s1_P3s1, P2s1_P3s2, P2s2_P3s1, P2s2_P3s2, P2s2_P3s3,
            P2s3_P3s2, P2s3_P3s3, P2s3_P3s4, P2s4_P3s3, P2s4_P3s4,
            P2s1_P3s3, P2s3_P3s1, P2s2_P3s4, P2s4_P3s2, P2s1_P3s4
        ]

        if create_plots:
            # Convert data to numpy arrays and filter
            pos_x = np.array(pos_x)
            pos_x = pos_x[(-200 < pos_x) & (pos_x < 200) & (pos_x != 0)]
            v_travel_time = np.array(v_travel_time)
            v_travel_time = v_travel_time[v_travel_time < 1.6]
            t_0 = np.array(t_0)
            t_0 = t_0[(-10 < t_0) & (t_0 < 10)]
            t_0 = t_0[t_0 != 0]
            
            # Prepare a figure with 1x3 subplots
            fig, axs = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
            
            # Plot histogram for positions (pos_x)
            axs[0].hist(pos_x, bins='auto', alpha=0.6, color='blue')
            axs[0].set_title('Positions')
            axs[0].set_xlabel('Position (units)')
            axs[0].set_ylabel('Frequency')
            
            # Plot histogram for travel time (v_travel_time)
            axs[1].hist(v_travel_time, bins=300, alpha=0.6, color='green')
            axs[1].set_title('Travel Time of a Particle at c')
            axs[1].set_xlabel('T / ns')
            axs[1].set_ylabel('Frequency')
            
            # Plot histogram for T0s (t_0)
            axs[2].hist(t_0, bins='auto', alpha=0.6, color='red')
            axs[2].set_title('T0s')
            axs[2].set_xlabel('T / ns')
            axs[2].set_ylabel('Frequency')
            
            # Show the combined figure
            plt.suptitle('Combined Histograms of Positions, Travel Time, and T0s')
            
            if save_plots:
                name_of_file = 'positions_travel_time_tzeros'
                final_filename = f'{fig_idx}_{name_of_file}.png'
                fig_idx += 1
                
                save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
                plot_list.append(save_fig_path)
                plt.savefig(save_fig_path, format='png')
            
            if show_plots: plt.show()
            plt.close()
        
            # No fit: loop over each vector and plot histogram
            for i, vector in enumerate(vectors):
                var_name = [name for name, val in globals().items() if val is vector][0]
                if i >= number_of_time_cal_figures: break
                hist_1d(vector, 100, var_name, "T / ns", var_name)


        # Dictionary to store CRT values
        crt_values = {}
        for i, vector in enumerate(vectors):
            var_name = [name for name, val in globals().items() if val is vector][0]
            vdat = np.array(vector)
            if len(vdat) > 1:
                try:
                    vdat = vdat[(vdat > np.quantile(vdat, CRT_gaussian_fit_quantile)) & (vdat < np.quantile(vdat, 1 - CRT_gaussian_fit_quantile))]
                except IndexError:
                    print(f"IndexError encountered for {var_name}, setting CRT to 0")
                    vdat = np.array([0])
            
            CRT = norm.fit(vdat)[1] / np.sqrt(2) if len(vdat) > 0 else 0
            # print(f"CRT for {var_name} is {CRT:.4g}")
            crt_values[f'CRT_{var_name}'] = CRT
        
        crt_df = pd.DataFrame(crt_values, index=working_df.index)
        working_df = pd.concat([working_df, crt_df], axis=1)
        crt_values = working_df.filter(like='CRT_').iloc[0].values
        Q1, Q3 = np.percentile(crt_values, [25, 75])
        crt_values = crt_values[crt_values <= 1]
        filtered_crt_values = crt_values[(crt_values >= Q1 - 1.5 * (Q3 - Q1)) & (crt_values <= Q3 + 1.5 * (Q3 - Q1))]
        
        global_variables['CRT_avg'] = np.mean(filtered_crt_values)*1000
        
        # print(f"CRT values: {crt_values}, Filtered: {filtered_crt_values}, Avg: {working_df['CRT_avg'][0]:.4g}")
        print("---------------------------")
        print(f"CRT Avg: {global_variables['CRT_avg']:.4g} ps")
        print("---------------------------")
        
        # Create row and column indices
        rows = ['P{}s{}'.format(i, j) for i in range(1, 5) for j in range(1, 5)]
        columns = ['P{}s{}'.format(i, j) for i in range(1, 5) for j in range(1, 5)]
        
        df = pd.DataFrame(index=rows, columns=columns)
        for vector in vectors:
            var_name = [name for name, val in globals().items() if val is vector][0]
            if var_name == "vector":
                continue
            current_prefix = str(var_name.split('_')[0])
            current_suffix = str(var_name.split('_')[1])
            # Key part: create the antisymmetric matrix
            df.loc[current_prefix, current_suffix] = summary(vector)
            df.loc[current_suffix, current_prefix] = -df.loc[current_prefix, current_suffix]
        
        print("Antisymmetric matrix:")
        print(df)
    
    else:
        # Create row and column indices
        rows = ['P{}s{}'.format(i, j) for i in range(1, 5) for j in range(1, 5)]
        columns = ['P{}s{}'.format(i, j) for i in range(1, 5) for j in range(1, 5)]
        
        df = pd.DataFrame(index=rows, columns=columns)
        
        # Fill df with antisymmetric c_offset values from slewing_fit_df
        for _, row in slewing_fit_df.iterrows():
            l1 = f'P{str(int(row["plane1"]))}s{str(int(row["strip1"]))}'
            l2 = f'P{str(int(row["plane2"]))}s{str(int(row["strip2"]))}'
            offset = row['c_offset']
            df.loc[l1, l2] = -offset
            df.loc[l2, l1] = offset
        
        print("Antisymmetric matrix:")
        print(df)
    
    # -----------------------------------------------------------------------------
    # Brute force method
    # -----------------------------------------------------------------------------
    brute_force_analysis = False
    if brute_force_analysis:
        # Main itinerary
        itinerary = ["P1s1", "P3s1", "P1s2", "P3s2", "P1s3", "P3s3", "P1s4", "P3s4","P4s4", "P2s4", "P4s3", "P2s3", "P4s2", "P2s2", "P4s1", "P2s1"]
        k = 0
        max_iter = 2000000
        brute_force_list = []
        # Create row and column indices
        rows = ['P{}'.format(i) for i in range(1, 5)]
        columns = ['s{}'.format(i) for i in range(1,5)]
        brute_force_df = pd.DataFrame(0, index=rows, columns=columns)
        jump = False
        while k < max_iter:
            if k % 50000 == 0: print(f"Itinerary {k}")
            brute_force_df[brute_force_df.columns] = 0
            step = itinerary
            a = []
            for i in range(len(itinerary)):
                if i > 0:
                    # Storing new values
                    a.append( df[step[i - 1]][step[i]] )
                relative_time = sum(a)
                if np.isnan(relative_time):
                    jump = True
                    break
                ind1 = str(step[i][0:2])
                ind2 = str(step[i][2:4])
                brute_force_df.loc[ind1,ind2] = brute_force_df.loc[ind1,ind2] + relative_time
            # If the path is succesful, print it, then we can copy it from terminal
            # and save it for the next step.
            if jump == False:
                print(itinerary)
            # Shuffle the path
            random.shuffle(itinerary)
            # Iterate
            k += 1
            if jump:
                jump = False
                continue
            # Substract a value from the entire DataFrame
            brute_force_df = brute_force_df.sub(brute_force_df.iloc[0, 0])
            # Append the matrix to the big list
            brute_force_list.append(brute_force_df.values)
        # Calculate the mean of all the paths
        calibrated_times_bf = np.nanmean(brute_force_list, axis=0)
        calibration_times = calibrated_times_bf
    
    # -----------------------------------------------------------------------------
    # Selected paths method
    # -----------------------------------------------------------------------------
    itineraries = [
    ['P1s1', 'P3s1', 'P1s2', 'P3s2', 'P1s3', 'P3s3', 'P1s4', 'P3s4', 'P4s4', 'P2s4', 'P4s3', 'P2s3', 'P4s2', 'P2s2', 'P4s1', 'P2s1'],
    ['P3s4', 'P1s4', 'P2s4', 'P4s4', 'P2s2', 'P4s3', 'P2s3', 'P1s3', 'P3s3', 'P2s1', 'P4s2', 'P1s2', 'P3s2', 'P1s1', 'P4s1', 'P3s1'],
    ['P3s2', 'P1s2', 'P2s2', 'P4s1', 'P3s1', 'P1s1', 'P3s3', 'P4s2', 'P2s3', 'P1s3', 'P3s4', 'P2s4', 'P4s4', 'P1s4', 'P4s3', 'P2s1'],
    ['P2s4', 'P4s2', 'P1s4', 'P4s4', 'P2s3', 'P4s1', 'P1s3', 'P3s3', 'P1s2', 'P2s2', 'P3s2', 'P2s1', 'P3s1', 'P1s1', 'P4s3', 'P3s4'],
    ['P2s4', 'P4s4', 'P2s2', 'P1s2', 'P3s1', 'P1s1', 'P4s3', 'P2s3', 'P4s1', 'P1s3', 'P3s4', 'P1s4', 'P3s3', 'P2s1', 'P4s2', 'P3s2'],
    ['P3s1', 'P2s1', 'P1s2', 'P4s3', 'P1s3', 'P2s2', 'P3s3', 'P4s1', 'P3s2', 'P1s1', 'P4s4', 'P2s3', 'P3s4', 'P2s4', 'P4s2', 'P1s4'],
    ['P2s3', 'P4s4', 'P2s4', 'P4s2', 'P1s1', 'P3s2', 'P2s1', 'P3s1', 'P4s1', 'P1s3', 'P2s2', 'P1s2', 'P3s3', 'P1s4', 'P4s3', 'P3s4'],
    ['P2s4', 'P3s4', 'P4s2', 'P1s1', 'P2s1', 'P3s1', 'P1s2', 'P4s1', 'P1s3', 'P4s4', 'P2s2', 'P3s3', 'P1s4', 'P2s3', 'P4s3', 'P3s2'],
    ['P3s3', 'P1s2', 'P3s2', 'P2s1', 'P4s3', 'P2s3', 'P4s4', 'P3s4', 'P2s4', 'P1s4', 'P4s2', 'P2s2', 'P1s3', 'P4s1', 'P1s1', 'P3s1'],
    ['P2s4', 'P3s4', 'P1s4', 'P3s3', 'P4s1', 'P2s3', 'P4s2', 'P2s1', 'P3s2', 'P1s3', 'P4s3', 'P2s2', 'P1s2', 'P4s4', 'P1s1', 'P3s1'],
    ['P4s2', 'P3s2', 'P4s3', 'P1s3', 'P2s2', 'P4s1', 'P1s1', 'P2s1', 'P3s3', 'P1s4', 'P2s3', 'P3s4', 'P2s4', 'P4s4', 'P1s2', 'P3s1'],
    ['P1s3', 'P2s3', 'P3s4', 'P1s4', 'P4s4', 'P2s4', 'P4s3', 'P1s2', 'P3s1', 'P4s1', 'P2s1', 'P4s2', 'P3s2', 'P1s1', 'P3s3', 'P2s2'],
    ['P2s4', 'P4s3', 'P1s2', 'P2s1', 'P3s2', 'P2s2', 'P4s2', 'P3s3', 'P1s4', 'P2s3', 'P1s3', 'P3s4', 'P4s4', 'P1s1', 'P3s1', 'P4s1'],
    ['P2s2', 'P1s2', 'P4s1', 'P1s1', 'P3s1', 'P2s1', 'P3s3', 'P4s2', 'P2s4', 'P4s4', 'P1s4', 'P2s3', 'P3s4', 'P4s3', 'P1s3', 'P3s2'],
    ['P3s1', 'P2s1', 'P3s3', 'P2s2', 'P4s2', 'P2s4', 'P4s4', 'P1s2', 'P3s2', 'P1s3', 'P3s4', 'P1s4', 'P2s3', 'P4s1', 'P1s1', 'P4s3'],
    ['P4s2', 'P3s2', 'P2s2', 'P4s4', 'P3s3', 'P1s4', 'P2s3', 'P1s3', 'P3s4', 'P2s4', 'P4s3', 'P2s1', 'P1s2', 'P3s1', 'P4s1', 'P1s1'],
    ['P1s2', 'P3s3', 'P4s4', 'P1s1', 'P4s1', 'P3s1', 'P2s1', 'P3s2', 'P1s3', 'P3s4', 'P2s3', 'P4s3', 'P2s2', 'P4s2', 'P2s4', 'P1s4'],
    ['P3s3', 'P1s2', 'P4s2', 'P3s2', 'P1s3', 'P2s2', 'P4s1', 'P1s1', 'P3s1', 'P2s1', 'P4s3', 'P1s4', 'P2s4', 'P3s4', 'P4s4', 'P2s3'],
    ['P3s4', 'P1s3', 'P4s2', 'P2s4', 'P4s3', 'P3s2', 'P1s2', 'P3s3', 'P2s2', 'P4s1', 'P2s3', 'P1s4', 'P4s4', 'P2s1', 'P1s1', 'P3s1'],
    ['P2s1', 'P1s1', 'P3s1', 'P1s2', 'P3s3', 'P1s4', 'P2s3', 'P4s4', 'P3s4', 'P4s2', 'P2s4', 'P4s3', 'P1s3', 'P2s2', 'P4s1', 'P3s2'],
    ['P3s3', 'P2s2', 'P1s2', 'P4s4', 'P2s1', 'P3s2', 'P1s3', 'P3s4', 'P1s4', 'P2s3', 'P4s1', 'P3s1', 'P1s1', 'P4s3', 'P2s4', 'P4s2'],
    ['P3s2', 'P2s2', 'P4s2', 'P2s4', 'P1s4', 'P3s4', 'P1s3', 'P4s1', 'P1s2', 'P3s1', 'P1s1', 'P3s3', 'P4s4', 'P2s3', 'P4s3', 'P2s1'],
    ['P3s2', 'P1s2', 'P4s2', 'P1s1', 'P4s4', 'P2s3', 'P1s4', 'P3s3', 'P2s1', 'P3s1', 'P4s1', 'P2s2', 'P1s3', 'P3s4', 'P2s4', 'P4s3'],
    ['P3s2', 'P2s2', 'P3s3', 'P1s1', 'P4s2', 'P1s3', 'P4s3', 'P3s4', 'P2s4', 'P1s4', 'P2s3', 'P4s4', 'P1s2', 'P4s1', 'P3s1', 'P2s1'],
    ['P1s3', 'P3s4', 'P2s4', 'P1s4', 'P3s3', 'P1s2', 'P2s1', 'P4s4', 'P2s3', 'P4s1', 'P3s2', 'P4s2', 'P2s2', 'P4s3', 'P1s1', 'P3s1'],
    ['P2s1', 'P3s3', 'P1s4', 'P2s3', 'P3s4', 'P1s3', 'P4s2', 'P1s1', 'P3s1', 'P4s1', 'P2s2', 'P3s2', 'P1s2', 'P4s3', 'P2s4', 'P4s4'],
    ['P3s1', 'P4s1', 'P3s2', 'P1s1', 'P4s2', 'P2s4', 'P1s4', 'P2s3', 'P1s3', 'P3s3', 'P2s2', 'P1s2', 'P4s4', 'P3s4', 'P4s3', 'P2s1'],
    ['P1s3', 'P3s4', 'P2s4', 'P4s2', 'P1s4', 'P4s4', 'P3s3', 'P2s3', 'P4s3', 'P3s2', 'P4s1', 'P2s1', 'P1s1', 'P3s1', 'P1s2', 'P2s2'],
    ['P3s2', 'P2s2', 'P1s3', 'P4s3', 'P1s4', 'P2s3', 'P4s2', 'P1s1', 'P4s1', 'P3s1', 'P2s1', 'P1s2', 'P3s3', 'P4s4', 'P2s4', 'P3s4'],
    ['P2s3', 'P3s3', 'P1s1', 'P3s1', 'P1s2', 'P4s2', 'P2s1', 'P3s2', 'P4s1', 'P2s2', 'P4s4', 'P1s3', 'P3s4', 'P4s3', 'P1s4', 'P2s4'],
    ['P1s1', 'P3s1', 'P1s2', 'P4s1', 'P2s1', 'P3s2', 'P1s3', 'P2s3', 'P1s4', 'P4s4', 'P2s2', 'P4s3', 'P2s4', 'P3s4', 'P4s2', 'P3s3'],
    ['P1s3', 'P3s3', 'P1s4', 'P2s4', 'P3s4', 'P4s2', 'P2s3', 'P4s4', 'P1s2', 'P3s2', 'P2s2', 'P4s3', 'P2s1', 'P4s1', 'P3s1', 'P1s1'],
    ['P2s3', 'P3s4', 'P2s4', 'P4s4', 'P1s1', 'P4s1', 'P2s2', 'P4s2', 'P1s2', 'P3s1', 'P2s1', 'P3s2', 'P1s3', 'P3s3', 'P4s3', 'P1s4'],
    ['P2s4', 'P4s4', 'P1s2', 'P4s2', 'P2s3', 'P3s4', 'P1s4', 'P3s3', 'P1s3', 'P4s1', 'P2s1', 'P4s3', 'P2s2', 'P3s2', 'P1s1', 'P3s1'],
    ['P4s3', 'P2s1', 'P1s2', 'P2s2', 'P3s2', 'P1s1', 'P3s1', 'P4s1', 'P3s3', 'P4s2', 'P2s4', 'P1s4', 'P4s4', 'P2s3', 'P3s4', 'P1s3'],
    ['P2s2', 'P4s4', 'P2s4', 'P4s3', 'P2s3', 'P4s1', 'P2s1', 'P1s1', 'P3s1', 'P1s2', 'P3s2', 'P4s2', 'P1s3', 'P3s3', 'P1s4', 'P3s4'],
    ['P3s1', 'P4s1', 'P2s3', 'P4s3', 'P1s1', 'P2s1', 'P1s2', 'P2s2', 'P4s2', 'P2s4', 'P4s4', 'P3s4', 'P1s4', 'P3s3', 'P1s3', 'P3s2'],
    ['P4s2', 'P3s3', 'P2s1', 'P1s2', 'P4s4', 'P2s2', 'P4s3', 'P1s3', 'P3s4', 'P2s4', 'P1s4', 'P2s3', 'P4s1', 'P3s1', 'P1s1', 'P3s2'],
    ['P1s3', 'P3s4', 'P2s4', 'P4s2', 'P1s1', 'P3s1', 'P1s2', 'P2s2', 'P4s4', 'P2s3', 'P1s4', 'P3s3', 'P4s3', 'P3s2', 'P4s1', 'P2s1'],
    ['P3s2', 'P1s3', 'P4s2', 'P3s3', 'P2s3', 'P3s4', 'P2s4', 'P1s4', 'P4s4', 'P2s2', 'P4s1', 'P2s1', 'P3s1', 'P1s2', 'P4s3', 'P1s1'],
    ['P2s3', 'P4s4', 'P2s4', 'P1s4', 'P3s4', 'P1s3', 'P3s2', 'P2s2', 'P4s2', 'P2s1', 'P4s3', 'P3s3', 'P1s1', 'P3s1', 'P1s2', 'P4s1'],
    ['P4s1', 'P3s1', 'P1s1', 'P2s1', 'P4s4', 'P1s3', 'P2s3', 'P4s3', 'P2s2', 'P3s2', 'P1s2', 'P4s2', 'P3s3', 'P1s4', 'P3s4', 'P2s4'],
    ['P2s4', 'P4s3', 'P2s3', 'P4s1', 'P1s3', 'P2s2', 'P3s2', 'P4s2', 'P1s2', 'P3s1', 'P2s1', 'P3s3', 'P1s1', 'P4s4', 'P1s4', 'P3s4'],
    ['P1s4', 'P2s4', 'P4s3', 'P2s3', 'P3s3', 'P1s1', 'P3s2', 'P4s1', 'P1s3', 'P3s4', 'P4s4', 'P2s2', 'P4s2', 'P2s1', 'P3s1', 'P1s2'],
    ['P2s2', 'P4s1', 'P2s3', 'P1s3', 'P3s2', 'P1s1', 'P3s1', 'P1s2', 'P3s3', 'P2s1', 'P4s3', 'P2s4', 'P3s4', 'P4s2', 'P1s4', 'P4s4'],
    ['P2s2', 'P1s2', 'P2s1', 'P3s2', 'P1s1', 'P4s3', 'P2s4', 'P4s2', 'P2s3', 'P3s4', 'P1s4', 'P3s3', 'P4s4', 'P1s3', 'P4s1', 'P3s1'],
    ['P2s1', 'P3s1', 'P4s1', 'P2s3', 'P3s3', 'P2s2', 'P3s2', 'P1s3', 'P4s4', 'P1s2', 'P4s2', 'P1s1', 'P4s3', 'P3s4', 'P1s4', 'P2s4'],
    ['P1s1', 'P3s3', 'P2s3', 'P1s3', 'P3s4', 'P4s4', 'P1s4', 'P2s4', 'P4s3', 'P2s1', 'P4s1', 'P3s1', 'P1s2', 'P2s2', 'P4s2', 'P3s2'],
    ['P2s2', 'P4s3', 'P2s3', 'P3s3', 'P4s4', 'P1s2', 'P4s2', 'P2s4', 'P1s4', 'P3s4', 'P1s3', 'P4s1', 'P3s1', 'P1s1', 'P3s2', 'P2s1'],
    ['P4s1', 'P1s1', 'P3s1', 'P1s2', 'P2s1', 'P3s2', 'P2s2', 'P4s3', 'P3s3', 'P4s4', 'P2s4', 'P1s4', 'P3s4', 'P1s3', 'P2s3', 'P4s2'],
    ['P4s4', 'P1s3', 'P3s3', 'P2s2', 'P1s2', 'P3s1', 'P2s1', 'P3s2', 'P1s1', 'P4s1', 'P2s3', 'P4s2', 'P1s4', 'P3s4', 'P2s4', 'P4s3'],
    ['P1s3', 'P4s4', 'P3s4', 'P2s4', 'P4s2', 'P2s2', 'P3s3', 'P1s1', 'P3s1', 'P1s2', 'P3s2', 'P4s3', 'P1s4', 'P2s3', 'P4s1', 'P2s1'],
    ['P3s2', 'P4s3', 'P2s1', 'P1s1', 'P3s1', 'P4s1', 'P1s3', 'P2s2', 'P1s2', 'P4s4', 'P2s4', 'P1s4', 'P3s4', 'P2s3', 'P3s3', 'P4s2'],
    ['P2s3', 'P4s2', 'P2s1', 'P4s4', 'P2s2', 'P1s2', 'P3s1', 'P1s1', 'P3s3', 'P4s3', 'P3s2', 'P4s1', 'P1s3', 'P3s4', 'P1s4', 'P2s4'],
    ['P2s2', 'P3s2', 'P4s1', 'P3s1', 'P2s1', 'P1s2', 'P4s4', 'P1s1', 'P4s3', 'P2s3', 'P3s3', 'P1s3', 'P3s4', 'P1s4', 'P4s2', 'P2s4'],
    ['P4s4', 'P2s2', 'P1s3', 'P3s3', 'P1s4', 'P3s4', 'P2s4', 'P4s2', 'P1s2', 'P4s1', 'P3s1', 'P1s1', 'P3s2', 'P2s1', 'P4s3', 'P2s3'],
    ['P2s2', 'P3s2', 'P1s1', 'P3s1', 'P2s1', 'P3s3', 'P4s1', 'P2s3', 'P1s4', 'P4s4', 'P2s4', 'P3s4', 'P1s3', 'P4s3', 'P1s2', 'P4s2'],
    ['P2s3', 'P3s3', 'P2s2', 'P1s3', 'P3s2', 'P1s2', 'P3s1', 'P4s1', 'P1s1', 'P2s1', 'P4s4', 'P1s4', 'P4s3', 'P3s4', 'P2s4', 'P4s2'],
    ['P2s4', 'P1s4', 'P3s3', 'P1s1', 'P3s1', 'P4s1', 'P2s2', 'P3s2', 'P4s3', 'P1s3', 'P3s4', 'P2s3', 'P4s4', 'P2s1', 'P4s2', 'P1s2'],
    ['P3s1', 'P1s1', 'P4s4', 'P2s1', 'P3s3', 'P4s1', 'P1s2', 'P4s2', 'P1s4', 'P2s3', 'P3s4', 'P1s3', 'P2s2', 'P3s2', 'P4s3', 'P2s4'],
    ['P2s2', 'P4s4', 'P2s1', 'P4s3', 'P2s4', 'P1s4', 'P4s2', 'P3s4', 'P2s3', 'P1s3', 'P3s3', 'P1s2', 'P3s2', 'P4s1', 'P3s1', 'P1s1'],
    ['P3s2', 'P1s3', 'P2s3', 'P4s2', 'P2s4', 'P1s4', 'P3s3', 'P1s1', 'P2s1', 'P4s4', 'P3s4', 'P4s3', 'P1s2', 'P3s1', 'P4s1', 'P2s2'],
    ['P1s4', 'P2s3', 'P4s4', 'P3s3', 'P1s1', 'P4s1', 'P3s1', 'P2s1', 'P4s3', 'P1s3', 'P3s4', 'P2s4', 'P4s2', 'P1s2', 'P3s2', 'P2s2'],
    ['P1s1', 'P3s1', 'P2s1', 'P3s3', 'P2s3', 'P4s2', 'P3s4', 'P1s4', 'P2s4', 'P4s4', 'P1s2', 'P2s2', 'P4s1', 'P3s2', 'P1s3', 'P4s3'],
    ['P1s4', 'P2s4', 'P4s2', 'P3s4', 'P2s3', 'P4s1', 'P3s1', 'P1s2', 'P2s1', 'P4s4', 'P3s3', 'P1s1', 'P4s3', 'P1s3', 'P3s2', 'P2s2'],
    ['P1s1', 'P3s1', 'P2s1', 'P3s2', 'P4s1', 'P2s3', 'P1s3', 'P3s3', 'P1s2', 'P4s2', 'P2s2', 'P4s4', 'P3s4', 'P1s4', 'P2s4', 'P4s3'],
    ['P1s3', 'P2s2', 'P3s2', 'P2s1', 'P4s3', 'P1s1', 'P4s1', 'P3s1', 'P1s2', 'P4s2', 'P1s4', 'P3s3', 'P4s4', 'P2s3', 'P3s4', 'P2s4'],
    ['P3s1', 'P1s2', 'P4s4', 'P1s4', 'P4s3', 'P2s2', 'P4s1', 'P2s1', 'P1s1', 'P3s3', 'P2s3', 'P1s3', 'P3s4', 'P2s4', 'P4s2', 'P3s2'],
    ['P4s4', 'P1s1', 'P3s1', 'P2s1', 'P3s2', 'P4s1', 'P1s2', 'P4s2', 'P3s4', 'P2s4', 'P1s4', 'P3s3', 'P2s2', 'P1s3', 'P4s3', 'P2s3'],
    ['P1s1', 'P4s1', 'P3s1', 'P2s1', 'P3s2', 'P4s2', 'P2s4', 'P4s4', 'P1s2', 'P2s2', 'P4s3', 'P2s3', 'P3s4', 'P1s3', 'P3s3', 'P1s4'],
    ['P2s4', 'P3s4', 'P4s3', 'P1s3', 'P2s2', 'P4s1', 'P3s2', 'P1s2', 'P2s1', 'P4s2', 'P1s4', 'P2s3', 'P4s4', 'P3s3', 'P1s1', 'P3s1'],
    ['P2s4', 'P4s3', 'P1s2', 'P3s2', 'P2s2', 'P3s3', 'P4s1', 'P3s1', 'P1s1', 'P2s1', 'P4s2', 'P2s3', 'P3s4', 'P1s3', 'P4s4', 'P1s4'],
    ['P2s2', 'P1s3', 'P4s1', 'P3s1', 'P2s1', 'P1s1', 'P3s2', 'P1s2', 'P3s3', 'P4s3', 'P3s4', 'P1s4', 'P4s4', 'P2s4', 'P4s2', 'P2s3'],
    ['P2s4', 'P4s4', 'P2s2', 'P4s2', 'P3s4', 'P1s3', 'P2s3', 'P1s4', 'P4s3', 'P3s3', 'P1s2', 'P3s2', 'P1s1', 'P3s1', 'P2s1', 'P4s1'],
    ['P3s2', 'P2s1', 'P3s3', 'P1s1', 'P4s4', 'P2s2', 'P4s3', 'P1s2', 'P3s1', 'P4s1', 'P2s3', 'P4s2', 'P1s3', 'P3s4', 'P2s4', 'P1s4'],
    ['P3s1', 'P4s1', 'P3s3', 'P2s2', 'P3s2', 'P1s1', 'P2s1', 'P1s2', 'P4s4', 'P3s4', 'P2s4', 'P4s3', 'P1s3', 'P2s3', 'P4s2', 'P1s4'],
    ['P2s3', 'P4s2', 'P2s4', 'P1s4', 'P4s4', 'P2s2', 'P4s3', 'P1s1', 'P3s2', 'P4s1', 'P3s1', 'P1s2', 'P2s1', 'P3s3', 'P1s3', 'P3s4'],
    ['P2s4', 'P4s2', 'P1s1', 'P3s1', 'P1s2', 'P3s2', 'P1s3', 'P3s4', 'P1s4', 'P4s4', 'P2s3', 'P3s3', 'P4s1', 'P2s2', 'P4s3', 'P2s1'],
    ['P2s1', 'P4s4', 'P1s3', 'P4s1', 'P1s2', 'P3s1', 'P1s1', 'P3s2', 'P2s2', 'P4s2', 'P3s3', 'P4s3', 'P1s4', 'P2s4', 'P3s4', 'P2s3'],
    ['P4s1', 'P3s3', 'P4s3', 'P2s4', 'P4s2', 'P1s3', 'P3s4', 'P2s3', 'P1s4', 'P4s4', 'P2s2', 'P1s2', 'P3s2', 'P1s1', 'P3s1', 'P2s1'],
    ['P4s3', 'P2s1', 'P1s1', 'P3s2', 'P2s2', 'P3s3', 'P1s4', 'P2s3', 'P3s4', 'P4s2', 'P2s4', 'P4s4', 'P1s3', 'P4s1', 'P3s1', 'P1s2'],
    ['P4s4', 'P1s2', 'P3s1', 'P2s1', 'P3s2', 'P2s2', 'P1s3', 'P3s4', 'P1s4', 'P4s3', 'P2s4', 'P4s2', 'P2s3', 'P4s1', 'P1s1', 'P3s3'],
    ['P1s1', 'P3s2', 'P1s2', 'P4s2', 'P2s2', 'P1s3', 'P4s3', 'P2s4', 'P1s4', 'P3s4', 'P4s4', 'P2s3', 'P3s3', 'P2s1', 'P3s1', 'P4s1'],
    ['P2s1', 'P3s1', 'P1s1', 'P3s2', 'P4s2', 'P2s4', 'P3s4', 'P4s4', 'P1s2', 'P2s2', 'P1s3', 'P4s1', 'P3s3', 'P2s3', 'P1s4', 'P4s3'],
    ['P2s4', 'P4s4', 'P1s2', 'P4s2', 'P3s3', 'P2s1', 'P3s2', 'P1s3', 'P2s3', 'P1s4', 'P3s4', 'P4s3', 'P2s2', 'P4s1', 'P3s1', 'P1s1'],
    ['P2s2', 'P3s3', 'P2s3', 'P1s4', 'P3s4', 'P4s2', 'P1s2', 'P2s1', 'P3s1', 'P4s1', 'P1s3', 'P3s2', 'P4s3', 'P2s4', 'P4s4', 'P1s1'],
    ['P4s3', 'P2s2', 'P3s3', 'P4s2', 'P2s4', 'P3s4', 'P1s4', 'P2s3', 'P1s3', 'P4s1', 'P2s1', 'P3s1', 'P1s1', 'P3s2', 'P1s2', 'P4s4'],
    ['P3s1', 'P4s1', 'P3s2', 'P1s1', 'P4s2', 'P2s4', 'P1s4', 'P2s3', 'P3s4', 'P4s4', 'P1s2', 'P2s2', 'P1s3', 'P4s3', 'P2s1', 'P3s3'],
    ['P2s4', 'P3s4', 'P1s4', 'P2s3', 'P4s3', 'P1s2', 'P3s2', 'P1s1', 'P2s1', 'P3s1', 'P4s1', 'P1s3', 'P2s2', 'P4s2', 'P3s3', 'P4s4'],
    ['P2s1', 'P4s2', 'P1s3', 'P3s3', 'P4s3', 'P1s2', 'P4s1', 'P2s3', 'P1s4', 'P3s4', 'P2s4', 'P4s4', 'P2s2', 'P3s2', 'P1s1', 'P3s1'],
    ['P3s3', 'P1s1', 'P3s1', 'P2s1', 'P4s4', 'P1s2', 'P4s3', 'P3s2', 'P4s2', 'P2s4', 'P1s4', 'P3s4', 'P1s3', 'P2s3', 'P4s1', 'P2s2'],
    ['P2s3', 'P3s4', 'P4s3', 'P2s1', 'P1s1', 'P3s1', 'P1s2', 'P3s3', 'P4s1', 'P2s2', 'P4s2', 'P3s2', 'P1s3', 'P4s4', 'P2s4', 'P1s4'],
    ['P1s4', 'P2s4', 'P4s2', 'P1s3', 'P3s4', 'P4s3', 'P3s2', 'P2s2', 'P1s2', 'P3s3', 'P2s3', 'P4s1', 'P3s1', 'P1s1', 'P2s1', 'P4s4'],
    ['P1s1', 'P3s3', 'P1s2', 'P2s1', 'P3s1', 'P4s1', 'P3s2', 'P4s3', 'P2s2', 'P1s3', 'P4s4', 'P3s4', 'P4s2', 'P2s4', 'P1s4', 'P2s3'],
    ['P2s2', 'P1s2', 'P3s1', 'P2s1', 'P1s1', 'P4s3', 'P3s2', 'P4s1', 'P2s3', 'P4s2', 'P3s3', 'P1s4', 'P2s4', 'P3s4', 'P1s3', 'P4s4'],
    ['P1s1', 'P3s2', 'P1s3', 'P4s4', 'P1s4', 'P4s3', 'P2s2', 'P4s2', 'P2s4', 'P3s4', 'P2s3', 'P3s3', 'P1s2', 'P2s1', 'P4s1', 'P3s1'],
    ['P1s3', 'P4s4', 'P2s2', 'P1s2', 'P3s2', 'P4s3', 'P2s4', 'P3s4', 'P1s4', 'P2s3', 'P3s3', 'P2s1', 'P4s2', 'P1s1', 'P4s1', 'P3s1'],
    ['P1s4', 'P2s4', 'P4s3', 'P3s4', 'P4s4', 'P2s2', 'P4s1', 'P1s3', 'P3s2', 'P1s1', 'P3s1', 'P1s2', 'P4s2', 'P2s1', 'P3s3', 'P2s3'],
    ['P2s3', 'P1s3', 'P4s2', 'P3s2', 'P4s1', 'P1s2', 'P4s3', 'P2s4', 'P1s4', 'P3s4', 'P4s4', 'P2s2', 'P3s3', 'P1s1', 'P3s1', 'P2s1'],
    ['P4s1', 'P3s1', 'P1s2', 'P4s4', 'P1s4', 'P2s4', 'P4s3', 'P1s1', 'P2s1', 'P3s3', 'P2s2', 'P4s2', 'P3s2', 'P1s3', 'P2s3', 'P3s4'],
    ['P1s4', 'P2s4', 'P3s4', 'P4s3', 'P2s2', 'P3s2', 'P2s1', 'P4s4', 'P1s2', 'P3s1', 'P1s1', 'P4s2', 'P1s3', 'P2s3', 'P3s3', 'P4s1'],
    ['P3s2', 'P1s1', 'P4s3', 'P1s3', 'P2s2', 'P1s2', 'P4s1', 'P3s1', 'P2s1', 'P4s4', 'P3s3', 'P4s2', 'P3s4', 'P2s3', 'P1s4', 'P2s4'],
    ['P4s3', 'P1s2', 'P4s1', 'P2s3', 'P3s4', 'P1s4', 'P4s4', 'P2s4', 'P4s2', 'P2s2', 'P3s3', 'P1s3', 'P3s2', 'P1s1', 'P3s1', 'P2s1'],
    ['P2s2', 'P4s1', 'P1s2', 'P3s3', 'P2s3', 'P1s3', 'P3s2', 'P4s3', 'P1s4', 'P4s2', 'P3s4', 'P2s4', 'P4s4', 'P2s1', 'P3s1', 'P1s1'],
    ['P2s2', 'P4s1', 'P3s1', 'P1s1', 'P4s3', 'P2s4', 'P3s4', 'P1s4', 'P4s4', 'P1s3', 'P4s2', 'P2s3', 'P3s3', 'P1s2', 'P2s1', 'P3s2'],
    ['P4s3', 'P1s4', 'P2s3', 'P3s4', 'P1s3', 'P2s2', 'P3s3', 'P4s1', 'P1s1', 'P3s2', 'P2s1', 'P3s1', 'P1s2', 'P4s2', 'P2s4', 'P4s4'],
    ['P3s1', 'P2s1', 'P1s1', 'P4s3', 'P2s2', 'P1s3', 'P4s1', 'P3s3', 'P4s2', 'P3s2', 'P1s2', 'P4s4', 'P2s4', 'P1s4', 'P2s3', 'P3s4'],
    ['P2s4', 'P1s4', 'P4s4', 'P1s3', 'P2s3', 'P3s4', 'P4s3', 'P1s1', 'P3s1', 'P4s1', 'P3s2', 'P1s2', 'P2s2', 'P3s3', 'P2s1', 'P4s2'],
    ['P4s2', 'P3s2', 'P2s1', 'P3s1', 'P1s2', 'P4s1', 'P1s3', 'P2s2', 'P4s4', 'P3s4', 'P2s4', 'P4s3', 'P1s4', 'P2s3', 'P3s3', 'P1s1'],
    ['P3s2', 'P2s2', 'P4s4', 'P3s3', 'P2s1', 'P4s1', 'P2s3', 'P4s2', 'P1s2', 'P3s1', 'P1s1', 'P4s3', 'P1s4', 'P2s4', 'P3s4', 'P1s3'],
    ['P2s2', 'P1s3', 'P4s1', 'P3s1', 'P2s1', 'P3s3', 'P4s2', 'P1s2', 'P3s2', 'P1s1', 'P4s4', 'P2s4', 'P1s4', 'P3s4', 'P2s3', 'P4s3'],
    ]
    
    def has_duplicate_sublists(lst):
        seen = set()
        for sub_list in lst:
            sub_list_tuple = tuple(sub_list)
            if sub_list_tuple in seen:
                return True
            seen.add(sub_list_tuple)
        return False
    
    if has_duplicate_sublists(itineraries):
        print("Duplicated itineraries.")
    
    selected_path_list = []
    
    # Create row and column indices
    rows = ['P{}'.format(i) for i in range(1, 5)]
    columns = ['s{}'.format(i) for i in range(1,5)]
    
    # Create DataFrame
    selected_path_df = pd.DataFrame(0, index=rows, columns=columns)
    
    for itinerary in itineraries:
        selected_path_df[selected_path_df.columns] = 0
        step = itinerary
        a = []
        for i in range(len(step)):
            if i > 0:
                a.append( df[step[i - 1]][step[i]] )
            
            relative_time = sum(a)
            ind1 = str(step[i][0:2])
            ind2 = str(step[i][2:4])
            
            selected_path_df[ind2] = selected_path_df[ind2].astype(float)
            # selected_path_df.loc[ind1,ind2] = selected_path_df.loc[ind1,ind2] - relative_time
            selected_path_df.loc[ind1,ind2] = selected_path_df.loc[ind1,ind2] + relative_time # ORIGINALLY THERE WAS A MINUS BUT STOPPED WORKING SO I PUT THE + TO TRY
        
        # Substract a value from the entire DataFrame
        selected_path_df = selected_path_df.sub(selected_path_df.iloc[0, 0])
        # Append
        selected_path_list.append(selected_path_df.values)
        
    # Calculate the mean of all the paths
    calibrated_times_sp = np.nanmean(selected_path_list, axis=0)
    calibration_times = calibrated_times_sp
    
    
    # Time calibration matrix calculated --------------------------------------
    print("------------------------")
    print("Calibration in times is:\n", calibration_times)
    
    diff = np.abs(calibration_times - time_sum_reference) > time_sum_distance
    nan_mask = np.isnan(calibration_times)
    values_replaced_t_sum = np.any(diff | nan_mask)
    calibration_times[diff | nan_mask] = time_sum_reference[diff | nan_mask]
    if values_replaced_t_sum:
        print("Some values were replaced in the calibration in times.")
    
    # Applying time calibration
    for i, key in enumerate(['T1', 'T2', 'T3', 'T4']):
        for j in range(4):
            mask = working_df[f'{key}_T_sum_{j+1}'] != 0
            working_df.loc[mask, f'{key}_T_sum_{j+1}'] += calibration_times[i][j]
    
    
    if create_plots:
        # Prepare a figure with 1x4 subplots (only for times, no positions)
        fig, axs = plt.subplots(1, 4, figsize=(20, 5), constrained_layout=True)
    
        # Plot histograms for T_sum values from working_df
        times = [working_df[f'P1_T_sum_{i+1}'] for i in range(4)]
        titles_times = ["T_sum P1", "T_sum P2", "T_sum P3", "T_sum P4"]
    
        for i, (time, title) in enumerate(zip(times, titles_times)):
            time_non_zero = time[time != 0]  # Filter out zeros
            ax = axs[i]  # Access the subplot
            ax.hist(time_non_zero, bins=100, alpha=0.75)
            ax.set_title(title)
            ax.set_xlabel("Time (ns)")
            ax.set_ylabel("Frequency")
    
        # Show the figure for the T_sum histograms
        plt.suptitle('Histograms of T_sum Values for Each Plane', fontsize=16)
    
        if save_plots:
            name_of_file = 'Tsum_times_calibrated'
            final_filename = f'{fig_idx}_{name_of_file}.png'
            fig_idx += 1

            save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
            plot_list.append(save_fig_path)
            plt.savefig(save_fig_path, format='png')
    
        if show_plots: 
            plt.show()
        plt.close()
    
else:
    calibration_times = time_sum_reference
    working_df['CRT_avg'] = 1000 # An extreme time to not crush the program
    print("Calibration in times was set to the reference! (calibration was not performed)\n", calibration_times)



print("----------------------------------------------------------------------")
print("----------------------- Time window filtering ------------------------")
print("----------------------------------------------------------------------")

time_window_fitting = True
if time_window_fitting:
    
    T_sum_columns = working_df.filter(regex='_T_sum_')

    t_sum_data = T_sum_columns.values  # shape: (n_events, n_detectors)
    widths = np.linspace(1, 10, 30)  # Scan range of window widths in ns

    counts_per_width = []
    counts_per_width_dev = []

    for w in widths:
        count_in_window = []
        for row in t_sum_data:
            row_no_zeros = row[row != 0]
            if len(row_no_zeros) == 0:
                count_in_window.append(0)
                continue

            stat = np.mean(row_no_zeros)  # or np.median(row_no_zeros)
            lower = stat - w / 2
            upper = stat + w / 2
            n_in_window = np.sum((row_no_zeros >= lower) & (row_no_zeros <= upper))
            count_in_window.append(n_in_window)

        counts_per_width.append(np.mean(count_in_window))
        counts_per_width_dev.append(np.std(count_in_window))

    counts_per_width = np.array(counts_per_width)
    counts_per_width_dev = np.array(counts_per_width_dev)
    counts_per_width_norm = counts_per_width / np.max(counts_per_width)

    # Define model function: signal (logistic) + linear background
    def signal_plus_background(w, S, w0, tau, B):
        return S / (1 + np.exp(-(w - w0) / tau)) + B * w

    # Initial guess: [signal_height, center, width, background_slope]
    p0 = [1.0, 1.0, 0.5, 0.02]

    # Fit
    popt, pcov = curve_fit(signal_plus_background, widths, counts_per_width_norm, p0=p0)

    # Extract parameters
    S_fit, w0_fit, tau_fit, B_fit = popt
    print(f"Fit parameters:\n  Signal amplitude S = {S_fit:.4f}\n  Sigmoid center w0 = {w0_fit:.4f} ns\n  Sigmoid width τ = {tau_fit:.4f} ns\n  Background slope B = {B_fit:.6f} per ns")

    global_variables['sigmoid_width'] = tau_fit
    global_variables['background_slope'] = B_fit

    if create_plots:
    # if create_essential_plots or create_plots:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(widths, counts_per_width_norm, label='Normalized average count in window')
        # ax.axvline(x=time_coincidence_window, color='red', linestyle='--', label='Time coincidence window')
        ax.set_xlabel("Window width (ns)")
        ax.set_ylabel("Normalized average # of T_sum values in window")
        ax.set_title("Fraction of hits within stat-centered window vs width")
        ax.grid(True)
        w_fit = np.linspace(min(widths), max(widths), 300)
        f_fit = signal_plus_background(w_fit, *popt)
        ax.plot(w_fit, f_fit, 'k--', label='Signal + background fit')
        ax.axhline(S_fit, color='green', linestyle=':', alpha=0.6, label=f'Signal plateau ≈ {S_fit:.2f}')
        s_vals = S_fit / (1 + np.exp(-(w_fit - w0_fit) / tau_fit))
        b_vals = B_fit * w_fit
        f_vals = s_vals + b_vals
        P_signal = s_vals / f_vals
        P_background = b_vals / f_vals
        fig = plt.figure(figsize=(10, 8))
        gs = GridSpec(2, 1, height_ratios=[1, 2], hspace=0.05)
        ax_fill = fig.add_subplot(gs[0])  # Top: signal vs. background fill
        ax_main = fig.add_subplot(gs[1], sharex=ax_fill)  # Bottom: your original plot
        ax_fill.fill_between(w_fit, 0, P_signal, color='green', alpha=0.4, label='Signal')
        ax_fill.fill_between(w_fit, P_signal, 1, color='red', alpha=0.4, label='Background')
        ax_fill.set_ylabel("Fraction")
        ax_fill.set_ylim(np.min(P_signal), 1)
        # ax_fill.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax_fill.legend(loc="upper right")
        ax_fill.set_title("Estimated Signal and Background Fractions per Window Width")
        plt.setp(ax_fill.get_xticklabels(), visible=False)
        ax_main.scatter(widths, counts_per_width_norm, label='Normalized average count in window')
        # ax_main.axvline(x=time_coincidence_window, color='red', linestyle='--', label='Time coincidence window')
        ax_main.plot(w_fit, f_fit, 'k--', label='Signal + background fit')
        ax_main.axhline(S_fit, color='green', linestyle=':', alpha=0.6, label=f'Signal plateau ≈ {S_fit:.2f}')
        ax_main.set_xlabel("Window width (ns)")
        ax_main.set_ylabel("Normalized average # of T_sum values in window")
        ax_main.grid(True)
        fit_summary = (f"Fit: S = {S_fit:.3f}, w₀ = {w0_fit:.3f} ns, " f"τ = {tau_fit:.3f} ns, B = {B_fit:.4f}/ns")
        ax_main.plot([], [], ' ', label=fit_summary)  # invisible handle to add text
        ax_main.legend()
        
        if save_plots:
            name_of_file = 'stat_window_accumulation'
            final_filename = f'{fig_idx}_{name_of_file}.png'
            fig_idx += 1
            save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
            plot_list.append(save_fig_path)
            plt.savefig(save_fig_path, format='png')
        if show_plots:
            plt.show()
        plt.close()



print("----------------------------------------------------------------------")
print("--------------- Cross-talk filtering, will be set to 0 ---------------")
print("----------------------------------------------------------------------")

crosstalk_removal_and_recalibration = True

if crosstalk_removal_and_recalibration:
    
    crstlk_th = crosstalk_threshold_ns

    crosstalk_pedestal = {
        "crstlk_pedestal_P1s1": 0, "crstlk_pedestal_P1s2": 0, "crstlk_pedestal_P1s3": 0, "crstlk_pedestal_P1s4": 0,
        "crstlk_pedestal_P2s1": 0, "crstlk_pedestal_P2s2": 0, "crstlk_pedestal_P2s3": 0, "crstlk_pedestal_P2s4": 0,
        "crstlk_pedestal_P3s1": 0, "crstlk_pedestal_P3s2": 0, "crstlk_pedestal_P3s3": 0, "crstlk_pedestal_P3s4": 0,
        "crstlk_pedestal_P4s1": 0, "crstlk_pedestal_P4s2": 0, "crstlk_pedestal_P4s3": 0, "crstlk_pedestal_P4s4": 0
    }

    crosstalk_limits = {
        "crstlk_limit_P1s1": 0, "crstlk_limit_P1s2": 0, "crstlk_limit_P1s3": 0, "crstlk_limit_P1s4": 0,
        "crstlk_limit_P2s1": 0, "crstlk_limit_P2s2": 0, "crstlk_limit_P2s3": 0, "crstlk_limit_P2s4": 0,
        "crstlk_limit_P3s1": 0, "crstlk_limit_P3s2": 0, "crstlk_limit_P3s3": 0, "crstlk_limit_P3s4": 0,
        "crstlk_limit_P4s1": 0, "crstlk_limit_P4s2": 0, "crstlk_limit_P4s3": 0, "crstlk_limit_P4s4": 0
    }
    
    crosstalk_mean = {
        "crstlk_mu_P1s1": 0, "crstlk_mu_P1s2": 0, "crstlk_mu_P1s3": 0, "crstlk_mu_P1s4": 0,
        "crstlk_mu_P2s1": 0, "crstlk_mu_P2s2": 0, "crstlk_mu_P2s3": 0, "crstlk_mu_P2s4": 0,
        "crstlk_mu_P3s1": 0, "crstlk_mu_P3s2": 0, "crstlk_mu_P3s3": 0, "crstlk_mu_P3s4": 0,
        "crstlk_mu_P4s1": 0, "crstlk_mu_P4s2": 0, "crstlk_mu_P4s3": 0, "crstlk_mu_P4s4": 0
    }
    
    crosstalk_std = {
        "crstlk_sigma_P1s1": 0, "crstlk_sigma_P1s2": 0, "crstlk_sigma_P1s3": 0, "crstlk_sigma_P1s4": 0,
        "crstlk_sigma_P2s1": 0, "crstlk_sigma_P2s2": 0, "crstlk_sigma_P2s3": 0, "crstlk_sigma_P2s4": 0,
        "crstlk_sigma_P3s1": 0, "crstlk_sigma_P3s2": 0, "crstlk_sigma_P3s3": 0, "crstlk_sigma_P3s4": 0,
        "crstlk_sigma_P4s1": 0, "crstlk_sigma_P4s2": 0, "crstlk_sigma_P4s3": 0, "crstlk_sigma_P4s4": 0
    }
    
    crosstalk_ampl = {
        "crstlk_ampl_P1s1": 0, "crstlk_ampl_P1s2": 0, "crstlk_ampl_P1s3": 0, "crstlk_ampl_P1s4": 0,
        "crstlk_ampl_P2s1": 0, "crstlk_ampl_P2s2": 0, "crstlk_ampl_P2s3": 0, "crstlk_ampl_P2s4": 0,
        "crstlk_ampl_P3s1": 0, "crstlk_ampl_P3s2": 0, "crstlk_ampl_P3s3": 0, "crstlk_ampl_P3s4": 0,
        "crstlk_ampl_P4s1": 0, "crstlk_ampl_P4s2": 0, "crstlk_ampl_P4s3": 0, "crstlk_ampl_P4s4": 0
    }
    
    crosstalk_linear = {
        "crstlk_mx_b_P1s1": [0, 0], "crstlk_mx_b_P1s2": [0, 0], "crstlk_mx_b_P1s3": [0, 0], "crstlk_mx_b_P1s4": [0, 0],
        "crstlk_mx_b_P2s1": [0, 0], "crstlk_mx_b_P2s2": [0, 0], "crstlk_mx_b_P2s3": [0, 0], "crstlk_mx_b_P2s4": [0, 0],
        "crstlk_mx_b_P3s1": [0, 0], "crstlk_mx_b_P3s2": [0, 0], "crstlk_mx_b_P3s3": [0, 0], "crstlk_mx_b_P3s4": [0, 0],
        "crstlk_mx_b_P4s1": [0, 0], "crstlk_mx_b_P4s2": [0, 0], "crstlk_mx_b_P4s3": [0, 0], "crstlk_mx_b_P4s4": [0, 0]
    }

    crosstalk_fitting = True

    # Gaussian + linear function
    def gaussian_linear(x, a, mu, sigma, m, b):
        return a * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) + m * x + b
    
    for i, key in enumerate(['1', '2', '3', '4']):
        for j in range(4):
            col = f'Q{key}_Q_sum_{j+1}'
            y = working_df[col]
            
            Q_clip_min = -2
            Q_clip_max = 3
            
            num_bins = 80
            data = y[(y != 0) & (y > Q_clip_min) & (y < Q_clip_max)]
            
            hist_vals, bin_edges = np.histogram(data, bins=num_bins)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            try:
                popt, _ = curve_fit(
                    gaussian_linear, 
                    bin_centers, 
                    hist_vals, 
                    p0=[max(hist_vals), 0, 1, 0, min(hist_vals)], 
                    bounds=([0, -1, 0, -np.inf, -np.inf], [2*max(hist_vals), 2, 3, np.inf, np.inf])
                )
                
                a, mu, sigma, m, b = popt
                
                crosstalk_ampl[f'crstlk_ampl_P{key}s{j+1}'] = a
                crosstalk_mean[f'crstlk_mu_P{key}s{j+1}'] = mu
                crosstalk_std[f'crstlk_sigma_P{key}s{j+1}'] = sigma
                crosstalk_linear[f'crstlk_mx_b_P{key}s{j+1}'] = [m, b]
                
                crosstalk_pedestal[f'crstlk_pedestal_P{key}s{j+1}'] = mu - 2 * sigma
                crosstalk_limits[f'crstlk_limit_P{key}s{j+1}'] = mu + 2 * sigma
                
            except RuntimeError:
                continue
    
    # if create_plots:
    if create_plots or create_essential_plots:
        fig_Q, axes_Q = plt.subplots(4, 4, figsize=(20, 10))  # Adjust the layout as necessary
        axes_Q = axes_Q.flatten()

        for i, key in enumerate(['1', '2', '3', '4']):
            for j in range(4):
                col = f'Q{key}_Q_sum_{j+1}'
                y = working_df[col]
                
                Q_clip_min = -2
                Q_clip_max = 3
                
                num_bins = 80
                data = y[(y != 0) & (y > Q_clip_min) & (y < Q_clip_max)]
                
                hist_vals, bin_edges = np.histogram(data, bins=num_bins)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

                axes_Q[i*4 + j].axvline(crosstalk_pedestal[f'crstlk_pedestal_P{key}s{j+1}'], color='blue', linestyle='--', alpha=0.5)
                axes_Q[i*4 + j].axvline(crosstalk_limits[f'crstlk_limit_P{key}s{j+1}'], color='blue', linestyle='--', alpha=0.5)
                
                a = crosstalk_ampl[f'crstlk_ampl_P{key}s{j+1}']
                mu = crosstalk_mean[f'crstlk_mu_P{key}s{j+1}']
                sigma = crosstalk_std[f'crstlk_sigma_P{key}s{j+1}']
                m, b = crosstalk_linear[f'crstlk_mx_b_P{key}s{j+1}']
                
                popt = a, mu, sigma, m, b
                
                x_fit = np.linspace(Q_clip_min, Q_clip_max, 500)
                y_fit = gaussian_linear(x_fit, *popt)
                axes_Q[i*4 + j].plot(x_fit, y_fit, 'r--', label='Gauss + Linear Fit')
                
                axes_Q[i*4 + j].hist(data, bins=num_bins, alpha=0.5, label=f'{col}')
                axes_Q[i*4 + j].set_title(f'{col}')
                axes_Q[i*4 + j].legend()
                axes_Q[i*4 + j].set_xlim([Q_clip_min, Q_clip_max])
                axes_Q[i*4 + j].set_ylim([0, None])
                axes_Q[i*4 + j].axvline(0, color='green', linestyle='--', alpha=0.5)
                
        # Display a vertical green dashed, alpha = 0.5 line at 0
        for ax in axes_Q:
            ax.axvline(0, color='green', linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.suptitle(f"Cross-talk study for filtering (zoom), mingo0{station}\n{start_time}", fontsize=16)
        if save_plots:
            final_filename = f'{fig_idx}_cross_talk_filtering_zoom.png'
            fig_idx += 1
            save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
            plot_list.append(save_fig_path)
            plt.savefig(save_fig_path, format='png')
        if show_plots: plt.show()
        plt.close(fig_Q)


    print("----------------------------------------------------------------------")
    print("-------------- Filter 5: charge sum crosstalk filtering --------------")
    print("----------------------------------------------------------------------")
    for i, key in enumerate(['1', '2', '3', '4']):
        for j in range(4):
            for col in working_df.columns:
                if f'Q{key}_Q_sum_{j+1}' == col:
                    working_df[col] = np.where( working_df[col] < crosstalk_limits[f'crstlk_limit_P{key}s{j+1}'] , 0, working_df[col])
    
    print("----------------------------------------------------------------------")
    print("------------------- Crosstalk pedestal recalibration -----------------")
    print("----------------------------------------------------------------------")
    # Apply the pedestal recalibration
    for i, key in enumerate(['1', '2', '3', '4']):
        for j in range(4):
            mask = working_df[f'Q{key}_Q_sum_{j+1}'] != 0
            working_df.loc[mask, f'Q{key}_Q_sum_{j+1}'] -= crosstalk_pedestal[f'crstlk_pedestal_P{key}s{j+1}']


    if create_plots or create_essential_plots:
    # if create_plots:
        fig_Q, axes_Q = plt.subplots(4, 4, figsize=(20, 10))  # Adjust the layout as necessary
        axes_Q = axes_Q.flatten()

        for i, key in enumerate(['1', '2', '3', '4']):
            for j in range(4):
                col = f'Q{key}_Q_sum_{j+1}'
                y = working_df[col]
                
                Q_clip_min = -2
                Q_clip_max = 3
                
                num_bins = 80
                data = y[(y != 0) & (y > Q_clip_min) & (y < Q_clip_max)]
                
                axes_Q[i*4 + j].hist(data, bins=num_bins, alpha=0.5, label=f'{col}')
                axes_Q[i*4 + j].set_title(f'{col}')
                axes_Q[i*4 + j].legend()
                axes_Q[i*4 + j].set_xlim([Q_clip_min, Q_clip_max])
                axes_Q[i*4 + j].set_ylim([0, None])
                axes_Q[i*4 + j].axvline(0, color='green', linestyle='--', alpha=0.5)
                
        # Display a vertical green dashed, alpha = 0.5 line at 0
        for ax in axes_Q:
            ax.axvline(0, color='green', linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.suptitle(f"Cross-talk check for filtering (zoom), mingo0{station}\n{start_time}", fontsize=16)
        if save_plots:
            final_filename = f'{fig_idx}_cross_talk_filtering_zoom_check.png'
            fig_idx += 1
            save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
            plot_list.append(save_fig_path)
            plt.savefig(save_fig_path, format='png')
        if show_plots: plt.show()
        plt.close(fig_Q)


print("----------------------------------------------------------------------")
print("---------------- Binary topology of active strips --------------------")
print("----------------------------------------------------------------------")

# Collect new columns in a dict first
active_strip_cols = {}

for plane_id in range(1, 5):
    cols = [f'Q{plane_id}_Q_sum_{i}' for i in range(1, 5)]
    Q_plane = working_df[cols].values  # shape (N, 4)

    # Binary activation: 1 if charge > threshold
    active_strips_binary = (Q_plane > 0).astype(int)

    # Convert to string representations
    binary_strings = [''.join(map(str, row)) for row in active_strips_binary]

    # Store in dict for later batch insertion
    active_strip_cols[f'active_strips_P{plane_id}'] = binary_strings

# Concatenate all new columns at once (column-wise)
working_df = pd.concat([working_df, pd.DataFrame(active_strip_cols, index=working_df.index)], axis=1)

# Print check
print("Active strips per plane calculated.")
print(working_df[['active_strips_P1', 'active_strips_P2', 'active_strips_P3', 'active_strips_P4']].head())

# if create_essential_plots or create_plots:
if create_plots:
    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(10, 12), sharex=True, sharey=True)
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    y_max = 0

    # First pass to determine global y-axis limit
    event_counts_list = []
    for i in [1, 2, 3, 4]:
        counts = working_df[f'active_strips_P{i}'].value_counts()
        counts = counts[counts.index != '0000']
        event_counts_list.append(counts)
        if not counts.empty:
            y_max = max(y_max, counts.max())
    
    # Get global label order from P1 (or any consistent source)
    label_order = working_df['active_strips_P1'].value_counts().drop('0000', errors='ignore').index.tolist()

    # Second pass to plot
    for i, ax in zip([1, 2, 3, 4], axes):
        event_counts_filt = event_counts_list[i - 1]
        event_counts_filt = event_counts_filt.reindex(label_order, fill_value=0)

        event_counts_filt.plot(kind='bar', ax=ax, color=colors[i - 1], alpha=0.7)
        ax.set_title(f'Plane {i}', fontsize=12)
        ax.set_ylabel('Counts')
        ax.set_ylim(0, y_max * 1.05)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        ax.tick_params(axis='x', labelrotation=45)

    axes[-1].set_xlabel('Active Strip Pattern')
    plt.tight_layout()

    if save_plots:
        final_filename = f'{fig_idx}_filtered_active_strips_all_planes.png'
        fig_idx += 1
        save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
        plot_list.append(save_fig_path)
        plt.savefig(save_fig_path, format='png')

    if show_plots:
        plt.show()
    plt.close()


print("----------------------------------------------------------------------")
print("----------------------- Y position calculation -----------------------")
print("----------------------------------------------------------------------")

y_new_method = True
blur_y = True

if y_new_method:
    y_columns = {}

    for plane_id in range(1, 5):
        # Retrieve and convert stored binary string to numeric array
        topo_binary = np.array([
            list(map(int, s)) for s in working_df[f'active_strips_P{plane_id}']
        ])  # shape (N, 4)

        # Select the corresponding Y position vector
        y_vec = y_pos_P1_and_P3 if plane_id in [1, 3] else y_pos_P2_and_P4

        # Compute weighted average
        weighted_y = topo_binary * y_vec
        active_counts = topo_binary.sum(axis=1)
        active_counts_safe = np.where(active_counts == 0, 1, active_counts)

        y_position = weighted_y.sum(axis=1) / active_counts_safe
        y_position[active_counts == 0] = 0  # enforce zero when no strips are active

        if blur_y:
            y_position_blurred = y_position.copy()
            nonzero_mask = y_position != 0
            y_position_blurred[nonzero_mask] = np.random.normal(
                loc=y_position[nonzero_mask],
                scale=anc_sy
            )
            y_columns[f'Y_{plane_id}'] = y_position_blurred
        else:
            y_columns[f'Y_{plane_id}'] = y_position

    # Insert all new Y_ columns at once
    working_df = pd.concat([working_df, pd.DataFrame(y_columns, index=working_df.index)], axis=1)

if create_essential_plots:
    plt.figure(figsize=(12, 8))
    for i, plane_id in enumerate(range(1, 5), 1):
        plt.subplot(2, 2, i)
        column_name = f'Y_{plane_id}'
        data = working_df[column_name]
        
        plt.hist(data[data != 0], bins=50, histtype='stepfilled', alpha=0.7)
        plt.title(f'Y Position Distribution - Plane {plane_id}')
        plt.xlabel('Y Position (a.u.)')
        plt.ylabel('Counts')
        plt.grid(True)

    plt.tight_layout()
    if save_plots:
        name_of_file = 'new_Y'
        final_filename = f'{fig_idx}_{name_of_file}.png'
        fig_idx += 1
        save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
        plot_list.append(save_fig_path)
        plt.savefig(save_fig_path, format='png')
    if show_plots:
        plt.show()
    plt.close()

print("Y position calculated.")


print("----------------------------------------------------------------------")
print("----------------- Some more tests (multi-strip data) -----------------")
print("----------------------------------------------------------------------")

# if create_essential_plots or create_plots:
if create_plots:
    for i_plane in range(1, 5):
        active_col = f'active_strips_P{i_plane}'
        print(f"\n--- Plane {i_plane} ---")

        # Column names
        T_sum_cols = [f'T{i_plane}_T_sum_{j+1}' for j in range(4)]
        T_diff_cols = [f'T{i_plane}_T_diff_{j+1}' for j in range(4)]
        Q_sum_cols = [f'Q{i_plane}_Q_sum_{j+1}' for j in range(4)]
        Q_dif_cols = [f'Q{i_plane}_Q_diff_{j+1}' for j in range(4)]

        # Get all unique multi-strip patterns
        patterns = working_df[active_col].unique()
        multi_patterns = [p for p in patterns if p != '0000' and p.count('1') > 1]

        for pattern in multi_patterns:
            active_strips = [i for i, c in enumerate(pattern) if c == '1']
            if len(active_strips) != 2:
                continue

            mask = working_df[active_col] == pattern
            n_events = mask.sum()
            if n_events == 0:
                continue

            print(f"Pattern {pattern} ({n_events} events):")

            for i, j in combinations(active_strips, 2):
                fig, axs = plt.subplots(1, 4, figsize=(20, 5), sharex=False, sharey=False)
                variable_sets = [
                    ('T_sum', T_sum_cols),
                    ('T_diff', T_diff_cols),
                    ('Q_sum', Q_sum_cols),
                    ('Q_dif', Q_dif_cols)
                ]

                for ax, (var_label, cols) in zip(axs, variable_sets):
                    xi = working_df.loc[mask, cols[i]].values
                    yi = working_df.loc[mask, cols[j]].values

                    ax.scatter(xi, yi, alpha=0.5, s=10)
                    ax.plot([min(xi.min(), yi.min()), max(xi.max(), yi.max())],
                            [min(xi.min(), yi.min()), max(xi.max(), yi.max())],
                            'k--', linewidth=1, label='y = x')
                    ax.set_xlabel(f'{var_label} Strip {i+1}')
                    ax.set_ylabel(f'{var_label} Strip {j+1}')
                    ax.set_title(f'{var_label}: Strip {i+1} vs {j+1}')
                    # Aspect ratio 1:1
                    ax.set_aspect('equal', adjustable='box')
                    ax.grid(True)
                    ax.legend()

                fig.suptitle(f'Plane {i_plane}, Pattern {pattern}', fontsize=14)
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])

                if save_plots:
                    name_of_file = f'rpc_variables_debug_P{i_plane}_{pattern}_s{i+1}s{j+1}.png'
                    final_filename = f'{fig_idx}_{name_of_file}'
                    fig_idx += 1

                    save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
                    plot_list.append(save_fig_path)
                    plt.savefig(save_fig_path, format='png')

                if show_plots:
                    plt.show()
                plt.close()
                
                fig, axs = plt.subplots(1, 4, figsize=(20, 5), sharex=False, sharey=False)

                # Other variables
                for ax, (var_label, cols) in zip(axs, variable_sets):
                    xi = working_df.loc[mask, cols[i]].values
                    yi = working_df.loc[mask, cols[j]].values

                    # ax.scatter(xi + yi, ( xi - yi ) / (xi + yi), alpha=0.5, s=10)
                    mask_nonzero = (xi + yi) != 0
                    ax.scatter(xi[mask_nonzero] + yi[mask_nonzero],
                            (xi[mask_nonzero] - yi[mask_nonzero]) / (xi[mask_nonzero] + yi[mask_nonzero]),
                            alpha=0.5, s=10)

                    # ax.plot([min(xi.min(), yi.min()), max(xi.max(), yi.max())],
                    #         [min(xi.min(), yi.min()), max(xi.max(), yi.max())],
                    #         'k--', linewidth=1, label='y = x')
                    ax.set_xlabel(f'{var_label} Strip {i+1}')
                    ax.set_ylabel(f'{var_label} Strip {j+1}')
                    ax.set_title(f'{var_label}: Strip {i+1} vs {j+1}')
                    # Aspect ratio 1:1
                    # ax.set_aspect('equal', adjustable='box')
                    ax.grid(True)
                    # ax.legend()

                fig.suptitle(f'Plane {i_plane}, Pattern {pattern}', fontsize=14)
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])

                if save_plots:
                    name_of_file = f'rpc_variables_calculations_P{i_plane}_{pattern}_s{i+1}s{j+1}.png'
                    final_filename = f'{fig_idx}_{name_of_file}'
                    fig_idx += 1

                    save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
                    plot_list.append(save_fig_path)
                    plt.savefig(save_fig_path, format='png')

                if show_plots:
                    plt.show()
                plt.close()


print("----------------------------------------------------------------------")
print("----------------------- Slewing correction 2/2 -----------------------")
print("----------------------------------------------------------------------")

print("WIP")


print("----------------------------------------------------------------------")
print("----------------- Setting the variables of each RPC ------------------")
print("----------------------------------------------------------------------")

# Prepare containers for final results
final_columns = {}

for i_plane in range(1, 5):
    # Column names
    T_sum_cols = [f'T{i_plane}_T_sum_{i+1}' for i in range(4)]
    T_dif_cols = [f'T{i_plane}_T_diff_{i+1}' for i in range(4)]
    Q_sum_cols = [f'Q{i_plane}_Q_sum_{i+1}' for i in range(4)]
    Q_dif_cols = [f'Q{i_plane}_Q_diff_{i+1}' for i in range(4)]

    # Extract data
    T_sums = working_df[T_sum_cols].astype(float).fillna(0).values
    T_difs = working_df[T_dif_cols].astype(float).fillna(0).values
    Q_sums = working_df[Q_sum_cols].astype(float).fillna(0).values
    Q_difs = working_df[Q_dif_cols].astype(float).fillna(0).values

    # Decode binary topology
    active_mask = np.array([
        list(map(int, s)) for s in working_df[f'active_strips_P{i_plane}']
    ])  # shape (N, 4)

    # Compute strip activation count
    n_active = active_mask.sum(axis=1)
    n_active_safe = np.where(n_active == 0, 1, n_active)

    # Apply mask and compute means
    T_sum_masked = T_sums * active_mask
    T_dif_masked = T_difs * active_mask
    Q_dif_masked = Q_difs * active_mask

    T_sum_final = T_sum_masked.sum(axis=1) / n_active_safe
    T_diff_final = T_dif_masked.sum(axis=1) / n_active_safe

    # Enforce zero where no active strips
    T_sum_final[n_active == 0] = 0
    T_diff_final[n_active == 0] = 0

    # Store final values in dictionary
    final_columns[f'P{i_plane}_T_sum_final'] = T_sum_final
    final_columns[f'P{i_plane}_T_diff_final'] = T_diff_final
    final_columns[f'P{i_plane}_Q_sum_final'] = (Q_sums * active_mask).sum(axis=1)
    final_columns[f'P{i_plane}_Q_diff_final'] = Q_dif_masked.sum(axis=1)

# Concatenate all new final columns at once
working_df = pd.concat([working_df, pd.DataFrame(final_columns, index=working_df.index)], axis=1)


if create_essential_plots or create_plots:
    fig, axes = plt.subplots(4, 10, figsize=(40, 20))  # 10 combinations per plane
    axes = axes.flatten()

    for i_plane in range(1, 5):
        # Column names
        t_sum_col = f'P{i_plane}_T_sum_final'
        t_diff_col = f'P{i_plane}_T_diff_final'
        q_sum_col = f'P{i_plane}_Q_sum_final'
        q_diff_col = f'P{i_plane}_Q_diff_final'
        y_col = f'Y_{i_plane}'

        # Filter valid rows (non-zero)
        valid_rows = working_df[[t_sum_col, t_diff_col, q_sum_col, q_diff_col, y_col]].replace(0, np.nan).dropna()
        
        # Extract variables and filter low charge
        cond = valid_rows[q_sum_col] < 50
        t_sum  = valid_rows.loc[cond, t_sum_col]
        t_diff = valid_rows.loc[cond, t_diff_col]
        q_sum  = valid_rows.loc[cond, q_sum_col]
        q_diff = valid_rows.loc[cond, q_diff_col]
        y      = valid_rows.loc[cond, y_col]

        base_idx = (i_plane - 1) * 10  # 10 plots per plane

        combinations = [
            (t_sum,  t_diff, f'{t_sum_col} vs {t_diff_col}'),
            (t_sum,  q_sum,  f'{t_sum_col} vs {q_sum_col}'),
            (t_sum,  y,      f'{t_sum_col} vs {y_col}'),
            (t_diff, q_sum,  f'{t_diff_col} vs {q_sum_col}'),
            (t_diff, y,      f'{t_diff_col} vs {y_col}'),
            (q_sum,  y,      f'{q_sum_col} vs {y_col}'),
            (t_sum,  q_diff, f'{t_sum_col} vs {q_diff_col}'),
            (t_diff, q_diff, f'{t_diff_col} vs {q_diff_col}'),
            (q_diff, y,      f'{q_diff_col} vs {y_col}'),
            (q_sum,  q_diff, f'{q_sum_col} vs {q_diff_col}')
        ]

        for offset, (x, yv, title) in enumerate(combinations):
            ax = axes[base_idx + offset]
            ax.hexbin(x, yv, gridsize=50, cmap='turbo')
            ax.set_title(title)

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.suptitle('Hexbin Plots for All Variable Combinations by Plane', fontsize=18)

    if save_plots:
        name_of_file = 'rpc_variables_hexbin_combinations'
        final_filename = f'{fig_idx}_{name_of_file}.png'
        fig_idx += 1

        save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
        plot_list.append(save_fig_path)
        plt.savefig(save_fig_path, format='png')

    if show_plots: plt.show()
    plt.close()


print("--------------------- Filter 6: calibrated data ----------------------")
for col in working_df.columns:
    if 'T_sum_final' in col:
        working_df[col] = np.where((working_df[col] < T_sum_RPC_left) | (working_df[col] > T_sum_RPC_right), 0, working_df[col])
    if 'T_diff_final' in col:
        working_df[col] = np.where((working_df[col] < T_diff_RPC_left) | (working_df[col] > T_diff_RPC_right), 0, working_df[col])
    if 'Q_sum_final' in col:
        working_df[col] = np.where((working_df[col] < Q_RPC_left) | (working_df[col] > Q_RPC_right), 0, working_df[col])
    if 'Q_diff_final' in col:
        working_df[col] = np.where((working_df[col] < Q_dif_RPC_left) | (working_df[col] > Q_dif_RPC_right), 0, working_df[col])
    if 'Y_' in col:
        working_df[col] = np.where((working_df[col] < Y_RPC_left) | (working_df[col] > Y_RPC_right), 0, working_df[col])

total_events = len(working_df)

for i_plane in range(1, 5):
    y_col      = f'Y_{i_plane}'
    t_sum_col  = f'P{i_plane}_T_sum_final'
    t_diff_col = f'P{i_plane}_T_diff_final'
    q_sum_col  = f'P{i_plane}_Q_sum_final'
    q_diff_col = f'P{i_plane}_Q_diff_final'

    cols = [y_col, t_sum_col, t_diff_col, q_sum_col, q_diff_col]

    # Identify affected rows
    mask = (working_df[cols] == 0).any(axis=1)
    num_affected = mask.sum()

    print(f"Plane {i_plane}: {num_affected} out of {total_events} events affected ({(num_affected / total_events) * 100:.2f}%)")

    # Apply zeroing
    working_df.loc[mask, cols] = 0


# ----------------------------------------------------------------------------------------------------------------
if stratos_save and station == 1:
    print("Saving X and Y for stratos.")
    
    stratos_df = working_df.copy()
    
    # Select columns that start with "Y_" or match "T<number>_T_diff_final"
    filtered_columns = [col for col in stratos_df.columns if col.startswith("Y_") or "_T_diff_final" in col or 'datetime' in col]

    # Create a new DataFrame with the selected columns
    filtered_stratos_df = stratos_df[filtered_columns]

    # Rename "T<number>_T_diff_final" to "X_<number>" and multiply by 200
    filtered_stratos_df.rename(columns=lambda col: f'X_{col.split("_")[0][1:]}' if "_T_diff_final" in col else col, inplace=True)
    filtered_stratos_df.loc[:, filtered_stratos_df.columns.str.startswith("X_")] *= 200

    # Define the save path
    save_stratos = os.path.join(stratos_list_events_directory, f'stratos_data_{save_filename_suffix}.csv')

    # Save DataFrame to CSV (correcting the method name)
    filtered_stratos_df.to_csv(save_stratos, index=False, float_format="%.1f")
# ----------------------------------------------------------------------------------------------------------------

# Same for hexbin
if create_plots or create_essential_plots:
    fig, axes = plt.subplots(4, 10, figsize=(40, 20))  # 10 combinations per plane
    axes = axes.flatten()

    for i_plane in range(1, 5):
        # Column names
        t_sum_col = f'P{i_plane}_T_sum_final'
        t_diff_col = f'P{i_plane}_T_diff_final'
        q_sum_col = f'P{i_plane}_Q_sum_final'
        q_diff_col = f'P{i_plane}_Q_diff_final'
        y_col = f'Y_{i_plane}'

        # Filter valid rows (non-zero)
        valid_rows = working_df[[t_sum_col, t_diff_col, q_sum_col, q_diff_col, y_col]].replace(0, np.nan).dropna()
        
        # Extract variables and filter low charge
        cond = valid_rows[q_sum_col] < 50
        t_sum  = valid_rows.loc[cond, t_sum_col]
        t_diff = valid_rows.loc[cond, t_diff_col]
        q_sum  = valid_rows.loc[cond, q_sum_col]
        q_diff = valid_rows.loc[cond, q_diff_col]
        y      = valid_rows.loc[cond, y_col]

        base_idx = (i_plane - 1) * 10  # 10 plots per plane

        combinations = [
            (t_sum,  t_diff, f'{t_sum_col} vs {t_diff_col}'),
            (t_sum,  q_sum,  f'{t_sum_col} vs {q_sum_col}'),
            (t_sum,  y,      f'{t_sum_col} vs {y_col}'),
            (t_diff, q_sum,  f'{t_diff_col} vs {q_sum_col}'),
            (t_diff, y,      f'{t_diff_col} vs {y_col}'),
            (q_sum,  y,      f'{q_sum_col} vs {y_col}'),
            (t_sum,  q_diff, f'{t_sum_col} vs {q_diff_col}'),
            (t_diff, q_diff, f'{t_diff_col} vs {q_diff_col}'),
            (q_diff, y,      f'{q_diff_col} vs {y_col}'),
            (q_sum,  q_diff, f'{q_sum_col} vs {q_diff_col}')
        ]

        for offset, (x, yv, title) in enumerate(combinations):
            ax = axes[base_idx + offset]
            ax.hexbin(x, yv, gridsize=50, cmap='turbo')
            ax.set_title(title)

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.suptitle('Hexbin Plots for All Variable Combinations by Plane, filtered', fontsize=18)

    if save_plots:
        name_of_file = 'rpc_variables_hexbin_combinations_filtered'
        final_filename = f'{fig_idx}_{name_of_file}.png'
        fig_idx += 1

        save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
        plot_list.append(save_fig_path)
        plt.savefig(save_fig_path, format='png')

    if show_plots: plt.show()
    plt.close()


print("----------------------------------------------------------------------")
print("----------------------------------------------------------------------")
print("-------------- Alternative angle and slowness fitting ----------------")
print("----------------------------------------------------------------------")
print("----------------------------------------------------------------------")

# Function to fit a straight line in 3D
def fit_3d_line(X, Y, Z, sX, sY, sZ):
    points = np.vstack((X, Y, Z)).T
    centroid = np.mean(points, axis=0)
    centered_points = points - centroid
    _, _, Vt = np.linalg.svd(centered_points)
    direction_vector = Vt[0]
    d_x, d_y, d_z = direction_vector
    theta = np.arccos(d_z / np.linalg.norm(direction_vector))
    phi = np.arctan2(d_y, d_x)
    
    if d_z != 0:  # Ensure no division by zero
        t_0 = -centroid[2] / d_z
        x_z0 = centroid[0] + t_0 * d_x
        y_z0 = centroid[1] + t_0 * d_y
    else:
        x_z0, y_z0 = np.nan, np.nan # Line is parallel to Z-plane
    
    distances = np.linalg.norm(np.cross(centered_points, direction_vector), axis=1)
    chi2 = np.sum((distances / np.sqrt(np.array(sX)**2 + np.array(sY)**2 + np.array(sZ)**2)) ** 2)
    return x_z0, y_z0, theta, phi, chi2

fit_results = {
    'alt_x': [], 'alt_y': [],
    'alt_theta': [], 'alt_phi': [],
    'alt_chi2': []
}

for idx, track in working_df.iterrows():
    planes_to_iterate = [
        i + 1 for i in range(nplan)
        if getattr(track, f'P{i+1}_Q_sum_final') > 4
    ]

    if len(planes_to_iterate) >= 2:
        X, Y, Z = [], [], []
        for iplane in planes_to_iterate:
            X.append(strip_speed * getattr(track, f'P{iplane}_T_diff_final'))
            Y.append(getattr(track, f'Y_{iplane}'))
            Z.append(z_positions[iplane - 1])

        sX = [anc_sx] * len(X)
        sY = [anc_sy] * len(Y)
        sZ = [anc_sz] * len(Z)

        x, y, theta, phi, chi2 = fit_3d_line(X, Y, Z, sX, sY, sZ)
    else:
        x = y = theta = phi = chi2 = 0.

    fit_results['alt_x'].append(x)
    fit_results['alt_y'].append(y)
    fit_results['alt_theta'].append(theta)
    fit_results['alt_phi'].append(phi)
    fit_results['alt_chi2'].append(chi2)

# Add all at once
working_df = pd.concat([working_df, pd.DataFrame(fit_results, index=working_df.index)], axis=1)

slow_results = {
    'alt_s': [],
    'chi2_tsum_fit': []
}

for idx, track in working_df.iterrows():
    planes_to_iterate = [
        i + 1 for i in range(nplan)
        if getattr(track, f'P{i+1}_Q_sum_final') > 0
    ]

    if len(planes_to_iterate) >= 2:
        tsum = [getattr(track, f'P{iplane}_T_sum_final') for iplane in planes_to_iterate]
        z = [z_positions[iplane - 1] for iplane in planes_to_iterate]

        theta = track['alt_theta']
        phi = track['alt_phi']

        v_dir = np.array([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta)
        ])

        positions = np.stack([np.zeros_like(z), np.zeros_like(z), z], axis=1)
        proj_dist = positions @ v_dir

        s_rel = proj_dist - proj_dist[0]
        t_rel = np.array(tsum) - tsum[0]

        slope, intercept = np.polyfit(s_rel, t_rel, deg=1)
        t_fit = slope * s_rel + intercept
        residuals = t_rel - t_fit
        chi2 = np.sum((residuals / anc_sts) ** 2)
    else:
        slope = chi2 = 0.

    slow_results['alt_s'].append(slope)
    slow_results['chi2_tsum_fit'].append(chi2)

# Add at once
working_df = pd.concat([working_df, pd.DataFrame(slow_results, index=working_df.index)], axis=1)

working_df = working_df.copy()
working_df['alt_th_chi'] = working_df['alt_chi2'] + working_df['chi2_tsum_fit']


# ---------------------------------------------------------------------------
# Put every value close to 0 to effectively 0 -------------------------------
# ---------------------------------------------------------------------------

# Filter the values inside the machine number window
eps = 1e-7  # Threshold

if create_plots:
# if create_essential_plots or create_plots:
    # Flatten all numeric values except 0
    flat_values = working_df.select_dtypes(include=[np.number]).values.ravel()
    flat_values = flat_values[flat_values != 0]

    cond = abs(flat_values) < eps
    flat_values = flat_values[cond]

    # Plot histogram
    plt.figure(figsize=(8, 6))
    plt.hist(flat_values, bins=300, alpha=0.7)
    plt.xlabel('Value')
    plt.ylabel('Counts')
    plt.title('Histogram of All Nonzero Values in working_df')
    plt.yscale('log')  # Optional: log scale to reveal structure
    plt.grid(True)
    plt.tight_layout()
    if save_plots:
            name_of_file = 'flat_values_histogram'
            final_filename = f'{fig_idx}_{name_of_file}.png'
            fig_idx += 1
            save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
            plot_list.append(save_fig_path)
            plt.savefig(save_fig_path, format='png')
    if show_plots:
        plt.show()
    plt.close()

def is_small_nonzero(x):
    return isinstance(x, (int, float)) and x != 0 and abs(x) < eps

# Create mask of small, non-zero numeric values
mask = working_df.applymap(is_small_nonzero)

# Count total non-zero numeric entries
nonzero_numeric_mask = working_df.applymap(lambda x: isinstance(x, (int, float)) and x != 0)
n_total = nonzero_numeric_mask.sum().sum()
n_small = mask.sum().sum()

# Apply the replacement
working_df = working_df.mask(mask, 0)

# Report
pct = 100 * n_small / n_total if n_total > 0 else 0
print(f"{n_small} out of {n_total} non-zero numeric values are below {eps} ({pct:.4f}%)")


# if create_essential_plots or create_plots:
if create_plots:
    
    print("Plotting...")
    
    # ---------------------------------------------------------------------------------------
    # ANGLES PLOTS ------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------------
    
    # PLOT 1 -------------------------------------------------------------------------------------------------------------------
    # Contour plot of alt_chi2 vs alt_theta and alt_phi
    theta_values = working_df['alt_theta'].values
    phi_values = working_df['alt_phi'].values
    chi2_values = np.clip(working_df['alt_chi2'].values, 0, 1)
    
    # Define grid resolution
    theta_bins = np.linspace(min(theta_values), max(theta_values), 40)
    phi_bins = np.linspace(min(phi_values), max(phi_values), 40)

    # Create a 2D histogram
    H, theta_edges, phi_edges = np.histogram2d(theta_values, phi_values, bins=[theta_bins, phi_bins], weights=chi2_values)
    counts, _, _ = np.histogram2d(theta_values, phi_values, bins=[theta_bins, phi_bins])

    # Compute the average chi2 in each bin
    with np.errstate(divide='ignore', invalid='ignore'):
        Chi2_binned = np.where(counts > 0, H / counts, 0)  # Average chi2 in each bin, 0 where no data

    # Define grid for plotting
    Theta_mid = (theta_edges[:-1] + theta_edges[1:]) / 2
    Phi_mid = (phi_edges[:-1] + phi_edges[1:]) / 2
    Theta_grid, Phi_grid = np.meshgrid(Theta_mid, Phi_mid, indexing='ij')

    # Plot binned contour
    plt.figure(figsize=(8, 6))
    contour = plt.contourf(Phi_grid, Theta_grid, Chi2_binned, levels=20, cmap='viridis')
    plt.colorbar(contour, label='Chi-Squared')
    plt.xlabel('Azimuth (φ) [radians]')
    plt.ylabel('Zenith (θ) [radians]')
    plt.title('Binned Contour Plot of Chi-Squared')
    plt.grid(True)

    if save_plots:
        name_of_file = 'alternative_fitting_results_hexbin_combination_projections'
        final_filename = f'{fig_idx}_{name_of_file}.png'
        fig_idx += 1
        save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
        plot_list.append(save_fig_path)
        plt.savefig(save_fig_path, format='png')

    # Show plot if enabled
    if show_plots:
        plt.show()
    plt.close()
    
    
    # ---------------------------------------------------------------------------------------
    # SLOWNESS PLOTS ------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------------
    
    # Histogram the slowness calculated
    plt.figure(figsize=(8, 6))

    v = working_df['alt_s']
    v = v[(v != 0) & v.notna()]  # Exclude exact 0 and NaNs

    n_total = len(v)
    n_small = ( (v > 0) & (abs(v) < eps) ).sum()
    print(f"{n_small} out of {n_total} values are below {eps} ({100 * n_small / n_total:.2f}%)")

    cond = (v > slowness_filter_left) & (v < slowness_filter_right)
    v = v[cond]

    plt.hist(v, bins=200, alpha=0.7)

    plt.xlabel('Slowness (ns/mm)')
    plt.ylabel('Counts')
    plt.title('Histogram of Slowness')
    plt.grid(True)
    plt.xlim(slowness_filter_left, slowness_filter_right)  # Adjust x-axis limits as needed
    # plt.ylim(slowness_filter_left, slowness_filter_right)  # Adjust y-axis limits as needed
    plt.tight_layout()
    
    if save_plots:
        name_of_file = 'alt_s'
        final_filename = f'{fig_idx}_{name_of_file}.png'
        fig_idx += 1

        save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
        plot_list.append(save_fig_path)
        plt.savefig(save_fig_path, format='png')

    # Show plot if enabled
    if show_plots:
        plt.show()
    plt.close()
    
    
    # Plot chi2_tsum_fit --------------------------------------------------------------------
    value_inter_med = eps
    
    # Histogram the slowness calculated
    plt.figure(figsize=(8, 6))
    v = working_df['chi2_tsum_fit'].replace(0, np.nan).dropna()
    cond = (v > value_inter_med) & (v < 10)
    v = v[cond]
    
    plt.hist(v, bins=100, alpha=0.7)
    plt.xlabel('Slowness chi2 (ns/mm)')
    plt.ylabel('Counts')
    plt.title('Histogram of chi2')
    plt.grid(True)
    # plt.xlim(slowness_filter_left, slowness_filter_right)  # Adjust x-axis limits as needed
    # plt.ylim(slowness_filter_left, slowness_filter_right)  # Adjust y-axis limits as needed
    plt.tight_layout()
    
    if save_plots:
        name_of_file = 'alt_s_chi2'
        final_filename = f'{fig_idx}_{name_of_file}.png'
        fig_idx += 1

        save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
        plot_list.append(save_fig_path)
        plt.savefig(save_fig_path, format='png')

    # Show plot if enabled
    if show_plots:
        plt.show()
    plt.close()
    
    
    # Plot chi2_tsum_fit vs alt_s --------------------------------------------------------
    plt.figure(figsize=(8, 6))
    plt.scatter(working_df['alt_s'], working_df['chi2_tsum_fit'], alpha=0.8, s = 0.1)
    plt.xlabel('Slowness (ns/mm)')
    plt.ylabel('Chi2 T_sum Fit')
    plt.title('Chi2 T_sum Fit vs Slowness')
    plt.grid(True)
    plt.xlim(slowness_filter_left, slowness_filter_right)  # Adjust x-axis limits as needed
    plt.ylim(0, 10)  # Adjust y-axis limits as needed
    plt.tight_layout()
    if save_plots:
        name_of_file = 'alt_s_vs_chi2_scatter'
        final_filename = f'{fig_idx}_{name_of_file}.png'
        fig_idx += 1

        save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
        plot_list.append(save_fig_path)
        plt.savefig(save_fig_path, format='png')

    # Show plot if enabled
    if show_plots:
        plt.show()
    plt.close()
    
    
    # HEXBIN
    x_min, x_max = slowness_filter_left, slowness_filter_right
    y_min, y_max = eps, 10

    # Filter data within the plotting region
    mask = (
        working_df['alt_s'].between(x_min, x_max) &
        working_df['chi2_tsum_fit'].between(y_min, y_max)
    )
    x = working_df.loc[mask, 'alt_s']
    y = working_df.loc[mask, 'chi2_tsum_fit']

    plt.figure(figsize=(8, 6))
    plt.hexbin(x, y, gridsize=80, cmap='viridis', bins='log')
    plt.xlabel('Slowness (ns/mm)')
    plt.ylabel('Chi2 T_sum Fit')
    plt.title('Chi2 T_sum Fit vs Slowness (Hexbin)')
    plt.grid(True)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.colorbar(label='log10(counts)')
    plt.tight_layout()

    if save_plots:
        name_of_file = 'alt_s_vs_chi2_hexbin'
        final_filename = f'{fig_idx}_{name_of_file}.png'
        fig_idx += 1
        save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
        plot_list.append(save_fig_path)
        plt.savefig(save_fig_path, format='png')

    if show_plots:
        plt.show()
    plt.close()

    
    # Plot chi2_tsum_fit vs angular fit chi2 --------------------------------------------------------
    plt.figure(figsize=(8, 6))
    plt.scatter(working_df['alt_chi2'], working_df['chi2_tsum_fit'], alpha=0.8, s = 0.1)
    plt.xlabel('Chi 2 angle')
    plt.ylabel('Chi2 slowness')
    plt.title('Chi2 angle vs chi2 slowness')
    plt.grid(True)
    plt.xlim(0, 10)  # Adjust x-axis limits as needed
    plt.ylim(0, 10)  # Adjust y-axis limits as needed
    plt.tight_layout()
    if save_plots:
        name_of_file = 'angle_chi2_vs_chi2_scatter'
        final_filename = f'{fig_idx}_{name_of_file}.png'
        fig_idx += 1

        save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
        plot_list.append(save_fig_path)
        plt.savefig(save_fig_path, format='png')

    # Show plot if enabled
    if show_plots:
        plt.show()
    plt.close()
    
    
    # HEXBIN
    x_min, x_max = eps, 2
    y_min, y_max = eps, 2

    # Filter data within the plotting region
    mask = (
        working_df['alt_chi2'].between(x_min, x_max) &
        working_df['chi2_tsum_fit'].between(y_min, y_max)
    )
    x = working_df.loc[mask, 'alt_chi2']
    y = working_df.loc[mask, 'chi2_tsum_fit']

    plt.figure(figsize=(8, 6))
    plt.hexbin(x, y, gridsize=100, cmap='viridis', bins='log')
    plt.xlabel('Angle chi2')
    plt.ylabel('Slowness chi2')
    plt.title('Chi2 angle vs chi2 slowness')
    plt.grid(True)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.colorbar(label='log10(counts)')
    plt.tight_layout()

    if save_plots:
        name_of_file = 'angle_chi2_vs_chi2_hexbin'
        final_filename = f'{fig_idx}_{name_of_file}.png'
        fig_idx += 1
        save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
        plot_list.append(save_fig_path)
        plt.savefig(save_fig_path, format='png')

    if show_plots:
        plt.show()
    plt.close()
    
    
    # Plot alt_theta vs alt_s --------------------------------------------------------
    plt.figure(figsize=(8, 6))
    plt.scatter(working_df['alt_theta'], working_df['alt_s'], alpha=0.8, s = 0.1)
    plt.ylabel('Slowness')
    plt.xlabel('Theta (zenith)')
    plt.title('slowness vs theta angle')
    plt.grid(True)
    plt.ylim(slowness_filter_left, slowness_filter_right)
    plt.xlim(0, np.pi)  # Adjust y-axis limits as needed
    plt.tight_layout()
    if save_plots:
        name_of_file = 's_vs_theta_scatter'
        final_filename = f'{fig_idx}_{name_of_file}.png'
        fig_idx += 1

        save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
        plot_list.append(save_fig_path)
        plt.savefig(save_fig_path, format='png')

    # Show plot if enabled
    if show_plots:
        plt.show()
    plt.close()
    
    # Print the most repeated value in alt_phi column
    most_repeated_value = working_df['alt_phi'].mode()[0]
    print(f"Most repeated value in alt_phi column: {most_repeated_value}")
    
    # Plot alt_theta vs alt_s --------------------------------------------------------
    plt.figure(figsize=(8, 6))
    plt.scatter(working_df['alt_phi'], working_df['alt_s'], alpha=0.8, s = 0.1)
    plt.ylabel('Slowness')
    plt.xlabel('Phi (Azimuth)')
    plt.title('slowness vs phi angle')
    plt.grid(True)
    plt.ylim(slowness_filter_left, slowness_filter_right)
    plt.xlim(-1*np.pi, np.pi)  # Adjust y-axis limits as needed
    plt.tight_layout()
    if save_plots:
        name_of_file = 's_vs_phi_scatter'
        final_filename = f'{fig_idx}_{name_of_file}.png'
        fig_idx += 1

        save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
        plot_list.append(save_fig_path)
        plt.savefig(save_fig_path, format='png')

    # Show plot if enabled
    if show_plots:
        plt.show()
    plt.close()


for col in working_df.columns:
    # Alternative fitting results
    if 'alt_x' == col or 'alt_y' == col:
        cond_bound = (working_df[col] > alt_pos_filter) | (working_df[col] < -1*alt_pos_filter)
        cond_zero = (working_df[col] == 0)
        working_df.loc[:, col] = np.where((cond_bound | cond_zero), 0, working_df[col])
    if 'alt_theta' == col:
        cond_bound = (working_df[col] > alt_theta_right_filter) | (working_df[col] < alt_theta_left_filter)
        cond_zero = (working_df[col] == 0)
        working_df.loc[:, col] = np.where((cond_bound | cond_zero), 0, working_df[col])
    if 'alt_phi' == col:
        cond_bound = (working_df[col] > alt_phi_right_filter) | (working_df[col] < alt_phi_left_filter)
        cond_zero = (working_df[col] == 0)
        working_df.loc[:, col] = np.where((cond_bound | cond_zero), 0, working_df[col])
    if 'alt_s' == col:
        cond_bound = (working_df[col] > alt_slowness_filter_right) | (working_df[col] < alt_slowness_filter_left)
        cond_zero = (working_df[col] == 0)
        working_df.loc[:, col] = np.where((cond_bound | cond_zero), 0, working_df[col])


# Make sure the full row is set to 0 or whatever
# cond = (working_df['alt_x'] != 0) & (working_df['alt_y'] != 0) &\
#        (working_df['alt_theta'] != 0) & (working_df['alt_phi'] != 0) &\
#        (working_df['alt_s'] != 0)

# working_df = working_df.loc[cond]

print("Alternative fitting done.")


print("----------------------------------------------------------------------")
print("----------------------------------------------------------------------")
print("------------------------- TimTrack fitting ---------------------------")
print("----------------------------------------------------------------------")
print("----------------------------------------------------------------------")

if fixed_speed:
    print("Fixed the slowness to the speed of light.")
else:
    print("Slowness not fixed.")

def fmgx(nvar, npar, vs, ss, zi): # G matrix for t measurements in X-axis
    mg = np.zeros([nvar, npar])
    XP = vs[1]; YP = vs[3]
    if fixed_speed:
        S0 = sc
    else:
        S0 = vs[5]
    kz = sqrt(1 + XP*XP + YP*YP)
    kzi = 1 / kz
    mg[0,2] = 1
    mg[0,3] = zi
    mg[1,1] = kzi * S0 * XP * zi
    mg[1,3] = kzi * S0 * YP * zi
    mg[1,4] = 1
    if fixed_speed == False: mg[1,5] = kz * zi
    mg[2,0] = ss
    mg[2,1] = ss * zi
    return mg

def fmwx(nvar, vsig): # Weigth matrix 
    sy = vsig[0]; sts = vsig[1]; std = vsig[2]
    mw = np.zeros([nvar, nvar])
    mw[0,0] = 1/(sy*sy)
    mw[1,1] = 1/(sts*sts)
    mw[2,2] = 1/(std*std)
    return mw

def fvmx(nvar, vs, lenx, ss, zi): # Fitting model array with X-strips
    vm = np.zeros(nvar)
    X0 = vs[0]; XP = vs[1]; Y0 = vs[2]; YP = vs[3]; T0 = vs[4]
    if fixed_speed:
        S0 = sc
    else:
        S0 = vs[5]
    kz = np.sqrt(1 + XP*XP + YP*YP)
    xi = X0 + XP * zi
    yi = Y0 + YP * zi
    ti = T0 + kz * S0 * zi
    th = 0.5 * lenx * ss   # tau half
    # lxmn = -lenx/2
    vm[0] = yi
    vm[1] = th + ti
    # vm[2] = ss * (xi-lxmn) - th
    vm[2] = ss * xi
    return vm

def fmkx(nvar, npar, vs, vsig, ss, zi): # K matrix
    mk  = np.zeros([npar,npar])
    mg  = fmgx(nvar, npar, vs, ss, zi)
    mgt = mg.transpose()
    mw  = fmwx(nvar, vsig)
    mk  = mgt @ mw @ mg
    return mk

def fvax(nvar, npar, vs, vdat, vsig, lenx, ss, zi): # va vector
    va = np.zeros(npar)
    mw = fmwx(nvar, vsig)
    vm = fvmx(nvar, vs, lenx, ss, zi)
    mg = fmgx(nvar, npar, vs, ss, zi)
    vg = vm - mg @ vs
    vdmg = vdat - vg
    va = mg.transpose() @ mw @ vdmg
    return va

def fs2(nvar, npar, vs, vdat, vsig, lenx, ss, zi):
    va = np.zeros(npar)
    mk = fmkx(nvar, npar, vs, vsig, ss, zi)
    va = fvax(nvar, npar, vs, vdat, vsig, lenx, ss, zi)
    vm = fvmx(nvar, vs, lenx, ss, zi)
    mg = fmgx(nvar, npar, vs, ss, zi)
    mw = fmwx(nvar, vsig)
    vg = vm - mg @ vs
    vdmg = vdat - vg
    mg = fmgx(nvar, npar, vs, ss, zi)
    sk = vs.transpose() @ mk @ vs
    sa = vs.transpose() @ va
    s0 = vdmg.transpose() @ mw @ vdmg
    s2 = sk - 2*sa + s0
    return s2

def fmahd(npar, vin1, vin2, merr): # Mahalanobis distance
    vdif  = np.subtract(vin1,vin2)
    vdsq  = np.power(vdif,2)
    verr  = np.diag(merr,0)
    vsig  = np.divide(vdsq,verr)
    dist  = np.sqrt(np.sum(vsig))
    return dist

def fres(vs, vdat, lenx, ss, zi):  # Residuals array
    X0 = vs[0]; XP = vs[1]; Y0 = vs[2]; YP = vs[3]; T0 = vs[4]
    if fixed_speed:
        S0 = sc
    else:
        S0 = vs[5]
    kz = sqrt(1 + XP*XP + YP*YP)
    # Fitted values
    xfit  = X0 + XP * zi
    yfit  = Y0 + YP * zi
    tffit = T0 + S0 * kz * zi + (lenx/2 + xfit) * ss
    tbfit = T0 + S0 * kz * zi + (lenx/2 - xfit) * ss
    tsfit = 0.5 * (tffit + tbfit)
    tdfit = 0.5 * (tffit - tbfit)
    # Data values
    ydat  = vdat[0]
    tsdat = vdat[1]
    tddat = vdat[2]
    # Residuals
    yr   = (yfit  - ydat)
    tsr  = (tsfit - tsdat)
    tdr  = (tdfit - tddat)
    # DeltaX_tsum = abs( (tsdat - ( T0 + S0 * kz * zi ) ) / 0.5 / ss - lenx)
    vres = [yr, tsr, tdr]
    return vres

if fixed_speed:
    npar = 5
else:
    npar = 6
nvar = 3

i = 0
ntrk  = len(working_df)
if limit and limit_number < ntrk: ntrk = limit_number
print("-----------------------------")
print(f"{ntrk} events to be fitted")

if res_ana_removing_planes:
    timtrack_results = ['x', 'xp', 'y', 'yp', 't0', 's',
                    'th_chi', 'res_y', 'res_ts', 'res_td', 'processed_tt',
                    'res_ystr_1', 'res_ystr_2', 'res_ystr_3', 'res_ystr_4',
                    'res_tsum_1', 'res_tsum_2', 'res_tsum_3', 'res_tsum_4',
                    'res_tdif_1', 'res_tdif_2', 'res_tdif_3', 'res_tdif_4',
                    'ext_res_ystr_1', 'ext_res_ystr_2', 'ext_res_ystr_3', 'ext_res_ystr_4',
                    'ext_res_tsum_1', 'ext_res_tsum_2', 'ext_res_tsum_3', 'ext_res_tsum_4',
                    'ext_res_tdif_1', 'ext_res_tdif_2', 'ext_res_tdif_3', 'ext_res_tdif_4']
else:
    timtrack_results = ['x', 'xp', 'y', 'yp', 't0', 's', 'processed_tt', 'th_chi',
                    'charge_1', 'charge_2', 'charge_3', 'charge_4', 'charge_event',
                    'res_ystr_1', 'res_ystr_2', 'res_ystr_3', 'res_ystr_4',
                    'res_tsum_1', 'res_tsum_2', 'res_tsum_3', 'res_tsum_4',
                    'res_tdif_1', 'res_tdif_2', 'res_tdif_3', 'res_tdif_4']

new_columns_df = pd.DataFrame(0., index=working_df.index, columns=timtrack_results)
working_df = pd.concat([working_df, new_columns_df], axis=1)

# TimTrack starts ------------------------------------------------------
repeat = number_of_TT_executions - 1 if timtrack_iteration else 0
for iteration in range(repeat + 1):
    working_df.loc[:, timtrack_results] = 0.0
    
    fitted = 0
    if timtrack_iteration:
        print("-----------------------------")
        print(f"TimTrack iteration {iteration}")
        print("-----------------------------")
    
    if crontab_execution:
        iterator = working_df.iterrows()
    else:
        iterator = tqdm(working_df.iterrows(), total=working_df.shape[0], desc="Processing events")
    
    for idx, track in iterator:
        # INTRODUCTION ------------------------------------------------------------------
        track_numeric = pd.to_numeric(track.drop('datetime'), errors='coerce')
        
        # -------------------------------------------------------------------------------
        name_type = ""
        planes_to_iterate = []
        
        charge_event = 0
        for i_plane in range(nplan):
            # Check if the sum of the charges in the current plane is non-zero
            charge_plane = getattr(track, f'P{i_plane + 1}_Q_sum_final')
            if charge_plane != 0:
                # Append the plane number to name_type and planes_to_iterate
                name_type += f'{i_plane + 1}'
                planes_to_iterate.append(i_plane + 1)
                working_df.at[idx, f'charge_{i_plane + 1}'] = charge_plane
                charge_event += charge_plane
                
        working_df.at[idx, 'charge_event'] = charge_event
        planes_to_iterate = np.array(planes_to_iterate)
        
        # FITTING -----------------------------------------------------------------------
        if len(planes_to_iterate) > 1:
            if fixed_speed:
                vs  = np.asarray([0,0,0,0,0])
            else:
                vs  = np.asarray([0,0,0,0,0,sc])
            mk  = np.zeros([npar, npar])
            va  = np.zeros(npar)
            istp = 0   # nb. of fitting steps
            dist = d0
            while dist>cocut:
                # for iplane, istrip in zip(planes_to_iterate, istrip_list):
                for iplane in planes_to_iterate:
                    
                    # Data --------------------------------------------------------
                    zi  = z_positions[iplane - 1]                              # z pos
                    yst = getattr(track, f'Y_{iplane}')                        # y position
                    sy  = anc_sy                                               # uncertainty in y               
                    ts  = getattr(track, f'P{iplane}_T_sum_final')             # t sum
                    sts = anc_std                                              # uncertainty in t sum
                    td  = getattr(track, f'P{iplane}_T_diff_final')            # t dif
                    std = anc_std                                              # uncertainty in tdif
                    # -------------------------------------------------------------
                    
                    vdat = [yst, ts, td]
                    vsig = [sy, sts, std]
                    mk = mk + fmkx(nvar, npar, vs, vsig, ss, zi)
                    va = va + fvax(nvar, npar, vs, vdat, vsig, lenx, ss, zi)
                istp = istp + 1
                merr = linalg.inv(mk)     # Error matrix
                vs0 = vs
                vs  = merr @ va          # sEa equation
                dist = fmahd(npar, vs, vs0, merr)
                if istp > 5:
                    continue
            dist = 10
            vsf = vs       # final saeta
            fitted += 1
        else:
            continue
        
        
        # RESIDUAL ANALYSIS ----------------------------------------------------------------------------
        
        # Standard residual analysis
        # chi2 = fs2(nvar, npar, vsf, vdat, vsig, lenx, ss, zi) # Theoretical chisq
        # chi2 = 0 # Theoretical chisq
        
        # Fit residuals
        res_ystr = 0
        res_tsum = 0
        res_tdif = 0
        ndat     = 0
        
        if len(planes_to_iterate) > 1:
            # for iplane, istrip in zip(planes_to_iterate, istrip_list):
            for iplane in planes_to_iterate:
                
                ndat = ndat + nvar
                
                # Data --------------------------------------------------------
                zi  = z_positions[iplane - 1]                                  # z pos
                yst = getattr(track, f'Y_{iplane}')                            # y position
                sy  = anc_sy                                                   # uncertainty in y               
                ts  = getattr(track, f'P{iplane}_T_sum_final')                 # t sum
                sts = anc_std                                                  # uncertainty in t sum
                td  = getattr(track, f'P{iplane}_T_diff_final')                # t dif
                std = anc_std                                                  # uncertainty in tdif
                # -------------------------------------------------------------
                
                vdat = [yst, ts, td]
                vsig = [sy, sts, std]
                vres = fres(vsf, vdat, lenx, ss, zi)
                
                working_df.at[idx, f'res_ystr_{iplane}'] = vres[0]
                working_df.at[idx, f'res_tsum_{iplane}'] = vres[1]
                working_df.at[idx, f'res_tdif_{iplane}'] = vres[2]
                
                res_ystr  = res_ystr  + vres[0]
                res_tsum  = res_tsum + vres[1]
                res_tdif  = res_tdif + vres[2]
                
            ndf  = ndat - npar    # number of degrees of freedom; was ndat - npar
            
            # working_df.at[idx, f'res_ystr_{iplane}'] = res_ystr
            # working_df.at[idx, f'res_tsum_{iplane}'] = res_tsum
            # working_df.at[idx, f'res_tdif_{iplane}'] = res_tdif
            
            working_df.at[idx, 'processed_tt'] = name_type
            
            chi2 = res_ystr**2 + res_tsum**2 + res_tdif**2
            working_df.at[idx, 'th_chi'] = chi2
            
            working_df.at[idx, 'x'] = vsf[0]
            working_df.at[idx, 'xp'] = vsf[1]
            working_df.at[idx, 'y'] = vsf[2]
            working_df.at[idx, 'yp'] = vsf[3]
            working_df.at[idx, 't0'] = vsf[4]
            
            if fixed_speed:
                working_df.at[idx, 's'] = sc
            else:
                working_df.at[idx, 's'] = vsf[5]
        
        # Residual analysis with 4-plane tracks (hide a plane and make a fit in the 3 remaining planes)
        if len(planes_to_iterate) == 4 and res_ana_removing_planes:
            
            # for iplane_ref, istrip_ref in zip(planes_to_iterate, istrip_list):
            for iplane_ref in planes_to_iterate:
                
                # Data --------------------------------------------------------
                z_ref  = z_positions[iplane_ref - 1]                               # z pos
                y_strip_ref = getattr(track, f'Y_{iplane_ref}')                    # y position
                sy  = anc_sy                                                       # uncertainty in y
                t_sum_ref  = getattr(track, f'P{iplane_ref}_T_sum_final')          # t sum
                sts = anc_sts                                                      # uncertainty in t sum
                t_dif_ref  = getattr(track, f'P{iplane_ref}_T_diff_final')         # t dif
                std = anc_std                                                      # uncertainty in tdif
                # -----------------------------------------------------------------
                
                vdat_ref = [ y_strip_ref, t_sum_ref, t_dif_ref]
                
                # istrip_list_short = istrip_list[ planes_to_iterate != iplane_ref ]
                planes_to_iterate_short = planes_to_iterate[planes_to_iterate != iplane_ref]
                
                vs     = vsf  # We start with the previous 4-planes fit
                mk     = np.zeros([npar, npar])
                va     = np.zeros(npar)
                isP3 = 0
                dist = d0
                while dist>cocut:
                    # for iplane, istrip in zip(planes_to_iterate_short, istrip_list_short):
                    for iplane in planes_to_iterate_short:
                    
                        # Data --------------------------------------------------------
                        zi  = z_positions[iplane - 1] - z_ref                           # z pos
                        yst = getattr(track, f'Y_{iplane}')                             # y position
                        sy  = anc_sy                                                    # uncertainty in y
                        ts  = getattr(track, f'P{iplane}_T_sum_final')                  # t sum
                        sts = anc_sts                                                   # uncertainty in t sum
                        td  = getattr(track, f'P{iplane}_T_diff_final')                 # t dif
                        std = anc_std                                                   # uncertainty in tdif
                        # -------------------------------------------------------------
                        
                        vdat = [yst, ts, td]
                        vsig = [sy, sts, std]
                        mk = mk + fmkx(nvar, npar, vs, vsig, ss, zi)
                        va = va + fvax(nvar, npar, vs, vdat, vsig, lenx, ss, zi)
                    isP3 = isP3 + 1
                    merr = linalg.inv(mk)    # Error matrix
                    vs0 = vs
                    vs  = merr @ va          # sEa equation
                    dist = fmahd(npar, vs, vs0, merr)
                    
                vsig = [sy, sts, std]
                # v_track  = [ iplane_ref, istrip_ref ]
                v_res    = fres(vs, vdat_ref, lenx, ss, 0)
                
                working_df.at[idx, f'ext_res_ystr_{iplane_ref}'] = v_res[0]
                working_df.at[idx, f'ext_res_tsum_{iplane_ref}'] = v_res[1]
                working_df.at[idx, f'ext_res_tdif_{iplane_ref}'] = v_res[2]
    
    
    print("----------------------------------------------------------------------")
    print("----------------- TimTrack result and residue plots ------------------")
    print("----------------------------------------------------------------------")
    
    if (create_plots and residual_plots):
    # if create_essential_plots or (create_plots and residual_plots):
        timtrack_columns = ['x', 'xp', 't0', 'y', 'yp', 's']
        residual_columns = [
            'res_ystr_1', 'res_ystr_2', 'res_ystr_3', 'res_ystr_4',
            'res_tsum_1', 'res_tsum_2', 'res_tsum_3', 'res_tsum_4',
            'res_tdif_1', 'res_tdif_2', 'res_tdif_3', 'res_tdif_4'
        ]
        
        # Combined plot for all types
        plot_histograms_and_gaussian(working_df, timtrack_columns, "Combined TimTrack Results", figure_number=1)
        # plot_histograms_and_gaussian(working_df, residual_columns, "Combined Residuals with Gaussian", figure_number=2, fit_gaussian=True, quantile=0.99)
        
        # Individual plots for each unique type - PROCESSED TT
        unique_types = working_df['processed_tt'].unique()
        for t in unique_types:
            subset_data = working_df[working_df['processed_tt'] == t]
            
            # Plot for the 'timtrack_columns' and 'residual_columns' based on type
            plot_histograms_and_gaussian(subset_data, timtrack_columns, f"TimTrack Results for Processed Type {t}", figure_number=1)
            plot_histograms_and_gaussian(subset_data, residual_columns, f"Residuals with Gaussian for Processed Type {t}", figure_number=2, fit_gaussian=True, quantile=0.99)
        
        # Individual plots for each unique type - ORIGINAL TT
        unique_types = working_df['original_tt'].unique()
        for t in unique_types:
            subset_data = working_df[working_df['original_tt'] == t]
            
            # Plot for the 'timtrack_columns' and 'residual_columns' based on type
            plot_histograms_and_gaussian(subset_data, timtrack_columns, f"TimTrack Results for Original Type {t}", figure_number=1)
            plot_histograms_and_gaussian(subset_data, residual_columns, f"Residuals with Gaussian for Original Type {t}", figure_number=2, fit_gaussian=True, quantile=0.99)
    # -------------------------------------------------------------------------
    
    
    # FILTER 6: TSUM, TDIF, QSUM, QDIF TIMTRACK X, Y, etc. FILTER --> IF THE
    # RESULT IS OUT OF RANGE, REMOVE THE MODULE WITH LARGEST RESIDUE
    # for index, row in working_df.iterrows():
    #     # Check if x, y, or t0 is outside the desired range
    #     if (row['t0'] > t0_right_filter or row['t0'] < t0_left_filter) or \
    #         (row['x'] > pos_filter or row['x'] < -pos_filter or row['x'] == 0) or \
    #         (row['y'] > pos_filter or row['y'] < -pos_filter or row['y'] == 0) or \
    #         (row['xp'] > proj_filter or row['xp'] < -proj_filter or row['xp'] == 0) or \
    #         (row['yp'] > proj_filter or row['yp'] < -proj_filter or row['yp'] == 0) or \
    #         (row['s'] > slowness_filter_right or row['s'] < slowness_filter_left or row['s'] == 0) or\
    #         (row['charge_event'] > charge_event_right_filter or row['charge_event'] < charge_event_left_filter or row['charge_event'] == 0):

    #         # Find the module with the largest absolute residue value
    #         max_residue = 0
    #         module_to_zero = None
            
    #         for i in range(1, 5):
    #             if res_ana_removing_planes:
    #                 res_tsum = abs(row[f'ext_res_tsum_{i}'])
    #                 res_tdif = abs(row[f'ext_res_tdif_{i}'])
    #                 res_ystr = abs(row[f'ext_res_ystr_{i}'])
    #             else:
    #                 res_tsum = abs(row[f'res_tsum_{i}'])
    #                 res_tdif = abs(row[f'res_tdif_{i}'])
    #                 res_ystr = abs(row[f'res_ystr_{i}'])
                
    #             # Calculate the maximum residue for the module
    #             max_module_residue = max(res_tsum, res_tdif, res_ystr)
                
    #             if max_module_residue > max_residue:
    #                 max_residue = max_module_residue
    #                 module_to_zero = i
    
    #         # If a module is identified, set related values to 0
    #         if module_to_zero:
    #             working_df.at[index, f'Y_{module_to_zero}'] = 0
    #             working_df.at[index, f'P{module_to_zero}_T_sum_final'] = 0
    #             working_df.at[index, f'P{module_to_zero}_T_diff_final'] = 0
    #             working_df.at[index, f'P{module_to_zero}_Q_sum_final'] = 0
    
    # FILTER 7: TSUM, TDIF, QSUM, QDIF TIMTRACK RESIDUE FILTER --> 0 THE COMPONENT THAT HAS LARGE RESIDUE
    for index, row in working_df.iterrows():
        for i in range(1, 5):
            if res_ana_removing_planes:
                if abs(row[f'ext_res_tsum_{i}']) > ext_res_tsum_filter or \
                    abs(row[f'ext_res_tdif_{i}']) > ext_res_tdif_filter or \
                    abs(row[f'ext_res_ystr_{i}']) > ext_res_ystr_filter:
                    
                    working_df.at[index, f'Y_{i}'] = 0
                    working_df.at[index, f'P{i}_T_sum_final'] = 0
                    working_df.at[index, f'P{i}_T_diff_final'] = 0
                    working_df.at[index, f'P{i}_Q_sum_final'] = 0
            else:
                if abs(row[f'res_tsum_{i}']) > res_tsum_filter or \
                    abs(row[f'res_tdif_{i}']) > res_tdif_filter or \
                    abs(row[f'res_ystr_{i}']) > res_ystr_filter:
                    
                    working_df.at[index, f'Y_{i}'] = 0
                    working_df.at[index, f'P{i}_T_sum_final'] = 0
                    working_df.at[index, f'P{i}_T_diff_final'] = 0
                    working_df.at[index, f'P{i}_Q_sum_final'] = 0
                    working_df.at[index, f'P{i}_Q_diff_final'] = 0
                    
    four_planes = len(working_df[working_df.processed_tt == 1234])
    print(f"Events that are 1234: {four_planes}")
    print(f"Events that are 123: {len(working_df[working_df.processed_tt == 123])}")
    print(f"Events that are 234: {len(working_df[working_df.processed_tt == 234])}")
    planes134 = len(working_df[working_df.processed_tt == 134])
    print(f"Events that are 134: {planes134}")
    planes124 = len(working_df[working_df.processed_tt == 124])
    print(f"Events that are 124: {planes124}")
    eff_2 = 1 - (planes134) / (four_planes + planes134 + planes124)
    print(f"First estimate of eff_2 ={eff_2}")
    eff_3 = 1 - (planes124) / (four_planes + planes134 + planes124)
    print(f"First estimate of eff_3 ={eff_3}")
    
    iteration += 1
    
    
# ------------------------------------------------------------------------------------
# End of TimTrack loop ---------------------------------------------------------------
# ------------------------------------------------------------------------------------

# Set the label to integer
working_df['processed_tt'] = working_df['processed_tt'].apply(builtins.int)

# Calculate angles
def calculate_angles(xproj, yproj):
    phi = np.arctan2(yproj, xproj)
    theta = np.arccos(1 / np.sqrt(xproj**2 + yproj**2 + 1))
    return theta, phi

theta, phi = calculate_angles(working_df['xp'], working_df['yp'])
new_columns_df = pd.DataFrame({'theta': theta, 'phi': phi}, index=working_df.index)
working_df = pd.concat([working_df, new_columns_df], axis=1)


print("----------------------------------------------------------------------")
print("---------------- Filter 8?. Timtrack results filter ------------------")
print("----------------------------------------------------------------------")

for col in working_df.columns:
    # TimTrack results
    if 't0' == col:
        working_df.loc[:, col] = np.where((working_df[col] > t0_right_filter) | (working_df[col] < t0_left_filter), 0, working_df[col])
    if 'x' == col or 'y' == col:
        cond_bound = (working_df[col] > pos_filter) | (working_df[col] < -1*pos_filter)
        cond_zero = (working_df[col] == 0)
        working_df.loc[:, col] = np.where((cond_bound | cond_zero), 0, working_df[col])
    if 'xp' == col or 'yp' == col:
        cond_bound = (working_df[col] > proj_filter) | (working_df[col] < -1*proj_filter)
        cond_zero = (working_df[col] == 0)
        working_df.loc[:, col] = np.where((cond_bound | cond_zero), 0, working_df[col])
    if 's' == col:
        cond_bound = (working_df[col] > slowness_filter_right) | (working_df[col] < slowness_filter_left)
        cond_zero = (working_df[col] == 0)
        working_df.loc[:, col] = np.where((cond_bound | cond_zero), 0, working_df[col])
        

# -----------------------------------------------------------------------------
# Define the last dataframe, the definitive one -------------------------------
# -----------------------------------------------------------------------------

definitive_df = working_df.copy()

cols_to_check = ['x', 'xp', 'y', 'yp', 's', 't0']

# cond = reduce(operator.and_, (working_df[col] != 0 for col in cols_to_check))
cond = (working_df[cols_to_check[0]] != 0)
for col in cols_to_check[1:]:
    cond &= (working_df[col] != 0)

definitive_df = definitive_df[cond]


print("----------------------------------------------------------------------")
print("----------------------- Calculating some stuff -----------------------")
print("----------------------------------------------------------------------")

df_plot_ancillary = definitive_df.copy()

cond = ( df_plot_ancillary['charge_1'] < 250 ) &\
    ( df_plot_ancillary['charge_2'] < 250 ) &\
    ( df_plot_ancillary['charge_3'] < 250 ) &\
    ( df_plot_ancillary['charge_4'] < 250 ) &\
    ( df_plot_ancillary['th_chi'] > eps ) &\
    ( df_plot_ancillary['th_chi'] < 0.03 ) &\
    ( df_plot_ancillary['alt_th_chi'] > eps ) &\
    ( df_plot_ancillary['alt_th_chi'] < 12 )

df_plot_ancillary = df_plot_ancillary.loc[cond].copy()
df_plot_ancillary = df_plot_ancillary[(df_plot_ancillary['charge_event'] > 0) & (df_plot_ancillary['charge_event'] < 600)]

# --------------------------------------------------------------

if create_plots:
# if create_plots or create_essential_plots:

    def plot_hexbin_matrix(df, columns_of_interest, filter_conditions, title, save_plots, show_plots, base_directories, fig_idx, plot_list, num_bins=60):
        """
        Generates a hexbin matrix plot with histograms on the diagonal.
        
        Parameters:
        - df: Pandas DataFrame containing the data
        - columns_of_interest: List of column names to include in the plot
        - filter_conditions: List of tuples (column, min_value, max_value) to filter df
        - title: Title of the plot
        - num_bins: Number of bins for histograms and hexbin plots (default: 60)
        - save_plots: Boolean to save the plot (default: False)
        - show_plots: Boolean to display the plot (default: True)
        - base_directory: Path to save the plot if save_plots is True
        - fig_idx: Index to differentiate saved plot filenames
        - plot_list: List to store the saved plot filenames
        """
        
        axis_limits = {
            'x': [-pos_filter, pos_filter],
            'y': [-pos_filter, pos_filter],
            'alt_x': [-pos_filter, pos_filter],
            'alt_y': [-pos_filter, pos_filter],
            'theta': [0, np.pi],
            'phi': [-np.pi, np.pi],
            'alt_theta': [0, np.pi],
            'alt_phi': [-np.pi, np.pi],
            'xp': [-2, 2],
            'yp': [-2, 2],
            'charge_event': [0, 600],
            'charge_1': [0, 250],
            'charge_2': [0, 250],
            'charge_3': [0, 250],
            'charge_4': [0, 250],
            's': [slowness_filter_left, slowness_filter_right],
            'alt_s': [slowness_filter_left, slowness_filter_right],
            'th_chi': [0, 0.03],
            'alt_th_chi': [0, 12]
        }
        
        # Apply filters
        for col, min_val, max_val in filter_conditions:
            df = df[(df[col] >= min_val) & (df[col] <= max_val)]
        
        num_var = len(columns_of_interest)
        fig, axes = plt.subplots(num_var, num_var, figsize=(15, 15))
        
        auto_limits = {}
        for col in columns_of_interest:
            if col in axis_limits:
                auto_limits[col] = axis_limits[col]
            else:
                auto_limits[col] = [df[col].min(), df[col].max()]
        
        for i in range(num_var):
            for j in range(num_var):
                ax = axes[i, j]
                x_col = columns_of_interest[j]
                y_col = columns_of_interest[i]
                
                if i < j:
                    ax.axis('off')  # Leave the lower triangle blank
                elif i == j:
                    # Diagonal: 1D histogram
                    hist_data = df[x_col]
                    # Remove nans
                    hist_data = hist_data[~np.isnan(hist_data)]
                    # Remove zeroes
                    hist_data = hist_data[hist_data != 0]
                    hist, bins = np.histogram(hist_data, bins=num_bins)
                    bin_centers = 0.5 * (bins[1:] + bins[:-1])
                    norm = plt.Normalize(hist.min(), hist.max())
                    cmap = plt.get_cmap('turbo')
                    
                    for k in range(len(hist)):
                        ax.bar(bin_centers[k], hist[k], width=bins[1] - bins[0], color=cmap(norm(hist[k])))
                    
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_xlim(auto_limits[x_col])
                    
                    # If the column is 'charge_1, 2, 3 or 4', set logscale in Y
                    if x_col.startswith('charge'):
                        ax.set_yscale('log')
                    
                else:
                    # Upper triangle: hexbin plots
                    x_data = df[x_col]
                    y_data = df[y_col]
                    # Remove zeroes and nans
                    cond = (x_data != 0) & (y_data != 0) & (~np.isnan(x_data)) & (~np.isnan(y_data))
                    x_data = x_data[cond]
                    y_data = y_data[cond]
                    ax.hexbin(x_data, y_data, gridsize=num_bins, cmap='turbo')
                    ax.set_facecolor(plt.cm.turbo(0))
                    
                    square_x = [-150, 150, 150, -150, -150]  # Closing the loop
                    square_y = [-150, -150, 150, 150, -150]
                    ax.plot(square_x, square_y, color='white', linewidth=1)  # Thin white line
                    
                    # Apply determined limits
                    ax.set_xlim(auto_limits[x_col])
                    ax.set_ylim(auto_limits[y_col])
                
                if i != num_var - 1:
                    ax.set_xticklabels([])
                if j != 0:
                    ax.set_yticklabels([])
                if i == num_var - 1:
                    ax.set_xlabel(x_col)
                if j == 0:
                    ax.set_ylabel(y_col)
        
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        plt.suptitle(title)
        if save_plots:
            name_of_file = 'timtrack_results_hexbin_combination_projections'
            final_filename = f'{fig_idx}_{name_of_file}.png'
            fig_idx += 1
            save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
            plot_list.append(save_fig_path)
            plt.savefig(save_fig_path, format='png')
        # Show plot if enabled
        if show_plots:
            plt.show()
        plt.close()
        return fig_idx


    # df_cases_2 = [
    #     ([("processed_tt", 12, 12)], "1-2 cases"),
    #     ([("processed_tt", 23, 23)], "2-3 cases"),
    #     ([("processed_tt", 34, 34)], "3-4 cases"),
    #     ([("processed_tt", 13, 13)], "1-3 cases"),
    #     ([("processed_tt", 14, 14)], "1-4 cases"),
    #     ([("processed_tt", 123, 123)], "1-2-3 cases"),
    #     ([("processed_tt", 234, 234)], "2-3-4 cases"),
    #     ([("processed_tt", 124, 124)], "1-2-4 cases"),
    #     ([("processed_tt", 134, 134)], "1-3-4 cases"),
    #     ([("processed_tt", 1234, 1234)], "1-2-3-4 cases"),
    # ]
    
    df_cases_2 = [
        # From original_tt = 1234
        ([("original_tt", 1234, 1234), ("processed_tt", 123, 123)], "original=1234, processed=123"),
        ([("original_tt", 1234, 1234), ("processed_tt", 124, 124)], "original=1234, processed=124"),
        ([("original_tt", 1234, 1234), ("processed_tt", 134, 134)], "original=1234, processed=134"),
        ([("original_tt", 1234, 1234), ("processed_tt", 234, 234)], "original=1234, processed=234"),
        ([("original_tt", 1234, 1234), ("processed_tt", 12, 12)],   "original=1234, processed=12"),
        ([("original_tt", 1234, 1234), ("processed_tt", 13, 13)],   "original=1234, processed=13"),
        ([("original_tt", 1234, 1234), ("processed_tt", 14, 14)],   "original=1234, processed=14"),
        ([("original_tt", 1234, 1234), ("processed_tt", 23, 23)],   "original=1234, processed=23"),
        ([("original_tt", 1234, 1234), ("processed_tt", 24, 24)],   "original=1234, processed=24"),
        ([("original_tt", 1234, 1234), ("processed_tt", 34, 34)],   "original=1234, processed=34"),
        ([("original_tt", 1234, 1234), ("processed_tt", 1234, 1234)], "original=1234, processed=1234"),

        # From original_tt = 124
        ([("original_tt", 124, 124), ("processed_tt", 12, 12)], "original=124, processed=12"),
        ([("original_tt", 124, 124), ("processed_tt", 14, 14)], "original=124, processed=14"),
        ([("original_tt", 124, 124), ("processed_tt", 24, 24)], "original=124, processed=24"),
        ([("original_tt", 124, 124), ("processed_tt", 124, 124)], "original=124, processed=124"),

        # From original_tt = 134
        ([("original_tt", 134, 134), ("processed_tt", 13, 13)], "original=134, processed=13"),
        ([("original_tt", 134, 134), ("processed_tt", 14, 14)], "original=134, processed=14"),
        ([("original_tt", 134, 134), ("processed_tt", 34, 34)], "original=134, processed=34"),
        ([("original_tt", 134, 134), ("processed_tt", 134, 134)], "original=134, processed=134"),

        # From original_tt = 123
        ([("original_tt", 123, 123), ("processed_tt", 12, 12)], "original=123, processed=12"),
        ([("original_tt", 123, 123), ("processed_tt", 13, 13)], "original=123, processed=13"),
        ([("original_tt", 123, 123), ("processed_tt", 23, 23)], "original=123, processed=23"),
        ([("original_tt", 123, 123), ("processed_tt", 123, 123)], "original=123, processed=123"),

        # From original_tt = 234
        ([("original_tt", 234, 234), ("processed_tt", 23, 23)], "original=234, processed=23"),
        ([("original_tt", 234, 234), ("processed_tt", 24, 24)], "original=234, processed=24"),
        ([("original_tt", 234, 234), ("processed_tt", 34, 34)], "original=234, processed=34"),
        ([("original_tt", 234, 234), ("processed_tt", 234, 234)], "original=234, processed=234"),

        # From original_tt = 12
        ([("original_tt", 12, 12), ("processed_tt", 12, 12)], "original=12, processed=12"),

        # From original_tt = 23
        ([("original_tt", 23, 23), ("processed_tt", 23, 23)], "original=23, processed=23"),

        # From original_tt = 34
        ([("original_tt", 34, 34), ("processed_tt", 34, 34)], "original=34, processed=34"),

        # From original_tt = 13
        ([("original_tt", 13, 13), ("processed_tt", 13, 13)], "original=13, processed=13"),
    ]

    # # Charge of each plane -------------------------------------------------------------------
    # for filters, title in df_cases_2:
    #     # Extract the relevant charge numbers from the title (e.g., "1-2 cases" -> [1, 2])
    #     relevant_charges = [f"charge_{n}" for n in map(int, title.split()[0].split('-'))]

    #     # Define the columns of interest dynamically
    #     columns_of_interest = ['x', 'y', 'theta', 'phi', 'xp', 'yp'] + relevant_charges

    #     # Keep the original filters (if needed) and apply them
    #     fig_idx = plot_hexbin_matrix(
    #         df_plot_ancillary,
    #         columns_of_interest,  # Dynamically set the columns to include relevant charges
    #         filters,  # Keep original filters
    #         title,
    #         save_plots,
    #         show_plots,
    #         base_directories,
    #         fig_idx,
    #         plot_list
    #     )


    # # Residues --------------------------------------------------------------------------------------
    # for filters, title in df_cases_2:
    #     relevant_residues_tsum = [f"res_tsum_{n}" for n in map(int, title.split()[0].split('-'))]
    #     relevant_residues_tdif = [f"res_tdif_{n}" for n in map(int, title.split()[0].split('-'))]
    #     relevant_residues_ystr = [f"res_ystr_{n}" for n in map(int, title.split()[0].split('-'))]
        
    #     columns_of_interest = ['x', 'y', 'theta', 'phi', 'xp', 'yp', 's'] + relevant_residues_tsum + relevant_residues_tdif + relevant_residues_ystr
        
    #     fig_idx = plot_hexbin_matrix(
    #         df_plot_ancillary,
    #         columns_of_interest,
    #         filters,
    #         title,
    #         save_plots,
    #         show_plots,
    #         base_directories,
    #         fig_idx,
    #         plot_list
    #     )

    # Comparison with alternative fitting -------------------------------------------------------------------
    for filters, title in df_cases_2:
        fig_idx = plot_hexbin_matrix(
            df_plot_ancillary,
            ['alt_x', 'alt_y', 'alt_phi', 'alt_theta', 'alt_s', 'charge_event', 's', 'theta', 'phi', 'y', 'x'],
            filters,
            title,
            save_plots,
            show_plots,
            base_directories,
            fig_idx,
            plot_list
        )
    
    # Comparison of chi2 with alternative fitting -------------------------------------------------------------------
    for filters, title in df_cases_2:
        fig_idx = plot_hexbin_matrix(
            df_plot_ancillary,
            ['alt_th_chi', 'alt_s', 's', 'th_chi'],
            filters,
            title,
            save_plots,
            show_plots,
            base_directories,
            fig_idx,
            plot_list
        )


# Display the trigger type before and after
if create_plots or create_essential_plots:
    analysis_data = working_df[['original_tt', 'processed_tt']]
    
    # Create a pivot table to count (original_tt, processed_tt) combinations
    counts = analysis_data.groupby(['original_tt', 'processed_tt']).size().unstack(fill_value=0)

    # Ensure consistent ordering for display
    original_order = sorted(analysis_data['original_tt'].unique())
    processed_order = sorted(analysis_data['processed_tt'].unique())
    counts = counts.reindex(index=original_order, columns=processed_order, fill_value=0)

    # Create the 2D colored plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Set ticks and labels
    ax.set_xticks(np.arange(len(counts.columns)))
    ax.set_yticks(np.arange(len(counts.index)))
    ax.set_xticklabels(counts.columns)
    ax.set_yticklabels(counts.index)

    # plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax.set_xlabel("processed_tt")
    ax.set_ylabel("original_tt")
    ax.set_title("Event counts per (original_tt, processed_tt) combination")

    im = ax.imshow(counts, cmap='plasma')
    for i in range(len(counts.index)):
        for j in range(len(counts.columns)):
            value = counts.iloc[i, j]
            if value > 0:
                ax.text(j, i, str(value), ha="center", va="center", color="black" if value > counts.values.max() * 0.5 else "white")

    # Colorbar
    # cbar = fig.colorbar(im, ax=ax)
    # cbar.set_label("Count")

    plt.tight_layout()
    if save_plots:
        name_of_file = 'trigger_types_og_and_processed'
        final_filename = f'{fig_idx}_{name_of_file}.png'
        fig_idx += 1
        save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
        plot_list.append(save_fig_path)
        plt.savefig(save_fig_path, format='png')
    # Show plot if enabled
    if show_plots:
        plt.show()
    plt.close()


print("----------------------------------------------------------------------")
print("----------------------- Final data statistics ------------------------")
print("----------------------------------------------------------------------")

data_purity = len(definitive_df) / raw_data_len*100
print(f"Data purity is {data_purity:.1f}%")

global_variables['purity_of_data_percentage'] = data_purity

if create_plots or create_essential_plots:
    plot_ancillary_df = definitive_df.copy()
    
    # Ensure datetime is proper and indexed
    plot_ancillary_df['datetime'] = pd.to_datetime(plot_ancillary_df['datetime'], errors='coerce')
    plot_ancillary_df = plot_ancillary_df.set_index('datetime')

    # Prepare a container for each group: 2-plane, 3-plane, 4-plane cases
    grouped_data = {
        "Two planes": defaultdict(list),
        "Three planes": defaultdict(list),
        "Four planes": defaultdict(list)
    }

    # Classify events by number of planes in original_tt
    for tt_code in plot_ancillary_df['original_tt'].unique():
        planes = str(tt_code)
        count = len(planes)
        label = f'Case {tt_code}'
        if count == 2:
            grouped_data["Two planes"][label] = plot_ancillary_df[plot_ancillary_df['original_tt'] == tt_code]
        elif count == 3:
            grouped_data["Three planes"][label] = plot_ancillary_df[plot_ancillary_df['original_tt'] == tt_code]
        elif count == 4:
            grouped_data["Four planes"][label] = plot_ancillary_df[plot_ancillary_df['original_tt'] == tt_code]

    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(24, 6))
    colors = plt.colormaps['tab10']

    for ax, (title, group_dict) in zip(axes, grouped_data.items()):
        for i, (label, df) in enumerate(group_dict.items()):
            df.index = pd.to_datetime(df.index, errors='coerce')
            events_per_second = df.index.floor('s').value_counts()
            hist_data = events_per_second.value_counts().sort_index()
            lambda_estimate = events_per_second.mean()
            x_values = np.arange(0, hist_data.index.max() + 1)
            poisson_pmf = poisson.pmf(x_values, lambda_estimate)
            poisson_pmf_scaled = poisson_pmf * len(events_per_second)

            ax.bar(hist_data.index, hist_data.values, label=label, alpha=0.8, color=colors(i % 10))
            ax.plot(x_values, poisson_pmf_scaled, '--', lw=1.5, color=colors(i % 10), alpha=0.6)

        ax.set_title(f'{title}')
        ax.set_xlabel('Number of Events per Second')
        ax.set_ylabel('Frequency')
        ax.legend(fontsize='small', loc='upper right')
        ax.grid(True)

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.suptitle('Event Rate Histograms by Original_tt Cardinality with Poisson Fits', fontsize=16)

    # Save and show
    if save_plots:
        final_filename = f'{fig_idx}_events_per_second_by_plane_cardinality.png'
        fig_idx += 1
        save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
        plot_list.append(save_fig_path)
        plt.savefig(save_fig_path, format='png')

    if show_plots:
        plt.show()
    plt.close()


# ------------------------------------------------------------------------------------------------
# ------------------------------------ Time window plotting --------------------------------------
# ------------------------------------------------------------------------------------------------

if create_plots or create_essential_plots:

    cases = [1234, 123, 234, 124, 134, 12, 23, 34, 13, 14, 24]
    cmap = plt.colormaps['turbo']
    colors = cmap(np.linspace(0, 1, len(cases)))

    # Define window widths
    widths = np.linspace(0, 0.1, 50)
    plt.figure(figsize=(10, 6))

    for idx, case in enumerate(cases):
        data_case = definitive_df[definitive_df["original_tt"] == case]
        
        # Extract only the _T_sum_ columns
        t_sum_columns = [col for col in data_case.columns if "_T_sum_" in col]
        t_sum_data = data_case[t_sum_columns].values  # shape: (n_events, 16)

        counts_per_width = []
        counts_per_width_dev = []
        
        for w in widths:
            count_in_window = []

            for row in t_sum_data:
                row_no_zeros = row[row != 0]
                if len(row_no_zeros) == 0:
                    count_in_window.append(0)
                    continue

                stat = np.mean(row_no_zeros)
                lower = stat - w / 2
                upper = stat + w / 2
                n_in_window = np.sum((row_no_zeros >= lower) & (row_no_zeros <= upper))
                count_in_window.append(n_in_window)

            counts_per_width.append(np.mean(count_in_window))
            counts_per_width_dev.append(np.std(count_in_window))

        plt.scatter(widths, counts_per_width / np.max(counts_per_width), color=colors[idx], label=f"type {case}")
        counts_per_width = np.array(counts_per_width)
        # counts_per_width_dev = np.array(counts_per_width_dev)
        # plt.fill_between( widths, (counts_per_width - counts_per_width_dev) / np.max(counts_per_width), (counts_per_width + counts_per_width_dev) / np.max(counts_per_width), color=colors[idx], alpha=0.2)

    plt.xlabel("Window width (ns)")
    plt.ylabel("Average number of non-zero T_sum values in window")
    plt.title("Counts inside statistic-centered window vs w")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    if save_plots:
        name_of_file = 'window_coincidence_count'
        final_filename = f'{fig_idx}_{name_of_file}.png'
        fig_idx += 1

        save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
        plot_list.append(save_fig_path)
        plt.savefig(save_fig_path, format='png')
        
    if show_plots:
        plt.show()
        

print("----------------------------------------------------------------------")
print("----------------------------------------------------------------------")
print("-------------------------- Save and finish ---------------------------")
print("----------------------------------------------------------------------")
print("----------------------------------------------------------------------")

# Put the global_variables as columns in the final data dataframe
for key, value in global_variables.items():
    if key not in definitive_df.columns:
        print(f"Adding {key} to the dataframe.")
        definitive_df[key] = value
    else:
        print(f"Warning: Column '{key}' already exists in the DataFrame. Skipping addition.")
        

# Round to 4 significant digits ---------------------------------------------------------------
print("Rounding the dataframe values.")

def round_to_4_significant_digits(x):
    try:
        # Use builtins.float to avoid any overridden names
        return builtins.float(f"{builtins.float(x):.4g}")
    except (builtins.ValueError, builtins.TypeError):
        return x
    
for col in definitive_df.select_dtypes(include=[np.number]).columns:
    definitive_df.loc[:, col] = definitive_df[col].apply(round_to_4_significant_digits)


# Change 'datetime' column to 'Time'
if 'datetime' in definitive_df.columns:
    definitive_df.rename(columns={'datetime': 'Time'}, inplace=True)
else:
    print("Column 'datetime' not found in DataFrame!")

# Save the data ----------------------------------------------------------------------------
if save_full_data: # Save a full version of the data, for different studies and debugging
    definitive_df.to_csv(save_full_path, index=False, sep=',', float_format='%.5g')
    print(f"Datafile saved in {save_full_filename}.")

# Save the main columns, relevant for the posterior analysis

# Loop on planes and strips to call the Q_P1s1 from Q1_Q_sum_1 etc
for i, module in enumerate(['1', '2', '3', '4']):
    for j in range(4):
        strip = j + 1
        definitive_df[f'Q_P{module}s{strip}'] = definitive_df[f'Q{module}_Q_sum_{strip}']

# Save a reduced version of the data always, to proceed with the analysis
columns_to_keep = [
    # Timestamp and identifiers
    'Time', 'original_tt', 'processed_tt',

    # Summary metrics and quality flags
    'CRT_avg', 'discarded_by_time_window_percentage', 'sigmoid_width',
    'background_slope', 'one_side_events', 'purity_of_data_percentage',
    'unc_y', 'unc_tsum', 'unc_tdif', 'th_chi',

    # Per-plane active strip patterns
    'active_strips_P1', 'active_strips_P2', 'active_strips_P3', 'active_strips_P4',

    # Y positions
    'Y_1', 'Y_2', 'Y_3', 'Y_4',

    # Final time and charge summaries per plane
    'P1_T_sum_final', 'P1_T_diff_final', 'P1_Q_sum_final', 'P1_Q_diff_final',
    'P2_T_sum_final', 'P2_T_diff_final', 'P2_Q_sum_final', 'P2_Q_diff_final',
    'P3_T_sum_final', 'P3_T_diff_final', 'P3_Q_sum_final', 'P3_Q_diff_final',
    'P4_T_sum_final', 'P4_T_diff_final', 'P4_Q_sum_final', 'P4_Q_diff_final',

    # Alternative reconstruction outputs
    'alt_x', 'alt_y', 'alt_theta', 'alt_phi', 'alt_chi2',
    'alt_s', 'chi2_tsum_fit', 'alt_th_chi',

    # Classical reconstruction outputs
    'x', 'xp', 'y', 'yp', 't0', 's', 'th_chi',

    # Strip-level time and charge info (ordered by plane and strip)
    *[f'Q_P{p}s{s}' for p in range(1, 5) for s in range(1, 5)]
]

reduced_df = definitive_df[columns_to_keep]

reduced_df.to_csv(save_list_path, index=False, sep=',', float_format='%.5g')
print(f"Datafile saved in {save_filename}. Path is {save_list_path}")


# -----------------------------------------------------------------------------
# Save the calibrations -------------------------------------------------------
# -----------------------------------------------------------------------------

new_row = {'Time': start_time}

for i, module in enumerate(['P1', 'P2', 'P3', 'P4']):
    for j in range(4):
        strip = j + 1
        if crosstalk_fitting:
            new_row[f'{module}_s{strip}_Q_sum'] = ( QF_pedestal[i][j] + QB_pedestal[i][j] ) / 2 - crosstalk_limits[f'crstlk_{module}s{strip}']
        else:
            new_row[f'{module}_s{strip}_Q_sum'] = ( QF_pedestal[i][j] + QB_pedestal[i][j] ) / 2
        new_row[f'{module}_s{strip}_T_sum'] = calibration_times[i, j]
        new_row[f'{module}_s{strip}_Q_dif'] = ( QF_pedestal[i][j] - QB_pedestal[i][j] ) / 2
        new_row[f'{module}_s{strip}_T_dif'] = Tdiff_cal[i][j]

if os.path.exists(csv_path):
    # Load the existing DataFrame
    calibrations_df = pd.read_csv(csv_path, parse_dates=['Time'])
else:
    columns = ['Time'] + [
        f'{module}_s{strip}_{var}'
        for module in ['P1', 'P2', 'P3', 'P4']
        for strip in range(1, 5)
        for var in ['Q_sum', 'T_sum', 'Q_dif', 'T_dif']
    ]
    calibrations_df = pd.DataFrame(columns=columns)

# Check if the current time already exists
existing_row_index = calibrations_df[calibrations_df['Time'] == start_time].index

if not existing_row_index.empty:
    # Update the existing row
    calibrations_df.loc[existing_row_index[0]] = new_row
    print(f"Updated existing calibration for date: {start_time}")
else:
    # Append the new row
    calibrations_df = pd.concat([calibrations_df, pd.DataFrame([new_row])], ignore_index=True)
    print(f"Added new calibration for date: {start_time}")

calibrations_df.sort_values(by='Time', inplace=True)
calibrations_df.to_csv(csv_path, index=False, float_format='%.5g')
print(f'{csv_path} updated with the calibrations for this folder.')     

a = 1/0

# Create and save the PDF -----------------------------------------------------
if create_pdf:
    if len(plot_list) > 0:
        with PdfPages(save_pdf_path) as pdf:
            if plot_list:
                for png in plot_list:
                    if os.path.exists(png) == False:
                        print(f"Error: {png} does not exist.")
                        continue
                    
                    # Open the PNG file directly using PIL to get its dimensions
                    img = Image.open(png)
                    fig, ax = plt.subplots(figsize=(img.width / 100, img.height / 100), dpi=100)  # Set figsize and dpi
                    ax.imshow(img)
                    ax.axis('off')  # Hide the axes
                    pdf.savefig(fig, bbox_inches='tight')  # Save figure tightly fitting the image
                    plt.close(fig)  # Close the figure after adding it to the PDF

        # Remove PNG files after creating the PDF
        for png in plot_list:
            try:
                os.remove(png)
                # print(f"Deleted {png}")
            except OSError as e:
                print(f"Error: {e.filename} - {e.strerror}.")


# Erase the figure_directory
if os.path.exists(figure_directory):
    print("Removing figure directory...")
    os.rmdir(figure_directory)

# Move the original datafile to PROCESSED -------------------------------------
print("Moving file to COMPLETED directory...")
# shutil.move(file_path, completed_path)
shutil.move(file_path, completed_file_path)
print("************************************************************")
print(f"File moved from\n{file_path}\nto:\n{completed_file_path}")
print("************************************************************")

if os.path.exists(temp_file):
    print("Removing temporary file...")
    os.remove(temp_file)

# Store the current time at the end
end_execution_time_counting = datetime.now()
time_taken = (end_execution_time_counting - start_execution_time_counting).total_seconds() / 60
print(f"Time taken for the whole execution: {time_taken:.2f} minutes")

print("----------------------------------------------------------------------")
print("------------------- Finished list_events creation --------------------")
print("----------------------------------------------------------------------\n\n\n")