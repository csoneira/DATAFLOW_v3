import os
import shutil

station = 1

working_directory = os.path.expanduser(f"~/DATAFLOW_v3/STATIONS/MINGO0{station}/FIRST_STAGE/EVENT_DATA")
acc_working_directory = os.path.join(working_directory, "LIST_TO_ACC")

# Define subdirectories relative to the working directory
base_directories = {
    "list_events_directory": os.path.join(working_directory, "LIST_EVENTS_DIRECTORY"),
    
    "unprocessed_directory": os.path.join(acc_working_directory, "ACC_FILES/ACC_UNPROCESSED"),
    "processing_directory": os.path.join(acc_working_directory, "ACC_FILES/ACC_PROCESSING"),
    "completed_directory": os.path.join(acc_working_directory, "ACC_FILES/ACC_COMPLETED"),
    
    "acc_events_directory": os.path.join(working_directory, "ACC_EVENTS_DIRECTORY"),
    "acc_rejected_directory": os.path.join(working_directory, "ACC_REJECTED"),
}

source_dir = base_directories["acc_events_directory"]
rejected_dir = base_directories["acc_rejected_directory"]

for filename in os.listdir(source_dir):
    file_path = os.path.join(source_dir, filename)
    
    # Check if it's a file
    if os.path.isfile(file_path):
        # Count the number of lines in the file
        with open(file_path, "r") as f:
            line_count = sum(1 for _ in f)

        # Move the file if it has < 15 or > 100 rows
        if line_count < 15 or line_count > 100:
            shutil.move(file_path, os.path.join(rejected_dir, filename))
            print(f"Moved: {filename}")
