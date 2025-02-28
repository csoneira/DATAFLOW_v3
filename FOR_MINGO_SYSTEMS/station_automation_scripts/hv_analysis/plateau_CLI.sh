#!/bin/bash

# Modify to perform a personalized study. (init HV, step, end HV [included]), all in kV

# Function to display usage and exit
usage() {
  echo "Usage: $0 [-s start_hv] [-e end_hv] [-i step] [-t time_min] [-f safe_hv]"
  echo "  -s start_hv  Starting HV value (kV) [default: 5.2]"
  echo "  -e end_hv    Ending HV value (kV) [default: 5.5]"
  echo "  -i step      Step size between HV values (kV) [default: 0.05]"
  echo "  -t time_min  Time per voltage (minutes) [default: 60]"
  echo "  -f safe_hv   Safe HV value to set at the end (kV) [default: 5.3]"
  exit 1
}

# Default values
start_hv=5.2
step=0.05
end_hv=5.5
time_per_voltage_in_min=60
safe_hv_value=5.3

# Parse command-line arguments
while getopts "s:e:i:t:f:" opt; do
  case "$opt" in
    s) start_hv="$OPTARG" ;;
    e) end_hv="$OPTARG" ;;
    i) step="$OPTARG" ;;
    t) time_per_voltage_in_min="$OPTARG" ;;
    f) safe_hv_value="$OPTARG" ;;
    \?) usage ;; # Invalid option
  esac
done

# Input validation
if ! [[ "$start_hv" =~ ^[0-9]+(\.[0-9]+)?$ ]] || ! [[ "$step" =~ ^[0-9]+(\.[0-9]+)?$ ]] || ! [[ "$end_hv" =~ ^[0-9]+(\.[0-9]+)?$ ]] || ! [[ "$time_per_voltage_in_min" =~ ^[0-9]+$ ]] || ! [[ "$safe_hv_value" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
  echo "Error: Invalid input. Please provide numeric values."
  usage
fi

# ----------------------------------------------------------------------------------
# Create an array of HV values using seq
w=$(seq "$start_hv" "$step" "$end_hv")
ww=($w)

time_per_voltage_in_sec=$((time_per_voltage_in_min * 60))

# Start of execution
event_count=$(echo "${#ww[@]}")

# Calculate total time in hours
total_time_in_sec=$((event_count * time_per_voltage_in_sec))
total_time_in_hours=$(echo "scale=2; $total_time_in_sec / 3600" | bc)

echo "There will be $total_time_in_hours hours of plateau analysis:"
echo "${ww[@]} kV"
read -p "Do you want to continue (Y/N)? " answer

# Convert the answer to uppercase for case-insensitive comparison
answer=${answer^^}

if [ "$answer" != "Y" ]; then
    echo "Exiting the script."
    exit 1
fi

# If confirmation is received, the analysis starts:

# Get the process IDs from the output of `pgrep dabc` and store them in an array
pids=($(pgrep dabc))

# Loop through each process ID and kill it using `kill -9`
for pid in "${pids[@]}"; do
      kill -9 "$pid"
done

# Time after stopping the startRun.sh to wait
#sleep 30

for v in $w; do
      echo '***************************'
      cd /home/rpcuser/bin/i2c/HV
      ./hv -b 0 -I 1 -V $v -on
      echo "V set to $v"

      # Time for the HV to settle is 5 min
      sleep 300

      cd /home/rpcuser/trbsoft/userscripts/trb/
      ./startRun.sh > /dev/null 2>&1 &

      echo 'Run started'
      date

      # Time of measurement at a certain HV, put 60s/min*20min=1200s
      echo "Sleeping $time_per_voltage_in_sec seconds"
      sleep $time_per_voltage_in_sec

      # Get the process IDs from the output of `pgrep dabc` and store them in an array
      pids=($(pgrep dabc))

      # Loop through each process ID and kill it using `kill -9`
      for pid in "${pids[@]}"; do
            kill -9 "$pid"
      done

      echo 'Run stopped'
      date
      # sleep 300
done

# We end setting a safe value for the voltage
/home/rpcuser/bin/i2c/HV/hv -b 0 -I 1 -V $safe_hv_value -on

# Time for the HV to settle
#sleep 300

# And starting the measurement storage to keep going
cd /home/rpcuser/trbsoft/userscripts/trb/
./startRun.sh > /dev/null 2>&1 &

echo 'Plateau measurement ended'