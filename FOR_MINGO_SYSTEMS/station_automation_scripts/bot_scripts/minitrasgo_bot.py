"""
DATAFLOW_v3 Script Header v1
Script: FOR_MINGO_SYSTEMS/station_automation_scripts/bot_scripts/minitrasgo_bot.py
Purpose: Run with.
Owner: DATAFLOW_v3 contributors
Sign-off: csoneira <csoneira@ucm.es>
Last Updated: 2026-03-02
Runtime: python3
Usage: python3 FOR_MINGO_SYSTEMS/station_automation_scripts/bot_scripts/minitrasgo_bot.py [options]
Inputs: CLI args, config files, environment variables, and/or upstream files.
Outputs: Files, logs, plots, or stdout/stderr side effects.
Notes: Keep behavior configuration-driven and reproducible.
"""

# Run with
# python3 minitrasgo_bot.py

import os
import subprocess
import sys

import io
import subprocess
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd

# Check if a station number is provided
if len(sys.argv) < 2:
    print("Error: No station number provided.")
    print("Usage: python3 script.py <station_number>")
    sys.exit(1)

# Get the station number
station = sys.argv[1]

# Load the API keys from API.txt
api_keys = {}
try:
    with open("/home/rpcuser/station_automation_scripts/bot_scripts/API.txt", "r") as file:
        for line in file:
            key_parts = line.strip().split()
            if len(key_parts) == 2:  # Ensure valid format: <station> <api_key>
                station_id, key = key_parts
                api_keys[station_id] = key
except FileNotFoundError:
    print("Error: API.txt not found.")
    sys.exit(1)

# Get the API key for the specified station
if station in api_keys:
    api_key = api_keys[station]
    print(f"Using API key for station {station}: {api_key}")
else:
    print(f"Error: No API key found for station {station}.")
    sys.exit(1)


# Initialize the bot with the API key
import telebot
bot = telebot.TeleBot(api_key)
#bot = telebot.TeleBot("8012230734:AAFN1MH_9zbQ3RRoXnKBqalhkJSiEB3V3DE")

passwords = {
    '6445882713': 'mingo@1234', # Cayetano
    '6888390801': 'mingo_ito', # Victor Nouvilas
    '5948691038': 'lidka_mingo', # Lidka
    # Add more users and passwords as needed
}


@bot.message_handler(commands=['get_username'])
def start(message):
    user_id = message.from_user.id  # Get the user's unique ID
    user_username = message.from_user.username  # Get the user's username (if available)
    user_first_name = message.from_user.first_name  # Get the user's first name
    user_last_name = message.from_user.last_name  # Get the user's last name (if available)

    # You can now use this user information as needed
    print(f"User ID: {user_id}")
    print(f"Username: {user_username}")
    print(f"First Name: {user_first_name}")
    print(f"Last Name: {user_last_name}")

    # Respond to the user
    bot.reply_to(message, f"Hello, {user_first_name}! Your ID is {user_id}. Your username is {user_username}.")

@bot.message_handler(commands=['get_chat_id'])
def send_chat_id(message):
    # Get the Chat ID
    chat_id = message.chat.id
    bot.reply_to(message, f"Your Chat ID is: {chat_id}")
    print(f"Chat ID: {chat_id}")  # Print Chat ID to console for debugging

def require_password(func):
    def wrapper(message):
        chat_id = message.chat.id
        user_id = message.from_user.id

        # Check if the user is in the passwords dictionary
        if str(user_id) in passwords:
            password = passwords[str(user_id)]
            bot.send_message(chat_id, "Please enter your password:")
            bot.register_next_step_handler(message, lambda m: check_password(m, password, func))
        else:
            bot.send_message(chat_id, "You are not authorized to perform this action.")

    return wrapper

def check_password(message, expected_password, func):
    user_id = message.from_user.id
    user_password = message.text

    if user_password == expected_password:
        func(message)
    else:
        bot.send_message(user_id, "Incorrect password. Action canceled.")


# Monitoring --------------------------------------------------------------------

@bot.message_handler(commands=['start'])
def send_welcome(message):
    chat_id = message.chat.id
    bot.reply_to(message, "Howdy, how are you doing? Type anything to see what you can do with the miniTRASGO bot.")

@bot.message_handler(commands=['send_voltage'])
def send_voltage(message):
    binary_path = '/home/rpcuser/bin/HV/hv -b 0'
    print(binary_path)

    try:
        # Use subprocess to execute the binary file
        result = subprocess.run(binary_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)

        # Check if the execution was successful
        if result.returncode == 0:
            print("Execution successful")
            # Output from the binary is available in result.stdout
        else:
            print("Execution failed")
            # Error message is available in result.stderr
            print("Error:", result.stderr)
    except Exception as e:
        print("An error occurred:", e)
        result = None  # Handle cases where the result variable might not exist

    # Send the response back to the user
    response = f'{result.stdout}' if result else "No response due to error."
    print(response)
    bot.send_message(message.chat.id, response)


@bot.message_handler(commands=['send_gas_flow'])
def send_gas_flow(message):
	# /home/rpcuser/bin/flowmeter/bin

	# bash_command = "tail /home/rpcuser/logs/Flow0*"
	# output = os.popen(bash_command).read()
	# bot.send_message(message.chat.id, output)

        binary_path='/home/rpcuser/bin/flowmeter/bin/GetData 0 4'
        print(binary_path)

        # Use subprocess to execute the binary file
        result = subprocess.run(binary_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)

        try:
            # Use subprocess to execute the binary file
            result = subprocess.run(binary_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)

            # Check if the execution was successful
            if result.returncode == 0:
                print("Execution successful")
                # Output from the binary is available in result.stdout
            else:
                print("Execution failed")
                # Error message is available in result.stderr
                print("Error:", result.stderr)
        except Exception as e:
            print("An error occurred:", e)

        response = f'{result.stdout}'
        print(response)
        bot.send_message(message.chat.id, response)


@bot.message_handler(commands=['send_internal_environment'])
def send_internal_environment(message):
        bash_command = "tail /home/rpcuser/logs/clean_sensors_bus1*"
        output = os.popen(bash_command).read()
        bot.send_message(message.chat.id, "Date --- T (ºC) --- RH (%) --- P (mbar)")
        bot.send_message(message.chat.id, output)


@bot.message_handler(commands=['send_external_environment'])
def send_external_environment(message):
        bash_command = "tail /home/rpcuser/logs/clean_sensors_bus0*"
        output = os.popen(bash_command).read()
        bot.send_message(message.chat.id, "Date --- T (ºC) --- RH (%) --- P (mbar)")
        bot.send_message(message.chat.id, output)


@bot.message_handler(commands=['send_TRB_rates'])
def send_TRB_rates(message):
        bash_command = "tail /home/rpcuser/logs/clean_rates*"
        output = os.popen(bash_command).read()
        bot.send_message(message.chat.id, "Date --- Asserted - Edge - Accepted --- Multiplexer 1 - M2 - M3 - M4 --- Coincidence Module 1 - CM2 - CM3 - CM4")
        bot.send_message(message.chat.id, output)

# Operation of miniTRASGO ----------------------------------------------------------------

#@bot.message_handler(commands=['set_data_control_system'])
#@require_password
#def set_data_control_system(message):
#        bash_command = "sh /media/externalDisk/gate/bin/dcs_om.sh"
#        bot.send_message(message.chat.id, 'Started...')
#        os.system(bash_command)
#        bot.send_message(message.chat.id, 'Executed.')


@bot.message_handler(commands=['start_daq'])
@require_password
def start_daq(message):
        bash_command = "/bin/bash /home/rpcuser/trbsoft/userscripts/trb399sc/startup_TRB399.sh"
        bot.send_message(message.chat.id, 'Started...')
        os.system(bash_command)
        bot.send_message(message.chat.id, 'Executed.')


@bot.message_handler(commands=['turn_on_hv'])
@require_password
def turn_on_HV(message):
        binary_path='/home/rpcuser/bin/HV/hv -b 0 -I 1 -V 5.52 -on'
        print(binary_path)

        # Use subprocess to execute the binary file
        result = subprocess.run(binary_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)

        try:
            # Use subprocess to execute the binary file
            result = subprocess.run(binary_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)

            # Check if the execution was successful
            if result.returncode == 0:
                print("Execution successful")
                # Output from the binary is available in result.stdout
            else:
                print("Execution failed")
                # Error message is available in result.stderr
                print("Error:", result.stderr)
        except Exception as e:
            print("An error occurred:", e)

        response = f'{result.stdout}'
        print(response)
        bot.send_message(message.chat.id, response)


@bot.message_handler(commands=['turn_off_hv'])
@require_password
def turn_off_HV(message):
        binary_path='/home/rpcuser/bin/HV/hv -b 0 -off'
        print(binary_path)

        # Use subprocess to execute the binary file
        result = subprocess.run(binary_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)

        try:
            # Use subprocess to execute the binary file
            result = subprocess.run(binary_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)

            # Check if the execution was successful
            if result.returncode == 0:
                print("Execution successful")
                # Output from the binary is available in result.stdout
            else:
                print("Execution failed")
                # Error message is available in result.stderr
                print("Error:", result.stderr)
        except Exception as e:
            print("An error occurred:", e)

        response = f'{result.stdout}'
        print(response)
        bot.send_message(message.chat.id, response)


@bot.message_handler(commands=['start_run'])
@require_password
def start_run(message):
        bash_command = "/bin/bash /home/rpcuser/trbsoft/userscripts/trb399sc/caye_startRun.sh"
        bot.send_message(message.chat.id, 'Started...')
        os.system(bash_command)
        bot.send_message(message.chat.id, 'Finished the run.')


@bot.message_handler(commands=['stop_run'])
@require_password
def stop_run(message):
        bash_command = "kill $(ps -ef | grep '[d]abc' | awk '{print $2}')"
        os.system(bash_command)

@bot.message_handler(commands=['any_reboot'])
@require_password
def any_reboot(message):
        bash_command = "reboot"
        bot.send_message(message.chat.id, 'Rebooting the system.')
        os.system(bash_command)

@bot.message_handler(commands=['safe_reboot'])
@require_password
def safe_reboot(message):
        binary_path='/home/rpcuser/bin/HV/hv -b 0 -off'
        print(binary_path)

        # Use subprocess to execute the binary file
        result = subprocess.run(binary_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)

        try:
            # Use subprocess to execute the binary file
            result = subprocess.run(binary_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)

            # Check if the execution was successful
            if result.returncode == 0:
                print("Execution successful")
                # Output from the binary is available in result.stdout
            else:
                print("Execution failed")
                # Error message is available in result.stderr
                print("Error:", result.stderr)
        except Exception as e:
            print("An error occurred:", e)

        response = f'{result.stdout}'
        print(response)
        bot.send_message(message.chat.id, response)

        bash_command = "sleep 300"
        bot.send_message(message.chat.id, 'Turning of the High Voltage... (waiting five minutes)')
        os.system(bash_command)

        bash_command = "reboot"
        bot.send_message(message.chat.id, 'Rebooting the system.')
        os.system(bash_command)

# Report and plots ------------------------------------------------------------------------

from datetime import datetime, timedelta

@bot.message_handler(commands=['send_original_report'])
def send_original_report(message):
        with open(f'/media/usb0/gate/system/devices/mingo01/reporting/report_mingo01.pdf', 'rb') as document:
                bot.send_document(message.chat.id, document)

@bot.message_handler(commands=['create_report'])
@require_password
def create_report(message):
        bash_command = "export RPCSYSTEM=sRPC;/home/rpcuser/gate/bin/createReport.sh"
        os.system(bash_command)

@bot.message_handler(commands=['send_daq_report'])
def send_daq_report(message):
	current_date = datetime.now()
	one_day_ago = current_date - timedelta(days=1)
	report_day = one_day_ago.strftime('20%y-%m-%d')

	with open(f'/home/rpcuser/caye_software/reports/report_daq_data_from_{report_day}.pdf', 'rb') as document:
		bot.send_document(message.chat.id, document)

@bot.message_handler(commands=['send_results_vs_time_report'])
def send_results_vs_time_report(message):
	current_date = datetime.now()
	one_day_ago = current_date - timedelta(days=1)
	report_day = one_day_ago.strftime('20%y-%m-%d')

	with open(f'/home/rpcuser/caye_software/reports/report_results_vs_time_from_{report_day}.pdf', 'rb') as document:
		bot.send_document(message.chat.id, document)

@bot.message_handler(commands=['send_weekly_results_report'])
def send_weekly_results_report(message):
	current_date = datetime.now()
	one_day_ago = current_date - timedelta(days=1)
	report_day = one_day_ago.strftime('20%y-%m-%d')

	with open(f'/home/rpcuser/caye_software/reports/weekly_report_results_vs_time_from_{report_day}.pdf', 'rb') as document:
		bot.send_document(message.chat.id, document)

@bot.message_handler(commands=['send_monitoring_report'])
def send_monitoring_report(message):
	current_date = datetime.now()
	one_day_ago = current_date - timedelta(days=1)
	report_day = one_day_ago.strftime('20%y-%m-%d')

	with open(f'/home/rpcuser/caye_software/reports/final_diagnosis_report_at_{report_day}.pdf', 'rb') as document:
		bot.send_document(message.chat.id, document)


# Emergency and assistance --------------------------------------------------------------------------------------
@bot.message_handler(commands=['restart_tunnel'])
def restart_tunnel(message):
    # Command to find the PID of the 'lipana' process and kill it
    find_pid_command = "pgrep -f 'autossh.*lipana'"
    try:
        import subprocess
        pid = subprocess.check_output(find_pid_command, shell=True).decode().strip()
        if pid:
            os.system(f"kill {pid}")
            bot.send_message(message.chat.id, f'Closing the tunnel... It automatically will be open in less than a minute.')
        else:
            bot.send_message(message.chat.id, 'No process found.')
    except Exception as e:
        bot.send_message(message.chat.id, f'Error killing process: {e}')

@bot.message_handler(commands=['gas_weight_measurement'])
@require_password
def gas_weight_measurement(message):
    chat_id = message.chat.id
    bot.send_message(chat_id, "Enter the gas weight in **kilograms** (kg). The number must include **at least one decimal digit** (e.g., 2.5, 0.8).")

    def process_input(msg):
        user_input = msg.text.strip()

        try:
            # Try converting to float and ensure it contains a decimal point
            if '.' not in user_input or user_input.count('.') != 1:
                raise ValueError("Invalid format: No decimal point found.")

            value = float(user_input)
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_entry = f"{now} {value:.3f}\n"

            log_dir = "/home/rpcuser/logs"
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, "clean_gas_weight_measurements.log")

            with open(log_file, "a") as f:
                f.write(log_entry)

            bot.send_message(chat_id, f"Logged: {log_entry.strip()}")
            
            bot.send_message(chat_id, f"Creating the log file division")
            SCRIPT_DIR = os.path.expanduser("~/station_automation_scripts/bot_scripts")
            script_path = os.path.join(SCRIPT_DIR, "split_gas_logs.sh")
            subprocess.run([script_path], check=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


        except ValueError:
            bot.send_message(chat_id, "Invalid input. Please enter a numeric value in kg with at least one decimal digit (e.g., 1.2).")

    bot.register_next_step_handler(message, process_input)



import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import io
import os

def parse_sensor_data(bash_command):
    """Parse sensor data from bash command output"""
    output = os.popen(bash_command).read()
    lines = output.strip().split('\n')
    
    timestamps = []
    temperatures = []
    humidities = []
    pressures = []
    
    for line in lines:
        if line.strip():
            parts = line.split()
            if len(parts) >= 4:
                try:
                    # Parse timestamp
                    timestamp_str = f"{parts[0]} {parts[1]}"
                    timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                    timestamps.append(timestamp)
                    
                    # Parse sensor values
                    temperatures.append(float(parts[2]))
                    humidities.append(float(parts[3]))
                    pressures.append(float(parts[4]))
                except (ValueError, IndexError):
                    continue
    
    return timestamps, temperatures, humidities, pressures

number_of_plot_lines = 10000

@bot.message_handler(commands=['plot_temperature'])
def plot_temperature(message):
    try:
        # Get internal data (bus1)
        internal_bash = f"tail -n {number_of_plot_lines} /home/rpcuser/logs/clean_sensors_bus1*"
        int_times, int_temps, _, _ = parse_sensor_data(internal_bash)
        
        # Get external data (bus0)
        external_bash = f"tail -n {number_of_plot_lines} /home/rpcuser/logs/clean_sensors_bus0*"
        ext_times, ext_temps, _, _ = parse_sensor_data(external_bash)
        
        # Create plot
        plt.figure(figsize=(12, 6))
        plt.plot(int_times, int_temps, 'r-o', label='Internal Temperature', linewidth=2, markersize=4)
        plt.plot(ext_times, ext_temps, 'b-o', label='External Temperature', linewidth=2, markersize=4)
        
        plt.title('Temperature Comparison - Internal vs External', fontsize=14, fontweight='bold')
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Temperature (°C)', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Format x-axis
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=1))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        # Save plot to bytes
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        
        # Send plot to telegram
        bot.send_photo(message.chat.id, buf, caption="Temperature Time Series")
        
    except Exception as e:
        bot.send_message(message.chat.id, f"Error generating temperature plot: {str(e)}")

@bot.message_handler(commands=['plot_pressure'])
def plot_pressure(message):
    try:
        # Get internal data (bus1)
        internal_bash = f"tail -n {number_of_plot_lines} /home/rpcuser/logs/clean_sensors_bus1*"
        int_times, _, _, int_pressures = parse_sensor_data(internal_bash)
        
        # Get external data (bus0)
        external_bash = f"tail -n {number_of_plot_lines} /home/rpcuser/logs/clean_sensors_bus0*"
        ext_times, _, _, ext_pressures = parse_sensor_data(external_bash)
        
        # Create plot
        plt.figure(figsize=(12, 6))
        plt.plot(int_times, int_pressures, 'r-o', label='Internal Pressure', linewidth=2, markersize=4)
        plt.plot(ext_times, ext_pressures, 'b-o', label='External Pressure', linewidth=2, markersize=4)
        
        plt.title('Pressure Comparison - Internal vs External', fontsize=14, fontweight='bold')
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Pressure (mbar)', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Format x-axis
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=1))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        # Save plot to bytes
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        
        # Send plot to telegram
        bot.send_photo(message.chat.id, buf, caption="Pressure Time Series")
        
    except Exception as e:
        bot.send_message(message.chat.id, f"Error generating pressure plot: {str(e)}")

@bot.message_handler(commands=['plot_RH'])
def plot_RH(message):
    try:
        # Get internal data (bus1)
        internal_bash = f"tail -n {number_of_plot_lines} /home/rpcuser/logs/clean_sensors_bus1*"
        int_times, _, int_humidity, _ = parse_sensor_data(internal_bash)
        
        # Get external data (bus0)
        external_bash = f"tail -n {number_of_plot_lines} /home/rpcuser/logs/clean_sensors_bus0*"
        ext_times, _, ext_humidity, _ = parse_sensor_data(external_bash)
        
        # Create plot
        plt.figure(figsize=(12, 6))
        plt.plot(int_times, int_humidity, 'r-o', label='Internal Relative Humidity', linewidth=2, markersize=4)
        plt.plot(ext_times, ext_humidity, 'b-o', label='External Relative Humidity', linewidth=2, markersize=4)
        
        plt.title('Relative Humidity Comparison - Internal vs External', fontsize=14, fontweight='bold')
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Relative Humidity (%)', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Format x-axis
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=1))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        # Save plot to bytes
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        
        # Send plot to telegram
        bot.send_photo(message.chat.id, buf, caption="Relative Humidity Time Series")
        
    except Exception as e:
        bot.send_message(message.chat.id, f"Error generating humidity plot: {str(e)}")


def parse_trb_rates_data(bash_command):
    """Parse TRB rates data from bash command output"""
    output = os.popen(bash_command).read()
    lines = output.strip().split('\n')
    
    timestamps = []
    asserted = []
    edge = []
    accepted = []
    m1 = []
    m2 = []
    m3 = []
    m4 = []
    cm1 = []
    cm2 = []
    cm3 = []
    cm4 = []
    
    for line in lines:
        if line.strip():
            parts = line.split()
            if len(parts) >= 13:  # Expecting 13 columns based on your header
                try:
                    # Parse timestamp
                    timestamp_str = f"{parts[0]} {parts[1]}"
                    timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                    timestamps.append(timestamp)
                    
                    # Parse TRB rates values
                    asserted.append(float(parts[2]))
                    edge.append(float(parts[3]))
                    accepted.append(float(parts[4]))
                    m1.append(float(parts[5]))
                    m2.append(float(parts[6]))
                    m3.append(float(parts[7]))
                    m4.append(float(parts[8]))
                    cm1.append(float(parts[9]))
                    cm2.append(float(parts[10]))
                    cm3.append(float(parts[11]))
                    cm4.append(float(parts[12]))
                except (ValueError, IndexError):
                    continue
    
    return (timestamps, asserted, edge, accepted, m1, m2, m3, m4, cm1, cm2, cm3, cm4)

@bot.message_handler(commands=['plot_asserted_edge_accepted'])
def plot_asserted_edge_accepted(message):
    try:
        # Get TRB rates data
        trb_bash = f"tail -n {number_of_plot_lines} /home/rpcuser/logs/clean_rates*"
        timestamps, asserted, edge, accepted, _, _, _, _, _, _, _, _ = parse_trb_rates_data(trb_bash)
        
        # Create plot
        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, asserted, 'r-o', label='Asserted', linewidth=2, markersize=4)
        plt.plot(timestamps, edge, 'g-o', label='Edge', linewidth=2, markersize=4)
        plt.plot(timestamps, accepted, 'b-o', label='Accepted', linewidth=2, markersize=4)
        
        plt.title('TRB Rates - Asserted, Edge, Accepted', fontsize=14, fontweight='bold')
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Rate', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Format x-axis
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=1))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        # Save plot to bytes
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        
        # Send plot to telegram
        bot.send_photo(message.chat.id, buf, caption="TRB Rates - Asserted, Edge, Accepted")
        
    except Exception as e:
        bot.send_message(message.chat.id, f"Error generating TRB rates plot: {str(e)}")

@bot.message_handler(commands=['plot_multiplexers'])
def plot_multiplexers(message):
    try:
        # Get TRB rates data
        trb_bash = f"tail -n {number_of_plot_lines} /home/rpcuser/logs/clean_rates*"
        timestamps, _, _, _, m1, m2, m3, m4, _, _, _, _ = parse_trb_rates_data(trb_bash)
        
        # Create plot
        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, m1, 'r-o', label='Multiplexer 1', linewidth=2, markersize=4)
        plt.plot(timestamps, m2, 'g-o', label='Multiplexer 2', linewidth=2, markersize=4)
        plt.plot(timestamps, m3, 'b-o', label='Multiplexer 3', linewidth=2, markersize=4)
        plt.plot(timestamps, m4, 'm-o', label='Multiplexer 4', linewidth=2, markersize=4)
        
        plt.title('TRB Rates - Multiplexers 1-4', fontsize=14, fontweight='bold')
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Rate', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Format x-axis
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=1))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        # Save plot to bytes
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        
        # Send plot to telegram
        bot.send_photo(message.chat.id, buf, caption="TRB Rates - Multiplexers 1-4")
        
    except Exception as e:
        bot.send_message(message.chat.id, f"Error generating multiplexers plot: {str(e)}")

@bot.message_handler(commands=['plot_coincidence_modules'])
def plot_coincidence_modules(message):
    try:
        # Get TRB rates data
        trb_bash = f"tail -n {number_of_plot_lines} /home/rpcuser/logs/clean_rates*"
        timestamps, _, _, _, _, _, _, _, cm1, cm2, cm3, cm4 = parse_trb_rates_data(trb_bash)
        
        # Create plot
        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, cm1, 'r-o', label='Coincidence Module 1', linewidth=2, markersize=4)
        plt.plot(timestamps, cm2, 'g-o', label='Coincidence Module 2', linewidth=2, markersize=4)
        plt.plot(timestamps, cm3, 'b-o', label='Coincidence Module 3', linewidth=2, markersize=4)
        plt.plot(timestamps, cm4, 'm-o', label='Coincidence Module 4', linewidth=2, markersize=4)
        
        plt.title('TRB Rates - Coincidence Modules 1-4', fontsize=14, fontweight='bold')
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Rate', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Format x-axis
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=1))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        # Save plot to bytes
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        
        # Send plot to telegram
        bot.send_photo(message.chat.id, buf, caption="TRB Rates - Coincidence Modules 1-4")
        
    except Exception as e:
        bot.send_message(message.chat.id, f"Error generating coincidence modules plot: {str(e)}")

@bot.message_handler(commands=['plot_all_trb_rates'])
def plot_all_trb_rates(message):
    try:
        # Get TRB rates data
        trb_bash = f"tail -n {number_of_plot_lines} /home/rpcuser/logs/clean_rates*"
        timestamps, asserted, edge, accepted, m1, m2, m3, m4, cm1, cm2, cm3, cm4 = parse_trb_rates_data(trb_bash)
        
        # Create subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12))
        
        # Plot 1: Asserted, Edge, Accepted
        ax1.plot(timestamps, asserted, 'r-o', label='Asserted', linewidth=2, markersize=3)
        ax1.plot(timestamps, edge, 'g-o', label='Edge', linewidth=2, markersize=3)
        ax1.plot(timestamps, accepted, 'b-o', label='Accepted', linewidth=2, markersize=3)
        ax1.set_title('TRB Rates - Asserted, Edge, Accepted', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Rate', fontsize=10)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot 2: Multiplexers
        ax2.plot(timestamps, m1, 'r-o', label='Multiplexer 1', linewidth=2, markersize=3)
        ax2.plot(timestamps, m2, 'g-o', label='Multiplexer 2', linewidth=2, markersize=3)
        ax2.plot(timestamps, m3, 'b-o', label='Multiplexer 3', linewidth=2, markersize=3)
        ax2.plot(timestamps, m4, 'm-o', label='Multiplexer 4', linewidth=2, markersize=3)
        ax2.set_title('TRB Rates - Multiplexers 1-4', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Rate', fontsize=10)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax2.tick_params(axis='x', rotation=45)
        
        # Plot 3: Coincidence Modules
        ax3.plot(timestamps, cm1, 'r-o', label='Coincidence Module 1', linewidth=2, markersize=3)
        ax3.plot(timestamps, cm2, 'g-o', label='Coincidence Module 2', linewidth=2, markersize=3)
        ax3.plot(timestamps, cm3, 'b-o', label='Coincidence Module 3', linewidth=2, markersize=3)
        ax3.plot(timestamps, cm4, 'm-o', label='Coincidence Module 4', linewidth=2, markersize=3)
        ax3.set_title('TRB Rates - Coincidence Modules 1-4', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Time', fontsize=10)
        ax3.set_ylabel('Rate', fontsize=10)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax3.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save plot to bytes
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        
        # Send plot to telegram
        bot.send_photo(message.chat.id, buf, caption="Complete TRB Rates Overview")
        
    except Exception as e:
        bot.send_message(message.chat.id, f"Error generating complete TRB rates plot: {str(e)}")


# -----------------------------------------------------------------------------------------


@bot.message_handler(func=lambda message: True)
def echo_all(message):
    string = '''
===========================================
            miniTRASGO Bot - Command Guide
===========================================

    General Commands:
    ---------------------
    - /start: Greet the bot and receive a welcome message.
    - /get_username: Show your username, ID, and other details (share this info with csoneira@ucm.es).
    - /get_chat_id: Retrieve your unique Chat ID.

    -------------------------------------------

    Monitoring Commands (Text Reports):
    -------------------------------------
    - /send_voltage: Get a report on the current high voltage.
    - /send_gas_flow: Receive the latest gas flow values for each RPC.
    - /send_internal_environment: Get data from the internal environment sensors.
    - /send_external_environment: Get data from the external environment sensors.
    - /send_TRB_rates: Receive the latest triggered TRB rates (Asserted, Edge, Accepted).

    -------------------------------------------

    Monitoring Commands (Plots):
    -------------------------------------
    - /plot_temperature: Plot temperature (°C) data from both internal and external sensors.
    - /plot_pressure: Plot pressure (mbar) data from both internal and external sensors.
    - /plot_RH: Plot relative humidity (%) data from both internal and external sensors.
    - /plot_asserted_edge_accepted: Plot TRB rates (Asserted, Edge, Accepted).
    - /plot_multiplexers: Plot the rates of all 4 multiplexers.
    - /plot_coincidence_modules: Plot the rates of all 4 coincidence modules.
    - /plot_all_trb_rates: View a comprehensive plot with all TRB rates in 3 subplots.

    -------------------------------------------

    Operation Commands:
    ---------------------
    - /set_data_control_system: (Currently Disabled) Connect via I2C from the mingo PC to the hub with environment, HV, and gas flow sensors.
    - /start_daq: Prepare the Data Acquisition System (DAQ) for measurements (data is not stored).
    - /turn_on_hv: Turn on the high voltage.
    - /turn_off_hv: Turn off the high voltage.
    - /start_run: Start recording data from the DAQ.
    - /stop_run: Stop the current data recording.
    - /any_reboot: Reboot the mingo PC (no impact on HV).

    -------------------------------------------

    Emergency & Assistance:
    ---------------------------
    - /restart_tunnel: Close the tunnel, it will automatically reopen.

    -------------------------------------------

    Logging Tools:
    -----------------
    - /gas_weight_measurement: Manually log the weight of the gas bottle in kilograms (ensure it includes at least one decimal place).

    ===========================================
    '''

    bot.send_message(message.chat.id, string)

bot.infinity_polling()
