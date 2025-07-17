# Run with
# python3 minitrasgo_bot.py

import os
import subprocess
import sys

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
    '6445882713': 'mingo@1234',
    'user2': 'password2',
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
    binary_path = '/home/rpcuser/bin/HV -b 0'
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

        except ValueError:
            bot.send_message(chat_id, "Invalid input. Please enter a numeric value in kg with at least one decimal digit (e.g., 1.2).")

    bot.register_next_step_handler(message, process_input)


# -----------------------------------------------------------------------------------------

@bot.message_handler(func=lambda message: True)
def echo_all(message):
	string = '\
--------------------------------------------\n\
Useful commands for miniTRASGO bot:\n\
--------------------------------------------\n\
\n\
General stuff:\n\
---------------------\n\
- /get_username: show your name, username and ID. Provide this info to csoneira@ucm.es.\n\
--------------------------------------------\n\
Monitoring:\n\
---------------------\n\
- /start: greetings.\n\
- /get_chat_id: send the ChatID.\n\
- /send_voltage: send a report on the high voltage in the moment.\n\
- /send_gas_flow: send the last values of the gas flow in each RPC.\n\
- /send_internal_environment: send the last values given by the internal environment sensors.\n\
- /send_external_environment: send the last values given by the external environment sensors.\n\
- /send_TRB_rates: send the last rates triggered in the TRB.\n\
--------------------------------------------\n\
\n\
Operation:\n\
---------------------\n\
- /set_data_control_system: script to perform the connection through the I2C protocol from the mingo PC to the hub containing the environment, HV and gas flow sensors. (CURRENTLY NOT IN ENABLED)\n\
- /start_daq: initiate the Data Acquisition System so it will be ready to measure. It does not store data.\n\
- /turn_on_hv: self-explanatory.\n\
- /turn_off_hv: self-explanatory, right?\n\
- /start_run: initiate the storing of the data that the daq is receiving.\n\
- /stop_run: kills the run.\n\
- /any_reboot: reboots the mingo PC. Does not affect the HV.\n\
- /safe_reboot: turns off the HV, waits five minutes and reboots the mingo PC.\n\
--------------------------------------------\n\
\n\
Report and plots:\n\
-----------------------------\n\
- /create_report: creates the pdf report.\n\
- /send_original_report: send the pdf report generated in the mingo itself.\n\
- /send_monitoring_report: send the pdf report with the information per channel.\n\
- /send_daq_report: send the pdf report of charges and multiplicities processed over one day.\n\
- /send_results_vs_time_report: send the pdf report of evolution of some quantities as well as logs.\n\
- /send_weekly_results_report: same, but for the previous week.\n\
--------------------------------------------\n\
\n\
Emergency and assistance:\n\
-----------------------------\n\
- /restart_tunnel: closes the tunnel so it reopens automatically.\n\
\n\
Logging tools:\n\
-----------------------------\n\
- /gas_weight_measurement: manually log the weight of the gas bottle in kilograms (must include at least one decimal digit).\n\
'
	bot.send_message(message.chat.id, string)

bot.infinity_polling()

