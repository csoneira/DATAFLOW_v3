#!/bin/bash
# ------------------------------------------------------------------
# daq_restart.sh - restart DAQ when data is stale or missing.
# ------------------------------------------------------------------

DATA_DIR="/home/rpcuser/gate/system/devices/TRB3/data/daqData/asci"
STARTUP_SCRIPT="/home/rpcuser/trbsoft/userscripts/trb/startup_TRB399.sh"
LOG_DIR="/home/rpcuser/logs"
LOG_FILE="${LOG_DIR}/daq_restart.log"
MAIL_TO="csoneira.alarms@gmail.com"
MAIL_COUNT_FILE="${LOG_DIR}/daq_restart_no_file_mail_count"
THRESHOLD_SEC=$((2 * 60 * 60))
LOCK_FILE="/tmp/daq_restart.lock"
VERBOSE="0"

for arg in "$@"; do
    case "${arg}" in
        -v|--verbose) VERBOSE="1" ;;
    esac
done

mkdir -p "${LOG_DIR}"

if command -v flock >/dev/null 2>&1; then
    exec 200>"${LOCK_FILE}"
    if ! flock -n 200; then
        [[ "${VERBOSE}" == "1" ]] && echo "INFO another instance is running; exiting."
        exit 0
    fi
fi

log_msg() {
    printf '%s %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >> "${LOG_FILE}"
}

vmsg() {
    [[ "${VERBOSE}" == "1" ]] && echo "$*"
}

send_mail() {
    local subject body
    subject="DAQ restart: no data files found"
    body="DAQ restart triggered at $(date '+%Y-%m-%d %H:%M:%S').\nNo files found in ${DATA_DIR}.\nHost: $(hostname)"
    if command -v mail >/dev/null 2>&1; then
        printf '%b\n' "${body}" | mail -s "${subject}" "${MAIL_TO}"
        return $?
    fi
    if command -v mailx >/dev/null 2>&1; then
        printf '%b\n' "${body}" | mailx -s "${subject}" "${MAIL_TO}"
        return $?
    fi
    log_msg "WARN mailer not found (mail/mailx)."
    return 1
}

last_file=""
last_mtime=""
no_files="false"

if [[ -d "${DATA_DIR}" ]]; then
    last_file=$(ls -1t "${DATA_DIR}" 2>/dev/null | head -n 1)
else
    log_msg "WARN data dir missing: ${DATA_DIR}"
    vmsg "WARN data dir missing: ${DATA_DIR}"
fi

if [[ -z "${last_file}" ]]; then
    no_files="true"
else
    if [[ -f "${DATA_DIR}/${last_file}" ]]; then
        last_mtime=$(stat -c %Y "${DATA_DIR}/${last_file}" 2>/dev/null || true)
    fi
    if [[ -z "${last_mtime}" ]]; then
        no_files="true"
    fi
fi

restart_needed="false"
reason=""
age_sec=""

if [[ "${no_files}" == "true" ]]; then
    restart_needed="true"
    reason="no_files"
else
    now_epoch=$(date +%s)
    age_sec=$((now_epoch - last_mtime))
    if [[ "${age_sec}" -ge "${THRESHOLD_SEC}" ]]; then
        restart_needed="true"
        reason="stale"
    fi
fi

if [[ "${restart_needed}" == "true" ]]; then
    log_msg "INFO restart triggered reason=${reason} last_file=${last_file:-NONE} age_sec=${age_sec:-NA}"
    vmsg "INFO restart triggered reason=${reason} last_file=${last_file:-NONE} age_sec=${age_sec:-NA}"
    if [[ -x "${STARTUP_SCRIPT}" ]]; then
        "${STARTUP_SCRIPT}" >> "${LOG_FILE}" 2>&1
        log_msg "INFO startup script executed."
        vmsg "INFO startup script executed."
    else
        log_msg "ERROR startup script missing or not executable: ${STARTUP_SCRIPT}"
        vmsg "ERROR startup script missing or not executable: ${STARTUP_SCRIPT}"
    fi

    if [[ "${no_files}" == "true" ]]; then
        count=0
        if [[ -f "${MAIL_COUNT_FILE}" ]]; then
            count=$(cat "${MAIL_COUNT_FILE}" 2>/dev/null || echo 0)
            [[ "${count}" =~ ^[0-9]+$ ]] || count=0
        fi
        if [[ "${count}" -lt 5 ]]; then
            if send_mail; then
                count=$((count + 1))
                echo "${count}" > "${MAIL_COUNT_FILE}"
                log_msg "INFO no-file mail sent count=${count}"
                vmsg "INFO no-file mail sent count=${count}"
            else
                log_msg "WARN no-file mail failed."
                vmsg "WARN no-file mail failed."
            fi
        else
            log_msg "INFO no-file mail suppressed count=${count}"
            vmsg "INFO no-file mail suppressed count=${count}"
        fi
    else
        echo "0" > "${MAIL_COUNT_FILE}"
        vmsg "INFO mail count reset (files present)."
    fi
else
    if [[ "${no_files}" == "false" ]]; then
        echo "0" > "${MAIL_COUNT_FILE}"
        vmsg "OK data present last_file=${last_file} age_sec=${age_sec} threshold_sec=${THRESHOLD_SEC}"
    fi
fi

exit 0
