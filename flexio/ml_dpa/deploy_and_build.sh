#!/bin/bash
#
# Copies this ml_dpa app into a DOCA "applications" source tree and builds only that app with
# meson/ninja. Run this ON the DPU / DOCA dev host (not on the machine holding this repo checkout
# unless it is the same one).
#
# Usage:
#   ./deploy_and_build.sh [SRC_DIR] [DOCA_APPS_DIR] [BUILD_DIR]
#
#   SRC_DIR       - path to this ml_dpa folder (default: directory this script lives in)
#   DOCA_APPS_DIR - path to the DOCA applications source tree (default: /opt/mellanox/doca/applications)
#   BUILD_DIR     - meson build output dir (default: /tmp/build)
#
# What it does:
#   1. Copies SRC_DIR -> DOCA_APPS_DIR/ml_dpa (backing up any existing ml_dpa dir first).
#   2. Adds an 'enable_ml_dpa' boolean option to DOCA_APPS_DIR/meson_options.txt (cloned from the
#      existing 'enable_pcc' option block, whatever its line-wrapping), and adds 'ml_dpa' to the
#      app_list array in DOCA_APPS_DIR/meson.build (cloned from the 'pcc' entry). Both files are
#      backed up (*.bak) before editing, and the script is idempotent -- if the entries already
#      exist it leaves them alone.
#   3. Runs: meson setup BUILD_DIR --reconfigure -Denable_all_applications=false -Denable_ml_dpa=true
#            ninja -C BUILD_DIR
#
# This does NOT run on Windows: it needs the real DOCA SDK / DPACC toolchain, which only exists
# on a Linux host or BlueField DPU with DOCA installed.

set -e -u -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${1:-$SCRIPT_DIR}"
DOCA_APPS_DIR="${2:-/opt/mellanox/doca/applications}"
BUILD_DIR="${3:-/tmp/build}"
APP_NAME="ml_dpa"
DEST_DIR="${DOCA_APPS_DIR}/${APP_NAME}"
OPTIONS_FILE="${DOCA_APPS_DIR}/meson_options.txt"
BUILD_FILE="${DOCA_APPS_DIR}/meson.build"

echo "== ml_dpa deploy_and_build =="
echo "SRC_DIR       = ${SRC_DIR}"
echo "DOCA_APPS_DIR = ${DOCA_APPS_DIR}"
echo "DEST_DIR      = ${DEST_DIR}"
echo "BUILD_DIR     = ${BUILD_DIR}"
echo

if [ ! -d "${SRC_DIR}/host" ] || [ ! -d "${SRC_DIR}/device" ]; then
	echo "ERROR: ${SRC_DIR} doesn't look like the ml_dpa app folder (missing host/ or device/)." >&2
	exit 1
fi

if [ ! -f "${BUILD_FILE}" ] || [ ! -f "${OPTIONS_FILE}" ]; then
	echo "ERROR: ${BUILD_FILE} or ${OPTIONS_FILE} not found. Is DOCA_APPS_DIR correct?" >&2
	echo "       (pass it as the 2nd argument, e.g. ./deploy_and_build.sh \"${SRC_DIR}\" /path/to/doca/applications)" >&2
	exit 1
fi

##############################
## 1. Copy the app in place ##
##############################

if [ -e "${DEST_DIR}" ]; then
	BACKUP="${DEST_DIR}.bak.$(date +%s)"
	echo "-- ${DEST_DIR} already exists, moving it to ${BACKUP}"
	mv "${DEST_DIR}" "${BACKUP}"
fi

echo "-- Copying ${SRC_DIR} -> ${DEST_DIR}"
mkdir -p "${DEST_DIR}"
cp -r "${SRC_DIR}/." "${DEST_DIR}/"
# Don't ship this deploy script into the DOCA tree copy.
rm -f "${DEST_DIR}/deploy_and_build.sh"

chmod +x "${DEST_DIR}/build_device_code.sh"

###############################################################
## 2. Wire the app into meson_options.txt / meson.build      ##
## (clone the existing 'pcc' entries, handling that the      ##
## option() call in meson_options.txt spans two lines)       ##
###############################################################

if grep -q "enable_${APP_NAME}" "${OPTIONS_FILE}"; then
	echo "-- ${OPTIONS_FILE} already has an enable_${APP_NAME} option, leaving it alone."
else
	start_line="$(grep -n "option('enable_pcc'" "${OPTIONS_FILE}" | head -1 | cut -d: -f1)"
	if [ -z "${start_line}" ]; then
		echo "ERROR: couldn't find \"option('enable_pcc', ...)\" in ${OPTIONS_FILE}." >&2
		echo "       Add an 'enable_${APP_NAME}' boolean option there yourself, modeled on the pcc one." >&2
		exit 1
	fi
	# The option('enable_pcc', ...) call may wrap onto a following line; find where it closes.
	close_line="$(awk -v start="${start_line}" 'NR>=start && /\)/{print NR; exit}' "${OPTIONS_FILE}")"

	cp "${OPTIONS_FILE}" "${OPTIONS_FILE}.bak"
	tmpfile="$(mktemp)"
	cat > "${tmpfile}" <<EOF
option('enable_${APP_NAME}', type: 'boolean', value: false,
        description: 'Enable ML DPA application.')
EOF
	sed -i "${close_line}r ${tmpfile}" "${OPTIONS_FILE}"
	rm -f "${tmpfile}"

	if grep -q "enable_${APP_NAME}" "${OPTIONS_FILE}"; then
		echo "-- Added enable_${APP_NAME} option to ${OPTIONS_FILE} (backup: ${OPTIONS_FILE}.bak)"
	else
		echo "ERROR: patch didn't take, restoring ${OPTIONS_FILE} from backup." >&2
		cp "${OPTIONS_FILE}.bak" "${OPTIONS_FILE}"
		exit 1
	fi
fi

if grep -Eq "['\"]${APP_NAME}['\"]" "${BUILD_FILE}"; then
	echo "-- ${BUILD_FILE} already references '${APP_NAME}', leaving it alone."
else
	pcc_line_num="$(grep -n "^[[:space:]]*'pcc',[[:space:]]*$" "${BUILD_FILE}" | head -1 | cut -d: -f1)"
	if [ -z "${pcc_line_num}" ]; then
		echo "ERROR: couldn't find a \"'pcc',\" entry in the app_list array in ${BUILD_FILE}." >&2
		echo "       Add a '${APP_NAME}', entry to app_list yourself, modeled on the pcc one." >&2
		exit 1
	fi
	pcc_line_text="$(sed -n "${pcc_line_num}p" "${BUILD_FILE}")"
	new_line="$(echo "${pcc_line_text}" | sed "s/'pcc'/'${APP_NAME}'/")"

	cp "${BUILD_FILE}" "${BUILD_FILE}.bak"
	tmpfile2="$(mktemp)"
	printf '%s\n' "${new_line}" > "${tmpfile2}"
	sed -i "${pcc_line_num}r ${tmpfile2}" "${BUILD_FILE}"
	rm -f "${tmpfile2}"

	if grep -Eq "['\"]${APP_NAME}['\"]" "${BUILD_FILE}"; then
		echo "-- Added '${APP_NAME}' to app_list in ${BUILD_FILE} (backup: ${BUILD_FILE}.bak)"
	else
		echo "ERROR: patch didn't take, restoring ${BUILD_FILE} from backup." >&2
		cp "${BUILD_FILE}.bak" "${BUILD_FILE}"
		exit 1
	fi
fi

##############################
## 3. Configure and build   ##
##############################

RECONFIGURE=""
if [ -d "${BUILD_DIR}" ]; then
	RECONFIGURE="--reconfigure"
fi

echo
echo "-- (cd ${DOCA_APPS_DIR} && meson setup ${BUILD_DIR} ${RECONFIGURE} -Denable_all_applications=false -Denable_${APP_NAME}=true)"
(cd "${DOCA_APPS_DIR}" && meson setup "${BUILD_DIR}" ${RECONFIGURE} -Denable_all_applications=false -Denable_${APP_NAME}=true)

echo "-- ninja -C ${BUILD_DIR}"
ninja -C "${BUILD_DIR}"

echo
echo "*** Build finished. Look for the '${APP_NAME}' binary under ${BUILD_DIR}/${APP_NAME}/ ***"
