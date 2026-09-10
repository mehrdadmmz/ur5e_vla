#!/usr/bin/env bash
# Validate/generate all three-cup HDF5 files, then replace each remote subset
# tar with a complete raw+HDF5 archive. Safe to restart.

set -Eeuo pipefail

ROOT=/home/joshua/Desktop/ur5e_vla
TASK=place_three_cups_in_bowls
REPO=mzxuan/real_world_data
ARCHIVE_DIR=/tmp/real_world_data_archives
STATE_DIR="$ROOT/data/$TASK/.hdf5_archive_upload_state"
LOCK_FILE="$STATE_DIR/pipeline.lock"
PYTHON="$ROOT/.venv/bin/python"
HF="$ROOT/.venv/bin/hf"
CONVERTER="$ROOT/convert_ep_hdf5.py"
AUDIT="$ROOT/audit_three_cups_dataset.py"
SUBSETS=(clean d1 d2 d3 d4)
EXPECTED_EPISODES=(100 25 25 25 25)

export HF_XET_HIGH_PERFORMANCE=1
export HF_HUB_ETAG_TIMEOUT=30

timestamp() {
    date '+%Y-%m-%d %H:%M:%S'
}

log() {
    printf '[%s] %s\n' "$(timestamp)" "$*"
}

require_file() {
    if [[ ! -f "$1" ]]; then
        log "ERROR: required file not found: $1"
        exit 1
    fi
}

count_hdf5() {
    find "$ROOT/data/$TASK" -type f -name 'episode*.hdf5' | wc -l
}

temp_signature() {
    find "$ROOT/data/$TASK" -type f -name 'episode*.hdf5.tmp' \
        -printf '%p:%s:%T@\n' | sort
}

source_fingerprint() {
    local source_dir=$1
    find "$source_dir" -type f -printf '%P|%s|%T@\0' \
        | LC_ALL=C sort -z \
        | sha256sum \
        | awk '{print $1}'
}

wait_for_external_conversion() {
    local previous_count=-1
    local stable_checks=0
    local previous_temp_signature=''
    local unchanged_temp_checks=0
    local current_count current_temp_signature temp_count

    while true; do
        current_count=$(count_hdf5)
        current_temp_signature=$(temp_signature)
        if [[ -n "$current_temp_signature" ]]; then
            temp_count=$(printf '%s\n' "$current_temp_signature" | wc -l)
            log "Existing conversion active: $current_count/200 complete, $temp_count temporary file(s)."
            stable_checks=0
            previous_count=$current_count

            if [[ "$current_temp_signature" == "$previous_temp_signature" ]]; then
                unchanged_temp_checks=$((unchanged_temp_checks + 1))
            else
                unchanged_temp_checks=0
            fi
            previous_temp_signature=$current_temp_signature

            # Only remove a ten-minute-stale temporary file when no process
            # has it open. A slow but live writer is always left untouched.
            if (( unchanged_temp_checks >= 20 )); then
                while IFS= read -r -d '' temporary_file; do
                    if lsof -- "$temporary_file" >/dev/null 2>&1; then
                        log "Temporary file is unchanged but still open; continuing to wait: $temporary_file"
                    else
                        log "Removing abandoned converter temporary file: $temporary_file"
                        rm -f -- "$temporary_file"
                    fi
                done < <(find "$ROOT/data/$TASK" -type f -name 'episode*.hdf5.tmp' -print0)
                previous_temp_signature=''
                unchanged_temp_checks=0
            fi
            sleep 30
            continue
        fi

        previous_temp_signature=''
        unchanged_temp_checks=0
        if (( current_count == 200 )); then
            log "Detected all 200 HDF5 files; no competing conversion remains."
            return
        fi
        if (( current_count > 200 )); then
            log "ERROR: found $current_count HDF5 files; expected no more than 200."
            exit 1
        fi

        if (( current_count == previous_count )); then
            stable_checks=$((stable_checks + 1))
        else
            stable_checks=0
        fi
        previous_count=$current_count

        if (( stable_checks >= 3 )); then
            log "No competing conversion activity for 90 seconds; safe to resume locally."
            return
        fi
        log "Waiting for conversion state to stabilize: $current_count/200 complete."
        sleep 30
    done
}

validate_hdf5_counts() {
    local index subset expected data_dir count temporary_count
    local total=0
    for index in "${!SUBSETS[@]}"; do
        subset=${SUBSETS[$index]}
        expected=${EXPECTED_EPISODES[$index]}
        data_dir="$ROOT/data/$TASK/$subset/data"
        count=$(find "$data_dir" -type f -name 'episode*.hdf5' | wc -l)
        temporary_count=$(find "$data_dir" -type f -name 'episode*.hdf5.tmp' | wc -l)
        if (( count != expected || temporary_count != 0 )); then
            log "ERROR: $subset has $count valid-name HDF5 files and $temporary_count temporary files; expected $expected and 0."
            exit 1
        fi
        total=$((total + count))
        log "HDF5 count verified: $subset=$count"
    done
    if (( total != 200 )); then
        log "ERROR: total HDF5 count is $total; expected 200."
        exit 1
    fi
    log "All 200 HDF5 files exist and passed converter validation."
}

remote_metadata() {
    local wanted_path=$1
    local listing="$ARCHIVE_DIR/.${TASK}_remote_listing.json"
    "$HF" datasets list "$REPO" --recursive --format json > "$listing"
    "$PYTHON" - "$listing" "$wanted_path" <<'PY'
import json
import sys

listing_path, wanted = sys.argv[1:]
entries = [entry for entry in json.load(open(listing_path)) if entry.get("path") == wanted]
if len(entries) != 1 or "size" not in entries[0]:
    raise SystemExit(1)
entry = entries[0]
sha256 = entry.get("lfs", {}).get("sha256", "")
print(entry["size"], sha256)
PY
}

build_and_verify_archive() {
    local subset=$1
    local expected=$2
    local subset_root="$ROOT/data/$TASK/$subset"
    local archive="$ARCHIVE_DIR/${TASK}_${subset}.tar"
    local partial="$archive.partial"
    local listing="$archive.list"
    local ready="$archive.ready"
    local source_files source_bytes source_fingerprint_value
    local available_bytes required_bytes
    local archived_files archived_hdf5 archived_sequences archive_bytes checksum

    source_files=$(find "$subset_root/data" -type f | wc -l)
    source_bytes=$(find "$subset_root/data" -type f -printf '%s\n' \
        | awk '{total += $1} END {printf "%.0f", total}')
    source_fingerprint_value=$(source_fingerprint "$subset_root/data")

    if [[ -s "$archive" && -s "$ready" ]]; then
        local ready_files ready_source_bytes ready_fingerprint ready_sha actual_sha
        ready_files=$(awk -F= '$1 == "source_files" {print $2}' "$ready")
        ready_source_bytes=$(awk -F= '$1 == "source_bytes" {print $2}' "$ready")
        ready_fingerprint=$(awk -F= '$1 == "source_fingerprint" {print $2}' "$ready")
        ready_sha=$(awk -F= '$1 == "sha256" {print $2}' "$ready")
        actual_sha=$(sha256sum "$archive" | awk '{print $1}')
        if [[ "$ready_files" == "$source_files" \
              && "$ready_source_bytes" == "$source_bytes" \
              && "$ready_fingerprint" == "$source_fingerprint_value" \
              && "$ready_sha" == "$actual_sha" ]]; then
            log "Reusing verified local archive for $subset."
            return
        fi
        log "Local source changed since $subset archive was built; rebuilding it."
    fi

    rm -f -- "$archive" "$partial" "$listing" "$ready"
    available_bytes=$(df -PB1 "$ARCHIVE_DIR" | awk 'NR == 2 {print $4}')
    required_bytes=$((source_bytes + 5368709120))
    if (( available_bytes < required_bytes )); then
        log "ERROR: $subset archive needs about $required_bytes free bytes; only $available_bytes available."
        exit 1
    fi

    log "Creating complete $subset tar from $source_files files ($source_bytes bytes)."
    tar -C "$subset_root" -cf "$partial" data/
    mv -- "$partial" "$archive"

    log "Verifying $subset tar inventory."
    tar -tf "$archive" > "$listing"
    archived_files=$(awk 'substr($0, length($0), 1) != "/" {count++} END {print count+0}' "$listing")
    archived_hdf5=$(grep -Ec '^data/[0-9]+/episode[0-9]+\.hdf5$' "$listing" || true)
    archived_sequences=$(grep -Ec '^data/[0-9]+/task_sequence\.json$' "$listing" || true)
    archive_bytes=$(stat -c '%s' "$archive")
    if (( archived_files != source_files )); then
        log "ERROR: $subset tar has $archived_files files; source has $source_files."
        exit 1
    fi
    if (( archived_hdf5 != expected )); then
        log "ERROR: $subset tar has $archived_hdf5 HDF5 files; expected $expected."
        exit 1
    fi
    if (( archived_sequences != expected )); then
        log "ERROR: $subset tar has $archived_sequences task_sequence files; expected $expected."
        exit 1
    fi
    checksum=$(sha256sum "$archive" | awk '{print $1}')
    {
        printf 'subset=%s\n' "$subset"
        printf 'source_files=%s\n' "$source_files"
        printf 'source_bytes=%s\n' "$source_bytes"
        printf 'source_fingerprint=%s\n' "$source_fingerprint_value"
        printf 'archive_bytes=%s\n' "$archive_bytes"
        printf 'sha256=%s\n' "$checksum"
    } > "$ready"
    rm -f -- "$listing"
    log "Verified $subset tar: HDF5=$archived_hdf5, task_sequence=$archived_sequences, bytes=$archive_bytes."
}

upload_and_verify_archive() {
    local subset=$1
    local archive="$ARCHIVE_DIR/${TASK}_${subset}.tar"
    local ready="$archive.ready"
    local uploaded="$STATE_DIR/${subset}.uploaded"
    local remote_path="$TASK/$subset/${TASK}_${subset}.tar"
    local local_bytes local_sha source_fingerprint_value
    local remote_info remote_bytes remote_sha attempt

    local_bytes=$(awk -F= '$1 == "archive_bytes" {print $2}' "$ready")
    local_sha=$(awk -F= '$1 == "sha256" {print $2}' "$ready")
    source_fingerprint_value=$(awk -F= '$1 == "source_fingerprint" {print $2}' "$ready")

    for attempt in 1 2 3 4; do
        log "Uploading $subset replacement (attempt $attempt/4) to $remote_path."
        if "$HF" upload "$REPO" "$archive" "$remote_path" \
            --type dataset \
            --commit-message "add HDF5: ${TASK}/${subset}"; then
            break
        fi
        if (( attempt == 4 )); then
            log "ERROR: upload failed four times; verified tar retained at $archive."
            exit 1
        fi
        log "Upload attempt failed; retrying in $((attempt * 15)) seconds."
        sleep $((attempt * 15))
    done

    if ! remote_info=$(remote_metadata "$remote_path"); then
        log "ERROR: uploaded path not found during remote verification; retaining $archive."
        exit 1
    fi
    read -r remote_bytes remote_sha <<< "$remote_info"
    if [[ "$remote_bytes" != "$local_bytes" || "$remote_sha" != "$local_sha" ]]; then
        log "ERROR: remote verification mismatch for $subset; retaining $archive."
        log "Local bytes/SHA: $local_bytes $local_sha"
        log "Remote bytes/SHA: $remote_bytes $remote_sha"
        exit 1
    fi

    {
        printf 'remote_path=%s\n' "$remote_path"
        printf 'bytes=%s\n' "$remote_bytes"
        printf 'sha256=%s\n' "$remote_sha"
        printf 'source_fingerprint=%s\n' "$source_fingerprint_value"
    } > "$uploaded"
    log "Remote size and SHA-256 verified for $subset."
    rm -f -- "$archive" "$ready"
    log "Removed uploaded local $subset tar to free disk space."
}

already_uploaded() {
    local subset=$1
    local uploaded="$STATE_DIR/${subset}.uploaded"
    local remote_path="$TASK/$subset/${TASK}_${subset}.tar"
    local subset_root="$ROOT/data/$TASK/$subset"
    local expected_bytes expected_sha expected_fingerprint current_fingerprint
    local remote_info remote_bytes remote_sha
    [[ -s "$uploaded" ]] || return 1
    expected_bytes=$(awk -F= '$1 == "bytes" {print $2}' "$uploaded")
    expected_sha=$(awk -F= '$1 == "sha256" {print $2}' "$uploaded")
    expected_fingerprint=$(awk -F= '$1 == "source_fingerprint" {print $2}' "$uploaded")
    current_fingerprint=$(source_fingerprint "$subset_root/data")
    [[ "$current_fingerprint" == "$expected_fingerprint" ]] || return 1
    remote_info=$(remote_metadata "$remote_path") || return 1
    read -r remote_bytes remote_sha <<< "$remote_info"
    [[ "$remote_bytes" == "$expected_bytes" && "$remote_sha" == "$expected_sha" ]]
}

main() {
    require_file "$PYTHON"
    require_file "$HF"
    require_file "$CONVERTER"
    require_file "$AUDIT"
    command -v tar >/dev/null
    command -v sha256sum >/dev/null
    command -v flock >/dev/null
    command -v lsof >/dev/null
    mkdir -p "$ARCHIVE_DIR" "$STATE_DIR"
    exec 9>"$LOCK_FILE"
    if ! flock -n 9; then
        log "ERROR: another $TASK archive/upload pipeline is already running."
        exit 1
    fi
    cd "$ROOT"

    log "Authenticated Hugging Face account:"
    "$HF" auth whoami
    log "Waiting for any existing converter to finish or become safely resumable."
    wait_for_external_conversion

    log "Running the resumable converter across all five subsets."
    "$PYTHON" -u "$CONVERTER" \
        --task "$TASK" \
        --subsets clean,d1,d2,d3,d4

    log "Re-running dataset metadata/instruction audit."
    "$PYTHON" "$AUDIT" --expect-rewritten \
        --report "$ROOT/data/$TASK/audit_pre_hdf5_archive_upload.json"
    validate_hdf5_counts

    local index subset expected
    for index in "${!SUBSETS[@]}"; do
        subset=${SUBSETS[$index]}
        expected=${EXPECTED_EPISODES[$index]}
        if already_uploaded "$subset"; then
            log "$subset was already uploaded and still matches its verified remote SHA; skipping."
            continue
        fi
        rm -f -- "$STATE_DIR/${subset}.uploaded"
        build_and_verify_archive "$subset" "$expected"
        upload_and_verify_archive "$subset"
    done

    log "SUCCESS: all five complete raw+HDF5 archives are verified on Hugging Face."
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
