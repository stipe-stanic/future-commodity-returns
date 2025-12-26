#!/bin/bash

# Script to create swap space on a GCE instance
# chmod +x scripts/create_swap.sh
# sudo ./scripts/create_swap.sh 64

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
  echo -e "${GREEN}[INFO]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

check_root() {
  if [[ $EUID -ne 0 ]]; then
    log_error "This script must be run with root priviliges"
    log_info "Used: sudo $0 [swap_size_in_gb]"
    exit 1
  fi
}

check_existing_swap() {
  if swapon --show | grep -q .; then
    log_warning "Existing swap detected:"
    swapon --show
    read -p "Do you want to delete existing swap (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
      log_info "Aborted"
      exit 0
    fi

    swapoff -a
    # Remove swap entry from /etc/fstab
    sed -i '/swap/d' /etc/fstab
  fi
}

get_swap_size() {
  local swap_size=${1:-4} # 4GB default

  if ! [[ "$swap_size" =~ ^[0-9]+$ ]]; then
    log_eror "Invalid swap size: $swap_size"
    log_info "Please specify an integer value"
    exit 1
  fi

  # Check if the size is too big
  if [ "$swap_size" -gt 32 ]; then
    log_warning "A swap size of 32GB or more was specified"
    read -p "Do you want to continue? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 0
    fi
  fi

  echo "$swap_size"
}

check_disk_space() {
  local required_space_gb=$1
  local available_space_kb=$(df / | awk 'NR==2 {print $4}')
  local available_space_gb=$((available_space_kb / 1024 / 1024))

  if [ "$available_space_gb" -lt "$required_space_gb" ]; then
    log_error "Insufficient disk space"
    log_info "Required: ${required_space_gb}GB, Available: ${available_space_gb}GB"
    exit 1
  fi
}

create_swap_file() {
  local swap_size_gb=$1
  local swap_file="/swapfile"

  log info "Creating swap file (${swap_size_gb}GB)..."

  # fallocate
  if command -v fallocate &> /dev/null; then
    fallocate -l ${swap_size_gb}G $swap_file
  else
    # Use dd if fallocate is not available
    dd if=/dev/zero of=$swap_file bs=1G count=$swap_size_gb status=progress
  fi

  chmod 600 $swap_file

  log_info "Initializing swap area"
  mkswap $swap_file

  log_info "Enabling swap"
  swapon $swap_file

  # makes swap file permanent
  log_info "/etc/fstab updating..."
  echo "$swap_file none swap sw 0 0" >> /etc/fstab
}

optimize_swap_settings() {
  log_info "Optimizing swap settings..."

  # Get the current memory size(MB)
  local mem_size_mb=$(free -m | awk 'NR==2 {print $2}')

  # Set swappiness (default:60, recommended:10-30)
  local swappiness=10
  if [ "$mem_size_mb" -lt 4096 ]; then
      swappiness=30
  fi

  echo "vm.swappiness=$swappiness" >> /etc/sysctl.conf

  # vfs_cache_pressure (default:10, recommended:50)
  echo "vm.vfs_cache_pressure=50" >> /etc/sysctl.conf

  # Apply settings instantly
  sysctl -p
}

show_result() {
  log_info "Swap creation complete"
  echo
  echo "=== Swap information ==="
  swapon -show
  echo
  echo "=== Memory usage ==="
  free -h
  echo
  echo "=== Current swap settings ==="
  echo "vm.swappiness = $(cate /proc/sys/vm/swappiness)"
  echo "vm.vfs_cache_pressure = $(cat /proc/sys/vm/vsf_cache_pressure)"
}

main() {
  log_info "Start the GCE swap creation script"

  check_root

  check_existing_swap

  local swap_size_gb=$(get_swap_size "$1")
  log_info "Swap size: ${swap_size_gb}GB"

  check_disk_space $swap_size_gb

  create_swap_file $swap_size_gb

  optimize_swap_settings

  show_result

  log_info "Processing complete"
}

# Run script
main "$@"