#!/bin/bash

# NeuralBlitz v50.0 - Backup and Restore Script
# File: backup_restore.sh
# Description: Backup and restore system for the Omega Singularity Architecture

set -e

# Configuration
BACKUP_DIR="/var/backups/neuralblitz"
DB_NAME="neuralblitz_v50"
DB_USER="neuralblitz"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_NAME="neuralblitz_backup_${TIMESTAMP}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Function to show usage
show_usage() {
    echo "NeuralBlitz v50.0 Backup and Restore Script"
    echo ""
    echo "Usage: $0 COMMAND [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  backup          Create full system backup"
    echo "  restore         Restore from backup"
    echo "  list            List available backups"
    echo "  clean           Remove old backups"
    echo ""
    echo "Options:"
    echo "  -d, --dir DIR     Backup directory (default: /var/backups/neuralblitz)"
    echo "  -b, --backup ID   Backup ID to restore"
    echo "  -k, --keep N     Keep last N backups when cleaning"
    echo "  -h, --help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 backup                         # Create backup"
    echo "  $0 restore -b backup_20240203_120000  # Restore specific backup"
    echo "  $0 clean -k 5                      # Keep last 5 backups"
}

# Function to create backup
create_backup() {
    local backup_path="${BACKUP_DIR}/${BACKUP_NAME}"
    
    print_info "Creating backup: $BACKUP_NAME"
    
    # Create backup directory
    mkdir -p "$backup_path"
    
    # Backup database
    print_info "Backing up database..."
    mysqldump -u "$DB_USER" -p "$DB_NAME" > "${backup_path}/${DB_NAME}.sql"
    
    # Backup configuration files
    print_info "Backing up configurations..."
    mkdir -p "${backup_path}/config"
    cp -r k8s/ "${backup_path}/config/" 2>/dev/null || true
    cp -r docker/ "${backup_path}/config/" 2>/dev/null || true
    cp *.yaml "${backup_path}/config/" 2>/dev/null || true
    
    # Backup deployment scripts
    print_info "Backing up deployment scripts..."
    mkdir -p "${backup_path}/scripts"
    cp -r scripts/ "${backup_path}/scripts/"
    
    # Create metadata
    print_info "Creating backup metadata..."
    cat > "${backup_path}/metadata.json" << EOF
{
    "name": "$BACKUP_NAME",
    "timestamp": "$(date -Iseconds)",
    "version": "50.0.0",
    "architecture": "Omega Singularity (OSA v2.0)",
    "golden_dag_seed": "a8d0f2a4c6b8d0f2a4c6b8d0f2a4c6b8d0f2a4c6b8d0f2a4c6b8d0",
    "contents": {
        "database": "${DB_NAME}.sql",
        "kubernetes": "config/k8s/",
        "docker": "config/docker/",
        "scripts": "scripts/"
    },
    "checksum": "$(find "$backup_path" -type f -exec sha256sum {} + | sort | sha256sum | cut -d' ' -f1)"
}
EOF
    
    # Compress backup
    print_info "Compressing backup..."
    cd "$BACKUP_DIR"
    tar -czf "${BACKUP_NAME}.tar.gz" "$BACKUP_NAME"
    rm -rf "$BACKUP_NAME"
    
    # Verify backup
    print_info "Verifying backup..."
    if [[ -f "${BACKUP_DIR}/${BACKUP_NAME}.tar.gz" ]]; then
        local size=$(du -h "${BACKUP_DIR}/${BACKUP_NAME}.tar.gz" | cut -f1)
        print_success "Backup created successfully: ${BACKUP_NAME}.tar.gz ($size)"
    else
        print_error "Backup creation failed"
        exit 1
    fi
}

# Function to restore backup
restore_backup() {
    local backup_id="$1"
    local backup_file="${BACKUP_DIR}/${backup_id}.tar.gz"
    
    if [[ -z "$backup_id" ]]; then
        print_error "Backup ID required for restore"
        exit 1
    fi
    
    if [[ ! -f "$backup_file" ]]; then
        print_error "Backup file not found: $backup_file"
        exit 1
    fi
    
    print_info "Restoring from backup: $backup_id"
    
    # Extract backup
    local temp_dir="/tmp/neuralblitz_restore_${TIMESTAMP}"
    mkdir -p "$temp_dir"
    
    print_info "Extracting backup..."
    tar -xzf "$backup_file" -C "$temp_dir"
    
    # Verify metadata
    local extracted_dir="${temp_dir}/${backup_id}"
    if [[ ! -f "${extracted_dir}/metadata.json" ]]; then
        print_error "Invalid backup: metadata missing"
        rm -rf "$temp_dir"
        exit 1
    fi
    
    # Show backup info
    print_info "Backup information:"
    grep -E '"(timestamp|version|architecture)"' "${extracted_dir}/metadata.json" | sed 's/^[[:space:]]*/  /'
    
    # Restore database
    if [[ -f "${extracted_dir}/${DB_NAME}.sql" ]]; then
        print_info "Restoring database..."
        mysql -u "$DB_USER" -p "$DB_NAME" < "${extracted_dir}/${DB_NAME}.sql"
        print_success "Database restored"
    fi
    
    # Restore configurations
    if [[ -d "${extracted_dir}/config" ]]; then
        print_info "Restoring configurations..."
        if [[ -d "${extracted_dir}/config/k8s" ]]; then
            cp -r "${extracted_dir}/config/k8s/"* ./k8s/ 2>/dev/null || true
        fi
        if [[ -d "${extracted_dir}/config/docker" ]]; then
            cp -r "${extracted_dir}/config/docker/"* ./docker/ 2>/dev/null || true
        fi
        print_success "Configurations restored"
    fi
    
    # Restore scripts
    if [[ -d "${extracted_dir}/scripts" ]]; then
        print_info "Restoring scripts..."
        cp -r "${extracted_dir}/scripts/"* ./scripts/ 2>/dev/null || true
        print_success "Scripts restored"
    fi
    
    # Cleanup
    rm -rf "$temp_dir"
    
    print_success "Restore completed successfully"
}

# Function to list backups
list_backups() {
    print_info "Available backups in $BACKUP_DIR:"
    echo ""
    
    if [[ ! -d "$BACKUP_DIR" ]]; then
        print_warning "Backup directory does not exist"
        return
    fi
    
    local count=0
    for backup_file in "$BACKUP_DIR"/*.tar.gz; do
        if [[ -f "$backup_file" ]]; then
            local basename=$(basename "$backup_file" .tar.gz)
            local size=$(du -h "$backup_file" | cut -f1)
            local date=$(stat -c %y "$backup_file" | cut -d' ' -f1,2)
            
            echo "  $basename"
            echo "    Size: $size"
            echo "    Date: $date"
            echo ""
            ((count++))
        fi
    done
    
    if [[ $count -eq 0 ]]; then
        print_warning "No backups found"
    else
        print_success "Found $count backup(s)"
    fi
}

# Function to clean old backups
clean_backups() {
    local keep_count="$1"
    
    if [[ -z "$keep_count" ]]; then
        keep_count=5
    fi
    
    print_info "Cleaning old backups (keeping last $keep_count)..."
    
    # Count and sort backups by date
    local backup_count=0
    local temp_list="/tmp/backup_list_${TIMESTAMP}"
    
    for backup_file in "$BACKUP_DIR"/*.tar.gz; do
        if [[ -f "$backup_file" ]]; then
            echo "$(stat -c %Y "$backup_file") $backup_file" >> "$temp_list"
            ((backup_count++))
        fi
    done
    
    if [[ $backup_count -le $keep_count ]]; then
        print_success "No cleanup needed (only $backup_count backups, keeping $keep_count)"
        rm -f "$temp_list"
        return
    fi
    
    # Remove oldest backups
    local to_delete=$((backup_count - keep_count))
    print_info "Deleting $to_delete oldest backup(s)..."
    
    sort -n "$temp_list" | head -n "$to_delete" | while read -r timestamp backup_file; do
        local basename=$(basename "$backup_file")
        print_info "Deleting: $basename"
        rm -f "$backup_file"
    done
    
    rm -f "$temp_list"
    print_success "Cleanup completed"
}

# Function to create backup directory
ensure_backup_dir() {
    if [[ ! -d "$BACKUP_DIR" ]]; then
        print_info "Creating backup directory: $BACKUP_DIR"
        mkdir -p "$BACKUP_DIR"
    fi
}

# Main execution
main() {
    local command=""
    local backup_id=""
    local keep_count=""
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            backup|restore|list|clean)
                command="$1"
                shift
                ;;
            -d|--dir)
                BACKUP_DIR="$2"
                shift 2
                ;;
            -b|--backup)
                backup_id="$2"
                shift 2
                ;;
            -k|--keep)
                keep_count="$2"
                shift 2
                ;;
            -h|--help)
                show_usage
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    # Validate command
    if [[ -z "$command" ]]; then
        print_error "No command specified"
        show_usage
        exit 1
    fi
    
    # Show banner
    echo "NeuralBlitz v50.0 - Omega Singularity Backup & Restore"
    echo "=================================================="
    
    # Execute command
    case $command in
        backup)
            ensure_backup_dir
            create_backup
            ;;
        restore)
            ensure_backup_dir
            restore_backup "$backup_id"
            ;;
        list)
            list_backups
            ;;
        clean)
            ensure_backup_dir
            clean_backups "$keep_count"
            ;;
        *)
            print_error "Unknown command: $command"
            show_usage
            exit 1
            ;;
    esac
}

# Check dependencies
if ! command -v mysqldump &> /dev/null; then
    print_error "mysqldump is required for database backups"
    exit 1
fi

if ! command -v mysql &> /dev/null; then
    print_error "mysql is required for database restores"
    exit 1
fi

# Run main function
main "$@"