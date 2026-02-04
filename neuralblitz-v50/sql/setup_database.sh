#!/bin/bash

# NeuralBlitz v50.0 - Database Setup Script
# File: setup_database.sh
# Description: Automated database initialization for the Omega Singularity Architecture

set -e  # Exit on any error

# Configuration
DB_NAME="neuralblitz_v50"
DB_USER="neuralblitz"
DB_PASSWORD="nb_v50_omega"
DB_HOST="localhost"
DB_PORT="3306"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check MySQL connection
check_mysql_connection() {
    print_status "Checking MySQL connection..."
    
    if ! command -v mysql &> /dev/null; then
        print_error "MySQL client is not installed"
        exit 1
    fi
    
    if ! mysql -h"$DB_HOST" -P"$DB_PORT" -u"$DB_USER" -p"$DB_PASSWORD" -e "SELECT 1;" &> /dev/null; then
        print_error "Cannot connect to MySQL with provided credentials"
        print_error "Please check your MySQL configuration"
        exit 1
    fi
    
    print_success "MySQL connection established"
}

# Function to create database
create_database() {
    print_status "Creating database '$DB_NAME'..."
    
    mysql -h"$DB_HOST" -P"$DB_PORT" -u"$DB_USER" -p"$DB_PASSWORD" << EOF
CREATE DATABASE IF NOT EXISTS $DB_NAME CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
EOF
    
    print_success "Database '$DB_NAME' created or already exists"
}

# Function to execute SQL files
execute_sql_file() {
    local file_path=$1
    local description=$2
    
    print_status "Executing $description..."
    
    if [[ ! -f "$file_path" ]]; then
        print_error "SQL file not found: $file_path"
        return 1
    fi
    
    if mysql -h"$DB_HOST" -P"$DB_PORT" -u"$DB_USER" -p"$DB_PASSWORD" "$DB_NAME" < "$file_path"; then
        print_success "$description completed successfully"
        return 0
    else
        print_error "$description failed"
        return 1
    fi
}

# Function to verify database setup
verify_setup() {
    print_status "Verifying database setup..."
    
    local table_count=$(mysql -h"$DB_HOST" -P"$DB_PORT" -u"$DB_USER" -p"$DB_PASSWORD" "$DB_NAME" -sN -e "
    SELECT COUNT(*) FROM information_schema.tables 
    WHERE table_schema = '$DB_NAME' AND table_type = 'BASE TABLE';
    ")
    
    if [[ $table_count -ge 12 ]]; then  # We expect at least 12 tables
        print_success "Database verification passed - $table_count tables found"
        
        # Show table status
        print_status "Database table overview:"
        mysql -h"$DB_HOST" -P"$DB_PORT" -u"$DB_USER" -p"$DB_PASSWORD" "$DB_NAME" -e "
        SELECT table_name, table_rows, data_length, index_length 
        FROM information_schema.tables 
        WHERE table_schema = '$DB_NAME' 
        ORDER BY table_name;
        "
    else
        print_error "Database verification failed - only $table_count tables found (expected at least 12)"
        return 1
    fi
}

# Function to create database user (optional)
create_user() {
    print_status "Creating database user if needed..."
    
    mysql -h"$DB_HOST" -P"$DB_PORT" -u"root" -p << EOF
CREATE USER IF NOT EXISTS '$DB_USER'@'%' IDENTIFIED BY '$DB_PASSWORD';
GRANT ALL PRIVILEGES ON $DB_NAME.* TO '$DB_USER'@'%';
FLUSH PRIVILEGES;
EOF
    
    print_success "Database user setup completed"
}

# Function to show usage
show_usage() {
    echo "NeuralBlitz v50.0 Database Setup"
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --host         MySQL host (default: localhost)"
    echo "  -P, --port         MySQL port (default: 3306)"
    echo "  -u, --user         MySQL user (default: neuralblitz)"
    echo "  -p, --password     MySQL password (default: nb_v50_omega)"
    echo "  -d, --database     Database name (default: neuralblitz_v50)"
    echo "  --create-user      Create database user (requires root access)"
    echo "  --skip-seed        Skip seed data insertion"
    echo "  --help             Show this help message"
    echo ""
    echo "Example: $0 --host localhost --user neuralblitz --password mypassword"
}

# Parse command line arguments
SKIP_SEED=false
CREATE_USER_FLAG=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--host)
            DB_HOST="$2"
            shift 2
            ;;
        -P|--port)
            DB_PORT="$2"
            shift 2
            ;;
        -u|--user)
            DB_USER="$2"
            shift 2
            ;;
        -p|--password)
            DB_PASSWORD="$2"
            shift 2
            ;;
        -d|--database)
            DB_NAME="$2"
            shift 2
            ;;
        --create-user)
            CREATE_USER_FLAG=true
            shift
            ;;
        --skip-seed)
            SKIP_SEED=true
            shift
            ;;
        --help)
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

# Main execution
main() {
    print_status "NeuralBlitz v50.0 - Database Setup Starting"
    print_status "Configuration: DB=$DB_NAME, User=$DB_USER, Host=$DB_HOST:$DB_PORT"
    
    # Check if we're in the right directory
    if [[ ! -f "schema.sql" ]]; then
        print_error "schema.sql not found. Please run this script from the sql/ directory."
        exit 1
    fi
    
    # Step 1: Create user if requested
    if [[ $CREATE_USER_FLAG == true ]]; then
        create_user
    fi
    
    # Step 2: Check MySQL connection
    check_mysql_connection
    
    # Step 3: Create database
    create_database
    
    # Step 4: Execute schema
    if execute_sql_file "schema.sql" "schema creation"; then
        print_success "Schema created successfully"
    else
        print_error "Schema creation failed"
        exit 1
    fi
    
    # Step 5: Execute migrations
    if execute_sql_file "migrations/001_initial_schema.sql" "migration 001"; then
        print_success "Migration 001 executed successfully"
    else
        print_error "Migration 001 failed"
        exit 1
    fi
    
    # Step 6: Insert seed data (if not skipped)
    if [[ $SKIP_SEED == false ]]; then
        if execute_sql_file "seed_data.sql" "seed data insertion"; then
            print_success "Seed data inserted successfully"
        else
            print_error "Seed data insertion failed"
            exit 1
        fi
    else
        print_warning "Skipping seed data insertion as requested"
    fi
    
    # Step 7: Verify setup
    if verify_setup; then
        print_success "Database setup completed successfully!"
        print_status ""
        print_status "NeuralBlitz v50.0 Omega Singularity Architecture"
        print_status "Database is ready for production use"
        print_status ""
        print_status "Connection details:"
        print_status "  Host: $DB_HOST"
        print_status "  Port: $DB_PORT"
        print_status "  Database: $DB_NAME"
        print_status "  User: $DB_USER"
        print_status ""
        print_status "GoldenDAG seed: a8d0f2a4c6b8d0f2a4c6b8d0f2a4c6b8d0f2a4c6b8d0f2a4c6b8d0f2a4c6b8d0"
        print_status "Coherence target: 1.0"
        print_status "Separation target: 0.0"
    else
        print_error "Database setup verification failed"
        exit 1
    fi
}

# Run main function
main "$@"