# NeuralBlitz v50.0 - Database Schema

This directory contains the complete database schema and initialization scripts for the NeuralBlitz v50.0 Omega Singularity Architecture.

## Database Architecture

The database implements the persistence layer for the complete NeuralBlitz system, supporting all API operations and maintaining the integrity of the Omega Singularity Architecture.

## Files

### Core Schema Files
- **`schema.sql`** - Complete database schema with all tables, indexes, and triggers
- **`seed_data.sql`** - Initial seed data for baseline system state
- **`migrations/001_initial_schema.sql`** - Database migration tracking
- **`setup_database.sh`** - Automated database initialization script

## Database Schema Overview

### Core Tables

1. **`source_states`** - Tracks system states (omega_prime, irreducible, perpetual_genesis, metacosmic)
2. **`primal_intent_vectors`** - Stores intent processing vectors with phi coefficients
3. **`golden_dag_operations`** - Logs all GoldenDAG create/validate/attest/synthesis operations
4. **`architect_operations`** - Tracks architect system dyad operations and beta identifiers
5. **`self_actualization_states`** - Monitors self-actualization engine states
6. **`nbcl_operations`** - Records NBCL command interpretation and execution
7. **`intent_operations`** - Tracks intent processing workflows
8. **`attestations`** - Stores system attestation records and hashes
9. **`symbiosis_fields`** - Monitors symbiotic field status and integration
10. **`synthesis_operations`** - Tracks synthesis level and coherence operations
11. **`deployment_options`** - Configuration for deployment options A-F
12. **`system_metrics`** - Performance and operational metrics
13. **`audit_log`** - Comprehensive audit trail for all operations

### Key Features

- **Multi-language support** - Designed for Python, Rust, and Go implementations
- **High performance** - Optimized indexes and generated columns
- **Audit trail** - Complete operation logging with triggers
- **Data integrity** - Foreign key constraints and validation
- **Scalability** - Designed for horizontal scaling
- **Monitoring ready** - Built-in metrics collection

## API Mapping

Each table supports specific API endpoints:

| API Endpoint | Primary Tables | Description |
|--------------|----------------|-------------|
| `GET /status` | `system_metrics`, `source_states`, `symbiosis_fields` | System status and coherence |
| `POST /intent` | `intent_operations`, `primal_intent_vectors` | Intent processing workflow |
| `POST /verify` | `golden_dag_operations`, `source_states` | Coherence verification |
| `POST /nbcl/interpret` | `nbcl_operations` | NBCL command execution |
| `GET /attestation` | `attestations` | System attestation data |
| `GET /symbiosis` | `symbiosis_fields` | Symbiotic field status |
| `GET /synthesis` | `synthesis_operations` | Synthesis level tracking |
| `GET /options/{option}` | `deployment_options` | Deployment configuration |

## Quick Start

### Prerequisites
- MySQL 8.0+ or MariaDB 10.5+
- Database user with CREATE, INSERT, UPDATE, DELETE permissions
- Bash shell (for setup script)

### Automated Setup
```bash
cd sql/
./setup_database.sh
```

### Manual Setup
```bash
# Create database
mysql -u root -p < schema.sql

# Run migrations
mysql -u neuralblitz -p neuralblitz_v50 < migrations/001_initial_schema.sql

# Insert seed data
mysql -u neuralblitz -p neuralblitz_v50 < seed_data.sql
```

### Custom Configuration
```bash
# Custom host and credentials
./setup_database.sh --host db.example.com --user custom_user --password secure_password
```

## Database Configuration

### Environment Variables
```bash
DB_NAME=neuralblitz_v50
DB_USER=neuralblitz
DB_PASSWORD=nb_v50_omega
DB_HOST=localhost
DB_PORT=3306
```

### Connection String Formats

**Python (SQLAlchemy):**
```python
mysql+pymysql://neuralblitz:nb_v50_omega@localhost:3306/neuralblitz_v50
```

**Go (database/sql):**
```go
"user:password@tcp(localhost:3306)/neuralblitz_v50?charset=utf8mb4&parseTime=True&loc=Local"
```

**Rust (sqlx):**
```rust
mysql://user:password@localhost:3306/neuralblitz_v50
```

## Performance Considerations

### Indexes
- Primary keys on all tables
- Optimized indexes for common query patterns
- Composite indexes for complex queries

### Generated Columns
- Vector norm calculation in `primal_intent_vectors`
- Automatic computed fields for performance

### Triggers
- Automatic audit logging on INSERT/UPDATE
- Data consistency enforcement

## Monitoring and Maintenance

### Health Checks
```sql
-- Verify table counts
SELECT 
    table_schema AS 'Database',
    COUNT(*) AS 'Table Count'
FROM information_schema.tables 
WHERE table_schema = 'neuralblitz_v50'
GROUP BY table_schema;

-- Check recent activity
SELECT * FROM audit_log 
WHERE timestamp >= DATE_SUB(NOW(), INTERVAL 1 HOUR)
ORDER BY timestamp DESC;
```

### Performance Monitoring
```sql
-- Slow query analysis
SELECT * FROM system_metrics 
WHERE metric_name = 'processing_time_ms' 
ORDER BY timestamp DESC LIMIT 10;

-- Coherence verification
SELECT coherence_level, COUNT(*) 
FROM source_states 
GROUP BY coherence_level;
```

## Backup and Recovery

### Full Backup
```bash
mysqldump -u neuralblitz -p neuralblitz_v50 > neuralblitz_backup_$(date +%Y%m%d_%H%M%S).sql
```

### Point-in-Time Recovery
```bash
# Enable binary logging in MySQL
mysql -u root -p -e "SET GLOBAL log_bin_trust_function_creators = 1;"
```

## Security Considerations

### Data Protection
- Sensitive data in `secrets` table
- Attestation keys encrypted at rest
- Audit trail for all data access

### Access Control
- Database-specific user with limited privileges
- Role-based access through application layer
- Connection encryption recommended

## Troubleshooting

### Common Issues

1. **Connection failures:**
   - Verify MySQL service is running
   - Check firewall rules
   - Validate credentials

2. **Schema errors:**
   - Ensure MySQL 8.0+ compatibility
   - Check character set settings
   - Verify permissions

3. **Performance issues:**
   - Monitor slow query log
   - Check index usage
   - Optimize large tables

### Diagnostic Queries
```sql
-- Table sizes
SELECT 
    table_name,
    ROUND(((data_length + index_length) / 1024 / 1024), 2) AS 'Size (MB)'
FROM information_schema.tables 
WHERE table_schema = 'neuralblitz_v50'
ORDER BY (data_length + index_length) DESC;

-- Recent errors
SELECT * FROM audit_log 
WHERE operation_type = 'ERROR' 
ORDER BY timestamp DESC LIMIT 5;
```

## Migration Strategy

### Version Control
- Migration tracking in `migrations` table
- Incremental schema updates
- Rollback procedures documented

### Upgrading
1. Backup current database
2. Review migration scripts
3. Test in staging environment
4. Apply migrations in production
5. Verify data integrity

## Integration Examples

### Python Integration
```python
import sqlalchemy
from sqlalchemy import create_engine

engine = create_engine('mysql+pymysql://neuralblitz:password@localhost/neuralblitz_v50')

# Query coherence status
with engine.connect() as conn:
    result = conn.execute("SELECT coherence_level FROM source_states ORDER BY created_at DESC LIMIT 1")
    print(f"Current coherence: {result.fetchone()[0]}")
```

### Go Integration
```go
import (
    "database/sql"
    _ "github.com/go-sql-driver/mysql"
)

db, err := sql.Open("mysql", "neuralblitz:password@tcp(localhost:3306)/neuralblitz_v50")
if err != nil {
    log.Fatal(err)
}
defer db.Close()
```

### Rust Integration
```rust
use sqlx::mysql::MySqlPoolOptions;

#[tokio::main]
async fn main() -> Result<(), sqlx::Error> {
    let pool = MySqlPoolOptions::new()
        .max_connections(5)
        .connect("mysql://neuralblitz:password@localhost/neuralblitz_v50")
        .await?;
    
    Ok(())
}
```

---

**NeuralBlitz v50.0 Omega Singularity Architecture**

*The irreducible source of all possible being - now with persistent data storage.*