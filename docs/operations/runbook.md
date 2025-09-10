# Operations Runbook

## Overview

This runbook provides operational procedures for managing the Dataset Generator service in production and development environments. The system consists of containerized services orchestrated via Docker Compose with GPU support for LLM inference.

## System Architecture

**Core Services:**
- `synthetic-data-service` - FastAPI application for dataset generation
- `ollama` - GPU-enabled LLM inference service
- `mlflow` - Experiment tracking (optional)

**Key Volumes:**
- `ollama_models` - Persistent model storage
- `./output_datasets` - Generated dataset outputs
- `./logs` - Application logs
- `./config`, `./templates`, `./user_configs` - Configuration and templates

## Service Management

### Starting Services

**Development Environment:**
```bash
# Clone repository and navigate to project root
git clone https://github.com/rootcodelabs/Dataset-Generator.git
cd Dataset-Generator

# Start all services in detached mode
docker compose up -d

# Verify services are running
docker compose ps
```

**Production Environment:**
```bash
# Pull latest images
docker compose pull

# Start services with restart policies
docker compose up -d --remove-orphans

# Check service status
docker compose ps
```

### Stopping Services

**Graceful Shutdown:**
```bash
# Stop all services (keeps containers)
docker compose stop

# Stop and remove containers (preserves volumes)
docker compose down

# Stop and remove everything including volumes (DATA LOSS WARNING)
docker compose down --volumes
```

**Emergency Stop:**
```bash
# Force stop all containers
docker kill $(docker ps -q --filter "name=dataset-gen")

# Clean up resources
docker compose down --remove-orphans
```

### Service Restart

**Individual Service Restart:**
```bash
# Restart API service only
docker compose restart synthetic-data-service

# Restart Ollama service only
docker compose restart ollama

# Restart with rebuild
docker compose up -d --build synthetic-data-service
```

**Full System Restart:**
```bash
# Restart all services
docker compose restart

# Restart with latest configurations
docker compose down && docker compose up -d
```

## Health Checks

### API Service Health

**Basic Health Check:**
```bash
# Check API responsiveness
curl -f http://localhost:8000/health

# Expected response:
# {"status": "healthy", "timestamp": "2025-08-10T12:34:56Z"}
```

**Detailed API Status:**
```bash
# Check all endpoints
curl http://localhost:8000/docs  # Swagger UI
curl http://localhost:8000/tasks # Background tasks
curl http://localhost:8000/metrics # Service metrics (if enabled)
```

**Container Health:**
```bash
# Check container status
docker compose ps synthetic-data-service

# View container logs
docker compose logs synthetic-data-service -f

# Execute health check inside container
docker exec synthetic-data-service curl -f localhost:8000/health
```

### Ollama Service Health

**Ollama API Health:**
```bash
# Check Ollama API
curl http://localhost:11434/api/version

# List available models
curl http://localhost:11434/api/tags

# Check specific model status
curl -X POST http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{"model": "gemma3:1b-it-qat", "prompt": "test", "stream": false}'
```

**GPU and Container Status:**
```bash
# Check GPU access in Ollama container
docker exec ollama nvidia-smi

# Check Ollama process status
docker exec ollama ps aux | grep ollama

# View Ollama logs
docker compose logs ollama -f
```

### End-to-End Health Check

```bash
# Test complete generation pipeline
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_structure": "single_question",
    "prompt_template": "institute_topic_question",
    "num_samples": 1,
    "language": "et"
  }'
```

## Model Management

### Ollama Model Operations

**List Available Models:**
```bash
# Inside Ollama container
docker exec ollama ollama list

# Via API
curl http://localhost:11434/api/tags
```

**Pull New Models:**
```bash
# Pull specific model
docker exec ollama ollama pull gemma3:12b-it-qat

# Pull model with environment variable override
docker exec -e MODEL_NAME=gemma3:7b ollama ollama pull $MODEL_NAME
```

**Remove Models:**
```bash
# Remove specific model
docker exec ollama ollama rm gemma3:1b-it-qat

# Remove unused models
docker exec ollama ollama rm $(docker exec ollama ollama list | grep -v "NAME" | awk '{print $1}' | head -n -1)
```

**Model Information:**
```bash
# Show model details
docker exec ollama ollama show gemma3:1b-it-qat

# Check model disk usage
docker exec ollama du -sh /root/.ollama/models/
```

### Model Configuration

**Change Active Model:**
```bash
# Update environment variable and restart
docker compose down
export MODEL_NAME=gemma3:7b-it-qat
docker compose up -d

# Or update docker-compose.yml and restart
```

**Model Performance Tuning:**
```bash
# Check GPU memory usage
docker exec ollama nvidia-smi --query-gpu=memory.used,memory.total --format=csv

# Adjust model concurrency (via Ollama config)
docker exec ollama ollama serve --help
```

## Cache and Storage Management

### Clear Application Caches

**Clear Generation Caches:**
```bash
# Clear output datasets (CAUTION: Deletes generated data)
rm -rf ./output_datasets/*

# Or use container
docker exec synthetic-data-service rm -rf /app/output_datasets/*
```

**Clear Model Caches:**
```bash
# Stop Ollama and clear model cache
docker compose stop ollama
docker volume rm dataset-generator_ollama_models
docker compose up -d ollama
```

**Clear MLflow Artifacts:**
```bash
# Clear MLflow data (if using MLflow)
rm -rf ./mlflow_artifacts/*
rm -rf ./mlflow_data/*
```

### Disk Usage Management

**Check Volume Usage:**
```bash
# Check Docker volume usage
docker system df

# Check specific volume sizes
docker exec synthetic-data-service df -h /app/output_datasets
docker exec ollama df -h /root/.ollama

# Check host directory sizes
du -sh ./output_datasets ./logs ./mlflow_artifacts
```

**Clean Up Disk Space:**
```bash
# Clean Docker system (removes unused images, containers, networks)
docker system prune -f

# Clean Docker volumes (CAUTION: May remove data)
docker volume prune -f

# Rotate logs
find ./logs -name "*.log" -mtime +30 -delete

# Compress old datasets
find ./output_datasets -name "*.json" -mtime +7 -exec gzip {} \;
```

**Set Storage Limits:**
```bash
# Monitor disk usage
df -h /

# Set up log rotation (example for production)
sudo logrotate -f /etc/logrotate.d/docker-logs
```

## Troubleshooting

### Common Failures

**Service Won't Start:**
```bash
# Check for port conflicts
netstat -tulpn | grep -E "(8000|11434)"

# Check Docker daemon
systemctl status docker

# Check GPU drivers (for Ollama)
nvidia-smi

# Verify Docker Compose syntax
docker compose config
```

**Ollama Model Loading Issues:**
```bash
# Check available disk space
df -h

# Check GPU memory
nvidia-smi --query-gpu=memory.free --format=csv

# Manually pull model
docker exec ollama ollama pull gemma3:1b-it-qat

# Check model file integrity
docker exec ollama ollama list -v
```

**API Service Timeouts:**
```bash
# Check API service logs
docker compose logs synthetic-data-service --tail=100

# Check resource usage
docker stats synthetic-data-service

# Verify Ollama connectivity
docker exec synthetic-data-service curl http://ollama:11434/api/version
```

**Dataset Generation Failures:**
```bash
# Check input data permissions
ls -la ./data/

# Verify template files exist
ls -la ./templates/ ./user_configs/

# Check configuration syntax
docker exec synthetic-data-service python -c "from src.core.config import app_config; print('Config OK')"
```

### Log Locations and Analysis

**Log File Locations:**
```bash
# Application logs (inside container)
/app/logs/YYYY-MM-DD_synthetic_data_service.log

# Host-mounted logs
./logs/YYYY-MM-DD_synthetic_data_service.log

# Docker container logs
docker compose logs synthetic-data-service
docker compose logs ollama
```

**Log Analysis Commands:**
```bash
# View recent errors
grep -i error ./logs/*.log | tail -20

# Monitor logs in real-time
tail -f ./logs/$(date +%Y-%m-%d)_synthetic_data_service.log

# Check API request patterns
grep "POST\|GET" ./logs/*.log | tail -50

# Analyze generation performance
grep "generation_time" ./logs/*.log | awk '{print $NF}' | sort -n
```

**Log Rotation Setup:**
```bash
# Create logrotate configuration
sudo tee /etc/logrotate.d/dataset-generator << EOF
./logs/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 0644 root root
}
EOF
```

### Performance Troubleshooting

**High Memory Usage:**
```bash
# Check container memory usage
docker stats --no-stream

# Check system memory
free -h

# Monitor GPU memory
watch nvidia-smi
```

**Slow Generation:**
```bash
# Check model loading time
time docker exec ollama ollama list

# Monitor CPU usage
htop

# Check disk I/O
iostat -x 1

# Profile generation request
time curl -X POST http://localhost:8000/generate -H "Content-Type: application/json" -d '{"dataset_structure": "single_question", "prompt_template": "institute_topic_question", "num_samples": 1}'
```

## Backup and Restore

### Backup Procedures

**Complete System Backup:**
```bash
#!/bin/bash
# Backup script: backup-dataset-generator.sh

BACKUP_DIR="/backups/dataset-generator/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

# Stop services
docker compose stop

# Backup volumes
docker run --rm -v dataset-generator_ollama_models:/source -v "$BACKUP_DIR":/backup alpine tar czf /backup/ollama_models.tar.gz -C /source .

# Backup host directories
tar czf "$BACKUP_DIR/output_datasets.tar.gz" ./output_datasets
tar czf "$BACKUP_DIR/logs.tar.gz" ./logs
tar czf "$BACKUP_DIR/config.tar.gz" ./config ./templates ./user_configs

# Backup configuration
cp docker-compose.yml "$BACKUP_DIR/"
cp -r config/ templates/ user_configs/ "$BACKUP_DIR/"

# Start services
docker compose up -d

echo "Backup completed: $BACKUP_DIR"
```

**Dataset-Only Backup:**
```bash
# Quick dataset backup
tar czf "datasets_backup_$(date +%Y%m%d).tar.gz" ./output_datasets

# Sync to remote storage
rsync -av ./output_datasets/ user@backup-server:/backup/datasets/
```

**Database Backup (if using MLflow):**
```bash
# Backup MLflow database
cp ./mlflow_data/mlflow.db "./mlflow_backup_$(date +%Y%m%d).db"

# Backup MLflow artifacts
tar czf "mlflow_artifacts_$(date +%Y%m%d).tar.gz" ./mlflow_artifacts
```

### Restore Procedures

**Complete System Restore:**
```bash
#!/bin/bash
# Restore script: restore-dataset-generator.sh

BACKUP_DIR="$1"
if [ -z "$BACKUP_DIR" ]; then
    echo "Usage: $0 <backup_directory>"
    exit 1
fi

# Stop services
docker compose down --volumes

# Restore configuration files
cp "$BACKUP_DIR/docker-compose.yml" ./
cp -r "$BACKUP_DIR/config/" "$BACKUP_DIR/templates/" "$BACKUP_DIR/user_configs/" ./

# Restore host directories
tar xzf "$BACKUP_DIR/output_datasets.tar.gz"
tar xzf "$BACKUP_DIR/logs.tar.gz"

# Create volume and restore
docker volume create dataset-generator_ollama_models
docker run --rm -v dataset-generator_ollama_models:/target -v "$BACKUP_DIR":/backup alpine tar xzf /backup/ollama_models.tar.gz -C /target

# Start services
docker compose up -d

echo "Restore completed from: $BACKUP_DIR"
```

**Selective Restore:**
```bash
# Restore only datasets
tar xzf datasets_backup_20250810.tar.gz

# Restore specific model
docker run --rm -v dataset-generator_ollama_models:/target -v ./backup:/backup alpine cp /backup/models/specific_model /target/

# Restore configuration
cp backup/config/config.yaml ./config/
docker compose restart synthetic-data-service
```

### Automated Backup

**Cron Job Setup:**
```bash
# Add to crontab (crontab -e)
# Daily backup at 2 AM
0 2 * * * /path/to/dataset-generator/backup-dataset-generator.sh

# Weekly full backup at Sunday 3 AM
0 3 * * 0 /path/to/dataset-generator/backup-full-system.sh
```

**Backup Retention:**
```bash
# Cleanup old backups (keep last 30 days)
find /backups/dataset-generator -name "202*" -mtime +30 -exec rm -rf {} \;

# Keep only last 5 dataset backups
ls -1t datasets_backup_*.tar.gz | tail -n +6 | xargs rm -f
```

## Monitoring and Alerting

### Health Monitoring

**Service Availability:**
```bash
# Create monitoring script: monitor-health.sh
#!/bin/bash
curl -f http://localhost:8000/health || echo "API service down" | mail -s "Dataset Generator Alert" admin@company.com
curl -f http://localhost:11434/api/version || echo "Ollama service down" | mail -s "Dataset Generator Alert" admin@company.com
```

**Resource Monitoring:**
```bash
# Check disk usage
df -h | grep -E "(output_datasets|ollama)" | awk '$5 > 80 {print "Disk usage high: " $0}'

# Check memory usage
free | awk 'NR==2{printf "Memory usage: %.2f%%\n", $3*100/$2}' | awk '$3 > 90 {print "Memory usage critical"}'
```

### Log Monitoring

**Error Detection:**
```bash
# Monitor for errors in logs
tail -f ./logs/$(date +%Y-%m-%d)_synthetic_data_service.log | grep -i "error\|exception\|failed"

# Count errors in last hour
grep "$(date -d '1 hour ago' +'%Y-%m-%d %H')" ./logs/*.log | grep -i error | wc -l
```

## Security Considerations

### Container Security

**Regular Updates:**
```bash
# Update base images
docker compose pull
docker compose up -d

# Check for security vulnerabilities
docker scout cves synthesisai/dataset-generator:latest
```

**Access Control:**
```bash
# Limit network exposure
# Only expose necessary ports in docker-compose.yml

# Use non-root users in containers
# Configure in Dockerfile: USER 1000:1000
```

### Data Protection

**Sensitive Data Handling:**
```bash
# Encrypt dataset backups
gpg --symmetric --cipher-algo AES256 datasets_backup.tar.gz

# Secure log files
chmod 640 ./logs/*.log
chown root:docker ./logs/*.log
```

**API Security:**
```bash
# Use reverse proxy with authentication
# Configure nginx/traefik with SSL and auth

# Limit API access
iptables -A INPUT -p tcp --dport 8000 -s trusted_network -j ACCEPT
iptables -A INPUT -p tcp --dport 8000 -j DROP
```
