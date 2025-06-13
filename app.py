from fastapi import FastAPI, File, UploadFile, HTTPException, Request, BackgroundTasks
from fastapi.responses import JSONResponse, Response
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os
import time
import psutil
import logging
import sys
import asyncio
import threading
from typing import Optional
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from dotenv import load_dotenv

# Configure Comprehensive Logging System
import logging.handlers
import platform

load_dotenv()

# Create formatters
detailed_formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s() - %(message)s'
)
simple_formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Clear any existing handlers
logger.handlers.clear()

# 1. FILE HANDLER - Application log file
file_handler = logging.FileHandler('app.log', mode='a', encoding='utf-8')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(detailed_formatter)

# 2. STDOUT HANDLER - Console output for INFO and above
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.INFO)
stdout_handler.setFormatter(simple_formatter)
# Filter to only show INFO and WARNING on stdout
stdout_handler.addFilter(lambda record: record.levelno <= logging.WARNING)

# 3. STDERR HANDLER - Error output for ERROR and CRITICAL
stderr_handler = logging.StreamHandler(sys.stderr)
stderr_handler.setLevel(logging.ERROR)
stderr_handler.setFormatter(detailed_formatter)

# 4. SYSLOG HANDLER - System log (platform dependent)
syslog_handler = None
try:
    if platform.system() == 'Windows':
        # Windows Event Log
        try:
            import win32evtlog
            import win32evtlogutil
            syslog_handler = logging.handlers.NTEventLogHandler(
                appname="ML_API_Monitoring",
                logtype="Application"
            )
        except ImportError:
            # Fallback to rotating file for Windows
            syslog_handler = logging.handlers.RotatingFileHandler(
                'system.log', 
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5
            )
    else:
        # Unix/Linux syslog
        syslog_handler = logging.handlers.SysLogHandler(
            address='/dev/log',
            facility=logging.handlers.SysLogHandler.LOG_LOCAL0
        )
    
    if syslog_handler:
        syslog_handler.setLevel(logging.WARNING)  # Only warnings and errors to syslog
        syslog_handler.setFormatter(logging.Formatter(
            'ML_API[%(process)d]: %(levelname)s - %(message)s'
        ))
        
except Exception as e:
    print(f"Warning: Could not setup syslog handler: {e}")

# 5. ROTATING FILE HANDLER - Prevent log files from growing too large
rotating_handler = logging.handlers.RotatingFileHandler(
    'app_detailed.log',
    maxBytes=50*1024*1024,  # 50MB
    backupCount=10,
    encoding='utf-8'
)
rotating_handler.setLevel(logging.DEBUG)
rotating_handler.setFormatter(detailed_formatter)

# Add all handlers to logger
handlers_to_add = [file_handler, stdout_handler, stderr_handler, rotating_handler]
if syslog_handler:
    handlers_to_add.append(syslog_handler)

for handler in handlers_to_add:
    logger.addHandler(handler)

# Configure root logger to avoid duplicate logs
logging.getLogger().handlers.clear()
logging.getLogger().addHandler(logging.NullHandler())

# Test logging setup
logger.info("=== ML API Logging System Initialized ===")
logger.info(f"Platform: {platform.system()}")
logger.info(f"Handlers configured: {len(logger.handlers)}")
for i, handler in enumerate(logger.handlers):
    logger.info(f"  Handler {i+1}: {type(handler).__name__} - Level: {logging.getLevelName(handler.level)}")

if syslog_handler:
    logger.warning("System logging test - this should appear in syslog")
else:
    logger.warning("Syslog handler not available on this platform")
app = FastAPI(title="ML Model API with Monitoring", version="1.0.0")

# ====== API MONITORING MIDDLEWARE ======

@app.middleware("http")
async def api_monitoring_middleware(request: Request, call_next):
    """Middleware to monitor all API requests"""
    global request_timestamps, error_timestamps, latency_samples
    
    # Extract request info
    method = request.method
    endpoint = request.url.path
    start_time = time.time()
    
    # Get request size
    request_size = get_request_size(request)
    
    # Record request timestamp
    request_timestamps.append(start_time)
    
    # Track active requests
    API_ACTIVE_REQUESTS.labels(method=method, endpoint=endpoint).inc()
    
    # Record request size
    API_REQUEST_SIZE_BYTES.labels(method=method, endpoint=endpoint).observe(request_size)
    
    try:
        # Process request
        response = await call_next(request)
        
        # Calculate metrics
        duration = time.time() - start_time
        status_code = response.status_code
        
        # Record successful request metrics
        API_REQUEST_COUNT.labels(
            method=method, 
            endpoint=endpoint, 
            status_code=status_code
        ).inc()
        
        API_REQUEST_DURATION.labels(
            method=method, 
            endpoint=endpoint
        ).observe(duration)
        
        # Record latency sample
        latency_samples.append({
            'timestamp': time.time(),
            'duration': duration
        })
        
        # Get response size
        response_size = 0
        if hasattr(response, 'headers') and 'content-length' in response.headers:
            try:
                response_size = int(response.headers['content-length'])
            except:
                pass
        
        API_RESPONSE_SIZE_BYTES.labels(
            method=method, 
            endpoint=endpoint, 
            status_code=status_code
        ).observe(response_size)
        
        # Log request
        logger.info(
            f"API Request: {method} {endpoint} - "
            f"Status: {status_code} - "
            f"Duration: {duration:.3f}s - "
            f"Request Size: {request_size}B - "
            f"Response Size: {response_size}B"
        )
        
        return response
        
    except Exception as e:
        # Calculate metrics for error
        duration = time.time() - start_time
        error_type = type(e).__name__
        status_code = getattr(e, 'status_code', 500)
        
        # Record error timestamp
        error_timestamps.append(time.time())
        
        # Record error metrics
        API_ERROR_COUNT.labels(
            method=method,
            endpoint=endpoint, 
            error_type=error_type,
            status_code=status_code
        ).inc()
        
        API_REQUEST_COUNT.labels(
            method=method,
            endpoint=endpoint,
            status_code=status_code
        ).inc()
        
        # Record latency sample even for errors
        latency_samples.append({
            'timestamp': time.time(),
            'duration': duration
        })
        
        # Log error
        logger.error(
            f"API Error: {method} {endpoint} - "
            f"Error: {error_type} - "
            f"Duration: {duration:.3f}s - "
            f"Message: {str(e)}"
        )
        
        # Re-raise the exception
        raise e
        
    finally:
        # Decrease active requests
        API_ACTIVE_REQUESTS.labels(method=method, endpoint=endpoint).dec()

# CPU/RAM
CPU_USAGE = Gauge('system_cpu_usage_percent', 'CPU usage percentage')
RAM_USAGE = Gauge('system_ram_usage_percent', 'RAM usage percentage')
RAM_USED_BYTES = Gauge('system_ram_used_bytes', 'RAM used in bytes')
RAM_TOTAL_BYTES = Gauge('system_ram_total_bytes', 'Total RAM in bytes')

# Disk
DISK_USAGE_PERCENT = Gauge('system_disk_usage_percent', 'Disk usage percentage', ['device', 'mountpoint'])
DISK_FREE_BYTES = Gauge('system_disk_free_bytes', 'Free disk space in bytes', ['device', 'mountpoint'])
DISK_TOTAL_BYTES = Gauge('system_disk_total_bytes', 'Total disk space in bytes', ['device', 'mountpoint'])

# Disk I/O
DISK_IO_READ_BYTES = Counter('system_disk_io_read_bytes_total', 'Total disk read bytes', ['device'])
DISK_IO_WRITE_BYTES = Counter('system_disk_io_write_bytes_total', 'Total disk write bytes', ['device'])
DISK_IO_READ_COUNT = Counter('system_disk_io_read_count_total', 'Total disk read operations', ['device'])
DISK_IO_WRITE_COUNT = Counter('system_disk_io_write_count_total', 'Total disk write operations', ['device'])

# Network I/O 
NETWORK_IO_SENT_BYTES = Counter('system_network_io_sent_bytes_total', 'Total network bytes sent', ['interface'])
NETWORK_IO_RECV_BYTES = Counter('system_network_io_received_bytes_total', 'Total network bytes received', ['interface'])
NETWORK_IO_SENT_PACKETS = Counter('system_network_io_sent_packets_total', 'Total network packets sent', ['interface'])
NETWORK_IO_RECV_PACKETS = Counter('system_network_io_received_packets_total', 'Total network packets received', ['interface'])

# GPU
GPU_UTILIZATION = Gauge('system_gpu_utilization_percent', 'GPU utilization percentage', ['gpu_id'])
GPU_MEMORY_USED = Gauge('system_gpu_memory_used_bytes', 'GPU memory used in bytes', ['gpu_id'])
GPU_MEMORY_TOTAL = Gauge('system_gpu_memory_total_bytes', 'GPU memory total in bytes', ['gpu_id'])
GPU_TEMPERATURE = Gauge('system_gpu_temperature_celsius', 'GPU temperature in Celsius', ['gpu_id'])
GPU_POWER_USAGE = Gauge('system_gpu_power_watts', 'GPU power usage in watts', ['gpu_id'])

# Store previous values for rate calculations
previous_disk_io = {}
previous_network_io = {}

# ====== API MONITORING METRICS ======

# Request Metrics
API_REQUEST_COUNT = Counter(
    'api_requests_total', 
    'Total number of API requests', 
    ['method', 'endpoint', 'status_code']
)

API_REQUEST_DURATION = Histogram(
    'api_request_duration_seconds',
    'API request duration in seconds',
    ['method', 'endpoint'],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)

# Error Metrics
API_ERROR_COUNT = Counter(
    'api_errors_total',
    'Total number of API errors',
    ['method', 'endpoint', 'error_type', 'status_code']
)

# Current metrics (Gauges)
API_REQUESTS_PER_SECOND = Gauge(
    'api_requests_per_second_current',
    'Current requests per second'
)

API_ERROR_RATE = Gauge(
    'api_error_rate_current',
    'Current error rate (percentage)'
)

API_AVERAGE_LATENCY = Gauge(
    'api_average_latency_seconds',
    'Current average response latency in seconds'
)

# Active requests
API_ACTIVE_REQUESTS = Gauge(
    'api_active_requests',
    'Number of requests currently being processed',
    ['method', 'endpoint']
)

# Request size metrics
API_REQUEST_SIZE_BYTES = Histogram(
    'api_request_size_bytes',
    'API request size in bytes',
    ['method', 'endpoint']
)

API_RESPONSE_SIZE_BYTES = Histogram(
    'api_response_size_bytes', 
    'API response size in bytes',
    ['method', 'endpoint', 'status_code']
)

# Rate calculation storage
request_timestamps = []
error_timestamps = []
latency_samples = []

# ====== MODEL MONITORING METRICS ======

# Model Inference Time Metrics
MODEL_INFERENCE_TIME_CPU = Histogram(
    'model_inference_time_cpu_seconds',
    'Model inference time on CPU in seconds',
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)

MODEL_INFERENCE_TIME_GPU = Histogram(
    'model_inference_time_gpu_seconds', 
    'Model inference time on GPU in seconds',
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)

MODEL_INFERENCE_TIME_TOTAL = Histogram(
    'model_inference_time_total_seconds',
    'Total model inference time in seconds',
    ['device'],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)

# Model Confidence Score Metrics
MODEL_CONFIDENCE_SCORE = Histogram(
    'model_confidence_score',
    'Model prediction confidence score',
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
)

MODEL_CONFIDENCE_CURRENT = Gauge(
    'model_confidence_current',
    'Current model confidence score'
)

MODEL_LOW_CONFIDENCE_COUNT = Counter(
    'model_low_confidence_total',
    'Total number of predictions with low confidence',
    ['threshold']
)

# Model Performance Gauges
MODEL_AVERAGE_INFERENCE_TIME = Gauge(
    'model_average_inference_time_seconds',
    'Average model inference time in seconds',
    ['device']
)

MODEL_AVERAGE_CONFIDENCE = Gauge(
    'model_average_confidence_score',
    'Average model confidence score'
)

MODEL_PREDICTIONS_COUNT = Counter(
    'model_predictions_total',
    'Total number of model predictions',
    ['device', 'predicted_class']
)

# Model inference samples storage
inference_samples = []
confidence_samples = []

# ====== GMAIL ALERTING SYSTEM ======

# Gmail Alert Configuration
GMAIL_ALERT_CONFIG = {
    "enabled": True,
    "gmail_user": "hellonghia321@gmail.com",
    "gmail_password": os.getenv('GMAIL_PASS'),  # App password
    "recipients": ["mouzfeed3r@gmail.com"],  # Danh sách email nhận alerts
    "smtp_server": "smtp.gmail.com",
    "smtp_port": 587
}

# Alert Thresholds
ALERT_THRESHOLDS = {
    "error_rate_critical": 50.0,      # 50% error rate
    "confidence_critical": 0.6,       # Below 60% confidence
    "cpu_critical": 95.0,             # 95% CPU usage
    "memory_critical": 95.0           # 95% memory usage
}

# Alert state (để tránh spam)
last_alert_times = {}
ALERT_COOLDOWN = 300  # 5 phút




# ==== Getting metrics =====
def get_cpu_metrics():
    """Collect CPU metrics"""
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        CPU_USAGE.set(cpu_percent)
        logger.debug(f"CPU Usage: {cpu_percent}%")
    except Exception as e:
        logger.error(f"Error collecting CPU metrics: {e}")

def get_memory_metrics():
    """Collect RAM metrics"""
    try:
        memory = psutil.virtual_memory()
        RAM_USAGE.set(memory.percent)
        RAM_USED_BYTES.set(memory.used)
        RAM_TOTAL_BYTES.set(memory.total)
        logger.debug(f"RAM Usage: {memory.percent}% ({memory.used}/{memory.total} bytes)")
    except Exception as e:
        logger.error(f"Error collecting memory metrics: {e}")

def get_disk_metrics():
    """Collect disk space and I/O metrics"""
    try:
        # Disk space metrics
        disk_partitions = psutil.disk_partitions()
        for partition in disk_partitions:
            try:
                partition_usage = psutil.disk_usage(partition.mountpoint)
                DISK_USAGE_PERCENT.labels(
                    device=partition.device, 
                    mountpoint=partition.mountpoint
                ).set(partition_usage.percent)
                DISK_FREE_BYTES.labels(
                    device=partition.device, 
                    mountpoint=partition.mountpoint
                ).set(partition_usage.free)
                DISK_TOTAL_BYTES.labels(
                    device=partition.device, 
                    mountpoint=partition.mountpoint
                ).set(partition_usage.total)
            except PermissionError:
                continue
        
        # Disk I/O metrics
        global previous_disk_io
        disk_io = psutil.disk_io_counters(perdisk=True)
        if disk_io:
            for device, io_stats in disk_io.items():
                # Update counters (Prometheus counters handle the rate calculation)
                DISK_IO_READ_BYTES.labels(device=device)._value._value = io_stats.read_bytes
                DISK_IO_WRITE_BYTES.labels(device=device)._value._value = io_stats.write_bytes
                DISK_IO_READ_COUNT.labels(device=device)._value._value = io_stats.read_count
                DISK_IO_WRITE_COUNT.labels(device=device)._value._value = io_stats.write_count
                
        logger.debug("Disk metrics updated")
    except Exception as e:
        logger.error(f"Error collecting disk metrics: {e}")

def get_network_metrics():
    """Collect network I/O metrics"""
    try:
        global previous_network_io
        network_io = psutil.net_io_counters(pernic=True)
        if network_io:
            for interface, io_stats in network_io.items():
                # Skip loopback interfaces
                if interface.lower().startswith('lo'):
                    continue
                
                # Skip vpn
                skip_keywords = [
                    'vpn', 'tap', 'tun', 'radmin', 'openvpn', 'nordvpn', 'expressvpn',
                    'vbox', 'vmware', 'hyper-v', 'docker', 'vethernet', 'bluetooth',
                    'teredo', 'isatap', '6to4'
                ]
                
                if any(keyword in interface.lower() for keyword in skip_keywords):
                    continue
                
                # monitor wifi, Lan things only
                physical_keywords = ['wi-fi', 'wireless', 'ethernet', 'local area connection', 'wlan', 'eth']
                is_physical = any(keyword in interface.lower() for keyword in physical_keywords)
                
                # Skip if not physical and has low traffic (likely virtual)
                total_traffic = io_stats.bytes_sent + io_stats.bytes_recv
                if not is_physical and total_traffic < 1024 * 1024:
                    continue
                    
                # Update counters (Prometheus counters handle the rate calculation)
                NETWORK_IO_SENT_BYTES.labels(interface=interface)._value._value = io_stats.bytes_sent
                NETWORK_IO_RECV_BYTES.labels(interface=interface)._value._value = io_stats.bytes_recv
                NETWORK_IO_SENT_PACKETS.labels(interface=interface)._value._value = io_stats.packets_sent
                NETWORK_IO_RECV_PACKETS.labels(interface=interface)._value._value = io_stats.packets_recv
                
        logger.debug("Network metrics updated")
    except Exception as e:
        logger.error(f"Error collecting network metrics: {e}")

def get_gpu_metrics():
    """Collect GPU metrics (optional)"""
    try:
        import GPUtil
        gpus = GPUtil.getGPUs()
        for i, gpu in enumerate(gpus):
            GPU_UTILIZATION.labels(gpu_id=str(i)).set(gpu.load * 100)
            GPU_MEMORY_USED.labels(gpu_id=str(i)).set(gpu.memoryUsed * 1024 * 1024)  # Convert MB to bytes
            GPU_MEMORY_TOTAL.labels(gpu_id=str(i)).set(gpu.memoryTotal * 1024 * 1024)  # Convert MB to bytes
            GPU_TEMPERATURE.labels(gpu_id=str(i)).set(gpu.temperature)
            
        logger.debug(f"GPU metrics updated for {len(gpus)} GPUs")
    except ImportError:
        logger.info("GPUtil not available, skipping GPU metrics")
    except Exception as e:
        logger.error(f"Error collecting GPU metrics: {e}")

def update_all_system_metrics():
    """Update all system metrics"""
    get_cpu_metrics()
    get_memory_metrics()
    get_disk_metrics()
    get_network_metrics()
    get_gpu_metrics()

# ====== API MONITORING FUNCTIONS ======

def calculate_current_rps():
    """Calculate current requests per second"""
    global request_timestamps
    current_time = time.time()
    
    # Keep only timestamps from last 60 seconds
    request_timestamps = [ts for ts in request_timestamps if current_time - ts <= 60]
    
    # Calculate RPS
    if len(request_timestamps) >= 2:
        time_span = current_time - min(request_timestamps)
        rps = len(request_timestamps) / max(time_span, 1)
    else:
        rps = 0
    
    API_REQUESTS_PER_SECOND.set(rps)
    return rps

def calculate_current_error_rate():
    """Calculate current error rate"""
    global error_timestamps, request_timestamps
    current_time = time.time()
    
    # Keep only timestamps from last 60 seconds
    error_timestamps = [ts for ts in error_timestamps if current_time - ts <= 60]
    recent_requests = [ts for ts in request_timestamps if current_time - ts <= 60]
    
    # Calculate error rate
    if len(recent_requests) > 0:
        error_rate = (len(error_timestamps) / len(recent_requests)) * 100
    else:
        error_rate = 0
    
    API_ERROR_RATE.set(error_rate)
    return error_rate

def calculate_average_latency():
    """Calculate current average latency"""
    global latency_samples
    current_time = time.time()
    
    # Keep only samples from last 60 seconds
    latency_samples = [sample for sample in latency_samples if current_time - sample['timestamp'] <= 60]
    
    if latency_samples:
        avg_latency = sum(sample['duration'] for sample in latency_samples) / len(latency_samples)
    else:
        avg_latency = 0
    
    API_AVERAGE_LATENCY.set(avg_latency)
    return avg_latency

def update_api_metrics():
    """Update all API metrics"""
    calculate_current_rps()
    calculate_current_error_rate()
    calculate_average_latency()

# ====== MODEL MONITORING FUNCTIONS ======

def calculate_model_metrics():
    """Calculate current model performance metrics"""
    global inference_samples, confidence_samples
    current_time = time.time()
    
    # Keep only samples from last 60 seconds
    inference_samples = [sample for sample in inference_samples if current_time - sample['timestamp'] <= 60]
    confidence_samples = [sample for sample in confidence_samples if current_time - sample['timestamp'] <= 60]
    
    # Calculate average inference time by device
    device_inference_times = {'cpu': [], 'gpu': []}
    for sample in inference_samples:
        device = sample['device']
        if device in device_inference_times:
            device_inference_times[device].append(sample['duration'])
    
    for device, times in device_inference_times.items():
        if times:
            avg_time = sum(times) / len(times)
            MODEL_AVERAGE_INFERENCE_TIME.labels(device=device).set(avg_time)
    
    # Calculate average confidence score
    if confidence_samples:
        avg_confidence = sum(sample['confidence'] for sample in confidence_samples) / len(confidence_samples)
        MODEL_AVERAGE_CONFIDENCE.set(avg_confidence)
        
        # Update current confidence with latest value
        if confidence_samples:
            latest_confidence = max(confidence_samples, key=lambda x: x['timestamp'])['confidence']
            MODEL_CONFIDENCE_CURRENT.set(latest_confidence)

def update_model_metrics():
    """Update all model metrics"""
    calculate_model_metrics()

# ====== GMAIL ALERT FUNCTIONS ======

def send_gmail_alert(subject: str, message: str):
    """Gửi alert qua Gmail"""
    if not GMAIL_ALERT_CONFIG["enabled"]:
        return False
    
    try:
        # Tạo email
        msg = MIMEMultipart()
        msg['From'] = GMAIL_ALERT_CONFIG["gmail_user"]
        msg['To'] = ", ".join(GMAIL_ALERT_CONFIG["recipients"])
        msg['Subject'] = subject
        
        msg.attach(MIMEText(message, 'plain'))
        
        # Kết nối SMTP và gửi
        server = smtplib.SMTP(GMAIL_ALERT_CONFIG["smtp_server"], GMAIL_ALERT_CONFIG["smtp_port"])
        server.starttls()
        server.login(GMAIL_ALERT_CONFIG["gmail_user"], GMAIL_ALERT_CONFIG["gmail_password"])
        text = msg.as_string()
        server.sendmail(GMAIL_ALERT_CONFIG["gmail_user"], GMAIL_ALERT_CONFIG["recipients"], text)
        server.quit()
        
        logger.info(f"Gmail alert sent successfully: {subject}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to send Gmail alert: {e}")
        return False

def should_send_alert(alert_type: str) -> bool:
    current_time = time.time()
    last_time = last_alert_times.get(alert_type, 0)
    
    if current_time - last_time >= ALERT_COOLDOWN:
        last_alert_times[alert_type] = current_time
        return True
    return False

def check_and_send_alerts():
    """Kiểm tra các điều kiện và gửi alerts nếu cần"""
    try:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.debug("Checking alerts...")
        
        # 1. Kiểm tra Error Rate
        error_rate = calculate_current_error_rate()
        logger.debug(f"Error rate: {error_rate:.1f}% (threshold: {ALERT_THRESHOLDS['error_rate_critical']}%)")
        if error_rate >= ALERT_THRESHOLDS["error_rate_critical"]:
            if should_send_alert("error_rate"):
                subject = "CRITICAL: High API Error Rate Detected"
                message = f"""
Time: {current_time}
Alert Type: High Error Rate
Severity: CRITICAL

Current Error Rate: {error_rate:.1f}%
Threshold: {ALERT_THRESHOLDS["error_rate_critical"]}%
"""
                send_gmail_alert(subject, message)
                logger.warning(f"ERROR RATE ALERT SENT: {error_rate:.1f}%")
        
        # 2. Kiểm tra Model Confidence
        if confidence_samples:
            recent_confidences = [
                sample['confidence'] for sample in confidence_samples 
                if time.time() - sample['timestamp'] <= 60
            ]
            
            logger.debug(f"Recent confidence samples: {len(recent_confidences)} samples")
            if recent_confidences:
                avg_confidence = sum(recent_confidences) / len(recent_confidences)
                logger.debug(f"Average confidence: {avg_confidence:.3f} (threshold: {ALERT_THRESHOLDS['confidence_critical']})")
                if avg_confidence < ALERT_THRESHOLDS["confidence_critical"]:
                    if should_send_alert("low_confidence"):
                        subject = "CRITICAL: Low Model Confidence Score"
                        message = f"""
Time: {current_time}
Alert Type: Low Model Confidence
Severity: CRITICAL

Average Confidence: {avg_confidence:.3f}
Samples Count: {len(recent_confidences)}
Threshold: {ALERT_THRESHOLDS["confidence_critical"]}
"""
                        send_gmail_alert(subject, message)
                        logger.warning(f"AVERAGE CONFIDENCE ALERT SENT: {avg_confidence:.3f}")
        
        # 3. Kiểm tra System Resources
        cpu_usage = psutil.cpu_percent()
        memory_usage = psutil.virtual_memory().percent
        
        if cpu_usage >= ALERT_THRESHOLDS["cpu_critical"]:
            if should_send_alert("high_cpu"):
                subject = "CRITICAL: High CPU Usage"
                message = f"""
Time: {current_time}
Alert Type: High CPU Usage
Severity: CRITICAL

Current CPU Usage: {cpu_usage:.1f}%
Threshold: {ALERT_THRESHOLDS["cpu_critical"]}%
"""
                send_gmail_alert(subject, message)
        
        if memory_usage >= ALERT_THRESHOLDS["memory_critical"]:
            if should_send_alert("high_memory"):
                subject = "CRITICAL: High Memory"
                message = f"""
Alert

Time: {current_time}
Alert Type: High Memory
Do nghiem trong: CRITICAL

Current Memory Usage: {memory_usage:.1f}%
Threshold: {ALERT_THRESHOLDS["memory_critical"]}%
"""
                send_gmail_alert(subject, message)
                
    except Exception as e:
        logger.error(f"Loi gui thong bao: {e}")

# ====== LOGGING MONITORING FUNCTIONS ======

def get_logging_status():
    """Get current logging system status"""
    status = {
        "total_handlers": len(logger.handlers),
        "handlers": [],
        "log_files": {},
        "system_logs": {}
    }
    
    for i, handler in enumerate(logger.handlers):
        handler_info = {
            "type": type(handler).__name__,
            "level": logging.getLevelName(handler.level),
            "active": True
        }
        
        # Add specific handler information
        if isinstance(handler, logging.FileHandler):
            handler_info["file_path"] = handler.baseFilename
            try:
                import os
                if os.path.exists(handler.baseFilename):
                    stat = os.stat(handler.baseFilename)
                    handler_info["file_size_mb"] = round(stat.st_size / (1024*1024), 2)
                    handler_info["last_modified"] = stat.st_mtime
            except:
                pass
        elif isinstance(handler, logging.StreamHandler):
            if handler.stream == sys.stdout:
                handler_info["stream"] = "stdout"
            elif handler.stream == sys.stderr:
                handler_info["stream"] = "stderr"
            else:
                handler_info["stream"] = "other"
        elif isinstance(handler, logging.handlers.SysLogHandler):
            handler_info["facility"] = "syslog"
        elif isinstance(handler, logging.handlers.NTEventLogHandler):
            handler_info["facility"] = "windows_event_log"
        
        status["handlers"].append(handler_info)
    
    # Check log file accessibility
    log_files = ['app.log', 'app_detailed.log', 'system.log']
    for log_file in log_files:
        try:
            import os
            if os.path.exists(log_file):
                stat = os.stat(log_file)
                status["log_files"][log_file] = {
                    "exists": True,
                    "size_mb": round(stat.st_size / (1024*1024), 2),
                    "last_modified": stat.st_mtime,
                    "readable": os.access(log_file, os.R_OK),
                    "writable": os.access(log_file, os.W_OK)
                }
            else:
                status["log_files"][log_file] = {"exists": False}
        except Exception as e:
            status["log_files"][log_file] = {"error": str(e)}
    
    return status

def test_all_log_levels():
    """Test logging to all configured handlers"""
    test_messages = {
        "DEBUG": "Debug level test message",
        "INFO": "Info level test message", 
        "WARNING": "Warning level test message",
        "ERROR": "Error level test message",
        "CRITICAL": "Critical level test message"
    }
    
    results = {}
    
    for level, message in test_messages.items():
        try:
            log_func = getattr(logger, level.lower())
            log_func(f"LOG TEST [{level}]: {message}")
            results[level] = "success"
        except Exception as e:
            results[level] = f"error: {str(e)}"
    
    return results

def log_system_event(event_type, details):
    """Log system-level events for monitoring"""
    if event_type == "startup":
        logger.info(f"=== SYSTEM STARTUP === {details}")
        logger.warning(f"System startup event logged to syslog: {details}")
    elif event_type == "shutdown":
        logger.info(f"=== SYSTEM SHUTDOWN === {details}")
        logger.warning(f"System shutdown event logged to syslog: {details}")
    elif event_type == "error":
        logger.error(f"=== SYSTEM ERROR === {details}", exc_info=True)
    elif event_type == "critical":
        logger.critical(f"=== SYSTEM CRITICAL === {details}", exc_info=True)
    else:
        logger.info(f"=== SYSTEM EVENT [{event_type}] === {details}")

def get_request_size(request: Request) -> int:
    """Get approximate request size"""
    try:
        # Get content length from headers
        content_length = request.headers.get('content-length')
        if content_length:
            return int(content_length)
        
        # Estimate from URL and headers
        url_size = len(str(request.url))
        headers_size = sum(len(f"{k}: {v}") for k, v in request.headers.items())
        return url_size + headers_size
    except:
        return 0


async def monitoring_background_task():
    """Background task to continuously update system, API and model metrics"""
    while True:
        try:
            # Update system metrics
            update_all_system_metrics()
            
            # Update API metrics
            update_api_metrics()
            
            # Update model metrics
            update_model_metrics()
            
            # Check alerts every 30 seconds
            if int(time.time()) % 30 == 0:
                check_and_send_alerts()
            
            await asyncio.sleep(10)  # Update every 10 seconds
        except Exception as e:
            logger.error(f"Error in monitoring background task: {e}")
            await asyncio.sleep(5)


# Load the trained model
def load_model(model_path):
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 10)
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

# Preprocess input image
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    return transform(image).unsqueeze(0)

# Class names for CIFAR-10
CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Load model when starting the app
model_path = './checkpoints/best_model_logging_demo.pth'
model = load_model(model_path) 


# ==== API endpoints ====
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    global inference_samples, confidence_samples
    
    if not file.filename:
        logger.warning("Prediction attempt with no file")
        raise HTTPException(status_code=400, detail="No file selected")

    try:
        # Read and preprocess the image
        image = Image.open(file.file).convert('RGB')
        image_tensor = preprocess_image(image)
        
        # Determine device and move model/tensor
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        image_tensor = image_tensor.to(device)
        model.to(device)
        
        # Track inference time with detailed timing
        start_total_time = time.time()
        
        # Make prediction with device-specific timing
        with torch.no_grad():
            if device == 'cuda':
                # GPU inference timing
                torch.cuda.synchronize()  # Ensure GPU operations are finished
                start_gpu_time = time.time()
                
                output = model(image_tensor)
                
                torch.cuda.synchronize()  # Wait for GPU operations to complete
                end_gpu_time = time.time()
                gpu_inference_time = end_gpu_time - start_gpu_time
                
                # Record GPU metrics
                MODEL_INFERENCE_TIME_GPU.observe(gpu_inference_time)
                MODEL_INFERENCE_TIME_TOTAL.labels(device='gpu').observe(gpu_inference_time)
                
                logger.debug(f"GPU inference time: {gpu_inference_time:.4f}s")
                
            else:
                # CPU inference timing
                start_cpu_time = time.time()
                
                output = model(image_tensor)
                
                end_cpu_time = time.time()
                cpu_inference_time = end_cpu_time - start_cpu_time
                
                # Record CPU metrics
                MODEL_INFERENCE_TIME_CPU.observe(cpu_inference_time)
                MODEL_INFERENCE_TIME_TOTAL.labels(device='cpu').observe(cpu_inference_time)
                
                logger.debug(f"CPU inference time: {cpu_inference_time:.4f}s")
        
        end_total_time = time.time()
        total_inference_time = end_total_time - start_total_time
        
        # Get prediction and confidence score
        probabilities = torch.softmax(output, dim=1)
        confidence_score, predicted = torch.max(probabilities, 1)
        predicted_class = CLASSES[predicted.item()]
        confidence_value = confidence_score.item()
        
        # Record model metrics
        MODEL_CONFIDENCE_SCORE.observe(confidence_value)
        MODEL_CONFIDENCE_CURRENT.set(confidence_value)
        MODEL_PREDICTIONS_COUNT.labels(device=device, predicted_class=predicted_class).inc()
        
        # Check for low confidence (multiple thresholds)
        low_confidence_thresholds = [0.5, 0.6, 0.7, 0.8]
        for threshold in low_confidence_thresholds:
            if confidence_value < threshold:
                MODEL_LOW_CONFIDENCE_COUNT.labels(threshold=str(threshold)).inc()
        
        # Store samples for rolling calculations
        inference_samples.append({
            'timestamp': time.time(),
            'duration': total_inference_time,
            'device': device
        })
        
        confidence_samples.append({
            'timestamp': time.time(),
            'confidence': confidence_value,
            'predicted_class': predicted_class
        })
        
        # Check for immediate low confidence alert
        if confidence_value < ALERT_THRESHOLDS["confidence_critical"]:
            if should_send_alert("low_confidence_immediate"):
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                subject = "CRITICAL: Low Confidence Prediction"
                message = f"""
Time: {current_time}
Alert Type: Low Model Confidence (Single Prediction)
Severity: CRITICAL

Prediction: {predicted_class}
Confidence Score: {confidence_value:.4f}
Threshold: {ALERT_THRESHOLDS["confidence_critical"]}
"""
                send_gmail_alert(subject, message)
                logger.warning(f"LOW CONFIDENCE ALERT SENT: {confidence_value:.4f} < {ALERT_THRESHOLDS['confidence_critical']}")
        
        # Log prediction with metrics
        logger.info(
            f"Prediction: {predicted_class} - "
            f"Confidence: {confidence_value:.4f} - "
            f"Device: {device} - "
            f"Inference time: {total_inference_time:.4f}s"
        )
        
        return JSONResponse(content={
            "class": predicted_class,
            "confidence": round(confidence_value, 4),
            "device": device,
            "inference_time_seconds": round(total_inference_time, 4),
            "timestamp": time.time()
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint"""
    # Update all metrics before serving them
    update_all_system_metrics()
    update_api_metrics()
    update_model_metrics()
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "cpu_usage": psutil.cpu_percent(),
        "memory_usage": psutil.virtual_memory().percent
    }

@app.get("/api-stats")
async def get_api_stats():
    """Get current API statistics"""
    try:
        # Update metrics first
        update_api_metrics()
        
        current_time = time.time()
        
        # Get recent data (last 60 seconds)
        recent_requests = [ts for ts in request_timestamps if current_time - ts <= 60]
        recent_errors = [ts for ts in error_timestamps if current_time - ts <= 60]
        recent_latencies = [sample['duration'] for sample in latency_samples if current_time - sample['timestamp'] <= 60]
        
        # Calculate statistics
        total_requests = len(recent_requests)
        total_errors = len(recent_errors)
        current_rps = calculate_current_rps()
        current_error_rate = calculate_current_error_rate()
        avg_latency = calculate_average_latency()
        
        # Latency percentiles
        percentiles = {}
        if recent_latencies:
            sorted_latencies = sorted(recent_latencies)
            percentiles = {
                "p50": sorted_latencies[int(len(sorted_latencies) * 0.5)] if sorted_latencies else 0,
                "p95": sorted_latencies[int(len(sorted_latencies) * 0.95)] if sorted_latencies else 0,
                "p99": sorted_latencies[int(len(sorted_latencies) * 0.99)] if sorted_latencies else 0,
                "max": max(sorted_latencies) if sorted_latencies else 0,
                "min": min(sorted_latencies) if sorted_latencies else 0
            }
        
        # Get active requests count safely
        active_requests_count = 0
        try:
            # Collect all samples from the active requests gauge
            for family in API_ACTIVE_REQUESTS.collect():
                for sample in family.samples:
                    active_requests_count += sample.value
        except Exception as e:
            logger.debug(f"Could not get active requests count: {e}")
            active_requests_count = 0
        
        return {
            "timestamp": current_time,
            "time_window_seconds": 60,
            "requests": {
                "total_last_60s": total_requests,
                "requests_per_second": round(current_rps, 2),
                "active_requests": int(active_requests_count)
            },
            "errors": {
                "total_last_60s": total_errors,
                "error_rate_percent": round(current_error_rate, 2)
            },
            "latency": {
                "average_seconds": round(avg_latency, 3),
                "percentiles_seconds": {k: round(v, 3) for k, v in percentiles.items()}
            },
            "status": "healthy" if current_error_rate < 50 and avg_latency < 2.0 else "degraded"
        }
        
    except Exception as e:
        logger.error(f"Error getting API stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model-stats")
async def get_model_stats():
    """Get current model performance statistics"""
    try:
        # Update model metrics first
        update_model_metrics()
        
        current_time = time.time()
        
        # Get recent data (last 60 seconds)
        recent_inferences = [sample for sample in inference_samples if current_time - sample['timestamp'] <= 60]
        recent_confidences = [sample for sample in confidence_samples if current_time - sample['timestamp'] <= 60]
        
        # Calculate inference statistics
        inference_stats = {
            "total_predictions_60s": len(recent_inferences),
            "by_device": {"cpu": {"count": 0, "times": []}, "gpu": {"count": 0, "times": []}},
            "average_inference_time": {"cpu": 0.0, "gpu": 0.0, "overall": 0.0}
        }
        
        for inference in recent_inferences:
            device = inference['device']
            duration = inference['duration']
            
            inference_stats["by_device"][device]["count"] += 1
            inference_stats["by_device"][device]["times"].append(duration)
        
        # Calculate averages and percentiles
        all_times = []
        for device in ['cpu', 'gpu']:
            times = inference_stats["by_device"][device]["times"]
            if times:
                avg_time = sum(times) / len(times)
                inference_stats["average_inference_time"][device] = round(avg_time, 4)
                all_times.extend(times)
        
        if all_times:
            inference_stats["average_inference_time"]["overall"] = round(sum(all_times) / len(all_times), 4)
            sorted_times = sorted(all_times)
            inference_stats["latency_percentiles"] = {
                "p50": round(sorted_times[int(len(sorted_times) * 0.5)], 4) if sorted_times else 0,
                "p95": round(sorted_times[int(len(sorted_times) * 0.95)], 4) if sorted_times else 0,
                "p99": round(sorted_times[int(len(sorted_times) * 0.99)], 4) if sorted_times else 0,
                "max": round(max(sorted_times), 4) if sorted_times else 0,
                "min": round(min(sorted_times), 4) if sorted_times else 0
            }
        
        # Calculate confidence statistics
        confidence_stats = {
            "total_predictions_60s": len(recent_confidences),
            "average_confidence": 0.0,
            "confidence_distribution": {
                "very_high": 0,  # > 0.9
                "high": 0,       # 0.8 - 0.9
                "medium": 0,     # 0.6 - 0.8
                "low": 0,        # 0.4 - 0.6
                "very_low": 0    # < 0.4
            },
            "low_confidence_alerts": {}
        }
        
        if recent_confidences:
            confidences = [sample['confidence'] for sample in recent_confidences]
            confidence_stats["average_confidence"] = round(sum(confidences) / len(confidences), 4)
            
            # Calculate confidence distribution
            for conf in confidences:
                if conf > 0.9:
                    confidence_stats["confidence_distribution"]["very_high"] += 1
                elif conf > 0.8:
                    confidence_stats["confidence_distribution"]["high"] += 1
                elif conf > 0.6:
                    confidence_stats["confidence_distribution"]["medium"] += 1
                elif conf > 0.4:
                    confidence_stats["confidence_distribution"]["low"] += 1
                else:
                    confidence_stats["confidence_distribution"]["very_low"] += 1
            
            # Calculate percentiles
            sorted_confidences = sorted(confidences)
            confidence_stats["confidence_percentiles"] = {
                "p50": round(sorted_confidences[int(len(sorted_confidences) * 0.5)], 4) if sorted_confidences else 0,
                "p95": round(sorted_confidences[int(len(sorted_confidences) * 0.95)], 4) if sorted_confidences else 0,
                "p99": round(sorted_confidences[int(len(sorted_confidences) * 0.99)], 4) if sorted_confidences else 0,
                "max": round(max(sorted_confidences), 4) if sorted_confidences else 0,
                "min": round(min(sorted_confidences), 4) if sorted_confidences else 0
            }
            
            # Check for low confidence alerts
            for threshold in [0.5, 0.6, 0.7, 0.8]:
                low_count = sum(1 for conf in confidences if conf < threshold)
                if low_count > 0:
                    confidence_stats["low_confidence_alerts"][f"below_{threshold}"] = {
                        "count": low_count,
                        "percentage": round((low_count / len(confidences)) * 100, 2)
                    }
        
        # Class distribution
        class_distribution = {}
        for sample in recent_confidences:
            class_name = sample['predicted_class']
            if class_name not in class_distribution:
                class_distribution[class_name] = {"count": 0, "confidences": []}
            class_distribution[class_name]["count"] += 1
            class_distribution[class_name]["confidences"].append(sample['confidence'])
        
        # Calculate average confidence per class
        for class_name, data in class_distribution.items():
            if data["confidences"]:
                data["average_confidence"] = round(sum(data["confidences"]) / len(data["confidences"]), 4)
                del data["confidences"]  # Remove raw data from response
        
        # Model health assessment
        avg_confidence = confidence_stats.get("average_confidence", 0)
        avg_inference_time = inference_stats["average_inference_time"]["overall"]
        
        model_health = "healthy"
        health_issues = []
        
        if avg_confidence < 0.6:
            model_health = "degraded"
            health_issues.append(f"Low average confidence: {avg_confidence}")
        
        if avg_inference_time > 2.0:
            model_health = "degraded" 
            health_issues.append(f"High inference time: {avg_inference_time}s")
        
        low_conf_count = sum(1 for sample in recent_confidences if sample['confidence'] < 0.5)
        if len(recent_confidences) > 0 and (low_conf_count / len(recent_confidences)) > 0.3:
            model_health = "degraded"
            health_issues.append(f"High proportion of low confidence predictions: {(low_conf_count / len(recent_confidences)) * 100:.1f}%")
        
        return {
            "timestamp": current_time,
            "time_window_seconds": 60,
            "inference_performance": inference_stats,
            "confidence_analysis": confidence_stats,
            "class_distribution": class_distribution,
            "model_health": {
                "status": model_health,
                "issues": health_issues if health_issues else ["No issues detected"]
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting model stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/logging-status")
async def get_logging_status_endpoint():
    """Get current logging system status and test all handlers"""
    try:
        # Get logging status
        status = get_logging_status()
        
        # Test all log levels
        logger.info("Logging status check initiated")
        test_results = test_all_log_levels()
        
        # Check requirements compliance
        requirements_check = {
            "stdout": False,
            "stderr": False, 
            "logfile": False,
            "syslog": False
        }
        
        for handler_info in status["handlers"]:
            handler_type = handler_info["type"]
            
            if handler_type == "StreamHandler":
                if handler_info.get("stream") == "stdout":
                    requirements_check["stdout"] = True
                elif handler_info.get("stream") == "stderr":
                    requirements_check["stderr"] = True
            elif handler_type in ["FileHandler", "RotatingFileHandler"]:
                requirements_check["logfile"] = True
            elif handler_type in ["SysLogHandler", "NTEventLogHandler"]:
                requirements_check["syslog"] = True
        
        # Calculate compliance score
        compliance_score = sum(requirements_check.values())
        compliance_percentage = (compliance_score / 4) * 100
        
        return {
            "timestamp": time.time(),
            "logging_system": status,
            "test_results": test_results,
            "requirements_compliance": {
                "checks": requirements_check,
                "score": f"{compliance_score}/4",
                "percentage": f"{compliance_percentage:.1f}%",
                "status": "compliant" if compliance_score == 4 else "partial" if compliance_score >= 2 else "non-compliant"
            },
            "recommendations": get_logging_recommendations(requirements_check)
        }
        
    except Exception as e:
        logger.error(f"Error getting logging status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

def get_logging_recommendations(requirements_check):
    """Get recommendations based on missing logging requirements"""
    recommendations = []
    
    if not requirements_check["stdout"]:
        recommendations.append("Add stdout handler for console output")
    
    if not requirements_check["stderr"]:
        recommendations.append("Add stderr handler for error output")
    
    if not requirements_check["logfile"]:
        recommendations.append("Add file handler for application logs")
    
    if not requirements_check["syslog"]:
        recommendations.append("Add syslog handler for system-level logging")
    
    if not recommendations:
        recommendations.append("All logging requirements are met")
    
    return recommendations

@app.get("/test-logging")
async def test_logging_endpoint():
    """Test all logging levels and handlers"""
    try:
        logger.info("=== LOGGING TEST INITIATED ===")
        
        # Test different log levels
        logger.debug("DEBUG: This is a debug message")
        logger.info("INFO: This is an info message")
        logger.warning("WARNING: This is a warning message")
        logger.error("ERROR: This is an error message")
        
        # Test exception logging
        try:
            raise ValueError("Test exception for logging")
        except ValueError as e:
            logger.error("EXCEPTION TEST: Caught test exception", exc_info=True)
        
        # Test system event logging
        log_system_event("test", "Manual logging test triggered")
        
        logger.info("=== LOGGING TEST COMPLETED ===")
        
        return {
            "status": "success",
            "message": "All log levels tested",
            "timestamp": time.time(),
            "check_locations": [
                "Console output (stdout/stderr)",
                "app.log file",
                "app_detailed.log file", 
                "system.log file (if Windows)",
                "System syslog (if Unix/Linux)"
            ]
        }
        
    except Exception as e:
        logger.critical(f"CRITICAL: Error during logging test: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/alerts/status")
async def get_gmail_alert_status():
    """Xem trạng thái Gmail alerting system"""
    try:
        current_time = time.time()
        
        # Lấy metrics hiện tại
        error_rate = calculate_current_error_rate()
        
        # Lấy confidence
        avg_confidence = 0
        if confidence_samples:
            recent_confidences = [
                sample['confidence'] for sample in confidence_samples 
                if current_time - sample['timestamp'] <= 60
            ]
            if recent_confidences:
                avg_confidence = sum(recent_confidences) / len(recent_confidences)
        
        # Lấy system metrics
        cpu_usage = psutil.cpu_percent()
        memory_usage = psutil.virtual_memory().percent
        
        return {
            "gmail_alerts_enabled": GMAIL_ALERT_CONFIG["enabled"],
            "gmail_user": GMAIL_ALERT_CONFIG["gmail_user"][:5] + "***" if GMAIL_ALERT_CONFIG["gmail_user"] else "Not configured",
            "recipients_count": len(GMAIL_ALERT_CONFIG["recipients"]),
            "current_metrics": {
                "error_rate": f"{error_rate:.1f}%",
                "avg_confidence": f"{avg_confidence:.3f}",
                "cpu_usage": f"{cpu_usage:.1f}%",
                "memory_usage": f"{memory_usage:.1f}%"
            },
            "alert_thresholds": {
                "error_rate_critical": f"{ALERT_THRESHOLDS['error_rate_critical']}%",
                "confidence_critical": ALERT_THRESHOLDS["confidence_critical"],
                "cpu_critical": f"{ALERT_THRESHOLDS['cpu_critical']}%",
                "memory_critical": f"{ALERT_THRESHOLDS['memory_critical']}%"
            },
            "alert_status": {
                "error_rate": "CRITICAL" if error_rate >= ALERT_THRESHOLDS["error_rate_critical"] else "NORMAL",
                "confidence": "CRITICAL" if avg_confidence < ALERT_THRESHOLDS["confidence_critical"] else "NORMAL",
                "cpu": "CRITICAL" if cpu_usage >= ALERT_THRESHOLDS["cpu_critical"] else "NORMAL",
                "memory": "CRITICAL" if memory_usage >= ALERT_THRESHOLDS["memory_critical"] else "NORMAL"
            },
            "last_alerts": {alert_type: time.time() - last_time for alert_type, last_time in last_alert_times.items()}
        }
        
    except Exception as e:
        logger.error(f"Error getting alert status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/alerts/test")
async def test_gmail_alert():
    """Test gửi Gmail alert"""
    try:
        if not GMAIL_ALERT_CONFIG["enabled"]:
            return {
                "status": "disabled",
                "message": "Disable thong bao qua gmail r. Chinh GMAIL_ALERT_CONFIG['enabled'] = True de enable."
            }
        
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        subject = "TEST: Gmail Alert"
        message = f"""
Test Alert

Time: {current_time}
Alert Type: Test Email
Severity: TEST

Day la test email
Neu ban nhan duoc email, he thong is working good

Configuration:
- Nguoi dung: {GMAIL_ALERT_CONFIG["gmail_user"]}
- Recipients: {len(GMAIL_ALERT_CONFIG["recipients"])} email(s)
"""
        
        success = send_gmail_alert(subject, message)
        
        if success:
            return {
                "status": "success",
                "message": "Email test thanh cong",
                "recipients": GMAIL_ALERT_CONFIG["recipients"],
                "timestamp": current_time
            }
        else:
            return {
                "status": "failed",
                "message": "Not gui duoc, check log de co details",
                "timestamp": current_time
            }
            
    except Exception as e:
        logger.error(f"Loi gui Email: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/system-info")
async def get_system_info():
    """Get current system information"""
    try:
        # CPU info
        cpu_info = {
            "usage_percent": psutil.cpu_percent(interval=1),
            "count": psutil.cpu_count(),
            "count_logical": psutil.cpu_count(logical=True)
        }
        
        # Memory info
        memory = psutil.virtual_memory()
        memory_info = {
            "total": memory.total,
            "used": memory.used,
            "available": memory.available,
            "percent": memory.percent
        }
        
        # Disk info
        disk_info = []
        for partition in psutil.disk_partitions():
            try:
                partition_usage = psutil.disk_usage(partition.mountpoint)
                disk_info.append({
                    "device": partition.device,
                    "mountpoint": partition.mountpoint,
                    "fstype": partition.fstype,
                    "total": partition_usage.total,
                    "used": partition_usage.used,
                    "free": partition_usage.free,
                    "percent": partition_usage.percent
                })
            except PermissionError:
                continue
        
        # Network info
        network_info = {}
        network_io = psutil.net_io_counters(pernic=True)
        for interface, io_stats in network_io.items():
            # Skip loopback interfaces
            if interface.lower().startswith('lo'):
                continue
            
            # Skip VPN and virtual interfaces
            skip_keywords = [
                'vpn', 'tap', 'tun', 'radmin', 'openvpn', 'nordvpn', 'expressvpn',
                'vbox', 'vmware', 'hyper-v', 'docker', 'vethernet', 'bluetooth',
                'teredo', 'isatap', '6to4'
            ]
            
            if any(keyword in interface.lower() for keyword in skip_keywords):
                continue
            
            # Only show interfaces that are likely physical
            physical_keywords = ['wi-fi', 'wireless', 'ethernet', 'local area connection', 'wlan', 'eth']
            is_physical = any(keyword in interface.lower() for keyword in physical_keywords)
            
            # Skip if not physical and has low traffic (likely virtual)
            total_traffic = io_stats.bytes_sent + io_stats.bytes_recv
            if not is_physical and total_traffic < 1024 * 1024:  # Less than 1MB total
                continue
                
            network_info[interface] = {
                "bytes_sent": io_stats.bytes_sent,
                "bytes_recv": io_stats.bytes_recv,
                "packets_sent": io_stats.packets_sent,
                "packets_recv": io_stats.packets_recv,
                "is_physical": is_physical,
                "total_traffic_mb": total_traffic / (1024**2)
            }
        
        # GPU info (if available)
        gpu_info = []
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            for i, gpu in enumerate(gpus):
                gpu_info.append({
                    "id": i,
                    "name": gpu.name,
                    "load": gpu.load,
                    "memory_used": gpu.memoryUsed,
                    "memory_total": gpu.memoryTotal,
                    "temperature": gpu.temperature
                })
        except ImportError:
            gpu_info = "GPU monitoring not available (GPUtil not installed)"
        except Exception as e:
            gpu_info = f"GPU monitoring error: {str(e)}"
        
        return {
            "cpu": cpu_info,
            "memory": memory_info,
            "disk": disk_info,
            "network": network_info,
            "gpu": gpu_info,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Error getting system info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ===== start ====
@app.on_event("startup")
async def startup_event():
    """Initialize monitoring on startup"""
    import datetime
    startup_time = datetime.datetime.now().isoformat()
    
    # Log system startup event
    log_system_event("startup", f"ML Model API with Monitoring started at {startup_time}")
    
    logger.info("Starting ML Model API with System Monitoring...")
    
    # Load model
    global model
    model_path = './checkpoints/best_model_logging_demo.pth'
    try:
        model = load_model(model_path)
        logger.info("Model loaded successfully")
        log_system_event("model_load", f"Model loaded from {model_path}")
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {e}")
        log_system_event("error", f"Model loading failed: {e}")
    
    # Start background monitoring task
    asyncio.create_task(monitoring_background_task())
    logger.info("Background monitoring task started")
    log_system_event("monitoring", "Background monitoring task initialized")
    
    # Initial metrics collection
    update_all_system_metrics()
    logger.info("Initial system metrics collected")
    
    # Test logging system
    logger.info("Testing logging system on startup...")
    try:
        test_results = test_all_log_levels()
        logger.info(f"Logging test results: {test_results}")
    except Exception as e:
        logger.error(f"Logging test failed: {e}")
    
    # Initialize Gmail alerting system
    logger.info("Initializing Gmail alerting system...")
    if GMAIL_ALERT_CONFIG["enabled"]:
        logger.info(f"Gmail alerts ENABLED - User: {GMAIL_ALERT_CONFIG['gmail_user']}")
        logger.info(f"Recipients: {len(GMAIL_ALERT_CONFIG['recipients'])} email(s)")
        logger.info(f"Thresholds - Error Rate: {ALERT_THRESHOLDS['error_rate_critical']}%, Confidence: {ALERT_THRESHOLDS['confidence_critical']}")
    else:
        logger.warning("Gmail alerts DISABLED - Set GMAIL_ALERT_CONFIG['enabled'] = True to enable")
    
    log_system_event("startup", f"Gmail alerting system initialized - Enabled: {GMAIL_ALERT_CONFIG['enabled']}")
    log_system_event("startup", "ML Model API startup completed successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    import datetime
    shutdown_time = datetime.datetime.now().isoformat()
    
    # Log system shutdown event
    log_system_event("shutdown", f"ML Model API shutdown initiated at {shutdown_time}")
    
    logger.info("Shutting down ML Model API...")
    
    try:
        # Log final statistics before shutdown
        logger.info("=== FINAL STATISTICS BEFORE SHUTDOWN ===")
        
        # Get final metrics
        logger.info(f"Total requests processed: {len(request_timestamps)}")
        logger.info(f"Total errors: {len(error_timestamps)}")
        logger.info(f"Total model inferences: {len(inference_samples)}")
        
        # Log system resource usage at shutdown
        import psutil
        cpu_usage = psutil.cpu_percent()
        memory_usage = psutil.virtual_memory().percent
        logger.info(f"Final system state - CPU: {cpu_usage}%, Memory: {memory_usage}%")
        
        log_system_event("shutdown", f"Final stats - CPU: {cpu_usage}%, Memory: {memory_usage}%")
        
    except Exception as e:
        logger.error(f"Error collecting final statistics: {e}")
        log_system_event("error", f"Shutdown statistics collection failed: {e}")
    
    log_system_event("shutdown", "ML Model API shutdown completed")
    logger.info("ML Model API shutdown completed")

# Initialize model variable
model = None