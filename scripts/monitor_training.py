"""Monitor neural network training progress."""

import sys
import time
import re
from pathlib import Path

log_file = Path('logs/neural_training.log')

print("Monitoring neural training...")
print("=" * 60)

last_size = 0
last_epoch = 0

try:
    while True:
        if log_file.exists():
            current_size = log_file.stat().st_size
            
            if current_size > last_size:
                with open(log_file) as f:
                    f.seek(last_size)
                    new_content = f.read()
                    last_size = current_size
                    
                    # Extract key information
                    for line in new_content.split('\n'):
                        if 'Epoch' in line and '/' in line and '%' not in line:
                            print(line)
                        elif 'Validation' in line:
                            print(line)
                        elif 'Best model saved' in line:
                            print("✅ " + line)
                        elif 'Training complete' in line:
                            print("\n" + "=" * 60)
                            print(line)
                            sys.exit(0)
        
        time.sleep(5)
        
except KeyboardInterrupt:
    print("\n\nMonitoring stopped.")
