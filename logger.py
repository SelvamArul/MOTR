'''
Script to generate tensorboard log from MetricLogger logs
'''

import argparse
import json
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import time

CURRENT_EPOCH = 0
def log(metric_log, writer):
    global CURRENT_EPOCH
    with open(metric_log, 'r') as f:
        for log_dict in  f.readlines():
            log_dict = json.loads(log_dict.strip())
            epoch = int(log_dict["epoch"]) // 2
            if epoch > CURRENT_EPOCH:
                print (epoch, log_dict["train_loss"], log_dict["test_loss"])
                for key in log_dict.keys():
                    if 'loss' in key and 'aux' not in key:  
                        writer.add_scalar('Loss/' + key, log_dict[key], epoch)
                CURRENT_EPOCH = epoch
    writer.flush()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Logger')
    parser.add_argument('metric_log', help='Metric log written by the training script')
    parser.add_argument('--output_path', help='output_path', default="logs/")

    args = parser.parse_args()

    metric_log = Path(args.metric_log)
    if not metric_log.exists():
        import sys
        sys.exit("Invalid metric log path")
    
    writer = SummaryWriter(args.output_path)
    print ("Staring writer") 
    while True:
        log(metric_log, writer)
        time.sleep(15)
