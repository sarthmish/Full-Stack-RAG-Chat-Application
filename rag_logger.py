# -*- coding: utf-8 -*-
"""

@author: sarthmish
"""
import json
import os
import logging.config

class StdOutLevelFilter:
    """Filters log records to only allow those at or below a certain level."""
    def __init__(self, level: str):
        self.level = getattr(logging, level.upper())

    def filter(self, record: logging.LogRecord) -> bool:
        """Returns True if the record should be logged, False otherwise."""
        return record.levelno <= self.level

# --- Setup logging directory and configuration ---

LOG_DIR = "logs"
CONFIG_FILE = "logger.json"

# Check if the log directory already exists and create it if not
if not os.path.exists(LOG_DIR):
    try:
        os.makedirs(LOG_DIR)
        print(f"Log directory '{LOG_DIR}' created successfully!")
    except OSError as e:
        print(f"Error creating directory {LOG_DIR}: {e}")
        # Exit or handle the error appropriately
        exit()

# Load logging configuration from a JSON file
try:
    with open(CONFIG_FILE, "r") as f:
        json_config = json.load(f)
        logging.config.dictConfig(json_config)
except FileNotFoundError:
    print(f"Error: Logging configuration file '{CONFIG_FILE}' not found.")
    # Fallback to basic configuration if the file is missing
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
except json.JSONDecodeError:
    print(f"Error: Could not decode JSON from '{CONFIG_FILE}'.")
    # Fallback to basic configuration
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


logger = logging.getLogger("root")
logger.info("Logging module imported and configured.")

# --- Main execution block ---

if __name__ == "__main__":
    # Redirect print statements to the logger (Note: this is not a standard feature)
    # The following print statement will go to standard output, not the logger.
    print("This is a standard print statement.")

    logger.debug("Testing the logging Module: This is a debug message.")
    logger.info("This is an info message.")
    logger.warning("This is a warning message.")
    logger.error("This is an error message.")