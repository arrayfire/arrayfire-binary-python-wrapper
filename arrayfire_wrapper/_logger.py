import logging

# TODO Add env variable, add function

# Configure the logger
logging.basicConfig(level=logging.INFO)

# Create a logger
logger = logging.getLogger(__name__)

# Create a console handler and set the level to DEBUG
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Create a formatter and attach it to the console handler
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)

# Add the console handler to the logger
logger.addHandler(console_handler)
