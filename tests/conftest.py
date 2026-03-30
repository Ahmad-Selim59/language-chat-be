import os
from dotenv import load_dotenv

test_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(test_dir)

# Load standard .env file before tests run
load_dotenv(os.path.join(project_dir, ".env"))
