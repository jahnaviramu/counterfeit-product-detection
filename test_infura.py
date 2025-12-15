from web3 import Web3
import os
from dotenv import load_dotenv

load_dotenv()

INFURA_URL = os.getenv("INFURA_URL")
w3 = Web3(Web3.HTTPProvider(INFURA_URL))

print("Connected to Ethereum:", w3.is_connected())
print("Current Block:", w3.eth.block_number)
