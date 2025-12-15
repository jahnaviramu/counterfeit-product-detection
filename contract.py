import os
from dotenv import load_dotenv
from web3 import Web3

# Load environment variables from .env
load_dotenv()

contract_address = os.getenv("CONTRACT_ADDRESS")
print("Contract Address:", contract_address)
print("Is valid:", Web3.is_address(contract_address))
