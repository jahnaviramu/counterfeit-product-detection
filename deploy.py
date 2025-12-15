from web3 import Web3
from dotenv import load_dotenv
import os
import json

load_dotenv()

w3 = Web3(Web3.HTTPProvider(os.getenv('INFURA_URL')))
private_key = os.getenv('PRIVATE_KEY')
account = w3.eth.account.from_key(private_key)

# Compile and get bytecode/abi
with open('ProductRegistry.json') as f:
    contract_data = json.load(f)

bytecode = contract_data['bytecode']
abi = contract_data['abi']

# Deploy contract
contract = w3.eth.contract(abi=abi, bytecode=bytecode)
nonce = w3.eth.get_transaction_count(account.address)

tx = contract.constructor().build_transaction({
    'chainId': 11155111,  # Sepolia
    'gas': 2000000,
    'gasPrice': w3.to_wei('50', 'gwei'),
    'nonce': nonce,
})

signed_tx = w3.eth.account.sign_transaction(tx, private_key)
tx_hash = w3.eth.send_raw_transaction(signed_tx.rawTransaction)
receipt = w3.eth.wait_for_transaction_receipt(tx_hash)

print(f"Contract deployed at: {receipt.contractAddress}")