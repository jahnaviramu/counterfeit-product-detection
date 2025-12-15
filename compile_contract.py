from solcx import compile_standard, install_solc
import json
import os

# Install Solidity compiler version (match this to your .sol version pragma)
install_solc("0.8.0")

# Load Solidity source code
contract_path = "E:\\majorfinal\\brand-auth-backend\\contracts\ProductRegistry.sol"  # No need for "contracts/"
with open(contract_path, "r") as file:
    product_registry_source = file.read()

# Compile the contract
compiled_sol = compile_standard(
    {
        "language": "Solidity",
        "sources": {
            "ProductRegistry.sol": {
                "content": product_registry_source
            }
        },
        "settings": {
            "outputSelection": {
                "*": {
                    "*": ["abi", "metadata", "evm.bytecode", "evm.sourceMap"]
                }
            }
        }
    },
    solc_version="0.8.0"
)

# Save the full output to a file (optional)
with open("contracts/ProductRegistry.json", "w") as f:
    json.dump(compiled_sol, f, indent=4)

# Extract and save just the ABI (optional)
abi = compiled_sol['contracts']['ProductRegistry.sol']['ProductRegistry']['abi']
with open("contracts/ProductRegistry.abi.json", "w") as f:
    json.dump(abi, f, indent=4)

print("✅ Contract compiled successfully and ABI saved.")
