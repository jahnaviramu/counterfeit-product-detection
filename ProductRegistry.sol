// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract ProductRegistry {
    struct Product {
        bool exists;
        string name;
        string brand;
        uint256 registrationDate;
    }
    
    mapping(bytes32 => Product) public products;
    
    event ProductRegistered(bytes32 indexed productId, string name, string brand);
    
    function registerProduct(
        bytes32 productId,
        string memory name,
        string memory brand,
        uint256 timestamp
    ) external {
        require(!products[productId].exists, "Product already registered");
        
        products[productId] = Product({
            exists: true,
            name: name,
            brand: brand,
            registrationDate: timestamp
        });
        
        emit ProductRegistered(productId, name, brand);
    }
    
    function getProduct(bytes32 productId) external view returns (
        bool exists,
        string memory name,
        string memory brand,
        uint256 registrationDate
    ) {
        Product memory p = products[productId];
        return (p.exists, p.name, p.brand, p.registrationDate);
    }
}