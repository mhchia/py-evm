from eth_utils import (
    to_checksum_address,
)

from eth_tester.exceptions import (
    ValidationError,
)

from eth_tester.backends.pyevm.main import (
    get_default_account_keys,
)

from evm.utils.address import generate_contract_address

from tests.sharding.mainchain_handler.fixtures import (  # noqa: F401
    mainchain_handler,
)


PASSPHRASE = '123'

test_keys = get_default_account_keys()

code = """
num_test: public(num)

@public
def __init__():
    self.num_test = 42

@public
def update_num_test(_num_test: num):
    self.num_test = _num_test
"""


def test_mainchain_handler(mainchain_handler):  # noqa: F811
    privkey = test_keys[0]
    sender_addr = privkey.public_key.to_canonical_address()
    try:
        mainchain_handler.w3.personal.importRawKey(privkey.to_hex(), PASSPHRASE)
    # Exceptions happen when the key is already imported.
    #   - ValueError: `web3.py`
    #   - ValidationError: `eth_tester`
    except (ValueError, ValidationError):
        pass
    # ensure that we can mine, btw. confirm all pending transactions
    mainchain_handler.mine(1)
    # bytecode of the code above
    bytecode = b'`\x005`\x1cRt\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00` Ro\x7f\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff`@R\x7f\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\x80\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00``Rt\x01*\x05\xf1\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xfd\xab\xf4\x1c\x00`\x80R\x7f\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xfe\xd5\xfa\x0e\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00`\xa0R4\x15a\x00\x9eW`\x00\x80\xfd[`*`\x00Ua\x01\xa1V`\x005`\x1cRt\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00` Ro\x7f\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff`@R\x7f\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\x80\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00``Rt\x01*\x05\xf1\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xfd\xab\xf4\x1c\x00`\x80R\x7f\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xfe\xd5\xfa\x0e\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00`\xa0Rc\x94\x1ap\x93`\x00Q\x14\x15a\x00\xd2W` `\x04a\x01@74\x15a\x00\xb4W`\x00\x80\xfd[``Q`\x045\x80`@Q\x90\x13XW\x80\x91\x90\x12XWPa\x01@Q`\x00U\x00[c/\xb8z\xae`\x00Q\x14\x15a\x00\xf8W4\x15a\x00\xebW`\x00\x80\xfd[`\x00T`\x00R` `\x00\xf3\x00[[a\x00\xa8a\x01\xa1\x03a\x00\xa8`\x009a\x00\xa8a\x01\xa1\x03`\x00\xf3'  # noqa: E501
    abi = [{'name': '__init__', 'outputs': [], 'inputs': [], 'constant': False, 'payable': False, 'type': 'constructor'}, {'name': 'update_num_test', 'outputs': [], 'inputs': [{'type': 'int128', 'name': '_num_test'}], 'constant': False, 'payable': False, 'type': 'function'}, {'name': 'get_num_test', 'outputs': [{'type': 'int128', 'name': 'out'}], 'inputs': [], 'constant': True, 'payable': False, 'type': 'function'}]  # noqa: E501
    contract_addr = to_checksum_address(
        generate_contract_address(
            sender_addr,
            mainchain_handler.get_nonce(sender_addr)
        )
    )
    tx_hash = mainchain_handler.deploy_contract(bytecode, privkey)
    mainchain_handler.mine(1)
    receipt = mainchain_handler.get_transaction_receipt(tx_hash)
    # notice: `contractAddress` in web3.py, but `contract_address` in eth_tester
    assert ('contractAddress' in receipt) and (contract_addr == receipt['contractAddress'])
    contract = mainchain_handler.contract(contract_addr, abi=abi, bytecode=bytecode)
    result = contract.call({'from': to_checksum_address(sender_addr), 'gas': 50000}).get_num_test()
    assert result == 42
    mainchain_handler.mine(1)

    unsigned_transaction = contract.buildTransaction({
        'nonce': mainchain_handler.get_nonce(sender_addr),
        'gas': 50000,
        'gas_price': 1,
        'value': 0,
        'chainId': None,
    }).update_num_test(4)
    signed_transaction_dict = mainchain_handler.w3.eth.account.signTransaction(
        unsigned_transaction,
        privkey.to_hex(),
    )
    tx_hash = mainchain_handler.w3.eth.sendRawTransaction(
        signed_transaction_dict['rawTransaction'],
    )
    mainchain_handler.mine(1)

    result = contract.call({'from': to_checksum_address(sender_addr), 'gas': 50000}).get_num_test()
    assert result == 4