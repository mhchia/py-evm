import logging
import sys

import pytest

import rlp

from eth_tester.backends.pyevm.main import get_default_account_keys

import eth_utils

from evm.utils.address import (
    generate_contract_address,
)

from evm.chains.sharding.mainchain_handler.config import (
    PERIOD_LENGTH,
    SHUFFLING_CYCLE_LENGTH,
)

from evm.chains.sharding.mainchain_handler.vmc_handler import (
    VMCHandler,
)

from evm.chains.sharding.mainchain_handler import (
    vmc_utils,
)

from evm.chains.sharding.mainchain_handler.backends.chain_handler import (
    RPCChainHandler,
)

from evm.chains.sharding.mainchain_handler.backends.tester_chain_handler import (
    TesterChainHandler,
)

test_keys = get_default_account_keys()

logger = logging.getLogger('evm.chain.sharding.mainchain_handler.VMCHandler')
stdout_handler = logging.StreamHandler(sys.stdout)
level = logging.DEBUG
logger.setLevel(level)
stdout_handler.setLevel(level)
logger.addHandler(stdout_handler)

@pytest.fixture
def chain_handler():
    return TesterChainHandler()

def do_withdraw(vmc_handler, validator_index):
    assert validator_index < len(test_keys)
    privkey = test_keys[validator_index]
    sender_addr = privkey.public_key.to_checksum_address()
    signature = vmc_utils.sign(vmc_utils.WITHDRAW_HASH, privkey)
    vmc_handler.withdraw(validator_index, signature, sender_addr)
    vmc_handler.chain_handler.mine(1)

def get_testing_colhdr(vmc_handler,
                       shard_id,
                       parent_collation_hash,
                       number,
                       collation_coinbase=test_keys[0].public_key.to_canonical_address(),
                       privkey=test_keys[0]):
    period_length = PERIOD_LENGTH
    expected_period_number = (vmc_handler.chain_handler.get_block_number() + 1) // period_length
    logger.debug("!@# add_header: expected_period_number=%s", expected_period_number)
    period_start_prevhash = vmc_handler.get_period_start_prevhash(expected_period_number)
    logger.debug("!@# add_header: period_start_prevhash=%s", period_start_prevhash)
    tx_list_root = b"tx_list " * 4
    post_state_root = b"post_sta" * 4
    receipt_root = b"receipt " * 4
    sighash = eth_utils.keccak(
        rlp.encode([
            shard_id,
            expected_period_number,
            period_start_prevhash,
            parent_collation_hash,
            tx_list_root,
            collation_coinbase,
            post_state_root,
            receipt_root,
            number,
        ])
    )
    sig = vmc_utils.sign(sighash, privkey)
    return rlp.encode([
        shard_id,
        expected_period_number,
        period_start_prevhash,
        parent_collation_hash,
        tx_list_root,
        collation_coinbase,
        post_state_root,
        receipt_root,
        number,
        sig,
    ])

def test_vmc_handler(chain_handler):
    shard_id = 0
    validator_index = 0
    primary_addr = test_keys[validator_index].public_key.to_checksum_address()
    zero_addr = eth_utils.to_checksum_address(b'\x00' * 20)

    vmc_handler = VMCHandler(chain_handler, primary_addr=primary_addr)
    logger.debug(
        "!@# viper_rlp_decoder_addr=%s",
        eth_utils.to_checksum_address(vmc_utils.viper_rlp_decoder_addr),
    )
    logger.debug(
        "!@# sighasher_addr=%s",
        eth_utils.to_checksum_address(vmc_utils.sighasher_addr)
    )

    if not vmc_handler.is_vmc_deployed():
        logger.debug('handler.is_vmc_deployed() == True')
        # import privkey
        for key in test_keys:
            vmc_handler.import_key_to_chain_handler(key)

        vmc_handler.deploy_initiating_contracts(test_keys[validator_index])
        vmc_handler.chain_handler.mine(1)
        vmc_handler.first_setup_and_deposit(test_keys[validator_index])

    assert vmc_handler.is_vmc_deployed()

    vmc_handler.chain_handler.mine(SHUFFLING_CYCLE_LENGTH)
    # handler.deploy_valcode_and_deposit(validator_index); handler.mine(1)

    assert vmc_handler.sample(shard_id) != zero_addr
    assert vmc_handler.get_num_validators() == 1
    logger.debug("!@# vmc_handler.get_num_validators()=%s", vmc_handler.get_num_validators())

    genesis_colhdr_hash = b'\x00' * 32
    header1 = get_testing_colhdr(vmc_handler, shard_id, genesis_colhdr_hash, 1)
    header1_hash = eth_utils.keccak(header1)
    vmc_handler.add_header(header1, primary_addr)
    vmc_handler.chain_handler.mine(SHUFFLING_CYCLE_LENGTH)

    header2 = get_testing_colhdr(vmc_handler, shard_id, header1_hash, 2)
    header2_hash = eth_utils.keccak(header2)
    vmc_handler.add_header(header2, primary_addr)
    vmc_handler.chain_handler.mine(SHUFFLING_CYCLE_LENGTH)

    assert vmc_handler.get_collation_header_score(shard_id, header1_hash) == 1
    assert vmc_handler.get_collation_header_score(shard_id, header2_hash) == 2

    vmc_handler.tx_to_shard(
        test_keys[1].public_key.to_checksum_address(),
        shard_id,
        100000,
        1,
        b'',
        1234567,
        primary_addr,
    )
    vmc_handler.chain_handler.mine(1)
    assert vmc_handler.get_receipt_value(0) == 1234567

    do_withdraw(vmc_handler, validator_index)
    vmc_handler.chain_handler.mine(1)
    assert vmc_handler.sample(shard_id) == zero_addr
