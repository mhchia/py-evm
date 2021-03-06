import asyncio
import logging
import pytest

from evm.vm.forks.sharding.windback_worker import (
    WindbackWorker,
)

from tests.sharding.fixtures import (  # noqa: F401
    default_shard_id,
    make_collation_header_chain,
    shard_tracker,
    smc_handler,
)


logger = logging.getLogger("evm.chain.sharding.windback_worker")
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("[%(levelname)s] %(module)s::%(funcName)s\t| %(message)s")
console.setFormatter(formatter)
logger.addHandler(console)


SIMULATED_COLLATION_DOWNLOADING_TIME = 2
SIMULATED_COLLATION_VERIFICATION_TIME = 0.01


async def mock_download_collation(collation_hash):
    logger.debug("Start downloading collation %s", collation_hash)
    await asyncio.sleep(SIMULATED_COLLATION_DOWNLOADING_TIME)
    logger.debug("Finished downloading collation %s", collation_hash)
    collation = collation_hash
    return collation


async def mock_verify_collation(collation, collation_hash):
    logger.debug("Verifying collation %s", collation_hash)
    await asyncio.sleep(SIMULATED_COLLATION_VERIFICATION_TIME)
    return True


@pytest.fixture  # noqa: F811
def windback_worker(smc_handler, monkeypatch):
    monkeypatch.setattr(
        'evm.vm.forks.sharding.windback_worker.download_collation',
        mock_download_collation,
    )
    monkeypatch.setattr(
        'evm.vm.forks.sharding.windback_worker.verify_collation',
        mock_verify_collation,
    )
    return WindbackWorker(
        smc_handler,
        shard_tracker(smc_handler, default_shard_id),
    )


def test_guess_head_no_new_collations(windback_worker, smc_handler):  # noqa: F811
    assert windback_worker.run_guess_head() is None


def test_guess_head_without_fork(windback_worker, smc_handler):  # noqa: F811
    header2_hash = make_collation_header_chain(smc_handler, default_shard_id, 2)
    assert windback_worker.run_guess_head() == header2_hash
    header3_hash = make_collation_header_chain(
        smc_handler,
        default_shard_id,
        1,
        header2_hash,
    )
    assert windback_worker.run_guess_head() == header3_hash
    # ensure all of the collations in the chain are verified
    assert len(windback_worker.collation_validity) == 3


def test_guess_head_with_fork(windback_worker, smc_handler):  # noqa: F811
    # without fork
    make_collation_header_chain(smc_handler, default_shard_id, 2)
    header3_prime_hash = make_collation_header_chain(smc_handler, default_shard_id, 3)
    # head changes
    assert windback_worker.run_guess_head() == header3_prime_hash
    # ensure only the chain of the best candidate is verified
    assert len(windback_worker.collation_validity) == 3


def test_guess_head_invalid_longest_chain(windback_worker, smc_handler):  # noqa: F811
    # setup two collation header chains, both having length=3.
    # originally, guess_head should return the hash of canonical chain head `header0_3_hash`
    header3_hash = make_collation_header_chain(smc_handler, default_shard_id, 3)
    header4_hash = make_collation_header_chain(
        smc_handler,
        default_shard_id,
        1,
        top_collation_hash=header3_hash,
    )
    header3_prime_hash = make_collation_header_chain(smc_handler, default_shard_id, 3)
    windback_worker.collation_validity[header3_hash] = False
    # the candidates is  [`header3`, `header3_prime`, `header2`, ...]
    # since the 1st candidate is invalid, `guess_head` should returns `header3_prime` instead
    assert windback_worker.run_guess_head() == header3_prime_hash
    assert not windback_worker.collation_validity[header4_hash]


def test_guess_head_new_only_candidate_is_invalid(windback_worker, smc_handler):  # noqa: F811
    head_header_hash = make_collation_header_chain(smc_handler, default_shard_id, 1)
    windback_worker.collation_validity[head_header_hash] = False
    assert windback_worker.run_guess_head() is None


def test_guess_head_new_only_candidate_is_not_longest(windback_worker, smc_handler):  # noqa: F811
    head_header_hash = make_collation_header_chain(smc_handler, default_shard_id, 3)
    windback_worker.run_guess_head()
    make_collation_header_chain(smc_handler, default_shard_id, 1)
    assert windback_worker.run_guess_head() == head_header_hash


def test_guess_head_windback_length(windback_worker, smc_handler):  # noqa: F811
    # mock WINDBACK_LENGTH to a number less than the height of head_collation,
    # to make testing faster
    smc_handler.config['WINDBACK_LENGTH'] = 3
    # build a chain with head_collation_height = 5
    make_collation_header_chain(smc_handler, default_shard_id, 5)
    windback_worker.run_guess_head()
    # the size of `collation_validity` should be the WINDBACK_LENGTH + 1(including the
    # `head_collation` itself), instead of the length of the chain
    num_verified_collations = len(windback_worker.collation_validity)
    assert num_verified_collations == smc_handler.config['WINDBACK_LENGTH'] + 1


def test_guess_head_invalid_collation_propagate_invalidity(windback_worker,  # noqa: F811
                                                           smc_handler):
    header2_hash = make_collation_header_chain(smc_handler, default_shard_id, 2)
    windback_worker.collation_validity[header2_hash] = False
    header4_hash = make_collation_header_chain(
        smc_handler,
        default_shard_id,
        2,
        header2_hash,
    )
    windback_worker.run_guess_head()
    assert not windback_worker.collation_validity[header2_hash]
    assert not windback_worker.collation_validity[header4_hash]


def test_guess_head_invalid_chain_propagate_invalidity(windback_worker, smc_handler):  # noqa: F811
    header2_hash = make_collation_header_chain(smc_handler, default_shard_id, 2)
    windback_worker.collation_validity[header2_hash] = False
    header3_hash = make_collation_header_chain(
        smc_handler,
        default_shard_id,
        1,
        header2_hash,
    )
    header4_hash = make_collation_header_chain(
        smc_handler,
        default_shard_id,
        1,
        header3_hash,
    )
    windback_worker.run_guess_head()
    assert not windback_worker.collation_validity[header3_hash]
    assert not windback_worker.collation_validity[header4_hash]


def test_guess_head_time_out(windback_worker, smc_handler, monkeypatch):  # noqa: F811
    header1_hash = make_collation_header_chain(smc_handler, default_shard_id, 1)

    async def mock_download_collation_nonstop(collation_hash):
        while True:
            await asyncio.sleep(1)

    monkeypatch.setattr(
        'evm.vm.forks.sharding.windback_worker.download_collation',
        mock_download_collation_nonstop,
    )

    time_up_event = asyncio.Event()
    time_up_event.set()
    # originally it should be blocked forever, but is stopped due to `time_up_event.set``
    assert windback_worker.run_guess_head(time_up_event) == header1_hash
