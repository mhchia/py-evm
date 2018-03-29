import asyncio
from collections import (
    defaultdict,
)
import time

import logging

from evm.vm.forks.sharding.constants import (
    GENESIS_COLLATION_HASH,
)

from evm.vm.forks.sharding.shard_tracker import (
    NoCandidateHead,
)

NUM_RESERVED_BLOCKS = 5


logger = logging.getLogger("evm.chain.sharding.guess_head_state_manager")


async def download_collation(collation_hash):
    # TODO: need to implemented after p2p part is done
    print("!@# Start downloading collation {}".format(collation_hash))
    await asyncio.sleep(2)
    print("!@# Finished downloading collation {}".format(collation_hash))
    collation = collation_hash
    return collation


async def verify_collation(collation, collation_hash):
    # TODO: to be implemented
    print("!@# Verifying collation {}".format(collation_hash))
    await asyncio.sleep(0.01)
    # data = get_from_p2p_network(collation_hash.data_root)
    # chunks = DATA_SIZE // 32
    # mtree = [0] * chunks + [data[chunks*32: chunks*32+32] for i in range(chunks)]
    # for i in range(chunks-1, 0, -1):
    #     mtree[i] = sha3(mtree[i*2] + mtree[i*2+1])
    # assert mtree[i] == collation.data_root
    return True


async def fetch_and_verify_collation(collation_hash):
    """Fetch the collation body and verify the collation

    :return: returns the collation's validity
    """
    collation = await download_collation(collation_hash)
    return await verify_collation(collation, collation_hash)


def create_collation(parent_hash, data):
    # return collation with data with head = candidate_head
    # ablalalala
    print("!@# create_collation: parent_hash={}, data={}".format(
        parent_hash,
        data,
    ))
    return True


def clean_done_task(tasks):
    return [task for task in tasks if not task.done()]


class GuessHeadStateManager:

    logger = logging.getLogger("evm.chain.sharding.GuessHeadStateManager")

    TIME_ONE_STEP = 0.1

    collation_validity_cache = None
    head_validity = None
    head_collation_hash = None
    current_collation_hash = None
    last_period_fetching_candidate_head = None
    tasks = None
    collation_task = None
    chain_collations = None

    def __init__(self, vmc, shard_id, shard_tracker, my_address):
        # a well-setup vmc handler with a shard_tracker in shard `shard_id`
        self.vmc = vmc
        self.shard_id = shard_id
        self.shard_tracker = shard_tracker
        self.my_address = my_address

        # map[collation] -> validity
        self.collation_validity_cache = {}
        # map[chain_head] -> validity
        # this should be able to be updated by the collations' validity
        self.head_validity = defaultdict(lambda: True)
        # list of chain head, to indicate priority
        # order: older -------> newer
        self.last_period_fetching_candidate_head = 0

        # current tasks
        self.tasks = []
        # collation task
        self.collation_task = {}
        self.chain_collations = defaultdict(list)

    def get_current_period(self):
        return self.vmc.web3.eth.blockNumber // self.vmc.config['PERIOD_LENGTH']

    def is_collator_in_period(self, period):
        collator_address = self.vmc.get_eligible_proposer(
            self.shard_id,
            period,
        )
        return collator_address == self.my_address

    def is_collator_in_lookahead_periods(self):
        """
        See if we are going to be a collator in the future periods
        """
        result = False
        for future_periods in range(self.vmc.config['LOOKAHEAD_PERIODS']):
            lookahead_period = self.get_current_period() + future_periods
            result |= self.is_collator_in_period(lookahead_period)
        return result

    def is_late_collator_period(self):
        current_period = self.get_current_period()
        current_block_num = self.vmc.web3.eth.blockNumber
        last_block_num_in_current_period = current_period * self.vmc.config['PERIOD_LENGTH']
        return (
            self.is_collator_in_period(current_period) and
            (current_block_num + NUM_RESERVED_BLOCKS >= last_block_num_in_current_period)
        )

    async def verify_collation(self, head_collation_hash, collation_hash):
        coroutine = fetch_and_verify_collation(
            collation_hash,
        )
        task = asyncio.ensure_future(coroutine)
        collation_validity = await asyncio.wait_for(task, None)
        self.collation_validity_cache[collation_hash] = collation_validity
        if not collation_validity:
            self.head_validity[head_collation_hash] = False

    def fetch_candidate_head_hash(self):
        head_collation_hash = None
        try:
            head_collation_dict = self.shard_tracker.fetch_candidate_head()
            head_collation_hash = head_collation_dict['header'].hash
        except NoCandidateHead:
            self.logger.debug("No candidate head available, `guess_head` stops")
        self.last_period_fetching_candidate_head = self.get_current_period()
        return head_collation_hash

    def try_change_head(self):
        # check for head changing
        current_period = self.get_current_period()
        # if head and current are None in the same time, it means it is just started up
        # should try to get the head for it
        if ((self.head_collation_hash is None and self.current_collation_hash is None) or
                (current_period > self.last_period_fetching_candidate_head) or
                (not self.head_validity[self.head_collation_hash])):
            if current_period > self.last_period_fetching_candidate_head:
                # TODO: should check if it is correct
                # flush old candidate heads first, since all those candidates are stale
                self.shard_tracker.clean_logs()
                pass
            # perform head changing
            self.head_collation_hash = self.fetch_candidate_head_hash()
            if self.head_collation_hash is not None:
                self.current_collation_hash = self.head_collation_hash

    def set_collation_task(self, head_collation_hash, current_collation_hash, task):
        self.tasks.append(task)
        self.collation_task[current_collation_hash] = task
        self.chain_collations[head_collation_hash].append(current_collation_hash)

    def get_chain_tasks(self, head_collation_hash):
        # ignore tasks that are done
        return [
            self.collation_task[collation_hash]
            for collation_hash in self.chain_collations[head_collation_hash]
        ]

    def process_current_collation(self):
        # only process collations when the node is collating
        if self.head_collation_hash is None:
            return
        if self.current_collation_hash is None:
            return
        if self.current_collation_hash not in self.collation_validity_cache:
            # process current collation
            coro = self.verify_collation(
                self.head_collation_hash,
                self.current_collation_hash,
            )
            task = asyncio.ensure_future(coro)
            asyncio.wait(task)
            self.set_collation_task(self.head_collation_hash, self.current_collation_hash, task)
        if self.current_collation_hash != GENESIS_COLLATION_HASH:
            self.current_collation_hash = self.vmc.get_parent_hash(
                self.shard_id,
                self.current_collation_hash,
            )

    def try_create_collation(self, enable_running_out_of_time=False):
        # Check if it is time to collate  #################################
        if self.head_collation_hash is None:
            return None
        # if verified enough collations
        # TODO: be replaced by the windback length checks
        if self.current_collation_hash == GENESIS_COLLATION_HASH:
            candidate_chain_tasks = self.get_chain_tasks(self.head_collation_hash)
            if len(candidate_chain_tasks) != 0:
                # check if all of the tasks in the chain are done
                is_tasks_done = True
                for task in candidate_chain_tasks:
                    is_tasks_done &= task.done()
                candidate_chain_tasks = clean_done_task(candidate_chain_tasks)
                if not is_tasks_done:
                    return None
            # ensure all of the collations are done and valid
            if not self.head_validity[self.head_collation_hash]:
                return None
        # or running out of time
        elif enable_running_out_of_time and self.is_late_collator_period():
            pass
        # other cases, should not create collation
        else:
            return None
        head_collation_hash = self.head_collation_hash
        create_collation(head_collation_hash, data="123")
        self.head_collation_hash = self.current_collation_hash = None
        return head_collation_hash

    async def async_loop_main(self, stop_after_create_collation=False):
        # At any time, there should be only one `head_collation`
        # The timing where `head_collation_hash` change are:
        #    1. `head_collation` is invalid, change to the new head
        #    2. There are new logs, and `fetch_candidate_head` is called
        # When to stop processing collation?
        self.head_collation_hash = self.current_collation_hash = None
        is_verifying_collations = False
        self.tasks = []
        while True:
            is_verifying_collations = (
                self.is_collator_in_lookahead_periods() or
                self.head_collation_hash is None
            )
            self.try_change_head()

            if is_verifying_collations:
                self.process_current_collation()

            parent_hash = self.try_create_collation()
            if parent_hash is not None:
                return parent_hash

            # clean up the finished tasks
            if len(self.tasks) != 0:
                await asyncio.wait(self.tasks, timeout=self.TIME_ONE_STEP)
                not_finished_tasks = clean_done_task(self.tasks)
                self.tasks = not_finished_tasks

            await asyncio.sleep(self.TIME_ONE_STEP)

    def async_daemon(self, stop_after_create_collation=False):
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(self.async_loop_main(stop_after_create_collation))
        return result
