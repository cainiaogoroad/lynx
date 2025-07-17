#-*- coding: utf-8 -*-


def connect_cluster(rank: int, local_rank: int, island_id: int, shard_id: int,
                    replica_id: int, lm_server: str, is_driver: bool):
  """Description: Connect to the lynx cluster

  Args:
    rank: The rank of the servitors in the global cluster
    local_rank: The rank of the servitors in the host
    island_id: The id of the island containing this servitor
    shard_id: The id of the shard the servitor belongs to
    replica_id: The id of the replica the servitor belongs to
    lm_server: The local manager server to connect, for example, "127.0.0.1"
    is_driver: Whether the servitor is a driver
  """
  from . import _LYNXC_RT
  succ = (
      _LYNXC_RT._connect_cluster(rank, local_rank, island_id, shard_id,
                                 replica_id, lm_server, is_driver) == 0)
  if not succ:
    raise RuntimeError("Lynx failed to connect cluster")
