import os


class World(object):

    @classmethod
    def set_world_state(cls, config):
        state = {
            "rank": int(os.environ["RANK"]),
            "local_rank": int(os.environ["LOCAL_RANK"]),
            "world_size": int(os.environ["WORLD_SIZE"]),
            "local_world_size": int(os.environ["LOCAL_WORLD_SIZE"]),
            "master_addr": os.environ["MASTER_ADDR"],
            "master_port": int(os.environ["MASTER_PORT"]),
            "is_master": int(os.environ["RANK"]) == 0,
        }
        for k, v in state.items():
            config[k] = v

