import os

class NgramConstants(object):
    TOKEN_DELIMITER = " "


class SkypilotConstants(object):

    RANK_ENV_VAR = "SKYPILOT_NODE_RANK"
    IPS_ENV_VAR = "SKYPILOT_NODE_IPS"

    @classmethod
    def get_rank(cls, default=None):
        rank = default
        rank_env = os.getenv(SkypilotConstants.RANK_ENV_VAR, None)
        if rank_env is not None:
            rank = int(rank_env)
        return rank

    @classmethod
    def get_world_size(cls, default=None):
        world_size = default
        ips = os.getenv(SkypilotConstants.IPS_ENV_VAR, None)
        if ips is not None:
            world_size = len(ips.split())
        return world_size
