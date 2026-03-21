from argparse import Namespace
from copy import deepcopy

from yacs.config import CfgNode


class CN(CfgNode):

    def __init__(self, init_dict=None, key_list=None, new_allowed=False):
        super().__init__(init_dict, key_list, new_allowed)
        self.recursive_cfg_update()

    def recursive_cfg_update(self):

        for k, v in self.items():
            if isinstance(v, list):
                for i, v_ in enumerate(v):
                    if isinstance(v_, dict):
                        new_v = CN(v_, new_allowed=True)
                        v[i] = new_v.recursive_cfg_update()
            elif isinstance(v, CN) or issubclass(type(v), CN):
                new_v = CN(v, new_allowed=True)
                self[k] = new_v.recursive_cfg_update()
        # self.freeze()
        return self

    def dump(self, *args, **kwargs):

        def change_back(cfg: CN) -> dict:
            for k, v in cfg.items():
                if isinstance(v, list):
                    for i, v_ in enumerate(v):
                        if isinstance(v_, CN):
                            new_v = change_back(v_)
                            v[i] = new_v
                elif isinstance(v, CN):
                    new_v = change_back(v)
                    cfg[k] = new_v
            return dict(cfg)

        cfg = change_back(deepcopy(self))
        return CfgNode(cfg).dump(*args, **kwargs)


def get_config(config_file: str) -> CN:
    """
    Read a config file and optionally merge it with the default config file.
    Args:
      config_file (str): Path to config file.
    Returns:
      CfgNode: Config as a yacs CfgNode object.
    """

    cfg = CN(new_allowed=True)
    cfg.merge_from_file(config_file)

    return cfg
