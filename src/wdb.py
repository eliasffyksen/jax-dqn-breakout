import wandb
from wandb.sdk.wandb_run import Run as WandbRun

from parameters import parse_parameter

_WDB: WandbRun | None
_WDB = None

def get_wdb(config, required=False) -> WandbRun | None:
    global _WDB

    if _WDB is not None:
        return _WDB
    
    wandb_api_key = parse_parameter('wandb-api-key', required)
    if wandb_api_key is None:
        return None

    wandb.login(key=wandb_api_key)

    _WDB = wandb.init(
        project='DQN-Breakout',
        config=config,        
    )
    
    return _WDB
