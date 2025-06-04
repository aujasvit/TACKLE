import argparse, os
from codesign.config.config import CodesignTestConfigurator
import wandb

if __name__ == '__main__':        
    parser = argparse.ArgumentParser(description='MRI Codesign')
    parser.add_argument(
        "--config", "-c", 
        type=str, 
        help="Path to config file"
    )
    parser.add_argument(
        "--data-config", "-d",
        type=str, 
        default=None,
        help="Name of the data module"
    )
    parser.add_argument(
        "--id", "-i",
        type=str, 
        default=None,
        help="Wandb ID of the run to be tested"
    )
    parser.add_argument(
        "--fix", "-f",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options using the command-line (e.g. python3 demo.py --config-file ***.py --opts key1 val1 key2 val2)"
    )
    args = parser.parse_args()
    cc = CodesignTestConfigurator(args)
    

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    wandb_mode = "online" if local_rank == 0 else "disabled"
    wandb.init(mode=wandb_mode)

    if local_rank == 0 and hasattr(cc.cfg, "exp_name"):       
        wandb.run.name = cc.cfg.exp_name
        wandb.run.save()


    # initialize exp, ckpt, model, and data_module
    exp, model, data_module = cc.init_all()
    
    exp.test_run(args, model, data_module)

    wandb.finish()