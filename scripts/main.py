import argparse, os
from codesign.config.config import CodesignConfigurator
import wandb

if __name__ == '__main__':        
    parser = argparse.ArgumentParser(description='MRI Codesign')
    parser.add_argument(
        "--config", "-c", 
        type=str, 
        help="Path to config file"
    )
    parser.add_argument(
        "--fix", "-f",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options using the command-line (e.g. python3 demo.py --config-file ***.py --opts key1 val1 key2 val2)"
    )
    args = parser.parse_args()

    local_rank = int(os.environ.get("SLURM_PROCID"))
    print("Local Rank is: ", local_rank, flush=True)
    wandb_mode = "online" if local_rank == 0 else "disabled"
    wandb.init(mode=wandb_mode)

    if args.fix is None:
        args.fix = []

    for k, v in wandb.config.items():
        args.fix.extend([k, str(v)])

    # Create configuration

    # Dynamically update exp_name based on sweep parameters
    # Reconstruct kernel shape from wandb.config
    ks = wandb.config.get("model.reconstructor.kernel_shape", [3,4])
    rc = wandb.config.get("model.reconstructor.radius_cutoff", 0.02)

    wandb.run.name = f"ks{ks}_rc{rc}"

    # Sanity check
    # if ks[0] is None or rc is None:
    #     raise ValueError("Sweep parameters kernel_shape or radius_cutoff are missing")

    # Format exp_name string
    exp_name = f"ks{ks}_rc{rc}"

    # Inject into args.fix
    args.fix.extend(["exp_name", exp_name])

    
    cc = CodesignConfigurator(args)

    # Save config to exp_dir
    os.makedirs(cc.cfg.exp_dir, exist_ok=True)
    with open(f'{cc.cfg.exp_dir}/config.yaml', 'w') as f:
        f.write(str(cc.cfg))

    exp, model, data_module = cc.init_all()
    exp(model, data_module)



# import argparse, os
# from codesign.config.config import CodesignConfigurator
# import wandb

# if __name__ == '__main__':        
#     parser = argparse.ArgumentParser(description='MRI Codesign')
#     parser.add_argument(
#         "--config", "-c", 
#         type=str, 
#         help="Path to config file"
#     )
#     parser.add_argument(
#         "--fix", "-f",
#         default=None,
#         nargs=argparse.REMAINDER,
#         help="Modify config options using the command-line (e.g. python3 demo.py --config-file ***.py --opts key1 val1 key2 val2)"
#     )
#     args = parser.parse_args()

#     local_rank = int(os.environ.get("LOCAL_RANK", 0))
#     wandb_mode = "online" if local_rank == 0 else "disabled"
#     wandb.init(mode=wandb_mode)

#     if args.fix is None:
#         args.fix = []

#     for k, v in wandb.config.items():
#         args.fix.extend([k, str(v)])

#     # Create configuration

#     # Dynamically update exp_name based on sweep parameters
#     # Reconstruct kernel shape from wandb.config
#     ks = wandb.config.get("model.reconstructor.kernel_shape")
#     rc = wandb.config.get("model.reconstructor.radius_cutoff")

#     wandb.run.name = f"ks{ks}_rc{rc}"

#     # Sanity check
#     # if ks[0] is None or rc is None:
#     #     raise ValueError("Sweep parameters kernel_shape or radius_cutoff are missing")

#     # Format exp_name string
#     exp_name = f"ks{ks[0]}x{ks[1]}_rc{rc}"

#     # Inject into args.fix
#     args.fix.extend(["exp_name", exp_name])

    
#     cc = CodesignConfigurator(args)

#     # Save config to exp_dir
#     os.makedirs(cc.cfg.exp_dir, exist_ok=True)
#     with open(f'{cc.cfg.exp_dir}/config.yaml', 'w') as f:
#         f.write(str(cc.cfg))

#     exp, model, data_module = cc.init_all()
#     exp(model, data_module)
