import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from codesign.engines.callbacks import TestCallback
from pytorch_lightning.strategies import DDPStrategy

class TrainValTest:
    def __init__(self, cfg) -> None:
        self.cfg = cfg

    def __call__(self, model, data_module):
        # default callbacks
        # checkpoint_callback = ModelCheckpoint(
        #     dirpath=self.cfg.exp_dir,
        #     filename=f'{{epoch}}-{{val_{model.val_test_loss.name}:.2f}}',
        #     monitor=f'val_{model.val_test_loss.name}',
        #     mode=model.val_test_loss.mode, 
        #     verbose=False
        # )
        checkpoint_callback=ModelCheckpoint(
            dirpath=self.cfg.exp_dir,
            filename='best',
            monitor=f'val_{model.val_test_loss.name}',
            mode=model.val_test_loss.mode,
            save_top_k=1,
            verbose=False,
        )

        early_stop_callback = EarlyStopping(
            monitor=f'val_{model.val_test_loss.name}',
            patience=self.cfg.patience, 
            mode=model.val_test_loss.mode, 
            verbose=False
        )

        # customized callbacks
        additional_callbacks = []
        from codesign.config import CodesignTestConfigurator
        if 'callbacks' in self.cfg.keys():
            for callback_name in self.cfg.callbacks:
                test_callback_type = CodesignTestConfigurator.str_to_class(
                    "codesign.engines.callbacks", 
                    callback_name
                )
                additional_callbacks.append(test_callback_type(type(data_module).__name__, self.cfg, self.cfg))
        else:
            additional_callbacks.append(TestCallback(type(data_module).__name__, self.cfg, self.cfg))



        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            logger =WandbLogger(**dict(self.cfg.logger), entity='aujasvitd')
        else:
            logger = False  # no logging on other ranks


        # train model
        trainer = pl.Trainer(
            accelerator='gpu', 
            # devices=4, 
            devices=4, 
            # strategy=DDPStrategy(find_unused_parameters=False), 
            logger=logger,
            callbacks=[
                checkpoint_callback, 
                early_stop_callback,
                *additional_callbacks
            ],
            # profiler=pl.profilers.SimpleProfiler(),
            **dict(self.cfg.trainer)
        )

        # train/val model
        trainer.fit(model, data_module)

        # test the model
        trainer.test(model, data_module, ckpt_path='best')