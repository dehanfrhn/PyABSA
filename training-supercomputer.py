from utils import Notification


def scenario_1():
    # Load checkpoint
    # checkpoint = 'D:\\project\\PyABSA\\checkpoints\\fast_lsa_t_v2_AttractionReviewEn_acc_90.84_f1_85.39'

    # Load dataset
    dataset = DatasetItem("AttractionReview", "520.attraction_en")

    # Load config model
    config = APC.APCConfigManager.get_apc_config_english()
    config.learning_rate = 2e-5
    config.model = APCModelList.FAST_LSA_T_V2
    # config.pretrained_bert = 'indobenchmark/indobert-base-p1'
    config.cache_dataset = False
    config.max_seq_len = 135
    # config.max_seq_len = 105
    config.num_epoch = 30
    config.batch_size = 16
    config.lsa = True

    # Train model
    Notification.send('Training started!')
    try:
        trainer = APC.APCTrainer(
            config=config,
            dataset=dataset,
            # from_checkpoint=checkpoint,  # if you want to resume training from our pretrained checkpoints, you can pass the checkpoint name here
            auto_device=DeviceTypeOption.CUDA,
            path_to_save=None,  # set a path to save checkpoints, if it is None, save checkpoints at 'checkpoints' folder
            checkpoint_save_mode=ModelSaveOption.SAVE_MODEL_STATE_DICT,
            load_aug=False,  # there are some augmentation dataset for integrated datasets, you use them by setting load_aug=True to improve performance
        )
        Notification.send('Training completed!')
    except Exception as e:
        Notification.send(f'Training failed! {e}')
        raise e

def scenario_2():
    # Load checkpoint
    # checkpoint = 'D:\\project\\PyABSA\\checkpoints\\fast_lsa_t_v2_AttractionReviewEn_acc_90.84_f1_85.39'

    # Load dataset
    dataset = DatasetItem("AttractionReview", "520.attraction_en")

    # Load config model
    config = APC.APCConfigManager.get_apc_config_english()
    config.learning_rate = 2e-5
    config.model = APCModelList.FAST_LSA_T_V2
    config.pretrained_bert = 'microsoft/deberta-v3-base'
    config.cache_dataset = False
    config.max_seq_len = 135
    # config.max_seq_len = 105
    config.num_epoch = 30
    config.batch_size = 16
    config.lsa = True

    # Train model
    Notification.send('Training started!')
    try:
        trainer = APC.APCTrainer(
            config=config,
            dataset=dataset,
            # from_checkpoint=checkpoint,  # if you want to resume training from our pretrained checkpoints, you can pass the checkpoint name here
            auto_device=DeviceTypeOption.CUDA,
            path_to_save=None,  # set a path to save checkpoints, if it is None, save checkpoints at 'checkpoints' folder
            checkpoint_save_mode=ModelSaveOption.SAVE_MODEL_STATE_DICT,
            load_aug=False,  # there are some augmentation dataset for integrated datasets, you use them by setting load_aug=True to improve performance
        )
        Notification.send('Training completed!')
    except Exception as e:
        Notification.send(f'Training failed! {e}')
        raise e

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')
    import discordwebhook

    # Importing for pyabsa
    from pyabsa import AspectPolarityClassification as APC
    from pyabsa.tasks.AspectPolarityClassification.models import APCModelList
    from pyabsa import ModelSaveOption, DeviceTypeOption
    from pyabsa import DatasetItem
    scenario_2()
