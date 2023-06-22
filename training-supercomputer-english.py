def baseline_scenario(pretrained):

    # Load checkpoint
    # checkpoint = 'D:\\project\\PyABSA\\checkpoints\\fast_lsa_t_v2_AttractionReviewEn_acc_90.84_f1_85.39'

    # Load dataset
    dataset = DatasetItem("AttractionReviewEn", "520.attraction_en")

    # Load config model
    config = APC.APCConfigManager.get_apc_config_english()
    config.learning_rate = 2e-5
    config.model = APCModelList.FAST_LSA_T_V2
    config.pretrained_bert = pretrained
    config.cache_dataset = False
    config.max_seq_len = 135
    # config.max_seq_len = 105
    config.num_epoch = 15
    config.batch_size = 16
    config.lsa = True
    config.use_amp = True
    config.scenario_case = 'baseline scenario'

    # Train model
    Notification.send(f'Training started! {pretrained}')
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


# continues epochs
def scenario_en_b_01():

    # Load checkpoint
    checkpoint = 'D:\\project\\PyABSA\\checkpoints\\fast_lsa_t_v2_AttractionReviewEn_acc_91.82_f1_86.05'

    # Load dataset
    dataset = DatasetItem("AttractionReviewEn", "520.attraction_en")

    # Load config model
    config = APC.APCConfigManager.get_apc_config_english()
    config.learning_rate = 2e-5
    config.model = APCModelList.FAST_LSA_T_V2
    config.pretrained_bert = 'yangheng/deberta-v3-base-absa-v1.1'
    config.cache_dataset = False
    config.max_seq_len = 135
    # config.max_seq_len = 105
    config.num_epoch = 15
    config.batch_size = 16
    config.lsa = True
    config.use_amp = True
    config.scenario_case = 'SCENARIO EN-B-01'

    # Train model
    Notification.send(f'Training started! {config.pretrained_bert}')
    try:
        trainer = APC.APCTrainer(
            config=config,
            dataset=dataset,
            from_checkpoint=checkpoint,  # if you want to resume training from our pretrained checkpoints, you can pass the checkpoint name here
            auto_device=DeviceTypeOption.CUDA,
            path_to_save=None,  # set a path to save checkpoints, if it is None, save checkpoints at 'checkpoints' folder
            checkpoint_save_mode=ModelSaveOption.SAVE_MODEL_STATE_DICT,
            load_aug=False,  # there are some augmentation dataset for integrated datasets, you use them by setting load_aug=True to improve performance
        ).destroy()
        Notification.send('Training completed!')
    except Exception as e:
        Notification.send(f'Training failed! {e}')
        raise e


# batch size = 8
def scenario_en_b_02():

    # Load checkpoint
    checkpoint = 'D:\\project\\PyABSA\\checkpoints\\fast_lsa_t_v2_AttractionReviewEn_acc_91.82_f1_86.05'

    # Load dataset
    dataset = DatasetItem("AttractionReviewEn", "520.attraction_en")

    # Load config model
    config = APC.APCConfigManager.get_apc_config_english()
    config.learning_rate = 2e-5
    config.model = APCModelList.FAST_LSA_T_V2
    config.pretrained_bert = 'yangheng/deberta-v3-base-absa-v1.1'
    config.cache_dataset = False
    config.max_seq_len = 135
    # config.max_seq_len = 105
    config.num_epoch = 15
    config.batch_size = 8
    config.lsa = True
    config.use_amp = True
    config.scenario_case = 'SCENARIO EN-B-02 (BATCH SIZE 8)'

    # Train model
    Notification.send(f'Training started! {config.pretrained_bert}')
    try:
        trainer = APC.APCTrainer(
            config=config,
            dataset=dataset,
            # from_checkpoint=checkpoint,  # if you want to resume training from our pretrained checkpoints, you can pass the checkpoint name here
            auto_device=DeviceTypeOption.CUDA,
            path_to_save=None,  # set a path to save checkpoints, if it is None, save checkpoints at 'checkpoints' folder
            checkpoint_save_mode=ModelSaveOption.SAVE_MODEL_STATE_DICT,
            load_aug=False,  # there are some augmentation dataset for integrated datasets, you use them by setting load_aug=True to improve performance
        ).destroy()
        Notification.send('Training completed!')
    except Exception as e:
        Notification.send(f'Training failed! {e}')
        raise e


# batch size = 24
def scenario_en_b_03():

    # Load checkpoint
    checkpoint = 'D:\\project\\PyABSA\\checkpoints\\fast_lsa_t_v2_AttractionReviewEn_acc_91.82_f1_86.05'

    # Load dataset
    dataset = DatasetItem("AttractionReviewEn", "520.attraction_en")

    # Load config model
    config = APC.APCConfigManager.get_apc_config_english()
    config.learning_rate = 2e-5
    config.model = APCModelList.FAST_LSA_T_V2
    config.pretrained_bert = 'yangheng/deberta-v3-base-absa-v1.1'
    config.cache_dataset = False
    config.max_seq_len = 135
    # config.max_seq_len = 105
    config.num_epoch = 15
    config.batch_size = 24
    config.lsa = True
    config.use_amp = True
    config.scenario_case = 'SCENARIO EN-B-03 (BATCH SIZE 24)'

    # Train model
    Notification.send(f'Training started! {config.pretrained_bert}')
    try:
        trainer = APC.APCTrainer(
            config=config,
            dataset=dataset,
            # from_checkpoint=checkpoint,  # if you want to resume training from our pretrained checkpoints, you can pass the checkpoint name here
            auto_device=DeviceTypeOption.CUDA,
            path_to_save=None,  # set a path to save checkpoints, if it is None, save checkpoints at 'checkpoints' folder
            checkpoint_save_mode=ModelSaveOption.SAVE_MODEL_STATE_DICT,
            load_aug=False,  # there are some augmentation dataset for integrated datasets, you use them by setting load_aug=True to improve performance
        ).destroy()
        Notification.send('Training completed!')
    except Exception as e:
        Notification.send(f'Training failed! {e}')
        raise e


# batch size it depends on best scenario (b01 - b03) & applying dropout 0.3
def scenario_en_b_04(batch_size):

    # Load checkpoint
    # checkpoint = 'D:\\project\\PyABSA\\checkpoints\\fast_lsa_t_v2_AttractionReviewEn_acc_91.82_f1_86.05'

    # Load dataset
    dataset = DatasetItem("AttractionReviewEn", "520.attraction_en")

    # Load config model
    config = APC.APCConfigManager.get_apc_config_english()
    config.learning_rate = 2e-5
    config.model = APCModelList.FAST_LSA_T_V2
    config.pretrained_bert = 'yangheng/deberta-v3-base-absa-v1.1'
    config.cache_dataset = False
    config.max_seq_len = 135
    # config.max_seq_len = 105
    config.num_epoch = 15
    config.batch_size = batch_size
    config.lsa = True
    config.use_amp = True
    config.dropout = 0.3
    config.scenario_case = 'SCENARIO EN-B-04 (BATCH SIZE ... , DROP OUT 0.3)'

    # Train model
    Notification.send(f'Training started! {config.pretrained_bert}')
    try:
        trainer = APC.APCTrainer(
            config=config,
            dataset=dataset,
            # from_checkpoint=checkpoint,  # if you want to resume training from our pretrained checkpoints, you can pass the checkpoint name here
            auto_device=DeviceTypeOption.CUDA,
            path_to_save=None,  # set a path to save checkpoints, if it is None, save checkpoints at 'checkpoints' folder
            checkpoint_save_mode=ModelSaveOption.SAVE_MODEL_STATE_DICT,
            load_aug=False,  # there are some augmentation dataset for integrated datasets, you use them by setting load_aug=True to improve performance
        ).destroy()
        Notification.send('Training completed!')
    except Exception as e:
        Notification.send(f'Training failed! {e}')
        raise e


def scenario_en_b_05(dropout):

    # Load checkpoint
    # checkpoint = 'D:\\project\\PyABSA\\checkpoints\\fast_lsa_t_v2_AttractionReviewEn_acc_91.82_f1_86.05'

    # Load dataset
    dataset = DatasetItem("AttractionReviewEn", "520.attraction_en")

    # Load config model
    config = APC.APCConfigManager.get_apc_config_english()
    config.model = APCModelList.FAST_LSA_T_V2
    config.pretrained_bert = 'yangheng/deberta-v3-base-absa-v1.1'
    config.learning_rate = 1e-5
    config.l2reg = 1e-8
    config.eta = -1
    config.eta_lr = 1e-3
    config.cache_dataset = False
    config.max_seq_len = 135
    # config.max_seq_len = 105
    config.num_epoch = 15
    config.batch_size = 16
    config.lsa = True
    config.use_amp = True
    config.dropout = dropout
    config.scenario_case = f'SCENARIO EN-B-05 DROPOUT {dropout}'

    # Train model
    Notification.send(f'Training started! {config.pretrained_bert}')
    try:
        trainer = APC.APCTrainer(
            config=config,
            dataset=dataset,
            # from_checkpoint=checkpoint,  # if you want to resume training from our pretrained checkpoints, you can pass the checkpoint name here
            auto_device=DeviceTypeOption.CUDA,
            path_to_save=None,  # set a path to save checkpoints, if it is None, save checkpoints at 'checkpoints' folder
            checkpoint_save_mode=ModelSaveOption.SAVE_MODEL_STATE_DICT,
            load_aug=False,  # there are some augmentation dataset for integrated datasets, you use them by setting load_aug=True to improve performance
        ).destroy()
        Notification.send('Training completed!')
    except Exception as e:
        Notification.send(f'Training failed! {e}')
        raise e


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')
    from utils import Notification

    # Importing for pyabsa
    from pyabsa import AspectPolarityClassification as APC
    from pyabsa.tasks.AspectPolarityClassification.models import APCModelList   
    from pyabsa import ModelSaveOption, DeviceTypeOption
    from pyabsa import DatasetItem
    # baseline_scenario('yangheng/deberta-v3-base-absa-v1.1')
    # baseline_scenario('microsoft/deberta-v3-base')
    # baseline_scenario('yangheng/deberta-v3-large-absa-v1.1')
    # baseline_scenario('microsoft/deberta-v3-large')

    # running scenario b
    # scenario_en_b_01()  # run on RTX 3090
    # scenario_en_b_02()  # run on colab
    # scenario_en_b_03()  # run on supercomputer
    # scenario_en_b_04(8)  # run on supercomputer
    # scenario_en_b_04(16)  # run on supercomputer
    # scenario_en_b_05(0.5)  # run on RTX 3090
    Notification.send('SCENARIO B5 DENGAN 0.3 DROPOUT')
    scenario_en_b_05(0.3)
