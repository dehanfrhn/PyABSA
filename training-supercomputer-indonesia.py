
def scenario_1():
    # Load checkpoint
    # checkpoint = ''

    # Load dataset
    dataset = DatasetItem("AttractionReviewId", "519.attraction_id")

    # Load config model
    config = APC.APCConfigManager.get_apc_config_english()
    config.learning_rate = 2e-5
    config.model = APCModelList.FAST_LSA_T_V2
    config.pretrained_bert = 'indobenchmark/indobert-base-p1'
    config.cache_dataset = False
    config.max_seq_len = 105
    config.num_epoch = 30
    config.lsa = True
    config.l2reg = 1e-5
    config.optimizer = 'adamw'
    config.eta_lr = 0.01
    config.batch_size = 16

    # Train model
    Notification.send('Training started!')
    try:
        trainer = APC.APCTrainer(
            config=config,
            dataset=dataset,
            # from_checkpoint=checkpoint,  # if you want to resume training from our pretrained checkpoints, you can pass the checkpoint name here
            auto_device=DeviceTypeOption.CUDA,
            path_to_save=None,
            # set a path to save checkpoints, if it is None, save checkpoints at 'checkpoints' folder
            checkpoint_save_mode=ModelSaveOption.SAVE_MODEL_STATE_DICT,
            load_aug=False,
            # there are some augmentation dataset for integrated datasets, you use them by setting load_aug=True to improve performance
        )
        Notification.send('Training completed!')
    except Exception as e:
        Notification.send(f'Training failed! {e}')
        raise e


def scenario_2():
    # Load checkpoint
    # checkpoint = ''

    # Load dataset
    dataset = DatasetItem("AttractionReviewId", "519.attraction_id")

    # Load config model
    config = APC.APCConfigManager.get_apc_config_english()
    config.learning_rate = 2e-5
    config.model = APCModelList.FAST_LSA_T_V2
    config.pretrained_bert = 'indobenchmark/indobert-large-p2'
    config.cache_dataset = False
    config.max_seq_len = 105
    config.num_epoch = 30
    config.lsa = True
    config.l2reg = 1e-5
    config.optimizer = 'adamw'
    config.eta_lr = 0.01
    config.batch_size = 16

    # Train model
    Notification.send('Training started!')
    try:
        trainer = APC.APCTrainer(
            config=config,
            dataset=dataset,
            # from_checkpoint=checkpoint,  # if you want to resume training from our pretrained checkpoints, you can pass the checkpoint name here
            auto_device=DeviceTypeOption.CUDA,
            path_to_save=None,
            # set a path to save checkpoints, if it is None, save checkpoints at 'checkpoints' folder
            checkpoint_save_mode=ModelSaveOption.SAVE_MODEL_STATE_DICT,
            load_aug=False,
            # there are some augmentation dataset for integrated datasets, you use them by setting load_aug=True to improve performance
        )
        Notification.send('Training completed!')
    except Exception as e:
        Notification.send(f'Training failed! {e}')
        raise e


def scenario_3():
    # Load checkpoint
    # checkpoint = ''

    # Load dataset
    dataset = DatasetItem("AttractionReviewId", "519.attraction_id")

    # Load config model
    config = APC.APCConfigManager.get_apc_config_english()
    config.learning_rate = 2e-5
    config.model = APCModelList.FAST_LSA_T_V2
    config.pretrained_bert = 'w11wo/indonesian-roberta-base-sentiment-classifier'
    config.cache_dataset = False
    config.max_seq_len = 105
    config.num_epoch = 30
    config.lsa = True
    config.l2reg = 1e-5
    config.optimizer = 'adamw'
    config.eta_lr = 0.01
    config.batch_size = 16

    # Train model
    Notification.send('Training started!')
    try:
        trainer = APC.APCTrainer(
            config=config,
            dataset=dataset,
            # from_checkpoint=checkpoint,  # if you want to resume training from our pretrained checkpoints, you can pass the checkpoint name here
            auto_device=DeviceTypeOption.CUDA,
            path_to_save=None,
            # set a path to save checkpoints, if it is None, save checkpoints at 'checkpoints' folder
            checkpoint_save_mode=ModelSaveOption.SAVE_MODEL_STATE_DICT,
            load_aug=False,
            # there are some augmentation dataset for integrated datasets, you use them by setting load_aug=True to improve performance
        )
        Notification.send('Training completed!')
    except Exception as e:
        Notification.send(f'Training failed! {e}')
        raise e


def scenario_4(pretrained):
    # Load checkpoint
    # checkpoint = ''

    # Load dataset
    dataset = DatasetItem("AttractionReviewId", "519.attraction_id")

    # Load config model
    config = APC.APCConfigManager.get_apc_config_english()
    config.learning_rate = 2e-5
    config.model = APCModelList.FAST_LSA_T_V2
    config.pretrained_bert = pretrained
    config.cache_dataset = False
    config.max_seq_len = 105
    config.num_epoch = 30
    config.lsa = True
    config.l2reg = 1e-5
    config.optimizer = 'adamw'
    config.eta_lr = 0.01
    config.batch_size = 16

    # Train model
    Notification.send('Training started!')
    try:
        trainer = APC.APCTrainer(
            config=config,
            dataset=dataset,
            # from_checkpoint=checkpoint,  # if you want to resume training from our pretrained checkpoints, you can pass the checkpoint name here
            auto_device=DeviceTypeOption.CUDA,
            path_to_save=None,
            # set a path to save checkpoints, if it is None, save checkpoints at 'checkpoints' folder
            checkpoint_save_mode=ModelSaveOption.SAVE_MODEL_STATE_DICT,
            load_aug=False,
            # there are some augmentation dataset for integrated datasets, you use them by setting load_aug=True to improve performance
        )
        Notification.send('Training completed!')
    except Exception as e:
        Notification.send(f'Training failed! {e}')
        raise e


def scenario_5(pretrained):
    # Load checkpoint
    # checkpoint = ''

    # Load dataset
    dataset = DatasetItem("AttractionReviewId", "519.attraction_id")

    # Load config model
    config = APC.APCConfigManager.get_apc_config_english()
    config.learning_rate = 2e-5
    config.model = APCModelList.FAST_LSA_T_V2
    config.pretrained_bert = pretrained
    config.cache_dataset = False
    config.max_seq_len = 105
    config.num_epoch = 30
    config.lsa = True
    config.l2reg = 1e-5
    config.optimizer = 'adamw'
    config.eta_lr = 0.01
    config.batch_size = 16

    # Train model
    Notification.send('Training started!')
    try:
        trainer = APC.APCTrainer(
            config=config,
            dataset=dataset,
            # from_checkpoint=checkpoint,  # if you want to resume training from our pretrained checkpoints, you can pass the checkpoint name here
            auto_device=DeviceTypeOption.CUDA,
            path_to_save=None,
            # set a path to save checkpoints, if it is None, save checkpoints at 'checkpoints' folder
            checkpoint_save_mode=ModelSaveOption.SAVE_MODEL_STATE_DICT,
            load_aug=False,
            # there are some augmentation dataset for integrated datasets, you use them by setting load_aug=True to improve performance
        )
        Notification.send('Training completed!')
    except Exception as e:
        Notification.send(f'Training failed! {e}')
        raise e


def scenario_6(pretrained):
    # Load checkpoint
    # checkpoint = ''

    # Load dataset
    dataset = DatasetItem("AttractionReviewId", "519.attraction_id")

    # Load config model
    config = APC.APCConfigManager.get_apc_config_english()
    config.learning_rate = 2e-5
    config.model = APCModelList.FAST_LSA_T_V2
    config.pretrained_bert = pretrained
    config.cache_dataset = False
    config.max_seq_len = 105
    config.num_epoch = 30
    config.lsa = True
    config.l2reg = 1e-5
    config.optimizer = 'adamw'
    config.eta_lr = 0.01
    config.batch_size = 16

    # Train model
    Notification.send('Training started!')
    try:
        trainer = APC.APCTrainer(
            config=config,
            dataset=dataset,
            # from_checkpoint=checkpoint,  # if you want to resume training from our pretrained checkpoints, you can pass the checkpoint name here
            auto_device=DeviceTypeOption.CUDA,
            path_to_save=None,
            # set a path to save checkpoints, if it is None, save checkpoints at 'checkpoints' folder
            checkpoint_save_mode=ModelSaveOption.SAVE_MODEL_STATE_DICT,
            load_aug=False,
            # there are some augmentation dataset for integrated datasets, you use them by setting load_aug=True to improve performance
        )
        Notification.send('Training completed!')
    except Exception as e:
        Notification.send(f'Training failed! {e}')
        raise e


def scenario_7(pretrained):
    # Load checkpoint
    # checkpoint = ''

    # Load dataset
    dataset = DatasetItem("AttractionReviewId", "519.attraction_id")

    # Load config model
    config = APC.APCConfigManager.get_apc_config_english()
    config.learning_rate = 2e-5
    config.model = APCModelList.FAST_LSA_T_V2
    config.pretrained_bert = pretrained
    config.cache_dataset = False
    config.max_seq_len = 105
    config.num_epoch = 30
    config.lsa = True
    config.l2reg = 1e-5
    config.optimizer = 'adamw'
    config.eta_lr = 0.01
    config.batch_size = 16

    # Train model
    Notification.send('Training started!')
    try:
        trainer = APC.APCTrainer(
            config=config,
            dataset=dataset,
            # from_checkpoint=checkpoint,  # if you want to resume training from our pretrained checkpoints, you can pass the checkpoint name here
            auto_device=DeviceTypeOption.CUDA,
            path_to_save=None,
            # set a path to save checkpoints, if it is None, save checkpoints at 'checkpoints' folder
            checkpoint_save_mode=ModelSaveOption.SAVE_MODEL_STATE_DICT,
            load_aug=False,
            # there are some augmentation dataset for integrated datasets, you use them by setting load_aug=True to improve performance
        )
        Notification.send('Training completed!')
    except Exception as e:
        Notification.send(f'Training failed! {e}')
        raise e


def baseline_scenario(pretrained):
    # Load checkpoint
    # checkpoint = ''

    # Load dataset
    dataset = DatasetItem("AttractionReviewId", "521.attraction_id")

    # Load config model
    config = APC.APCConfigManager.get_apc_config_english()
    config.num_epoch = 2
    config.learning_rate = 2e-5
    config.model = APCModelList.FAST_LSA_T_V2
    config.pretrained_bert = pretrained
    config.cache_dataset = False
    config.max_seq_len = 105
    config.lsa = True
    config.l2reg = 1e-5
    config.optimizer = 'adamw'
    config.eta_lr = 0.01
    config.batch_size = 16

    config.scenario_case = "baseline scenario"

    # Train model
    Notification.send('Training started!')
    try:
        trainer = APC.APCTrainer(
            config=config,
            dataset=dataset,
            # from_checkpoint=checkpoint,  # if you want to resume training from our pretrained checkpoints, you can pass the checkpoint name here
            auto_device=DeviceTypeOption.CUDA,
            path_to_save=None,
            # set a path to save checkpoints, if it is None, save checkpoints at 'checkpoints' folder
            checkpoint_save_mode=ModelSaveOption.SAVE_MODEL_STATE_DICT,
            load_aug=False,
            # there are some augmentation dataset for integrated datasets, you use them by setting load_aug=True to improve performance
        )

        Notification.send('Training completed!')
    except Exception as e:
        Notification.send(f'Training failed! {e}')
        raise e


if __name__ == '__main__':
    from utils import Notification
    import warnings
    warnings.filterwarnings('ignore')
    import discordwebhook

    # Importing for pyabsa
    from pyabsa import AspectPolarityClassification as APC
    from pyabsa.tasks.AspectPolarityClassification.models import APCModelList
    from pyabsa import ModelSaveOption, DeviceTypeOption
    from pyabsa import DatasetItem

    # pemilihan pretrained model
    Notification.send('indobenchmark/indobert-base-p1 started!')
    baseline_scenario('indobenchmark/indobert-base-p1')

    Notification.send('indobenchmark/indobert-base-p2 started!')
    baseline_scenario('indobenchmark/indobert-base-p2')

    Notification.send('indobenchmark/indobert-lite-large-p2 started!')
    baseline_scenario('indobenchmark/indobert-lite-large-p2')  # dipilih karena memiliki akurasi yang lebih baik untuk dataset SmSA (review)


    # baseline_scenario('indobenchmark/indobert-large-p2')  # metrics paling tinggi dari indobenchmark
    # baseline_scenario('w11wo/indonesian-roberta-base-sentiment-classifier')
    # baseline_scenario('rizalmilyardi/IndobertTypeNews')
    # baseline_scenario('cahya/bert-base-indonesian-1.5G')