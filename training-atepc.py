
def _training_english():
    config = ATEPC.ATEPCConfigManager.get_atepc_config_english()
    config.model = ATEPC.ATEPCModelList.FAST_LCF_ATEPC  # improved version of LCF-ATEPC
    config.batch_size = 16
    config.verbose = True
    config.scenario_case = "ENGLISH ATEPC"

    dataset = DatasetItem('attractionEN', '520.attraction_en')

    aspect_extractor = ATEPC.ATEPCTrainer(
        config=config,
        from_checkpoint="english",
        dataset=dataset,
        checkpoint_save_mode=1,
        auto_device=True,
        load_aug=False,
    )

def _training_indonesia():
    config = ATEPC.ATEPCConfigManager.get_atepc_config_english()
    config.model = ATEPC.ATEPCModelList.FAST_LCF_ATEPC  # improved version of LCF-ATEPC
    config.batch_size = 32
    config.verbose = True
    config.scenario_case = "INDO ATEPC"
    config.pretrained_bert = "indobenchmark/indobert-large-p2"
    config.log_step = 15

    dataset = DatasetItem('attractionEN', '520.attraction_en')

    aspect_extractor = ATEPC.ATEPCTrainer(
        config=config,
        # from_checkpoint="english",
        dataset=dataset,
        checkpoint_save_mode=1,
        auto_device=True,
        load_aug=False,
    )



def predict(target_file, bahasa="English"):
    if bahasa == "English":
        aspect_extractor = ATEPC.AspectExtractor(checkpoint='english')
        aspect_extractor.extract_aspect(dataset=target_file)
    elif bahasa == "Indonesia":
        aspect_extractor = ATEPC.AspectExtractor(checkpoint='indonesia')
        aspect_extractor.extract_aspect(dataset=target_file)

if __name__ == '__main__':
    from pyabsa import AspectTermExtraction as ATEPC
    from pyabsa import DatasetItem
    from pyabsa import ModelSaveOption, DeviceTypeOption

    # _training_english()

    _training_indonesia()