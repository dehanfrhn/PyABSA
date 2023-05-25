from pyabsa import DatasetItem
from pyabsa import AspectTermExtraction as ATEPC
from pyabsa import ModelSaveOption, DeviceTypeOption
import warnings

warnings.filterwarnings("ignore")

dataset = DatasetItem('attraction_en', '512_attraction_en')
print(dataset)

config = (
    ATEPC.ATEPCConfigManager.get_atepc_config_english()
)  # this config contains 'pretrained_bert', it is based on pretrained models
config.model = ATEPC.ATEPCModelList.FAST_LCF_ATEPC  # improved version of LCF-ATEPC
# config.optimizer = "adamw"
# config.learning_rate = 0.00002
config.pretrained_bert = "microsoft/deberta-v3-base"
config.cache_dataset = False
# config.warmup_step = -1
# config.user_bert_spc = True
config.show_metric = True
# config.max_seq_len = 80
# config.SRD = 3
# config.use_syntax_based_SRD = False
config.lcf = "cdw"
# config.cross_validate_fold = 10
# config.window = "lr"
# config.dropout = 0.5
# config.l2reg = 0.00001
config.num_epoch = 10
# config.batch_size = 16
# config.initializer = "xavier_uniform_"
# config.seed = 52
config.output_dim = 3
config.log_step = -1
# config.patience = 99999
# config.gradient_accumulation_steps = 1
# config.dynamic_truncate = True
# config.srd_alignment = True
# config.evaluate_begin = 0

config.verbose = False  # If verbose == True, PyABSA will output the model strcture and seversal processed data examples
config.notice = (
    "This is an training example for aspect term extraction"  # for memos usage
)

try:
    trainer = ATEPC.ATEPCTrainer(
    config=config,
    dataset=dataset,
    # from_checkpoint="english",  # if you want to resume training from our pretrained checkpoints, you can pass the checkpoint name here
    auto_device=DeviceTypeOption.AUTO,  # use cuda if available
    checkpoint_save_mode=ModelSaveOption.SAVE_MODEL_STATE_DICT,  # save state dict only instead of the whole model
    load_aug=False,  # there are some augmentation dataset for integrated datasets, you use them by setting load_aug=True to improve performance
)
except Exception as e:
    raise e