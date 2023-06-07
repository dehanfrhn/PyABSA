import warnings
warnings.filterwarnings('ignore')
from pyabsa import AspectPolarityClassification as APC
sentiment_classifier = APC.SentimentClassifier(
    checkpoint="../checkpoints/fast_lsa_t_v2_AttractionReviewEn_acc_90.72_f1_84.53"
)

# TRAIN PATH
TRAIN_PATH = 'D:\\project\\PyABSA\\integrated_datasets\\apc_datasets\\500.Tripadvisor\\505.attraction_en\\train.tourism_review_en.txt.apc.inference'

# TEST PATH
TEST_PATH = 'D:\\project\\PyABSA\\integrated_datasets\\test_datasets\\test.tourism_review_en.txt.apc.inference'
# batch predict on training to check acc training
sentiment_classifier.batch_predict(
    target_file=TRAIN_PATH,  # the batch_predict() is only available for a file only, please put the examples in a file
    # target_file=TEST_PATH,  # the batch_predict() is only available for a file only, please put the examples in a file
    print_result=True,
    save_result=True,
    ignore_error=True,
)

# sentiment_classifier.batch_predict(
#     # target_file=TRAIN_PATH,  # the batch_predict() is only available for a file only, please put the examples in a file
#     target_file=TEST_PATH,  # the batch_predict() is only available for a file only, please put the examples in a file
#     print_result=True,
#     save_result=True,
#     ignore_error=True,
# )