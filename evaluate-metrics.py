
def _evaluate_indonesia(checkpoint_name):
    # find a suitable checkpoint and use the name:
    sentiment_classifier = APC.SentimentClassifier(
        checkpoint=f"D:\\project\\PyABSA\\checkpoints\\{checkpoint_name}",
    )

    sentiment_classifier.batch_predict(
        target_file=DatasetItem('attraction', '522.attraction_id_test'),
        # the batch_predict() is only available for a file only, please put the examples in a file
        print_result=True,
        save_result=True,
        ignore_error=True,
        eval_batch_size=32,
    )

def _evaluate_inggris(checkpoint_name):
# find a suitable checkpoint and use the name:
    sentiment_classifier = APC.SentimentClassifier(
        checkpoint=f"D:\\project\\PyABSA\\checkpoints\\{checkpoint_name}",
    )

    sentiment_classifier.batch_predict(
        target_file=DatasetItem('attraction', '523.attraction_en_test'),
        # the batch_predict() is only available for a file only, please put the examples in a file
        print_result=True,
        save_result=True,
        ignore_error=True,
        eval_batch_size=32,
    )


if __name__ ==  '__main__':
    from pyabsa import AspectPolarityClassification as APC
    from pyabsa.tasks.AspectPolarityClassification import SentimentClassifier
    from pyabsa import available_checkpoints
    from pyabsa import DatasetItem

    _evaluate_inggris('EN_B_04_fast_lsa_t_v2_AttractionReviewEn_acc_91.42_f1_85.41')
    # _evaluate_indonesia('ID_A_02_fast_lsa_t_v2_AttractionReviewId_acc_80.8_f1_76.25')