

def load_model_aspect_extractor(checkpoint='english'):
    return ATEPC.AspectExtractor(checkpoint=checkpoint)


def load_model_sentiment_classifier(checkpoint='english'):
    return APC.SentimentClassifier(checkpoint=checkpoint)


def extract_aspect(text, model):
    text_aspect_extractor = model.predict(
        text=text,
        print_result=False,
        ignore_error=True,
        eval_batch_size=32,
        pred_sentiment=False,
        save_result=False,
    )
    return text_aspect_extractor


def extract_aspects(texts, model):
    texts_aspect_extractor = model.batch_predict(
        target_file=texts,  # list of text
        save_result=False,
        print_result=False,  # print the result
        pred_sentiment=False,  # Predict the sentiment of extracted aspect terms
        )
    return texts_aspect_extractor

def get_inference_format(extracted_aspect):
    text = extracted_aspect
    # find the index
    idxs = []
    IOB_LIST = text['IOB']
    for i, tag in enumerate(IOB_LIST):
        if tag == 'B-ASP':
            idxs.append(i)

    # add the tag
    for idx in idxs:
        token = text['tokens'][idx]
        new_token = f'[B-ASP]{token}[E-ASP]'
        text['tokens'][idx] = new_token
    # convert to string
    sentence = " ".join(text['tokens'])
    return sentence


def get_inference_format_from_list(list_of_extracted_aspects):
    inference_format = []
    for row in list_of_extracted_aspects:
        inference_format.append(get_inference_format(row))
    return inference_format


def predict_sentiment_from_list(texts, model):
    result_sentiments_classifier = []
    for text in texts:
        result_sentiments_classifier.append(predict_sentiment_from_text(text, model))
    return result_sentiments_classifier


def predict_sentiment_from_text(text, model):
    result_sentiment_classifier = model.predict(
        text=text,  # list of text
        print_result=False,  # print the result
        save_result=False,
        ignore_error=True,
    )
    return result_sentiment_classifier


def read_txt_file(file):
    with open(file, 'r') as f:
        texts = f.readlines()
    return texts


if __name__ == '__main__':
    import pandas as pd
    from utils import Notification
    import sys
    from pyabsa import AspectTermExtraction as ATEPC
    from pyabsa import AspectPolarityClassification as APC
    from pyabsa.tasks.AspectTermExtraction.prediction import aspect_extractor
    from pyabsa.tasks.AspectPolarityClassification.prediction import sentiment_classifier

    english_aspect_extractor = load_model_aspect_extractor("checkpoints/ATE_EN")
    english_sentiment_classifier = load_model_sentiment_classifier("checkpoints/EN_B_04_fast_lsa_t_v2_AttractionReviewEn_acc_91.42_f1_85.41")

    indonesia_aspect_extractor = load_model_aspect_extractor("checkpoints/ATE_ID")
    indonesia_sentiment_classifier = load_model_sentiment_classifier("checkpoints/ID_A_02_fast_lsa_t_v2_AttractionReviewId_acc_80.8_f1_76.25")
    print('success import')

    token = True
    while token:
        print("1. Aspect Term Extraction")
        print("2. Sentiment Classification")
        print("3. Exit")
        choice = int(input("Choose Number: "))
        if choice == 1:
            print("1. English")
            print("2. Indonesia")
            choice = int(input("Choose Number: "))
            if choice == 1:
                print(choice)
                # target_file = input("Target file:  (e.g. data/EN/520.attraction_en.txt)")
                # english_aspect_extractor.extract_aspect(dataset=target_file)
            elif choice == 2:
                print(choice)
                target_file = input("Target file: ")
                indonesia_aspect_extractor.extract_aspect(dataset=target_file)
        elif choice == 2:
            print("1. English")
            print("2. Indonesia")
            choice = int(input("Choose Number: "))
            if choice == 1:
                print('1. single text')
                print('2. file')
                choice = int(input("Choose Number: "))
                if choice == 1:
                    text = input("Input text: ")
                    extracted_aspect = extract_aspect(text, english_aspect_extractor)
                    inference_format = get_inference_format(extracted_aspect)
                    print(inference_format)
                    result_sentiment_classifier = predict_sentiment_from_text(inference_format, english_sentiment_classifier)
                    print('text:', result_sentiment_classifier['text'])
                    print('aspect:', ' '.join(result_sentiment_classifier['aspect']))
                    print('sentiment:', ' '.join(result_sentiment_classifier['sentiment']))
                    print('confidence:', result_sentiment_classifier['confidence'][0])
                    # print(result_sentiment_classifier)
                elif choice == 2:
                    target_file = input("Target file: ")
                    texts = read_txt_file(target_file)
                    extracted_aspects = extract_aspects(texts, english_aspect_extractor)
                    inference_format = get_inference_format_from_list(extracted_aspects)
                    result_sentiments_classifier = predict_sentiment_from_list(inference_format, english_sentiment_classifier)
                    print(result_sentiments_classifier)
                else:
                    print("Invalid number")
                # target_file = input("Target file: ")
                # english_sentiment_classifier.extract_sentiment(dataset=target_file)
            elif choice == 2:  # bahasa
                print('1. single text')
                print('2. file')
                choice = int(input("Choose Number: "))

                if choice == 1:
                    text = input("Input text: ")
                    extracted_aspect = extract_aspect(text, indonesia_aspect_extractor)
                    inference_format = get_inference_format(extracted_aspect)
                    # print(inference_format)
                    result_sentiment_classifier = predict_sentiment_from_text(inference_format, indonesia_sentiment_classifier)
                    print('text:', result_sentiment_classifier['text'])
                    print('aspect:', ' '.join(result_sentiment_classifier['aspect']))
                    print('sentiment:', ' '.join(result_sentiment_classifier['sentiment']))
                    print('confidence:', result_sentiment_classifier['confidence'][0])
                    # text = result_sentiments_class2ifier['text']
                    # aspect = result_sentiments_classifier['aspect']
                    # polarity = result_sentiments_classifier['polarity']
                    # confidence = result_sentiments_classifier['confidence']
                    # print(text, aspect, polarity, confidence)

                elif choice == 2:
                    target_file = input("Target file: ")
                    texts = read_txt_file(target_file)
                    extracted_aspects = extract_aspects(texts, indonesia_aspect_extractor)
                    inference_format = get_inference_format_from_list(extracted_aspects)
                    result_sentiments_classifier = predict_sentiment_from_list(inference_format, indonesia_sentiment_classifier)

                    print(result_sentiment_classifier)
                else:
                    print("Invalid number")
        elif choice == 3:
            token = False
            sys.exit()
        else:
            print("Invalid number")


