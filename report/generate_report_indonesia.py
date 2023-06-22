# Generate report for the training process
def get_metrics(line):
    temp = line.split(' | ')
    print(temp)

    epoch = int(temp[0][-2:])
    avg_training_loss = float(temp[1][-6:])
    avg_validation_acc = float(temp[2][8:-11])
    avg_validation_f1 = float(temp[3][7:-11])
    max_validation_acc = float(temp[2][-6:-1])
    max_validation_f1 = float(temp[3][-6:-1])

    return int(epoch), avg_training_loss, avg_validation_acc, avg_validation_f1, max_validation_acc, max_validation_f1


def _visualize_single():
    with open(f'../report/indonesia/{file_name}.txt', encoding='utf-8', mode='r') as f:
        lines = f.readlines()
        lines = [line.replace('\n', '') for line in lines]

    result = []
    for line in lines:
        result.append(get_metrics(line))

    df = pd.DataFrame(result, columns=['epoch', 'avg_training_loss', 'avg_validation_acc', 'avg_validation_f1',
                                       'max_validation_acc', 'max_validation_f1'])

    # save the dataframe
    df.to_csv(f'../report/indonesia/{file_name}-metrics_report.csv', index=False)

    # visualize validation performance
    sns.set_theme(style="darkgrid")
    sns.lineplot(data=df[['avg_validation_acc', 'avg_validation_f1']], palette="tab10", linewidth=2.5)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Performance')
    plt.savefig(f'../report/indonesia/{file_name}-validation_performance.jpg', dpi=300)
    plt.show()

    # visualize best validation performance (max)
    sns.set_theme(style="darkgrid")
    sns.lineplot(data=df[['max_validation_acc', 'max_validation_f1']], palette="tab10", linewidth=2.5)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Best Validation Performance')
    plt.savefig(f'../report/indonesia/{file_name}-best_validation_performance.jpg', dpi=300)
    plt.show()

    # visualize the loss
    sns.set_theme(style="darkgrid")
    sns.lineplot(data=df[['avg_training_loss']], palette="tab10", linewidth=2.5)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig(f'../report/indonesia/{file_name}-training_loss.jpg', dpi=300)
    plt.show()

    print('Done! Check the report folder for the report.')


def generate_visual(data, title, file_name, y_label, legend_labels):
    sns.set_theme(style="darkgrid")
    for i in range(len(data)):
        sns.lineplot(data=data[i], linewidth=2.5, legend=True, dashes=False, label=legend_labels[i])
    plt.xlabel('Epoch')
    plt.ylabel(y_label)
    plt.title(title)
    plt.savefig(f'../report/indonesia/{file_name}.jpg', dpi=300)
    plt.show()


def _visualize_combine():

    # load the data
    glob_path = '..\\report\\indonesia\\*.csv'
    all_files = glob.glob(glob_path)
    df_from_each_file = [
        pd.read_csv(f, sep=',') for f in all_files
    ]

    # get the data
    list_training_loss = [df['avg_training_loss'].tolist() for df in df_from_each_file]
    list_validation_acc = [df['avg_validation_acc'].tolist() for df in df_from_each_file]
    list_validation_f1 = [df['avg_validation_f1'].tolist() for df in df_from_each_file]
    legend_labels = ['indobenchmark/indobert-base-p2', 'indobenchmark/indobert-large-p2', 'w11wo/indonesian-roberta-base-sentiment-classifier']

    print('generating report...')

    # visualize training loss on all files
    generate_visual(list_training_loss, 'Training Loss', 'indonesia-training_loss', 'Loss', legend_labels)

    # visualize validation performance on all files (accuracy)
    generate_visual(list_validation_acc, 'Validation Performance (Accuracy)', 'indonesia-validation_performance_acc', 'Accuracy', legend_labels)

    # visualize validation performance on all files (f1)
    generate_visual(list_validation_f1, 'Validation Performance (F1)', 'indonesia-validation_performance_f1', 'F1', legend_labels)


if __name__ == '__main__':
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import glob

    mode = input("Enter mode (single/combine): ")
    if mode == 'single':
        file_name = input("Enter file name: ")[:-4]

    if mode == 'single':
        _visualize_single()
    elif mode == 'combine':
        _visualize_combine()
    else:
        print('Invalid mode!')