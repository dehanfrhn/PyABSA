file_name = input("Enter file name: ")[:-4]

with open(f'../report/{file_name}.txt') as f:
    lines = f.readlines()
    lines = [line[:-2] for line in lines]

def get_smooth_loss(line):
    return line[25:31]


def get_val_acc(line):
    return line[111:116]


def get_val_f1(line):
    return line[135:140]

list = []
for line in lines:
    list.append([get_smooth_loss(line), get_val_acc(line), get_val_f1(line)])

import pandas as pd
df = pd.DataFrame(list, columns=['smooth_loss', 'val_acc', 'val_f1'], dtype=float)

# save the dataframe
df.to_csv(f'../report/{file_name}-metrics_report.csv', index=False)

import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="darkgrid")
sns.lineplot(data=df[['val_acc', 'val_f1']], palette="tab10", linewidth=2.5)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Validation Performance')
plt.savefig(f'../report/{file_name}-validation_performance.jpg', dpi=300)
plt.show()


# visualize the loss
sns.set_theme(style="darkgrid")
sns.lineplot(data=df[['smooth_loss']], palette="tab10", linewidth=2.5)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.savefig(f'../report/{file_name}-training_loss.jpg', dpi=300)
plt.show()

print('Done! Check the report folder for the report.')