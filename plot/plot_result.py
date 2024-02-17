import seaborn as sns
import matplotlib.pyplot as plt
import csv
import pandas as pd

plt.style.use("classic")
plt.rcParams['font.sans-serif'] = 'STSong'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 16
plt.rcParams['figure.dpi'] = 200

sns.set_style("whitegrid")

def plot_from_csv(csv_file):
    train_loss = []
    val_loss = []
    train_accuracy = []
    val_accuracy = []

    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        rows = list(reader)
        
    for i in range(len(rows[0])):
        value_str = rows[0][i]
        start_index = value_str.find('(') + 1
        end_index = value_str.find(',')
        value = float(value_str[start_index:end_index])
        train_loss.append(value)

        value_str2 = rows[1][i]
        start_index2 = value_str2.find('(') + 1
        end_index2 = value_str2.find(',')
        value2 = float(value_str2[start_index2:end_index2])
        val_loss.append(value2)
        
        value3 = float(rows[2][i])
        train_accuracy.append(value3)
        
        value4 = float(rows[3][i])
        val_accuracy.append(value4)

    # 將數據轉換為 Pandas DataFrame
    df = pd.DataFrame({'Train Loss': train_loss, 'Val Loss': val_loss, 'Train Accuracy': train_accuracy, 'Val Accuracy': val_accuracy, 'Epoch': range(len(train_loss))})
    print(df.head())
    plt.figure(figsize=((20, 8)))
    
    plt.subplot(1, 2, 1)
    sns.lineplot(data=df, x='Epoch', y='Train Loss', label='Train Loss', linewidth=2, color='blue')
    sns.lineplot(data=df, x='Epoch', y='Val Loss', label='Val Loss',  linewidth=2, color='red')
    plt.xlabel('Epoch', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.ylim(0, 0.1)
    plt.legend()

    plt.subplot(1, 2, 2)
    sns.lineplot(data=df, x='Epoch', y='Train Accuracy', label='Train Accuracy',linewidth=2, color='blue')
    sns.lineplot(data=df, x='Epoch', y='Val Accuracy', label='Val Accuracy',linewidth=2, color='red')
    plt.xlabel('Epoch', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.ylim(0, 1)
    plt.legend(loc='lower right')

    plt.show()
    plt.savefig("net_huber_BS2*4.png") 

# 調用函數並傳入 CSV 文件路徑
plot_from_csv('/home/ccy/cellcycle/Predict_Mito_nuclei/saved_model_mito/record/net_huber_BS2*4_record.csv')

