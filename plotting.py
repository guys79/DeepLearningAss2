import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

results_dir = 'C:\\Users\\Jonathan\\Documents\\BGU\\Semester 8\\courses\\DL\\assignments\\2\\results\\L2 values'
result_names = [
    'results_00000.csv',
    'results_00001.csv',
    'results_00005.csv',
    'results_00010.csv',
    'results_00050.csv',
]
l2_values = [
    '0.00000',
    '0.00001',
    '0.00005',
    '0.00010',
    '0.00050',
]
result_types = {
    'val_loss': 'validation loss',
    'val_accuracy': 'validation accuracy',
    'loss': 'training loss',
    'accuracy': 'training accuracy',
}
for column in result_types.keys():
    for i in range(len(result_names)):
        result_name = result_names[i]
        l2_value = l2_values[i]
        df_result = pd.read_csv('%s\\%s' % (results_dir, result_name))
        plt.plot(df_result['epoch'], df_result[column], label='L2=%s' % l2_value)
    result_type = result_types[column]
    plt.xlabel('epoch')
    plt.ylabel(result_type)
    plt.legend()
    plt.title('%s, various L2 penalties' % result_type)
    plt.savefig('%s\\%s.png' % (results_dir, result_type))
    plt.show()
    plt.clf()

