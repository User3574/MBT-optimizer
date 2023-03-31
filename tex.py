import json
import pandas as pd
import os
import glob


def make_lr_table(path):
    filename = os.path.basename(path).replace('_', '-')
    data = json.load(open(path))
    df = pd.DataFrame.from_dict(data, orient="index")
    df = df.round(decimals=3)
    df.columns.name = 'Batch\LR'
    df_latex = df.to_latex(caption=filename)
    print(df_latex)


def make_optim_table(path, stat_name='acc_valid', fn=max):
    filename = os.path.basename(path).replace('_', '-')
    data = json.load(open(path))
    df = pd.DataFrame(columns=[lr for lr in data],
                      index=[optimizer for optimizer in data[next(iter(data.keys()))]])
    for lr in data:
        for optimizer in data[lr]:
            df.loc[optimizer, str(lr)] = fn(data[lr][optimizer][stat_name])
    df = df.round(decimals=3)
    df.columns.name = 'Method\LR'
    df_latex = df.to_latex(caption=filename + '-' + fn.__name__ + '-' + stat_name)
    print(df_latex)


if __name__ == '__main__':
    # for lr_filepath in glob.glob('/home/user3574/Private/Disk/MBT-optimizer/history/lr/*.json'):
    #     make_lr_table(lr_filepath)

    make_optim_table('/home/user3574/Private/Disk/MBT-optimizer/history/optimizers/all_optimizersResNet18_dFashionMNIST_b200.json')
