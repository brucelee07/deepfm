import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

from models import Deep_Fm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sparse_feas_map = {
    "creativeID": 100000,
    "positionID": 100000,
    "connectionType": 10,
    "telecomsOperator": 10
}

sparse_feas = list(sparse_feas_map.keys())

dense_feas = []

feature_info = [
    dense_feas,
    sparse_feas_map,
]

hidden_units = [256, 128, 64, 32, 1]
net = Deep_Fm(feature_info, hidden_units)

net.to(device)

loss_func = nn.BCELoss()
optimizer = torch.optim.Adam(params=net.parameters(), lr=0.001)


def metric_func(predictions, trues):
    predictions = predictions > 0.5
    auc = np.sum(predictions == trues) / len(trues)
    return auc


def logloss(predictions, trues):
    results = []
    for i in range(len(trues)):
        if trues[i] == 1:
            results.append(-np.log(predictions[i]))
        else:
            results.append(-np.log(1 - predictions[i]))
    if results:
        return sum(results) / len(results)
    else:
        return 0.0


batch_size = 20
with pd.read_csv("train.csv", chunksize=1000) as reader:
    for chunk in reader:
        # break

        df = chunk[sparse_feas + ['label']]

        train_df, val_df = train_test_split(df,
                                            test_size=0.2,
                                            random_state=2022)

        dl_train_dataset = TensorDataset(
            torch.tensor(train_df[sparse_feas].to_numpy()).float(),
            torch.tensor(train_df["label"].to_numpy()).float(),
        )
        dl_val_dataset = TensorDataset(
            torch.tensor(val_df[sparse_feas].to_numpy()).float(),
            torch.tensor(val_df["label"].to_numpy()).float(),
        )

        dl_train = DataLoader(dl_train_dataset,
                              shuffle=True,
                              batch_size=batch_size)
        dl_vaild = DataLoader(dl_val_dataset,
                              shuffle=True,
                              batch_size=batch_size)
        # 测试一下模型
        # print("测试model")
        # fea, label = next(iter(dl_train))
        # out = net(fea)
        # print(out)

        epochs = 4
        log_step_freq = 10

        print("Start Training...")
        nowtime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print("=========" * 8 + "%s" % nowtime)

        for epoch in range(1, epochs + 1):
            # 训练阶段
            net.train()
            loss_sum = 0.0
            metric_sum = 0.0
            logloss_sum = 0.0
            step = 1
            auc_step = 1

            for step, (features, labels) in enumerate(dl_train, 1):

                optimizer.zero_grad()
                features = features.to(device)
                labels = torch.unsqueeze(labels, dim=-1)
                predictions = net(features)
                loss = loss_func(predictions, labels)

                try:  # 这里就是如果当前批次里面的y只有一个类别， 跳过去
                    metric = metric_func(predictions.detach().cpu().numpy(),
                                         labels.numpy())
                    logloss_value = logloss(predictions.detach().cpu().numpy(),
                                            labels.numpy())
                    metric_sum += metric
                    logloss_sum += logloss_value
                    # metric_2 += mape(predictions, labels).item()
                    auc_step += 1
                except ValueError:
                    pass

                loss.backward()
                optimizer.step()

                loss_sum += loss.item()

                if step % log_step_freq == 0:
                    print(("[step = %d] loss: %.3f, " + "auc" + ": %.3f" +
                           "  logloss:" + ": %.3f") % (
                               step,
                               loss_sum / step,
                               metric_sum / auc_step,
                               logloss_sum / auc_step,
                           ))

            # 验证阶段
            net.eval()
            val_loss_sum = 0.0
            val_metric_sum = 0.0
            # val_metric_2 = 0.0
            val_step = 1

            for val_step, (features, labels) in enumerate(dl_vaild, 1):
                with torch.no_grad():
                    features = features.to(device)
                    predictions = net(features)
                    labels = torch.unsqueeze(labels, dim=-1)
                    val_loss = loss_func(predictions, labels)
                    try:
                        val_metric = metric_func(predictions.cpu().numpy(),
                                                 labels.numpy())
                        val_metric_sum += val_metric.item()
                        # val_metric_2 += mape(predictions, labels).item()
                    except ValueError:
                        pass
                val_loss_sum += val_loss.item()

            # 打印epoch级别日志
            print(f"\nEPOCH = {epoch}, loss = {loss_sum/ step:.3f},\
        auc = {metric_sum/step:.3f}, val_loss = {val_loss_sum/step:.3f},\
        val-auc = {val_metric_sum/step:.3f}, ")
            nowtime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print("\n" + "==========" * 8 + "%s" % nowtime)

        print("Finished Training...")
