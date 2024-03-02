import numpy as np
import pandas as pd
import evaluate
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


############################################################################################################
# 使うモデル
model_name = "FacebookAI/xlm-mlm-17-1280"

# マージ元
from_emoji = ["😂", "💕", "☺️", "😃", "😆", "😁", "🥲", "👍", "✨", "☀️", "😅", "🥺"]
# マージ先
to_emoji = ["🤣", "🥰", "😊", "😊", "😊", "😊", "😭", "😊", "😊", "😊", "🤣", "😭"]
# ラベルマッピング
label2id = {"😭": 0, "🤣": 1, "🎉": 2, "🥰": 3, "😇": 4, "💦": 5, "😊": 6, "🤔": 7}

"""
# マージ元
from_emoji = ["😂", "💕", "☺️", "😃", "😆", "😁", "🥲", "👍", "✨", "☀️", "😅", "🥺", "😇", "💦"]
# マージ先
to_emoji = ["🤣", "🥰", "😊", "😊", "😊", "😊", "😭", "😊", "😊", "😊", "🤣", "😭", "😭", "😭"]
# ラベルマッピング
label2id = {"😭": 0, "🤣": 1, "🎉": 2, "🥰": 3, "😊": 4, "🤔": 5}
"""
############################################################################################################

# ラベル
label = [e for e in label2id]
# ラベル数
num_labels = len(label)


# json ファイルから pandas に読み込む
def load_json_to_pandas(json_path):

    data = pd.read_json(json_path)

    # 絵文字マージ
    if from_emoji:
        for index, row in data.iterrows():
            if row["label"] in from_emoji:
                data.at[index, "label"] = to_emoji[from_emoji.index(row["label"])]

    return data


# pandas から Dataset オブジェクト化する
def pandas_to_Dataset(dataset):

    dataset = Dataset.from_pandas(dataset)

    return dataset


# DataFrame 型のデータセットのラベルを整数に変える
def label_to_int(dataset):

    for index, row in dataset.iterrows():
        dataset.at[index, "label"] = label2id[row["label"]]

    return dataset


# トークン化
def tokenize(data):

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return tokenizer(data["text"], padding="max_length", truncation=True, max_length=128)


# 評価
def compute_metric(eval_pred):

    metric = evaluate.load("accuracy")

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    print(classification_report(y_true=labels, y_pred=predictions, target_names=label, zero_division=0.0))
    
    return metric.compute(predictions=predictions, references=labels)


# パラメータサーチのためのモデルの読み込み
def model_init():
    return AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)


############################################################################################################
# パラメータサーチスペース
def optuna_hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [8, 16, 32]),
    }
############################################################################################################


def main():

    ############################################################################################################
    # パラメータ設定
    # 訓練データパス
    train_data_path = "./dataset/train_cls/R8.json"
    # テストデータのパス
    test_data_path = "./dataset/dev/V8h.json"
    # 訓練データ数
    num_of_train_data = 313770
    # テストデータ数
    num_of_test_data = 360
    # 結果ファイルのパス
    result_path = "./log_optuna"
    # エポック数
    num_epoch = 2
    # パラメータサーチ回数
    n_trials = 5
    ############################################################################################################

    # 生データの読み込み（絵文字マージ含む）
    train_data = load_json_to_pandas(train_data_path)
    test_data = load_json_to_pandas(test_data_path)

    # ラベルを整数に変える
    train_data = label_to_int(train_data)
    test_data = label_to_int(test_data)
    
    # Dataset オブジェクトに変える
    train_data = pandas_to_Dataset(train_data)
    test_data = pandas_to_Dataset(test_data)
    #print(train_data[1])

    # トークン化
    train_data = train_data.map(tokenize, batched=True)
    test_data = test_data.map(tokenize, batched=True)
    #print(train_data[0])

    # より小さいデータセットを作成する
    train_data = train_data.shuffle(seed=42).select(range(num_of_train_data))
    test_data = test_data.shuffle(seed=42).select(range(num_of_test_data))
    #print(len(train_data), len(test_data))
    #print(train_data[0])
    #print(test_data[0])

    # training args の作成
    training_args = TrainingArguments(
        output_dir=result_path,
        evaluation_strategy="epoch",
        num_train_epochs=num_epoch,
        logging_strategy="steps",
        logging_steps=0.1,
        report_to="none"
    )

    # 訓練データ
    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=test_data,
        compute_metrics=compute_metric
    )
    best_trial = trainer.hyperparameter_search(
        hp_space=optuna_hp_space,
        n_trials=n_trials,
        direction="maximize",
        backend="optuna"
    )
    print(best_trial)

if __name__ == "__main__":
    main()