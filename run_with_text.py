import sys

from models.DeepDense import DeepDense
from models.TextLSTM import TextLSTM
from models.Transformer import Transformer
from models.TransformerEncoder import TransformerEncoder
from models.Wide import Wide

from models.WideDeep import WideDeep
from optim.Initializer import KaimingNormal, XavierNormal
from optim.radam import RAdam
from preprocessing.Preprocessor import WidePreprocessor, DeepPreprocessor, DeepTextPreprocessor

sys.path.append('.')

import numpy as np
import pandas as pd
import torch

embedding_path = '/Users/tanzhen/Desktop/pai_pytorch/data/embedding_SougouNews.npz'
traindata_path = '/Users/tanzhen/Desktop/pai_pytorch/data/adult_train_bak.csv'
summary_path = '/Users/tanzhen/Desktop/code/wdl_pytorch/log/transformer'


if __name__ == '__main__':
    df = pd.read_csv(traindata_path)
    df.columns = [c.replace("-", "_") for c in df.columns]
    df["age_buckets"] = pd.cut(df.age, bins=[16, 25, 30, 35, 40, 45, 50, 55, 60, 91], labels=np.arange(9))
    df["income_label"] = (df["income"].apply(lambda x: ">50K" in x)).astype(int)
    df.drop("income", axis=1, inplace=True)
    print(df.head())

    # wide 列名
    wide_cols = ["age_buckets", "education", "relationship", "workclass", "occupation", "native_country", "gender"]
    crossed_cols = [("education", "occupation"), ("native_country", "occupation")]
    # deep 列名
    cat_embed_cols = [("education", 10), ("relationship", 8), ("workclass", 10), ("occupation", 10), ("native_country", 10)]
    continuous_cols = ["age", "hours_per_week"]
    # text 列名
    text_cols = 'desc'
    target = "income_label"
    target = df[target].values

    # Wide 输入
    prepare_wide = WidePreprocessor(wide_cols=wide_cols, crossed_cols=crossed_cols)
    X_wide = prepare_wide.fit_transform(df)
    """
        [[0. 0. 0. 0. 1. 0. 0. 1. 0. 0. 0. 0. 1. 0. 1. 0. 0. 0. 0. 0. 1. 0. 0. 1. 0. 1. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
         [0. 0. 0. 0. 0. 1. 0. 0. 0. 1. 0. 1. 0. 0. 0. 1. 0. 0. 1. 0. 0. 0. 0. 0. 1. 0. 1. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 1. 0. 0.]]
    """

    # Deep 输入
    prepare_deep = DeepPreprocessor(embed_cols_list=cat_embed_cols, continuous_cols=continuous_cols)
    X_deep = prepare_deep.fit_transform(df)
    """
    [[ 2. 0. 2. 2. 1. -0.346393 0.31569232]
     [ 2. 1. 1. 1. 1. 0.9675134 -2.0520001]]
    """

    # lstm 输入
    prepare_text = DeepTextPreprocessor(text_cols_list=text_cols, pad_size=16, vocab_path='/Users/tanzhen/Desktop/pai_pytorch/data/vocab.pkl')
    X_text = prepare_text.fit_transform(df)
    """
    [[  66  440 4761 4761 4761 4761 4761 4761 4761 4761 4761 4761 4761 4761 4761 4761]
     [   5  440 4761 4761 4761 4761 4761 4761 4761 4761 4761 4761 4761 4761 4761 4761]]
    """

    # Build model
    wide = Wide(wide_dim=X_wide.shape[1], output_dim=1)
    deepdense = DeepDense(hidden_layers=[64, 32], dropout=[0.2, 0.2], deep_column_idx=prepare_deep.deep_column_idx, embed_input=prepare_deep.emb_col_val_dim_tuple, continuous_cols=continuous_cols)
    # lstm = TextLSTM(embedding_path)
    lstm = TransformerEncoder(embedding_path)
    wide_deep_model = WideDeep(wide=wide, deepdense=deepdense, deeptext=lstm)

    # 1.设定 optimizer ==> 2.Init 子 model 各层种参数 ==> 3.StepLR
    wide_opt = torch.optim.Adam(wide_deep_model.wide.parameters())
    text_opt = torch.optim.Adam(wide_deep_model.deeptext.parameters())
    deep_opt = RAdam(wide_deep_model.deepdense.parameters())

    wide_sch = torch.optim.lr_scheduler.StepLR(wide_opt, step_size=3)
    deep_sch = torch.optim.lr_scheduler.StepLR(deep_opt, step_size=5)
    text_sch = torch.optim.lr_scheduler.StepLR(text_opt, step_size=5)

    optimizers = {"wide": wide_opt, "deepdense": deep_opt, 'deeptext': text_opt}
    schedulers = {"wide": wide_sch, "deepdense": deep_sch, 'deeptext': text_sch}
    initializers = {"wide": KaimingNormal, "deepdense": XavierNormal, 'deeptext': KaimingNormal}

    wide_deep_model.compile(method='binary', optimizers_dic=optimizers, lr_schedulers_dic=schedulers, initializers_dic=initializers)

    wide_deep_model.fit(X_wide=X_wide, X_deep=X_deep, X_text=X_text, target=target, n_epochs=10, batch_size=128,val_split=0.2, summary_path=summary_path)










