def model_check(X_train, y_train, X_val, y_val):   
    
    """Check the performance of different models with no tuning of hyperparameters [Logisitic_Regression, DecisionTreeClassifier, KNeghborsClassifier, RandomForestClassifier, XGBClassifier, LGBMClassifier]"""


    import time
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
    import pandas as pd
    import warnings
    warnings.filterwarnings('ignore')


    model1_lr = LogisticRegression()
    model1_dt = DecisionTreeClassifier()
    model1_knn = KNeighborsClassifier()
    model1_rf = RandomForestClassifier()
    model1_xgb = XGBClassifier()
    model1_lgbm = LGBMClassifier()

    models = [ model1_lr, model1_dt, model1_knn, model1_rf, model1_xgb, model1_lgbm]

    def model_single(model, train_x, train_y, val_x, val_y):
        start = time.time()
        model.fit(train_x, train_y)
        cp_preds = model.predict(val_x)
        cp_preds_train = model.predict(train_x)
        total_time = time.time() - start

        as_train = accuracy_score(cp_preds_train, train_y)
        as_val = accuracy_score(cp_preds, val_y)

        ps_train = precision_score(cp_preds_train, train_y)
        ps_val = precision_score(cp_preds, val_y)

        rs_train = recall_score(cp_preds_train, train_y)
        rs_val = recall_score(cp_preds, val_y)

        f1_train = f1_score(cp_preds_train, train_y)
        f1_val = f1_score(cp_preds, val_y)

        auc_train = roc_auc_score(cp_preds_train, train_y)
        auc_val = roc_auc_score(cp_preds, val_y)

        return as_train, as_val, ps_train, ps_val, rs_train, rs_val, f1_train, f1_val, auc_train, auc_val, total_time

    train_acc = []
    val_acc = []

    train_ps = []
    val_ps = []

    train_rs = []
    val_rs = []

    train_f1 = []
    val_f1 = []

    train_auc = []
    val_auc = []

    model_time = []

    for model in models:
        train_as, val_as, precision_train, presision_val, recall_train, recall_val, f1_train, f1_val, auc_train, auc_val, total_time = model_single(model, X_train, y_train, X_val, y_val)
        train_acc.append(train_as)
        val_acc.append(val_as)

        train_ps.append(precision_train)
        val_ps.append(presision_val)

        train_rs.append(recall_train)
        val_rs.append(recall_val)

        train_f1.append(f1_train)
        val_f1.append(f1_val)

        train_auc.append(auc_train)
        val_auc.append(auc_val)

        model_time.append(total_time)

    model_df = pd.DataFrame()
    model_names = [str(model)[0:str(model).find('(')] for model in models]

    model_df['Model'] =  model_names
    model_df['train_accuracy'] = train_acc
    model_df['val_accuracy'] = val_acc

    model_df['train_precision'] = train_ps
    model_df['val_precision'] = val_ps

    model_df['train_recall'] = train_rs
    model_df['val_recall'] = val_rs

    model_df['train_f1'] = train_f1
    model_df['val_f1'] = val_f1

    model_df['train_auc'] = train_auc
    model_df['val_auc'] = val_auc

    model_df['total_time'] = model_time


    return model_df