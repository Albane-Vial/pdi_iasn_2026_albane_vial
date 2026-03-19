import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import (
    f1_score, 
    precision_score, 
    recall_score, 
    classification_report,
    confusion_matrix
)

def evaluer_detection_binaire(df, col_vrai='label_vrai', col_pred='label_pred'):
    y_true = df[col_vrai].astype(int)
    y_pred = df[col_pred].astype(int)
    
    resultats_binaires = {
        'Accuracy': (y_true == y_pred).mean(),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1_Score': f1_score(y_true, y_pred, zero_division=0),
        'Matrice_Confusion': confusion_matrix(y_true, y_pred).tolist()
    }
    return resultats_binaires

def preparer_labels_multilabel(df, col_vraies_valeurs, col_predictions):
    y_true_list = df[col_vraies_valeurs].apply(
        lambda x: str(x).split('|') if pd.notna(x) and x != "none" else []
    ).tolist()
    
    y_pred_list = df[col_predictions].apply(
        lambda x: str(x).split('|') if pd.notna(x) and x != "none" else []
    ).tolist()

    mlb = MultiLabelBinarizer()
    mlb.fit(y_true_list + y_pred_list)
    
    y_true_bin = mlb.transform(y_true_list)
    y_pred_bin = mlb.transform(y_pred_list)
    
    return y_true_bin, y_pred_bin, mlb.classes_

def evaluer_caracterisation_multilabel(df, col_vrai='error_types', col_pred='error_types_pred'):
    y_true_bin, y_pred_bin, classes = preparer_labels_multilabel(df, col_vrai, col_pred)

    if len(classes) == 0:
        return {"Erreur": "Aucune classe à évaluer."}

    rapport_detaille = classification_report(y_true_bin, y_pred_bin, target_names=classes, zero_division=0)
    
    resultats_multi = {
        #'F1_Macro': f1_score(y_true_bin, y_pred_bin, average='macro', zero_division=0),
        #'F1_Micro': f1_score(y_true_bin, y_pred_bin, average='micro', zero_division=0),
        'Rapport_Detaille': rapport_detaille
    }
    return resultats_multi

def evaluer_pipeline_complet(df):
    rapport_global = {}

    rapport_global['Detection_Binaire'] = evaluer_detection_binaire(
        df, col_vrai='label_vrai', col_pred='label_pred'
    )
    
    rapport_global['Caracterisation_Multilabel'] = evaluer_caracterisation_multilabel(
        df, col_vrai='error_types', col_pred='error_types_pred'
    )
    
    return rapport_global