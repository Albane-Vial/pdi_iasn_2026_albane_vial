import torch
from captum.attr import LayerIntegratedGradients
import pandas as pd
from pathlib import Path
import shap
import scipy as sp
import numpy as np
import torch.nn.functional as F
from lime.lime_text import LimeTextExplainer



def generer_explicabilite_ig_deberta(df_audit: pd.DataFrame, modele, tokenizer, device, chemin_export: str):
    modele.eval()
    modele.zero_grad()
    
    def custom_forward_func(inputs, attention_mask=None):
        outputs = modele(inputs, attention_mask=attention_mask)
        return outputs.logits.mean(dim=1) 

    lig = LayerIntegratedGradients(custom_forward_func, modele.deberta.embeddings.word_embeddings)
    resultats = []

    for index, row in df_audit.iterrows():
        texte = str(row['phrase_clinique'])
        label_vrai = row['label_vrai']
        label_pred = row['label_pred']
        
        encodage = tokenizer(texte, return_tensors="pt")
        input_ids = encodage["input_ids"].to(device)
        attention_mask = encodage["attention_mask"].to(device)
        baseline_input_ids = torch.zeros_like(input_ids).to(device)

        attributions, delta = lig.attribute(
            inputs=input_ids,
            baselines=baseline_input_ids,
            additional_forward_args=(attention_mask,),
            target=1,
            n_steps=50,
            method="gausslegendre",
            return_convergence_delta=True
        )
        
        attributions_sum = attributions.sum(dim=-1).squeeze(0)
        attributions_norm = (attributions_sum / torch.norm(attributions_sum)).cpu().detach().numpy()
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
        
        resultats.append({
            "phrase": texte, 
            "label_vrai": label_vrai,
            "label_pred": label_pred,
            "tokens": tokens, 
            "attributions": attributions_norm.tolist()
        })

    pd.DataFrame(resultats).to_excel(chemin_export, index=False)
def generer_explicabilite_shap_deberta(df_audit: pd.DataFrame, modele, tokenizer, device, chemin_export: str):
    modele.eval()

    def prediction_pipeline_shap(texts):
        if isinstance(texts, np.ndarray):
            texts = texts.tolist()
            
        encoded_inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = modele(**encoded_inputs)
            
        logits = outputs.logits.mean(dim=1).detach().cpu().numpy() 
        probabilities = (np.exp(logits).T / np.exp(logits).sum(axis=-1)).T
        return sp.special.logit(probabilities[:, 1])

    masqueur_texte = shap.maskers.Text(tokenizer)
    explainer_shap = shap.Explainer(prediction_pipeline_shap, masqueur_texte)
    
    # Limitation de l'échantillon à 10 pour la performance
    df_echantillon = df_audit.head(10)
    textes = df_echantillon['phrase_clinique'].astype(str).tolist()
    labels_vrais = df_echantillon['label_vrai'].tolist()
    labels_preds = df_echantillon['label_pred'].tolist()
    
    shap_values = explainer_shap(textes)
    
    df_shap = pd.DataFrame({
        "phrase": textes, 
        "label_vrai": labels_vrais,
        "label_pred": labels_preds,
        "shap_values": [v.values.tolist() for v in shap_values]
    })
    df_shap.to_excel(chemin_export, index=False)
    
def generer_explicabilite_lime_deberta(df_audit: pd.DataFrame, modele, tokenizer, device, chemin_export: str):
    modele.eval()

    def prediction_probabiliste_lime(text_array):
        inputs = tokenizer(
            text_array.tolist() if isinstance(text_array, np.ndarray) else text_array,
            padding=True, truncation=True, max_length=512, return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            outputs = modele(**inputs)
            
        logits_mean = outputs.logits.mean(dim=1)
        probs = F.softmax(logits_mean, dim=-1)
        return probs.cpu().numpy()

    explainer_lime = LimeTextExplainer(class_names=["Classe Négative", "Classe Positive"], split_expression=r'\W+')
    
    df_echantillon = df_audit.head(5)
    textes = df_echantillon['phrase_clinique'].astype(str).tolist()
    
    explications_excel = []
    for idx, row in df_echantillon.iterrows():
        texte = str(row['phrase_clinique'])
        label_vrai = row['label_vrai']
        label_pred = row['label_pred']

        exp = explainer_lime.explain_instance(texte, prediction_probabiliste_lime, num_features=10, num_samples=200)
        
        # Export HTML
        fichier_html = str(chemin_export).replace(".xlsx", f"_{idx}.html")
        exp.save_to_file(fichier_html)
        
        # Ajout des données pour l'export Excel
        explications_excel.append({
            "phrase": texte,
            "label_vrai": label_vrai,
            "label_pred": label_pred,
            "mots_importants_lime": str(exp.as_list()), 
            "fichier_html_associe": Path(fichier_html).name
        })

    pd.DataFrame(explications_excel).to_excel(chemin_export, index=False)

def extraire_erreurs_pour_explicabilite(chemin_predictions_parquet: str):
    df = pd.read_parquet(chemin_predictions_parquet, engine='pyarrow')
    
    df_erreurs = df[(df['label_vrai'] != df['label_pred']) & (df['nb_errors'] == 1)].copy()    
    df_fp = df_erreurs[df_erreurs['label_pred'] == 1].copy()
    df_fp['nature_erreur'] = 'FP'
    
    df_fn = df_erreurs[df_erreurs['label_vrai'] == 1].copy()
    df_fn['nature_erreur'] = 'FN'
    
    exemples_fp = df_fp.groupby('error_types').first().reset_index()
    exemples_fn = df_fn.groupby('error_types').first().reset_index()
    print(exemples_fn.shape)
        
    df_correcte = df[(df['error_types'] == df['error_types_pred']) & (df['nb_errors'] == 1)].copy()
    df_correcte['nature_erreur'] = 'nul'
    
    df_correcte_multi = df_correcte.groupby('error_types').head(5).reset_index(drop=True)
    return pd.concat([exemples_fp, exemples_fn, df_correcte], ignore_index=True)

def executer_audit_organise(df_audit, modele, tokenizer, device, dossier_racine, args):
    for _, row in df_audit.iterrows():
   
        nature = row['nature_erreur'] 
        e_type = str(row['error_types']).replace("/", "_") 
        
        dossier_cible = Path(dossier_racine) / nature / e_type
        dossier_cible.mkdir(parents=True, exist_ok=True)
        
        nom_base = f"{nature}_{e_type}"
        
        df_ligne = pd.DataFrame([row])

        if args.xai_method in ['ig', 'all']:
            generer_explicabilite_ig_deberta(
                df_ligne, modele, tokenizer, device, 
                str(dossier_cible / f"{nom_base}_ig.xlsx")
            )
        
        if args.xai_method in ['shap', 'all']:
            generer_explicabilite_shap_deberta(
                df_ligne, modele, tokenizer, device, 
                str(dossier_cible / f"{nom_base}_shap.xlsx")
            )
            
        if args.xai_method in ['lime', 'all']:
            generer_explicabilite_lime_deberta(
                df_ligne, modele, tokenizer, device, 
                str(dossier_cible / f"{nom_base}_lime.xlsx")
            )