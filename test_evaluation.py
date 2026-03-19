import torch
from transformers import AutoTokenizer, DebertaV2ForTokenClassification
import pandas as pd

df_test_final = pd.read_parquet('/teamspace/studios/this_studio/df_test.parquet', engine='pyarrow')


# 1. Filtrage des prescriptions d'Aspirine sans erreurs
# On cherche le mot 'Aspirin' (insensible à la casse) et nb_errors == 0
masque_aspirine_saine = (
    df_test_final['phrase_clinique'].str.contains('Aspirin', case=False, na=False) & 
    (df_test_final['nb_errors'] == 1)
)

df_aspirine_ok = df_test_final[masque_aspirine_saine]

if not df_aspirine_ok.empty:
    # On prend la première phrase trouvée
    phrase_cible = df_aspirine_ok.iloc[0]['phrase_clinique']
    print("=== PHRASE SAINE TROUVÉE ===")
    print(phrase_cible)
    print(f"\nLabels réels : {df_aspirine_ok.iloc[0]['error_types']}")
else:
    print("Aucune phrase d'Aspirine sans erreur n'a été trouvée dans le dataset de test.")
# 1. Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_path = "/teamspace/studios/this_studio/deberta_rtd_medical_checkpoints/checkpoint-6250"

# 2. Chargement

tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
model = DebertaV2ForTokenClassification.from_pretrained(checkpoint_path).to(device)
model.eval()

# 3. Phrase de test
phrase_test = df_aspirine_ok.iloc[0]['phrase_clinique']

# 4. Inférence (reste identique)
inputs = tokenizer(phrase_test, return_tensors="pt", truncation=True, padding=True).to(device)

with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)[0]

# 5. Affichage détaillé (reste identique)
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
print(f"{'TOKEN':<20} | {'PRÉDICTION':<10}")
print("-" * 35)
for token, pred in zip(tokens, predictions):
    label = "ERREUR (1)" if pred == 1 else "OK (0)"
    print(f"{token:<20} | {label:<10}")