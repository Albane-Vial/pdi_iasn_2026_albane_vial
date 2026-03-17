# Prescription Anomaly Detection — MIMIC-IV

Goal: detect outlier/anomalous drug prescriptions using both structured and **textual** data from MIMIC-IV v2.1.

---

## 1. Variable Selection

### Core table: `prescriptions`

| Variable | Type | Why |
|---|---|---|
| `drug` | free text ⭐ | Main textual signal — messy, normalized drug names |
| `prod_strength` | free text ⭐ | "125mg/5mL Suspension" — encode + extract numeric |
| `route` | text/categorical ⭐ | IV vs PO vs SQ — anomalous route = strong signal |
| `dose_val_rx` | numeric | Outlier detection per drug group |
| `dose_unit_rx` | categorical | Needed to normalize doses |
| `doses_per_24_hrs` | numeric | Unusual frequency |
| `form_rx` | categorical | TABLET, VIAL, etc. |
| `drug_type` | categorical | Filter on `MAIN` first |
| `starttime`, `stoptime` | timestamps | Compute duration |
| `gsn`, `ndc` | codes | Group drugs reliably (less messy than `drug` text) |

### From `patients`

| Variable | Why |
|---|---|
| `anchor_age` | Dose anomalies are age-dependent |
| `gender` | Drug appropriateness |

### From `admissions`

| Variable | Why |
|---|---|
| `admission_type` | ELECTIVE vs EMER. = very different prescription patterns |
| `race` | Demographic context |
| `hospital_expire_flag` | Weak supervision signal (careful: not a label for bad prescriptions, but correlates) |

### From `diagnoses_icd` + `d_icd_diagnoses` ⭐ (Key for textual analysis)

| Variable | Why |
|---|---|
| `long_title` (aggregated) | **Core textual feature**: "Unspecified atrial fibrillation" → should predict expected drugs |
| `seq_num` | Take top 3–5 diagnoses (primary first) |

### From `services`

| Variable | Why |
|---|---|
| `curr_service` | "CSURG" vs "PSYCH" — drug norms differ radically by service |

---

## 2. Anomaly Definition Strategies

Since there is no ground-truth "bad prescription" label, combine multiple signals:

| Signal | Method |
|---|---|
| **Dose outlier** | IQR/z-score of `dose_val_rx` grouped by `ndc` or `gsn` |
| **Drug–Diagnosis mismatch** | Embed `drug` + `prod_strength` vs aggregated `long_title` (BioBERT/sentence-transformers), flag low cosine similarity |
| **Unusual route** | Most frequent route per `gsn`; flag deviations |
| **Unusual duration** | Outlier on `(stoptime - starttime)` per drug group |
| **Service mismatch** | Morphine in PSYCH, chemotherapy in MEDICINE, etc. |

---

## 3. Joins & SQL Query

```sql
-- Step 1: One service per admission (take first)
WITH first_service AS (
    SELECT hadm_id, curr_service
    FROM services
    WHERE prev_service IS NULL  -- first service of the admission
),

-- Step 2: Top-5 diagnoses per admission, concatenated
admission_diagnoses AS (
    SELECT
        di.hadm_id,
        STRING_AGG(dd.long_title, ' | ' ORDER BY di.seq_num) AS diagnoses_text
    FROM diagnoses_icd di
    JOIN d_icd_diagnoses dd USING (icd_code, icd_version)
    WHERE di.seq_num <= 5
    GROUP BY di.hadm_id
)

-- Main query
SELECT
    -- Identifiers
    pr.subject_id,
    pr.hadm_id,
    pr.pharmacy_id,

    -- Prescription core (textual)
    pr.drug,
    pr.prod_strength,
    pr.route,
    pr.form_rx,
    pr.drug_type,

    -- Prescription numeric
    pr.dose_val_rx,
    pr.dose_unit_rx,
    pr.doses_per_24_hrs,
    pr.gsn,
    pr.ndc,

    -- Computed duration
    EXTRACT(EPOCH FROM (pr.stoptime - pr.starttime)) / 3600.0 AS duration_hours,

    -- Patient
    p.gender,
    p.anchor_age,

    -- Admission context
    a.admission_type,
    a.race,
    a.hospital_expire_flag,

    -- Service
    fs.curr_service,

    -- Diagnoses (textual) ⭐
    ad.diagnoses_text

FROM prescriptions pr
LEFT JOIN patients             p  USING (subject_id)
LEFT JOIN admissions           a  USING (hadm_id)
LEFT JOIN first_service        fs USING (hadm_id)
LEFT JOIN admission_diagnoses  ad USING (hadm_id)

WHERE pr.drug_type = 'MAIN'           -- skip additives
  AND pr.drug IS NOT NULL
  AND pr.dose_val_rx IS NOT NULL;
```

---

## 4. Train / Test Split Strategy

For **anomaly detection** (mostly unsupervised), the split logic differs from classic supervised ML:

```
┌─────────────────────────────────────────────────────────────────────┐
│  Option A – Temporal split (recommended for realistic evaluation)   │
│  Train: admittime < 2016                                            │
│  Test:  admittime >= 2016                                           │
│  → Avoids leakage, mirrors real deployment                          │
├─────────────────────────────────────────────────────────────────────┤
│  Option B – Semi-supervised                                         │
│  Train: ELECTIVE admissions only (cleanest, most "routine" Rx)      │
│  Test:  ALL admission types                                         │
│  → Train the "normal" distribution on planned care                  │
├─────────────────────────────────────────────────────────────────────┤
│  Option C – Patient-level random split (80/20)                      │
│  Split on subject_id (NOT on prescription rows!)                    │
│  → Prevents same-patient data bleeding across sets                  │
└─────────────────────────────────────────────────────────────────────┘
```

> Always split on `subject_id` or `hadm_id`, never on individual prescription rows.

---

## 5. Feature Engineering Pipeline

```python
# Textual features → NLP encoder
text_features = [
    "drug",           # "Metoprolol Succinate"
    "prod_strength",  # "25mg Tablet"
    "route",          # "PO"
    "diagnoses_text", # "Chronic systolic heart failure | Hypertension | ..."
    "curr_service",   # "MED"
]

# Concatenate into a single document per prescription
df["text_input"] = (
    "Drug: "      + df["drug"].fillna("") + " | " +
    "Strength: "  + df["prod_strength"].fillna("") + " | " +
    "Route: "     + df["route"].fillna("") + " | " +
    "Service: "   + df["curr_service"].fillna("") + " | " +
    "Diagnoses: " + df["diagnoses_text"].fillna("")
)

# Numeric features (scale + impute)
numeric_features = [
    "dose_val_rx", "doses_per_24_hrs", "duration_hours", "anchor_age"
]
```

Encode `text_input` with **BioBERT** or `sentence-transformers/all-MiniLM-L6-v2`, concatenate with scaled numeric features, then feed into an anomaly detector (Isolation Forest, Autoencoder, LOF).

---

## Summary Checklist

- [x] **Base table**: `prescriptions` filtered to `drug_type = 'MAIN'`
- [x] **Textual signals**: `drug`, `prod_strength`, `route`, `diagnoses_text` (aggregated `long_title`)
- [x] **Context**: `curr_service`, `admission_type`, `anchor_age`, `gender`
- [x] **Drug grouping**: use `gsn`/`ndc` to normalize messy `drug` free text
- [x] **Split unit**: `subject_id` level, preferably temporal
