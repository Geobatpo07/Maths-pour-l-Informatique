
# %% [markdown]
# # Mini‑projet — Classifieur Bayésien Naïf (Bernoulli) 🧠🧮
# **Mathématiques pour l'informatique — FSGA / Université Quisqueya**  
# **Enseignant : Geovany Batista Polo LAGUERRE — Semestre 1 — 2025–2026**
#
# Ce notebook contient **tout le nécessaire** pour votre mini‑projet :
# - classe `NaiveBayesBernoulli` (avec **lissage de Laplace** déjà implémenté),
# - fonctions utilitaires pour lire les données,
# - **pipeline robuste** de prédiction (texte libre ou vecteurs binaires) avec `safe_predict`,
# - TODO structurés pour vous guider.

# %% [markdown]
# ## Règles & rendu
# - Travail **individuel**. Autorisés : `csv`, `math`, `collections`, `itertools`, `random`, `matplotlib` (facultatif). **Interdit :** `scikit-learn`.
# - Rendez ce notebook **exécuté** (toutes les sorties présentes).  
# - Nommez le fichier : `NOM_Prenom_NaiveBayes_Pantalon.ipynb`.
#
# ### Barème (rappel)
# - Implémentation correcte (utilisation de la classe + pipeline) — 35 pts  
# - Lissage de Laplace **utilisé** et **interprété** — 15 pts  
# - Dénombrements & fréquences affichés — 15 pts  
# - Démo et cas tests pertinents — 15 pts  
# - Qualité du code & commentaires — 10 pts  
# - Analyse & limites/pistes — 10 pts

# %% [markdown]
# ## Données (jeu jouet + possibilité de jeu 150 lignes)
# La cellule suivante **écrit** un jeu **minimal** de 8 lignes dans `/mnt/data/train_pantalon.csv`.  
# Si votre enseignant fournit `train_pantalon.csv`, vous pouvez **remplacer le chemin** dans la suite.

# %%
import csv, pandas as pd
from pathlib import Path

csv_small = Path("/mnt/data/train_pantalon.csv")
rows_small = [
    {"id":1, "pas_cher":1, "anglais":0, "achat":"OUI"},
    {"id":2, "pas_cher":0, "anglais":1, "achat":"NON"},
    {"id":3, "pas_cher":0, "anglais":1, "achat":"NON"},
    {"id":4, "pas_cher":0, "anglais":1, "achat":"NON"},
    {"id":5, "pas_cher":1, "anglais":0, "achat":"NON"},
    {"id":6, "pas_cher":1, "anglais":1, "achat":"OUI"},
    {"id":7, "pas_cher":1, "anglais":0, "achat":"OUI"},
    {"id":8, "pas_cher":1, "anglais":0, "achat":"OUI"},
]

csv_small.parent.mkdir(parents=True, exist_ok=True)
with open(csv_small, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["id","pas_cher","anglais","achat"])
    w.writeheader(); w.writerows(rows_small)

print("Fichier écrit :", csv_small)
pd.read_csv(csv_small).head()

# %% [markdown]
# ## Classe fournie : `NaiveBayesBernoulli` (lissage de **Laplace** déjà implémenté)
# - **À VOUS** d’**utiliser** cette classe dans le pipeline (chargement, fit, prédiction, affichage des comptes, etc.).
# - Vous pouvez ajouter de **nouvelles features binaires** (facultatif).

# %%
import csv
from collections import Counter, defaultdict
from math import log, exp

class NaiveBayesBernoulli:
    """
    Classifieur Bayésien Naïf (modèle Bernoulli binaire)
    - X : dict binaire {'pas_cher':0/1, 'anglais':0/1, ...}
    - y : 'OUI' / 'NON'
    - Lissage de Laplace (alpha) : déjà implémenté.
    """
    def __init__(self, alpha=1.0):
        self.alpha = float(alpha)
        self.classes_ = []
        self.features_ = []
        self.class_counts_ = Counter()
        self.feature_counts_ = defaultdict(lambda: Counter())
        self.n_ = 0

    def fit(self, X_list, y_list):
        self.n_ = len(y_list)
        self.classes_ = sorted(set(y_list))
        feat = set()
        for X in X_list:
            feat |= set(X.keys())
        self.features_ = sorted(feat)
        self.class_counts_.clear()
        self.feature_counts_.clear()
        for X, y in zip(X_list, y_list):
            self.class_counts_[y] += 1
            for f in self.features_:
                v = int(X.get(f, 0))
                self.feature_counts_[y][(f, v)] += 1
        return self

    def _p_class(self, c):
        return self.class_counts_[c] / self.n_

    def _p_feat_given_class(self, feat, val, c):
        c1 = self.feature_counts_[c][(feat, 1)]
        c0 = self.feature_counts_[c][(feat, 0)]
        tot = c1 + c0
        num = (c1 + self.alpha) if val == 1 else (c0 + self.alpha)
        den = tot + 2*self.alpha
        return num / den

    def predict_proba(self, X):
        scores = {}
        for c in self.classes_:
            s = log(self._p_class(c))
            for f in self.features_:
                v = int(X.get(f, 0))
                s += log(self._p_feat_given_class(f, v, c))
            scores[c] = s
        m = max(scores.values())
        exps = {c: exp(v - m) for c, v in scores.items()}
        Z = sum(exps.values())
        return {c: exps[c]/Z for c in self.classes_}

    def predict(self, X):
        proba = self.predict_proba(X)
        return max(proba, key=proba.get)

def load_csv_binary(path, feature_names=("pas_cher","anglais")):
    X_list, y_list = [], []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            X = {feat: int(row[feat]) for feat in feature_names}
            y = row["achat"].strip()
            X_list.append(X); y_list.append(y)
    return X_list, y_list

def pretty_counts(nb: NaiveBayesBernoulli):
    print("Classes :", nb.classes_)
    print("Features:", nb.features_)
    for c in nb.classes_:
        print(f"\nClasse {c} (count={nb.class_counts_[c]})")
        for f in nb.features_:
            c1 = nb.feature_counts_[c][(f,1)]
            c0 = nb.feature_counts_[c][(f,0)]
            tot = c1 + c0
            p1 = (c1 + nb.alpha) / (tot + 2*nb.alpha)
            p0 = (c0 + nb.alpha) / (tot + 2*nb.alpha)
            print(f"  {f}: #1={c1}, #0={c0}, p(1|{c})={p1:.3f}, p(0|{c})={p0:.3f}")

# %% [markdown]
# ## Pipeline robuste : texte libre → features → prédiction sûre (`safe_predict`)
# Cette partie gère les **entrées ad hoc** : texte libre ou dictionnaires imparfaits.
# - Normalisation de texte (`normalize_text`)
# - Règles simples de détection de mots‑clés (`KEYWORD_RULES`)
# - Vectorisation (`text_to_features`)
# - Validation de vecteurs (`validate_instance`)
# - Prédiction avec **abstention** possible (`safe_predict`)

# %%
import re, unicodedata
from typing import Dict, Tuple

def normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

KEYWORD_RULES = {
    "pas_cher": [r"\bpas\s*cher\b", r"\bprix\b", r"\bbon\s*plan\b", r"\breduction\b", r"\bpromo\b"],
    "anglais":  [r"\banglais\b", r"\btraduction\b", r"\benglish\b", r"\btranslate\b"],
}
compiled_rules = {feat: [re.compile(pat) for pat in pats] for feat, pats in KEYWORD_RULES.items()}

def text_to_features(query: str, expected_features=("pas_cher","anglais")) -> Dict[str, int]:
    q = normalize_text(query)
    X = {f: 0 for f in expected_features}
    for feat in expected_features:
        if feat in compiled_rules:
            X[feat] = int(any(pat.search(q) for pat in compiled_rules[feat]))
    return X

def validate_instance(x: Dict[str, int], expected_features=("pas_cher","anglais")) -> Tuple[bool, str]:
    if set(x.keys()) != set(expected_features):
        extra = set(x.keys()) - set(expected_features)
        missing = set(expected_features) - set(x.keys())
        return False, f"Features inattendues: {sorted(extra)} ; manquantes: {sorted(missing)}"
    for f in expected_features:
        if x[f] not in (0,1):
            return False, f"Feature {f} doit valoir 0 ou 1 (reçu: {x[f]!r})"
    return True, "OK"

def confidence_from_proba(proba: Dict[str, float]) -> float:
    vals = sorted(proba.values(), reverse=True)
    if len(vals) < 2:
        return 1.0
    return vals[0] - vals[1]

def safe_predict(nb_model, query_or_dict, expected_features=("pas_cher","anglais"), threshold=0.15):
    if isinstance(query_or_dict, str):
        x = text_to_features(query_or_dict, expected_features)
        source = "text"
    else:
        x = dict(query_or_dict)
        source = "dict"
        ok, msg = validate_instance(x, expected_features)
        if not ok:
            return {"status":"INVALID", "reason":msg, "features":x}
    proba = nb_model.predict_proba(x)
    yhat  = max(proba, key=proba.get)
    conf  = confidence_from_proba(proba)
    if conf < threshold:
        return {"status":"ABSTAIN", "reason":f"marge={conf:.3f} < seuil={threshold:.3f}", "features":x, "proba":proba}
    return {"status":"OK", "prediction":yhat, "proba":proba, "confidence":conf, "features":x, "source":source}

# %% [markdown]
# ## ✅ TODO 1 — Charger les données et entraîner le modèle
# 1. Charger `/mnt/data/train_pantalon.csv` **ou** remplacer par votre `train_pantalon.csv`.
# 2. Entraîner `NaiveBayesBernoulli(alpha=1.0)`.  
# 3. Afficher **comptes** et **probabilités lissées** via `pretty_counts`.

# %%
data_path = "/mnt/data/train_pantalon.csv"   # <- remplacez par "/mnt/data/train_pantalon.csv" si fourni
X_train, y_train = load_csv_binary(data_path, feature_names=("pas_cher","anglais"))
nb = NaiveBayesBernoulli(alpha=1.0).fit(X_train, y_train)
pretty_counts(nb)

# %% [markdown]
# ## ✅ TODO 2 — Prédire et comparer (`alpha=1` vs `alpha=0`)
# Tester sur : `[1,1]`, `[1,0]`, `[0,1]`, `[0,0]`.  
# **Comparer** les probabilités et la classe prédite avec/sans lissage. **Interpréter** en 4–6 lignes.

# %%
tests = [
    {"pas_cher":1, "anglais":1},
    {"pas_cher":1, "anglais":0},
    {"pas_cher":0, "anglais":1},
    {"pas_cher":0, "anglais":0},
]

def run_preds(alpha):
    nb = NaiveBayesBernoulli(alpha=alpha).fit(X_train, y_train)
    out = []
    for x in tests:
        proba = nb.predict_proba(x)
        yhat = nb.predict(x)
        out.append((x, {k:round(v,3) for k,v in proba.items()}, yhat))
    return out

print("== Avec lissage alpha=1.0 ==")
for x, proba, yhat in run_preds(1.0):
    print(x, "→", yhat, "| proba:", proba)

print("\n== Sans lissage alpha=0.0 ==")
for x, proba, yhat in run_preds(0.0):
    print(x, "→", yhat, "| proba:", proba)

# %% [markdown]
# ## ✅ TODO 3 — Entrées “ad hoc” (texte libre & validation)
# 1. Tester `safe_predict` avec des **requêtes texte** (orthographes et variantes).  
# 2. Tester `safe_predict` avec des **dicts** corrects/incorrects.  
# 3. Ajuster le **seuil** `threshold` (ex. 0.15 → 0.25) et commenter l'effet.

# %%
nb = NaiveBayesBernoulli(alpha=1.0).fit(X_train, y_train)

queries = [
    "pantalon pas cher pour homme",
    "patron pantalon en anglais svp",
    "traduction du mot pantalon",
    "prix d'un pantalon en promo",
    "je cherche des infos sur pantalon"
]

for q in queries:
    res = safe_predict(nb, q, expected_features=("pas_cher","anglais"), threshold=0.15)
    print(q, "=>", res)

print("\n-- Dict invalide --")
bad = {"pas_cher": 2, "anglais": "oui"}
print(safe_predict(nb, bad, threshold=0.15))

print("\n-- Dict valide --")
good = {"pas_cher": 1, "anglais": 0}
print(safe_predict(nb, good, threshold=0.15))

# %% [markdown]
# ## ✅ TODO 4 — Rapport court (5–8 lignes)
# - Hypothèse **naïve** (indépendance conditionnelle **à classe fixée**).  
# - Pourquoi elle est pratique (et sa **limite**).  
# - Interpréter l’effet du **lissage** à partir de vos résultats.  
# - Intérêt de `safe_predict` pour gérer des entrées **bruitées/ad hoc**.

# %% [markdown]
# ## (Option) Visualisation
# Carte des scores `p(OUI | pas_cher, anglais)` pour `alpha=1`.

# %%
import matplotlib.pyplot as plt
import numpy as np

nb = NaiveBayesBernoulli(alpha=1.0).fit(X_train, y_train)
grid = [(a,b) for a in [0,1] for b in [0,1]]
scores = [ nb.predict_proba({"pas_cher":a,"anglais":b})["OUI"] for a,b in grid ]

fig, ax = plt.subplots(figsize=(4,3))
im = ax.imshow(np.array(scores).reshape(2,2), vmin=0, vmax=1, origin="lower")
ax.set_xticks([0,1]); ax.set_xticklabels(["anglais=0","anglais=1"])
ax.set_yticks([0,1]); ax.set_yticklabels(["pas_cher=0","pas_cher=1"])
for i,(a,b) in enumerate(grid):
    ax.text(b, a, f"{scores[i]:.2f}", ha="center", va="center", color="w")
ax.set_title("p(OUI | pas_cher, anglais) — alpha=1")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## ✅ TODO 5 — Enrichir les features binaires (règles mots‑clés)
# Objectif: améliorer la vectorisation texte→features sans sklearn.
# - Ajouter 1–3 nouvelles features binaires pertinentes (ex. `soldes`, `livraison`, `marque`).
# - Étendre/ajuster les regex de `KEYWORD_RULES` et recompiler.
# - Re‑entraîner et observer l'effet sur les proba/prédictions et sur la matrice de confusion (TODO 6).
# Conseils: partez simple; évitez d'introduire des features corrélées si possible.

# %%
RUN_TODO_5 = False  # Mettez True pour exécuter ce bloc
if RUN_TODO_5:
    import re
    # Copie sûre des règles existantes pour expérimentation
    KEYWORD_RULES_STUDENT = {
        k: list(v) for k, v in KEYWORD_RULES.items()
    }
    # EXEMPLES — À MODIFIER/COMPLÉTER
    # Nouvelle feature "soldes" (prix réduit)
    KEYWORD_RULES_STUDENT["soldes"] = [r"\bsoldes?\b", r"\bremise\b", r"\bblack\s*friday\b"]
    # Renforcer "anglais"
    KEYWORD_RULES_STUDENT["anglais"] += [r"\bangl\.", r"\buk\b"]

    compiled_rules = {feat: [re.compile(pat) for pat in pats]
                      for feat, pats in KEYWORD_RULES_STUDENT.items()}

    # Revectoriser une requête pour vérifier la prise en compte
    print("Features attendues:", list(KEYWORD_RULES_STUDENT.keys()))
    print(text_to_features("soldes pantalon pas cher uk" , KEYWORD_RULES_STUDENT.keys()))

# %% [markdown]
# ## ✅ TODO 6 — Découpage Train/Test et métriques sans sklearn
# - Créer un split aléatoire (ex. 70/30) reproductible (fixer la graine).
# - Calculer Accuracy et Matrice de confusion 2x2 avec de simples compteurs.
# - Afficher les exemples mal classés et réfléchir aux causes (alimentera TODO 9).

# %%
RUN_TODO_6 = False
if RUN_TODO_6:
    import random
    random.seed(42)

    # Charger les données (adaptez feature_names si vous avez ajouté des features en TODO 5)
    feats = ("pas_cher","anglais")  # ajustez: ("pas_cher","anglais","soldes", ...)
    X_all, y_all = load_csv_binary(data_path, feature_names=feats)

    # Split 70/30
    idx = list(range(len(y_all)))
    random.shuffle(idx)
    cut = int(0.7*len(idx))
    idx_tr, idx_te = idx[:cut], idx[cut:]

    X_tr = [X_all[i] for i in idx_tr]; y_tr = [y_all[i] for i in idx_tr]
    X_te = [X_all[i] for i in idx_te]; y_te = [y_all[i] for i in idx_te]

    nb_tt = NaiveBayesBernoulli(alpha=1.0).fit(X_tr, y_tr)

    # Confusion et accuracy
    conf = {('OUI','OUI'):0, ('OUI','NON'):0, ('NON','OUI'):0, ('NON','NON'):0}
    errors = []
    ok = 0
    for x, y in zip(X_te, y_te):
        yhat = nb_tt.predict(x)
        ok += int(yhat == y)
        conf[(y, yhat)] += 1
        if yhat != y:
            errors.append((x, y, yhat, nb_tt.predict_proba(x)))

    acc = ok / max(1, len(y_te))
    print("Accuracy test:", round(acc,3))
    print("Matrice de confusion (y_true, y_pred):", conf)
    print("Exemples mal classés (max 5):")
    for e in errors[:5]:
        print(e)

# %% [markdown]
# ## ✅ TODO 7 — Effet du seuil d'abstention (coverage vs précision)
# - Balayer plusieurs valeurs de `threshold` pour `safe_predict` (ex. 0.00→0.50).
# - Mesurer: taux d'abstention, précision conditionnelle sur les cas où le modèle ne s'abstient pas.
# - Option: tracer la courbe (coverage -> précision) si `matplotlib` est disponible.

# %%
RUN_TODO_7 = False
if RUN_TODO_7:
    import numpy as np
    nb_thr = NaiveBayesBernoulli(alpha=1.0).fit(X_train, y_train)
    thresholds = [round(t,2) for t in np.linspace(0.0, 0.5, 11)]
    results = []
    for th in thresholds:
        n, abstain, ok = 0, 0, 0
        for x, y in zip(X_train, y_train):
            res = safe_predict(nb_thr, x, expected_features=("pas_cher","anglais"), threshold=th)
            n += 1
            if res["status"] == "ABSTAIN":
                abstain += 1
            elif res["status"] == "OK":
                ok += int(res["prediction"] == y)
        coverage = 1 - abstain/max(1,n)
        prec = ok / max(1, (n - abstain))
        results.append((th, coverage, prec))
    print("threshold | coverage | precision")
    for th, cov, pre in results:
        print(f"{th:8.2f} | {cov:8.3f} | {pre:9.3f}")

# %% [markdown]
# ## ✅ TODO 8 — Balayage du lissage (alpha)
# - Évaluer plusieurs `alpha` (0, 0.5, 1, 2, 5) sur le split de TODO 6 ou via validation croisée simple.
# - Observer stabilité des proba et la robustesse des prédictions.

# %%
RUN_TODO_8 = False
if RUN_TODO_8:
    import numpy as np, random
    random.seed(42)
    feats = ("pas_cher","anglais")
    X_all, y_all = load_csv_binary(data_path, feature_names=feats)
    idx = list(range(len(y_all))); random.shuffle(idx)
    cut = int(0.7*len(idx))
    X_tr = [X_all[i] for i in idx[:cut]]; y_tr = [y_all[i] for i in idx[:cut]]
    X_te = [X_all[i] for i in idx[cut:]]; y_te = [y_all[i] for i in idx[cut:]]

    for alpha in [0.0, 0.5, 1.0, 2.0, 5.0]:
        nb_a = NaiveBayesBernoulli(alpha=alpha).fit(X_tr, y_tr)
        ok = sum(nb_a.predict(x) == y for x, y in zip(X_te, y_te))
        acc = ok / max(1, len(y_te))
        print(f"alpha={alpha:.1f} → accuracy test={acc:.3f}")

# %% [markdown]
# ## ✅ TODO 9 — Analyse d'erreurs et amélioration des règles texte
# - À partir des erreurs (TODO 6), identifier 2–3 motifs récurrents.
# - Proposer puis tester une amélioration des regex ou une nouvelle feature (voir TODO 5) pour corriger au moins 1 motif.
# - Documenter l'avant/après (quelques lignes + comptage des erreurs corrigées).

# %%
RUN_TODO_9 = False
if RUN_TODO_9:
    # Exemple d'esquisse d'analyse (à adapter selon vos erreurs)
    from collections import Counter
    # Ici, rejouer une prédiction et collecter les erreurs comme dans TODO 6 puis agréger
    feats = ("pas_cher","anglais")
    X_all, y_all = load_csv_binary(data_path, feature_names=feats)
    nb_err = NaiveBayesBernoulli(alpha=1.0).fit(X_all, y_all)
    errors = []
    for x, y in zip(X_all, y_all):
        yhat = nb_err.predict(x)
        if yhat != y:
            errors.append((tuple(sorted(x.items())), y, yhat))
    print("Nb d'erreurs:", len(errors))
    print("Top motifs (features, y_true→y_pred):")
    cnt = Counter(((e[0], e[1], e[2])) for e in errors)
    for (f,y,yh), c in cnt.most_common(5):
        print(c, f, f"{y}->{yh}")

# %% [markdown]
# ## ✅ TODO 10 — Sauvegarde/chargement simple du modèle (JSON) et reproductibilité
# - Écrire 2 fonctions: `save_nb(model, path)` et `load_nb(path)` qui sérialisent le modèle sans pickle.
# - Sauvegarder `alpha`, `classes_`, `features_`, `class_counts_`, `feature_counts_`, `n_`.
# - Pour `feature_counts_`, convertir les clés `(feat, val)` en chaîne `"feat|val"`.
# - Tester: sauvegarder puis recharger et vérifier que les prédictions sont identiques sur 4 cas tests.

# %%
RUN_TODO_10 = False
if RUN_TODO_10:
    import json

    def save_nb(model: NaiveBayesBernoulli, path: str):
        data = {
            "alpha": model.alpha,
            "classes_": model.classes_,
            "features_": model.features_,
            "class_counts_": dict(model.class_counts_),
            "feature_counts_": {
                c: {f"{k[0]}|{k[1]}": v for k, v in d.items()}
                for c, d in model.feature_counts_.items()
            },
            "n_": model.n_,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_nb(path: str) -> NaiveBayesBernoulli:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        nb2 = NaiveBayesBernoulli(alpha=data["alpha"])
        nb2.classes_ = data["classes_"]
        nb2.features_ = data["features_"]
        from collections import Counter, defaultdict
        nb2.class_counts_ = Counter(data["class_counts_"])
        fc = defaultdict(lambda: Counter())
        for c, d in data["feature_counts_"].items():
            for k, v in d.items():
                feat, sval = k.split("|")
                fc[c][(feat, int(sval))] = int(v)
        nb2.feature_counts_ = fc
        nb2.n_ = int(data["n_"])
        return nb2

    # Test rapide de round-trip
    X_train, y_train = load_csv_binary(data_path, feature_names=("pas_cher","anglais"))
    nb0 = NaiveBayesBernoulli(alpha=1.0).fit(X_train, y_train)
    save_nb(nb0, "/mnt/data/nb_model.json")
    nb1 = load_nb("/mnt/data/nb_model.json")
    tests = [
        {"pas_cher":1, "anglais":1},
        {"pas_cher":1, "anglais":0},
        {"pas_cher":0, "anglais":1},
        {"pas_cher":0, "anglais":0},
    ]
    for x in tests:
        p0, p1 = nb0.predict(x), nb1.predict(x)
        print(x, "→", p0, p1, "OK" if p0==p1 else "MISMATCH")
