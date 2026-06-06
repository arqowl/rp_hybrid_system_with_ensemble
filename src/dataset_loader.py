"""
T1 — Pipeline de Dados & Pré-processamento (fonte única de verdade)
==================================================================

Carrega as 5 bases de Estimação de Esforço de Software (SEE) a partir dos
arquivos REAIS em data/raw/, aplica o mesmo pré-processamento a todas e
expõe um contrato de interface único para as etapas seguintes (pool, GA,
DES, combinação, testes).

NÃO há geração de dados sintéticos. Se um arquivo de base não existir em
data/raw/, a base é apenas PULADA (com aviso). Nada é inventado — números
de execução têm que vir de dados reais.

Regras de pré-processamento (decididas e fixas para as 5 bases)
---------------------------------------------------------------
- Alvo (target) detectado automaticamente por nome normalizado.
- Colunas de IDENTIFICAÇÃO (nº de caso, id de projeto, nome) → descartadas.
- Colunas de DURAÇÃO (Duration / Length) → descartadas do X
  (tempo decorrido só é conhecido ao fim do projeto; correlaciona com o
   esforço → vazamento/ otimismo irreal numa estimativa a priori).
- Colunas de esforço normalizado redundante (ex.: N_effort no China) →
  descartadas (vazamento trivial do alvo).
- Finnish: colunas nominais de texto (YK, Business/Protype/Hardware_names)
  → descartadas (alta cardinalidade em base pequena; fora do escopo do artigo).
- Maxwell / China / COCOMO81: códigos categóricos e cost drivers tratados
  como numéricos/ordinais (consistente com o artigo de referência).
- Fatores de ajuste ordinais (T01.., cost drivers) → mantidos como numéricos.
- Holdout 70/30 com random_state=42.
- MinMax [0,1] com fit APENAS no treino (sem data leakage). y também é
  escalado; y_*_raw guarda os valores originais para métricas de negócio.

API pública
-----------
    DATASETS                      # registro de configuração das 5 bases
    load_raw(name)  -> DataFrame  # lê o arquivo cru (csv/arff)
    build_dataset(name) -> dict   # raw -> splits prontos (contrato)
    load_all(verbose) -> dict     # todas as bases disponíveis
    run_pipeline()                # executa, salva artefatos e metadata
"""

from __future__ import annotations

import os
import re
import json
import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# ─── Configuração global ─────────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE = 0.30
LOG_TARGET = False  # fidelidade ao artigo: MinMax no esforço cru (sem log)

# Diretórios (relativos à raiz do projeto)
RAW_DIR = os.path.join("data", "raw")
OUT_DIR = os.path.join("data", "processed")

# ─── Registro das 5 bases ────────────────────────────────────────────────────
# target_candidates : nomes possíveis do alvo (comparação por nome normalizado)
# drop_cols         : colunas a remover do X (ids, duração, esforço redundante,
#                     nominais de texto). Comparação por nome normalizado.
# fmt               : "csv" ou "arff"
DATASETS: dict[str, dict] = {
    "finnish": {
        "fmt": "csv",
        "file": "Finnish407.csv",
        "target_candidates": ["Worksup", "Effort"],
        "drop_cols": [
            "Case_Number", "Project_tech_ID", "Project_name",   # identificadores
            "YK", "Business_names", "Protype_names", "Hardware_names",  # nominais texto
            "Duration",                                          # duração
        ],
    },
    "maxwell": {
        "fmt": "arff",
        "file": "maxwell.arff",
        "target_candidates": ["Effort"],
        "drop_cols": ["Duration"],     # códigos categóricos mantidos como numéricos
    },
    "china": {
        "fmt": "arff",
        "file": "china.arff",
        "target_candidates": ["Effort"],
        # ID e N_effort (esforço normalizado) saem por vazamento; Duration sai.
        "drop_cols": ["ID", "N_effort", "Duration"],
    },
    "desharnais": {
        "fmt": "csv",
        "file": "desharnais.csv",
        "target_candidates": ["Effort"],
        # Project/id são identificadores; Length é a duração.
        "drop_cols": ["Project", "id", "Length"],
    },
    "cocomo81": {
        "fmt": "arff",
        "file": "cocomo81.arff",
        # No COCOMO81 do PROMISE o alvo se chama "actual" (pessoa-mês).
        "target_candidates": ["actual", "Effort", "effort"],
        "drop_cols": [],               # cost drivers ordinais mantidos como numéricos
    },
}


# ─── Utilitários ─────────────────────────────────────────────────────────────
def _norm(name: str) -> str:
    """Normaliza um nome de coluna para comparação robusta."""
    return re.sub(r"[^a-z0-9]+", "", str(name).lower().strip())


def _decode_arff(df: pd.DataFrame) -> pd.DataFrame:
    """Converte bytes (ARFF) para str e tudo que der para numérico."""
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(
                lambda x: x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else x
            )
    return df


def _find_target(df: pd.DataFrame, candidates: list[str]) -> str | None:
    norm_map = {_norm(c): c for c in df.columns}
    for cand in candidates:
        if _norm(cand) in norm_map:
            return norm_map[_norm(cand)]
    return None


# ─── Leitura crua ────────────────────────────────────────────────────────────
def raw_path(name: str) -> str:
    return os.path.join(RAW_DIR, DATASETS[name]["file"])


def is_available(name: str) -> bool:
    return os.path.exists(raw_path(name))


def load_raw(name: str) -> pd.DataFrame:
    """Lê o arquivo real da base `name`. Levanta erro se não existir."""
    cfg = DATASETS[name]
    path = raw_path(name)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"[{name}] arquivo real ausente: {path}. "
            f"Baixe-o do PROMISE e coloque em {RAW_DIR}/ — sem dados sintéticos."
        )
    if cfg["fmt"] == "csv":
        df = pd.read_csv(path)
    elif cfg["fmt"] == "arff":
        data, _meta = arff.loadarff(path)
        df = _decode_arff(pd.DataFrame(data))
    else:
        raise ValueError(f"Formato não suportado: {cfg['fmt']}")
    return df


# ─── Pré-processamento ───────────────────────────────────────────────────────
def preprocess(df: pd.DataFrame, cfg: dict, name: str) -> dict:
    """raw DataFrame -> dicionário-contrato com splits normalizados."""
    target_col = _find_target(df, cfg["target_candidates"])
    if target_col is None:
        raise ValueError(
            f"[{name}] alvo não encontrado. Candidatos={cfg['target_candidates']}. "
            f"Colunas disponíveis: {df.columns.tolist()}"
        )

    # Conjunto de colunas a descartar (normalizado), + o próprio alvo do X
    drop_norm = {_norm(c) for c in cfg["drop_cols"]}
    feature_cols = [
        c for c in df.columns
        if c != target_col and _norm(c) not in drop_norm
    ]
    dropped = [c for c in df.columns if c != target_col and _norm(c) in drop_norm]

    # Tudo numérico; o que não converter vira NaN e some no dropna
    work = df[feature_cols + [target_col]].apply(pd.to_numeric, errors="coerce")
    work = work.dropna(axis=1, how="all")               # colunas 100% NaN
    feature_cols = [c for c in work.columns if c != target_col]
    work = work.dropna().reset_index(drop=True)          # linhas com NaN

    if len(feature_cols) == 0:
        raise ValueError(f"[{name}] nenhuma feature numérica restou após limpeza.")

    # Qualidade: descarta projetos com esforço <= 0 (faltante/implausível).
    n_nonpos = int((work[target_col].values <= 0).sum())
    if n_nonpos:
        work = work[work[target_col] > 0].reset_index(drop=True)

    X = work[feature_cols].values.astype(float)
    y = work[target_col].values.astype(float)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    sx, sy = MinMaxScaler(), MinMaxScaler()
    X_tr_s = sx.fit_transform(X_tr)
    X_te_s = sx.transform(X_te)

    # Log-target: o esforço é fortemente assimétrico (skewness ~3.7 no artigo).
    # Modelamos log1p(esforço) e depois MinMax. y_*_raw guardam o esforço original.
    y_tr_m = np.log1p(y_tr) if LOG_TARGET else y_tr
    y_te_m = np.log1p(y_te) if LOG_TARGET else y_te
    y_tr_s = sy.fit_transform(y_tr_m.reshape(-1, 1)).ravel()
    y_te_s = sy.transform(y_te_m.reshape(-1, 1)).ravel()

    return {
        "name": name,
        "target_name": target_col,
        "feature_names": feature_cols,
        "dropped_cols": dropped,
        "n_rows_raw": int(len(df)),
        "n_rows_used": int(len(work)),
        "n_dropped_nonpositive": n_nonpos,
        "X_train": X_tr_s, "X_test": X_te_s,
        "y_train": y_tr_s, "y_test": y_te_s,
        "y_train_raw": y_tr, "y_test_raw": y_te,
        "scaler_X": sx, "scaler_y": sy,
        "n_train": int(X_tr_s.shape[0]),
        "n_test": int(X_te_s.shape[0]),
        "n_features": int(X_tr_s.shape[1]),
    }


def build_dataset(name: str) -> dict:
    """Pipeline completo de uma base: leitura real -> splits-contrato."""
    return preprocess(load_raw(name), DATASETS[name], name)


def load_all(verbose: bool = True) -> dict[str, dict]:
    """Constrói todas as bases disponíveis em data/raw/. Pula as ausentes."""
    out: dict[str, dict] = {}
    for name in DATASETS:
        if not is_available(name):
            if verbose:
                print(f"[T1] (pulada) {name}: arquivo ausente em {raw_path(name)}")
            continue
        s = build_dataset(name)
        out[name] = s
        if verbose:
            warn = ("  ⚠ %d linha(s) com alvo<=0 descartada(s)" % s["n_dropped_nonpositive"]
                    if s["n_dropped_nonpositive"] else "")
            print(f"[T1] {name:<11} alvo={s['target_name']:<8} "
                  f"linhas {s['n_rows_raw']}→{s['n_rows_used']} | "
                  f"features={s['n_features']} | "
                  f"treino/teste={s['n_train']}/{s['n_test']}{warn}")
            if s["dropped_cols"]:
                print(f"              descartadas: {s['dropped_cols']}")
    return out


# ─── Persistência (artefatos + contrato) ─────────────────────────────────────
def save_dataset(splits: dict):
    os.makedirs(OUT_DIR, exist_ok=True)
    p = os.path.join(OUT_DIR, splits["name"])
    np.save(f"{p}_X_train.npy", splits["X_train"])
    np.save(f"{p}_X_test.npy", splits["X_test"])
    np.save(f"{p}_y_train.npy", splits["y_train"])
    np.save(f"{p}_y_test.npy", splits["y_test"])
    np.save(f"{p}_y_train_raw.npy", splits["y_train_raw"])
    np.save(f"{p}_y_test_raw.npy", splits["y_test_raw"])


def save_metadata(all_splits: dict[str, dict]) -> dict:
    os.makedirs(OUT_DIR, exist_ok=True)
    meta = {
        "contract_version": "2.0",
        "source": "PROMISE SEE datasets (arquivos reais em data/raw/)",
        "holdout_ratio": f"{int((1-TEST_SIZE)*100)}/{int(TEST_SIZE*100)}",
        "random_state": RANDOM_STATE,
        "scaling": "MinMax [0,1], fit apenas no treino",
        "rules": {
            "duration_removed": True,
            "ids_removed": True,
            "finnish_text_nominals_removed": True,
            "nonpositive_target_rows_removed": True,
            "categorical_codes": "tratados como numéricos/ordinais",
            "synthetic_fallback": False,
        },
        "datasets": {
            name: {
                "target": s["target_name"],
                "n_train": s["n_train"], "n_test": s["n_test"],
                "n_features": s["n_features"],
                "feature_names": s["feature_names"],
                "dropped_cols": s["dropped_cols"],
                "n_rows_used": s["n_rows_used"],
            } for name, s in all_splits.items()
        },
    }
    path = os.path.join(OUT_DIR, "pool_metadata.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return meta


# ─── Helpers de leitura para etapas seguintes (T3+) ──────────────────────────
def datasets_with_pool() -> list[str]:
    """Bases que já têm matrizes gabarito geradas pelo train_pool (T2)."""
    return [n for n in DATASETS
            if os.path.exists(os.path.join(OUT_DIR, f"{n}_pred_matrix_train.npy"))]


def load_pool_arrays(name: str) -> tuple:
    """(pred_train, pred_test, y_train, y_test) reais de uma base já treinada."""
    p = os.path.join(OUT_DIR, name)
    return (
        np.load(f"{p}_pred_matrix_train.npy"),
        np.load(f"{p}_pred_matrix_test.npy"),
        np.load(f"{p}_y_train.npy"),
        np.load(f"{p}_y_test.npy"),
    )


def load_xy(name: str) -> tuple:
    """(X_train, X_test, y_train, y_test) reais de uma base."""
    p = os.path.join(OUT_DIR, name)
    return (
        np.load(f"{p}_X_train.npy"),
        np.load(f"{p}_X_test.npy"),
        np.load(f"{p}_y_train.npy"),
        np.load(f"{p}_y_test.npy"),
    )


def load_oof_train(name: str) -> np.ndarray:
    """Matriz de predições OUT-OF-FOLD do treino (gerada pelo train_pool)."""
    return np.load(os.path.join(OUT_DIR, f"{name}_pred_matrix_train_oof.npy"))


def inverse_target(name: str, y_norm: np.ndarray) -> np.ndarray:
    """Converte predição (espaço normalizado/log) de volta para esforço original."""
    y_norm = np.asarray(y_norm, dtype=float)
    ytr_raw = np.load(os.path.join(OUT_DIR, f"{name}_y_train_raw.npy"))
    base = np.log1p(ytr_raw) if LOG_TARGET else ytr_raw
    lo, hi = float(base.min()), float(base.max())
    m = y_norm * (hi - lo) + lo
    return np.expm1(m) if LOG_TARGET else m


def run_pipeline(verbose: bool = True) -> dict[str, dict]:
    print("=" * 64)
    print("  T1 — Carregamento e pré-processamento (dados reais, 5 bases)")
    print("=" * 64)
    all_splits = load_all(verbose=verbose)
    for s in all_splits.values():
        save_dataset(s)
    save_metadata(all_splits)
    print("-" * 64)
    print(f"[T1] ✓ {len(all_splits)}/{len(DATASETS)} base(s) processada(s) "
          f"→ artefatos em {OUT_DIR}/")
    ausentes = [n for n in DATASETS if n not in all_splits]
    if ausentes:
        print(f"[T1] Faltam (suba em {RAW_DIR}/): {ausentes}")
    return all_splits


if __name__ == "__main__":
    run_pipeline()
