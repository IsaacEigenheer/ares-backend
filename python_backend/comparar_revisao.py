"""
comparador_planilhas_excel.py

Algoritmo para comparar abas (sheets) de dois arquivos Excel extraídos de PDFs
(revisões diferentes de um desenho técnico). O script identifica quais abas
correspondem entre os arquivos usando uma combinação de métricas de similaridade
(Jaccard, cosseno sobre n-grams TF-IDF, Levenshtein, Jaro-Winkler e embeddings
(BERTimbau quando disponível / fallback multilingual)). Depois gera um arquivo
Excel de saída contendo:
  - para cada par de abas correspondentes: as duas tabelas lado a lado
  - uma aba de diff listando as células alteradas (posição, cabeçalho, valor antigo, novo)
  - listas de abas não emparelhadas (removidas / adicionadas)

Requisitos (instalar com pip):
  pip install pandas openpyxl scikit-learn rapidfuzz tqdm
  # para embeddings (opcional, melhora qualidade):
  pip install transformers torch sentence-transformers

Uso:
  python comparador_planilhas_excel.py --old arquivo_antigo.xlsx --new arquivo_novo.xlsx --out relatorio_differences.xlsx

Obs:
  - O algoritmo faz normalização de texto e tenta alinhar colunas por nome
    (fuzzy) antes de comparar célula a célula.
  - Ajuste pesos e thresholds conforme necessidade.

"""

from __future__ import annotations
import argparse
import re
import os
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz, distance
from tqdm import tqdm

# Tenta importar transformers/sentence-transformers para embeddings (opcional)
try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    _HAS_TRANSFORMERS = True
except Exception:
    _HAS_TRANSFORMERS = False

try:
    from sentence_transformers import SentenceTransformer
    _HAS_SENTENCE_TRANSFORMERS = True
except Exception:
    _HAS_SENTENCE_TRANSFORMERS = False


# ----------------------------- utilitários -----------------------------

def normalize_text(s: str) -> str:
    """Normaliza uma string: minusculas, remove espaços redundantes e pontuação básica."""
    if pd.isna(s):
        return ""
    s = str(s)
    s = s.strip().lower()
    # remover múltiplos espaços
    s = re.sub(r"\s+", " ", s)
    # remover caracteres não alfanuméricos básicos (mas preservar .,-/())
    s = re.sub(r"[^0-9a-záàâãéèêíïóôõöúçñç\-\./(),% ]+", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s)
    return s


def df_to_text(df: pd.DataFrame) -> str:
    """Converter DataFrame para um texto representativo (headers + linhas)"""
    parts = []
    # headers
    headers = [normalize_text(c) for c in df.columns]
    parts.append(" | ".join(headers))
    # sample rows (ou todas se pequenas)
    for _, row in df.iterrows():
        vals = [normalize_text(v) for v in row.tolist()]
        parts.append(" | ".join(vals))
    return "\n".join(parts)


# --------------------------- métricas -------------------------------

def jaccard_similarity(a: str, b: str) -> float:
    tokens_a = set(a.split())
    tokens_b = set(b.split())
    if not tokens_a and not tokens_b:
        return 1.0
    inter = tokens_a.intersection(tokens_b)
    uni = tokens_a.union(tokens_b)
    return len(inter) / len(uni) if uni else 0.0


def tfidf_cosine_similarity(a: str, b: str, analyzer: str = "char_wb", ngram_range=(3, 5)) -> float:
    # usa TF-IDF em character n-grams para tolerância a pequenas mudanças
    vect = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram_range)
    X = vect.fit_transform([a, b])
    sim = cosine_similarity(X[0], X[1])[0][0]
    return float(sim)


def levenshtein_ratio(a: str, b: str) -> float:
    # rapidfuzz.distance.Levenshtein.normalized_similarity retorna 0..100
    try:
        r = distance.Levenshtein.normalized_similarity(a, b) / 100.0
    except Exception:
        r = 0.0
    return float(r)


def jaro_winkler_similarity(a: str, b: str) -> float:
    try:
        r = fuzz.WRatio(a, b) / 100.0  # WRatio é robusto; aproxima Jaro-Winkler/combinação
    except Exception:
        r = 0.0
    return float(r)


# ---------------------- embeddings BERTimbau / fallback -----------------

def get_text_embeddings(texts: List[str], model_name: Optional[str] = None) -> np.ndarray:
    """
    Tenta retornar embeddings para uma lista de textos.
    - Se transformers estiver instalado, tenta carregar 'neuralmind/bert-base-portuguese-cased' (BERTimbau)
      e faz mean pooling dos tokens.
    - Caso contrário, tenta sentence-transformers (paraphrase-multilingual-MiniLM-L12-v2).
    - Se nada disponível, retorna None.
    """
    if model_name is None:
        model_name = "neuralmind/bert-base-portuguese-cased"

    if _HAS_TRANSFORMERS:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
            model.eval()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)

            all_embeddings = []
            with torch.no_grad():
                for t in texts:
                    inputs = tokenizer(t, return_tensors="pt", truncation=True, padding=True, max_length=512)
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    out = model(**inputs)
                    # mean pooling over tokens (attention mask aware)
                    token_embeds = out.last_hidden_state  # (1, seq_len, dim)
                    mask = inputs.get("attention_mask", None)
                    if mask is not None:
                        mask = mask.unsqueeze(-1)
                        summed = (token_embeds * mask).sum(1)
                        counts = mask.sum(1).clamp(min=1)
                        emb = summed / counts
                    else:
                        emb = token_embeds.mean(1)
                    emb = emb.cpu().numpy()[0]
                    all_embeddings.append(emb)
            return np.vstack(all_embeddings)
        except Exception as exc:
            print("Falha ao gerar embeddings com transformers:", exc)
            # fallback

    if _HAS_SENTENCE_TRANSFORMERS:
        try:
            st_model_name = "paraphrase-multilingual-MiniLM-L12-v2"
            st = SentenceTransformer(st_model_name)
            embs = st.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            return embs
        except Exception as exc:
            print("Falha com sentence-transformers:", exc)

    # se não for possível, retorna None
    return None


# --------------------- combinar métricas e emparelhamento ----------------

def combined_similarity(a_text: str, b_text: str, emb_a: Optional[np.ndarray] = None, emb_b: Optional[np.ndarray] = None,
                        weights: Dict[str, float] = None) -> float:
    """Computa uma similaridade combinada (0..1) usando várias métricas."""
    if weights is None:
        weights = {
            "jaccard": 0.15,
            "tfidf": 0.25,
            "levenshtein": 0.15,
            "jaro": 0.15,
            "embedding": 0.3
        }
    scores = {}
    scores["jaccard"] = jaccard_similarity(a_text, b_text)
    scores["tfidf"] = tfidf_cosine_similarity(a_text, b_text)
    scores["levenshtein"] = levenshtein_ratio(a_text, b_text)
    scores["jaro"] = jaro_winkler_similarity(a_text, b_text)

    if emb_a is not None and emb_b is not None:
        # cosine entre embeddings
        try:
            num = np.dot(emb_a, emb_b)
            den = np.linalg.norm(emb_a) * np.linalg.norm(emb_b)
            emb_sim = float(num / den) if den != 0 else 0.0
        except Exception:
            emb_sim = 0.0
        scores["embedding"] = emb_sim
    else:
        # se não houver embeddings, reduzir peso de embedding proporcionalmente
        scores["embedding"] = 0.0

    # normalizar pesos se embedding não fornecido
    total_w = sum(weights.values())
    # if embedding score is zero but weight>0, we should renormalize to keep comparability
    if scores["embedding"] == 0 and weights.get("embedding", 0) > 0:
        # distribuir o peso de embedding proporcionalmente para os outros
        emb_w = weights["embedding"]
        other_sum = total_w - emb_w
        if other_sum <= 0:
            norm_weights = {k: 1.0 / len(weights) for k in weights}
        else:
            norm_weights = {k: (w / other_sum) for k, w in weights.items() if k != "embedding"}
            # reintroduzir embedding com 0
            norm_weights["embedding"] = 0.0
    else:
        norm_weights = {k: (w / total_w) for k, w in weights.items()}

    combined = 0.0
    for k, v in scores.items():
        w = norm_weights.get(k, 0.0)
        combined += w * v

    return float(combined)


def match_sheets(dfs_old: Dict[str, pd.DataFrame], dfs_new: Dict[str, pd.DataFrame],
                 weights: Dict[str, float] = None, threshold: float = 0.60,
                 use_embeddings: bool = True) -> Tuple[List[Tuple[str, str, float]], List[str], List[str]]:
    """
    Retorna:
      - lista de triples (sheet_old_name, sheet_new_name, similarity_score)
      - lista de old-only sheet names (removidas)
      - lista de new-only sheet names (adicionadas)

    Estratégia:
      - converte cada df para texto representativo
      - (opcional) calcula embeddings
      - calcula matriz de similaridade combinada
      - para cada sheet do old, pega melhor candidato do new se acima de threshold
      - garante matching 1:1 (se conflito, escolhe maior score globalmente)
    """
    names_old = list(dfs_old.keys())
    names_new = list(dfs_new.keys())

    texts_old = [df_to_text(dfs_old[n]) for n in names_old]
    texts_new = [df_to_text(dfs_new[n]) for n in names_new]

    emb_old = None
    emb_new = None
    if use_embeddings:
        embs = get_text_embeddings(texts_old + texts_new)
        if embs is not None:
            emb_old = embs[: len(texts_old)]
            emb_new = embs[len(texts_old):]
        else:
            use_embeddings = False

    # pré-cálculo TF-IDF coseno em batch pode ser feito, porém para simplicidade usamos par-a-par
    candidates = []
    for i, a in enumerate(tqdm(texts_old, desc="calculando similaridade")):
        for j, b in enumerate(texts_new):
            emb_a = emb_old[i] if (emb_old is not None) else None
            emb_b = emb_new[j] if (emb_new is not None) else None
            score = combined_similarity(a, b, emb_a=emb_a, emb_b=emb_b, weights=weights)
            candidates.append((names_old[i], names_new[j], score))

    # ordena por score decrescente e faz greedy matching 1:1
    candidates = sorted(candidates, key=lambda x: x[2], reverse=True)
    matched_old = set()
    matched_new = set()
    matches = []
    for old_n, new_n, sc in candidates:
        if sc < threshold:
            break
        if old_n in matched_old or new_n in matched_new:
            continue
        matched_old.add(old_n)
        matched_new.add(new_n)
        matches.append((old_n, new_n, sc))

    old_only = [n for n in names_old if n not in matched_old]
    new_only = [n for n in names_new if n not in matched_new]
    return matches, old_only, new_only


# --------------------- comparação célula a célula -----------------------

def align_columns(df_old: pd.DataFrame, df_new: pd.DataFrame, col_threshold: float = 0.7) -> List[Tuple[str, Optional[str]]]:
    """Tenta alinhar nomes de colunas por similaridade fuzzy. Retorna lista de pares (old_col, matched_new_col_or_None)."""
    old_cols = list(df_old.columns)
    new_cols = list(df_new.columns)
    pairs = []
    used_new = set()
    for oc in old_cols:
        best = None
        best_score = 0.0
        for nc in new_cols:
            if nc in used_new:
                continue
            score = jaro_winkler_similarity(normalize_text(str(oc)), normalize_text(str(nc)))
            if score > best_score:
                best_score = score
                best = nc
        if best_score >= col_threshold:
            pairs.append((oc, best))
            used_new.add(best)
        else:
            pairs.append((oc, None))
    # também apontar new cols não usados as (None, newcol)
    for nc in new_cols:
        if nc not in used_new:
            pairs.append((None, nc))
    return pairs


def generate_diff_rows(df_old: pd.DataFrame, df_new: pd.DataFrame) -> List[Dict]:
    """Gera uma lista de diferenças entre duas tabelas após alinhar colunas.
    Cada diferença é um dict: {row_key, column_name, old_value, new_value, type}

    Observação: Tenta igualar por índice; se índices diferentes, trata como linhas distintas.
    """
    diffs = []
    # alinha colunas
    col_pairs = align_columns(df_old, df_new)

    # construir normalized column mapping: map old_col -> new_col or None
    mapping_old_to_new = {oc: nc for oc, nc in col_pairs if oc is not None}
    # observar new-only columns
    new_only_columns = [nc for oc, nc in col_pairs if oc is None and nc is not None]

    # vamos comparar por índice. Se índices iguais (mesmo comprimento e tipos), compara por posição
    max_rows = max(len(df_old), len(df_new))
    for i in range(max_rows):
        row_old = df_old.iloc[i] if i < len(df_old) else None
        row_new = df_new.iloc[i] if i < len(df_new) else None
        # para cada coluna do conjunto (old mapped + new-only)
        cols_to_check = list(mapping_old_to_new.keys()) + new_only_columns
        for col in cols_to_check:
            new_col = mapping_old_to_new.get(col, None)
            old_val = row_old[col] if (row_old is not None and col in df_old.columns) else None
            new_val = row_new[new_col] if (row_new is not None and new_col in df_new.columns) else None
            # normalizar valores para comparação
            old_n = normalize_text(old_val) if old_val is not None else ""
            new_n = normalize_text(new_val) if new_val is not None else ""
            if old_n != new_n:
                change_type = "modified"
                if row_old is None:
                    change_type = "added_row"
                elif row_new is None:
                    change_type = "removed_row"
                elif col not in df_old.columns:
                    change_type = "added_column"
                diffs.append({
                    "row_index": i,
                    "old_column": col,
                    "new_column": new_col,
                    "old_value": old_val,
                    "new_value": new_val,
                    "change_type": change_type
                })
    return diffs


# --------------------- gerar arquivo Excel de saída --------------------

def generate_report(matches: List[Tuple[str, str, float]], dfs_old: Dict[str, pd.DataFrame], dfs_new: Dict[str, pd.DataFrame],
                    old_only: List[str], new_only: List[str], out_path: str):
    """Gera um relatório Excel com os pares comparados e diffs."""
    writer = pd.ExcelWriter(out_path, engine="openpyxl")

    # resumo
    resumo_rows = []
    for old, new, sc in matches:
        resumo_rows.append({"old_sheet": old, "new_sheet": new, "score": sc})
    resumo_df = pd.DataFrame(resumo_rows)
    resumo_df.to_excel(writer, sheet_name="RESUMO_MATCHES", index=False)

    # abas não casadas
    pd.DataFrame({"removed_sheets": old_only}).to_excel(writer, sheet_name="REMOVIDAS", index=False)
    pd.DataFrame({"added_sheets": new_only}).to_excel(writer, sheet_name="ADICIONADAS", index=False)

    # para cada match, colocar lado a lado e diffs
    for old, new, sc in matches:
        df_old = dfs_old[old]
        df_new = dfs_new[new]
        # side-by-side summary
        maxr = max(len(df_old), len(df_new))
        # preparar df lado a lado: prefixar colunas
        left = df_old.copy()
        right = df_new.copy()
        left.columns = [f"old__{c}" for c in left.columns]
        right.columns = [f"new__{c}" for c in right.columns]
        side = pd.concat([left.reset_index(drop=True), right.reset_index(drop=True)], axis=1)
        sheet_name_side = f"side_{old[:15]}_vs_{new[:15]}"
        # garantir nome de aba até 31 chars
        sheet_name_side = sheet_name_side[:31]
        side.to_excel(writer, sheet_name=sheet_name_side, index=False)

        # diffs detalhados
        diffs = generate_diff_rows(df_old, df_new)
        diffs_df = pd.DataFrame(diffs)
        sheet_name_diffs = f"diff_{old[:12]}_{new[:12]}"[:31]
        if diffs_df.empty:
            pd.DataFrame([{"message": "sem_diferencas_detectadas"}]).to_excel(writer, sheet_name=sheet_name_diffs, index=False)
        else:
            diffs_df.to_excel(writer, sheet_name=sheet_name_diffs, index=False)

    writer.close()


# --------------------- script principal / CLI -------------------------

def main(old_path, new_path, id):
    # leitura de todas as abas
    dfs_old = pd.read_excel(old_path, sheet_name=None)
    dfs_new = pd.read_excel(new_path, sheet_name=None)

    # executar matching
    matches, old_only, new_only = match_sheets(dfs_old, dfs_new, threshold=0.60, use_embeddings="use_embeddings")

    # gerar relatório
    generate_report(matches, dfs_old, dfs_new, old_only, new_only, f"Excel/relatorio_final{id}.xlsx")


if __name__ == '__main__':
    main()
