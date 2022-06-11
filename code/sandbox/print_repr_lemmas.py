import pandas as pd
import gspread
import pickle
import numpy as np

with open('conjugation_local/repr_lemmas/repr_lemmas_adj_msa.pkl', 'rb') as f:
    repr_lemmas = pickle.load(f)
    repr_lemmas = pd.DataFrame(list(repr_lemmas.values()))
    repr_lemmas["line"] = np.arange(repr_lemmas.shape[0])
    repr_lemmas = repr_lemmas[["line", "lemma", "form", "pos", "cond_t", "rat", "index", "freq", "cond_s", "gen", "num", "cas", "gloss", "other_lemmas"]]
    # repr_lemmas = repr_lemmas[["line", "lemma", "freq", "form", "pos", "gen", "num", "cond_t", "cond_s", "cas", "rat", "gloss", "other_lemmas"]]

    sheet_name = 'Adj-Repr-Lemmas-Stem-Based'
    sa = gspread.service_account(
        "/Users/chriscay/.config/gspread/service_account.json")
    sh = sa.open('Nouns-Sandbox')
    worksheet = sh.add_worksheet(title=sheet_name, rows="100", cols="20")
    # worksheet = sh.worksheet(title=sheet_name)
    worksheet.clear()
    worksheet.update(
        [repr_lemmas.columns.values.tolist()] + repr_lemmas.values.tolist())

    # worksheet.freeze(rows=1, cols=5)
    worksheet.freeze(rows=1, cols=8)
