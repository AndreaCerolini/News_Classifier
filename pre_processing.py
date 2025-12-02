import os
import numpy as np
import pandas as pd
import pickle
import unicodedata

PATH1 = 'Fake.csv' 

PATH2 = 'True.csv'

SAVE_PATH = os.path.join(os.getcwd(), "preprocessed_data")

os.makedirs(SAVE_PATH, exist_ok=True)

# ----------------------------- CARICAMENTO DATI FAKE -----------------------------

df_fake = pd.read_csv(PATH1)

# Elimino subject e date

df_fake = df_fake.drop(columns=["subject", "date"])

# Devo creare ora le etichette associate

df_fake["label"] = 1

# ------------------------ CARICAMENTO DATI TRUE -----------------------------

df_true = pd.read_csv(PATH2)

df_true = df_true.drop(columns=["subject", "date"])

df_true["label"] = 0

# ------------------------ UNIONE DEI DATI -----------------------------

df = pd.concat([df_fake, df_true], ignore_index=True)

print("Numero di esempi:", len(df))

# ------------------------ NORMALIZZAZIONE UNICODE --------------------------

def norm(s):

    if pd.isna(s):

            return ""

    s = unicodedata.normalize("NFKC", str(s))

    s = s.replace("\u00A0", " ")

    return s

# Applica la normalizzazione

df["title"] = df["title"].map(norm)

df["text"] = df["text"].map(norm)

# ------------------------- ENCODING DELLE LABEL -----------------------------

y = np.array(df["label"])

print(y.ndim, y.shape)

print("Esempi di etichette dopo conversione:", y[:10])

print(df.head())

# ----------------------------- VOCABOLARIO ----------------------------------

X_title = np.array(df["title"].astype(str))

X_text = np.array(df["text"].astype(str))

# Costruiamo il set dei caratteri presenti negli URL, riservando 0=PAD (riempimento) e 1=UNK (caratteri non visti)

PAD, UNK = 0, 1

# Calcoliamo ora il vocabolario per i titoli

# La funzione sorted ordina i caratteri rendendoli un set 

char_set = sorted({char for string in X_title for char in string})  

# Mappiamo ogni carattere a un indice intero, iniziando da 2 per lasciare spazio a PAD e UNK

string_to_index_title = {char: i+2 for i, char in enumerate(char_set)} 

# Calcoliamo la dimensione del vocabolario includendo PAD e UNK

vocab_size_title = len(string_to_index_title) + 2

# Calcoliamo ora il vocabolario per i testi

char_set = sorted({char for string in X_text for char in string})

string_to_index_text = {char: i+2 for i, char in enumerate(char_set)}

vocab_size_text = len(string_to_index_text) + 2

# Ma perchè li teniamo separati? Perchè i testi sono molto più lunghi e quindi hanno bisogno di un vocabolario più ampio
# quindi magari caratteri che appaiono nei testi non appaiono nei titoli e viceversa, e potrebbero
# essere significativi per la classificazione.

# ------------------------ ENCODING A SEQUENZA DI INDICI ------------------------

lengths_title = np.array([len(s) for s in X_title])

lengths_text  = np.array([len(s) for s in X_text])

MAX_LEN_TITLE = int(np.percentile(lengths_title, 95))

MAX_LEN_TEXT  = int(np.percentile(lengths_text, 95))

print("Nuova lunghezza massima titoli:", MAX_LEN_TITLE)

print("Nuova lunghezza massima testi:", MAX_LEN_TEXT)

# Funzione per convertire una stringa in una sequenza di indici interi

def encode(string,max_len, string_to_index):

    # Mappiamo ogni carattere a un indice intero, iniziando da 2 per lasciare spazio a PAD e UNK

    # .get(char, UNK) restituisce l'indice del carattere o UNK se non è presente
    # di ogni stringa di cui prendiamo solo i primi MAX_LEN caratteri (quindi tronchiamo se più lunga)

    idx = [string_to_index.get(char, UNK) for char in string[:max_len]]  

    # Se la stringa è più corta di MAX_LEN, aggiungiamo PAD fino a raggiungere la lunghezza desiderata

    if len(idx) < max_len:

        idx += [PAD] * (max_len - len(idx))

    return idx

X_title_idx = np.array([encode(string, MAX_LEN_TITLE, string_to_index_title) for string in X_title], dtype=np.int64)

X_text_idx = np.array([encode(string, MAX_LEN_TEXT, string_to_index_text) for string in X_text], dtype=np.int64)

print(f"Esempi di X_title_idx (shape={X_title_idx.shape}):")

print(X_title_idx[:3],"\n")

# ------------------------------- SALVATAGGI ---------------------------------

np.save(f"{SAVE_PATH}/X_title.npy", X_title_idx)           

np.save(f"{SAVE_PATH}/X_text.npy", X_text_idx)

np.save(f"{SAVE_PATH}/y.npy", y)

# Salviamo il mapping dei caratteri e i metadati utili in un file pickle

with open(f"{SAVE_PATH}/meta_data.pkl", "wb") as f:

    pickle.dump({
        
        "PAD": PAD,
        
        "UNK": UNK,
        
        "string_to_index_title": string_to_index_title,
        
        "string_to_index_text": string_to_index_text,
        
        "vocab_size_title": vocab_size_title,
        
        "vocab_size_text": vocab_size_text,
        
        "MAX_LEN_TITLE": MAX_LEN_TITLE,
        
        "MAX_LEN_TEXT": MAX_LEN_TEXT,
        
        "labels": sorted(list(set(df["label"])))
    }, f)

print("Salvato in:", SAVE_PATH,"\n")

print("File scritti: X_title.npy, X_text.npy, y.npy, meta_data.pkl","\n")

print("X_title shape:", X_title_idx.shape, "| y shape:", y.shape,)