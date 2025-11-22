import pandas as pd

def cargar_datos(path):
    try:
        df = pd.read_csv(path)
        return df
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo {path}")
        return None
