import pandas as pd
from utils.file_handler import check_file

def load_data(path):
    check_file(path)
    df = pd.read_csv(path)
    return df
