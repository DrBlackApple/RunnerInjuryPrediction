import os
import zipfile
import requests
import pandas as pd

DATA_FOLDER = "data/"
DATA_ARCHIVE = DATA_FOLDER + "data.zip"

if __name__ == "__main__":
    if not os.path.exists(DATA_FOLDER):
        os.mkdir(DATA_FOLDER)

    if not os.path.exists(DATA_ARCHIVE):
        print("Gathering Data ", end="")
        # Url de téléchargement
        url = "https://www.kaggle.com/api/v1/datasets/download/shashwatwork/injury-prediction-for-competitive-runners?datasetVersionNumber=1"

        # Télécharger l'archive zip
        response = requests.get(url, allow_redirects=True)

        # Vérifier si le téléchargement a réussi
        if response.status_code == 200:
            # Écrire le contenu téléchargé dans un fichier local
            with open(DATA_ARCHIVE, "wb") as f:
                f.write(response.content)

            # Extraire les fichiers de l'archive zip
            with zipfile.ZipFile(DATA_ARCHIVE, "r") as zip_ref:
                zip_ref.extractall(DATA_FOLDER)
            print("[Done]")
        else:
            print(f"\nLe téléchargement a échoué avec le statut {response.status_code}")
    else:
        print("Data already downloaded !")

    df_week = pd.read_csv(DATA_FOLDER + "week_approach_maskedID_timeseries.csv")
    df_week.info()
