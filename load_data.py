import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from get_data import DATA_FOLDER

WEEK_TRAINING_CSV = "week_approach_maskedID_timeseries.csv"


def get_data(test_size=0.2):
    df_week = pd.read_csv(DATA_FOLDER + WEEK_TRAINING_CSV)
    # filtre les colonnes finissant par .x
    df_week = df_week.filter(regex=".+[^\.][^1-6]$")

    # drop les colonnes suivantes
    df_sans_injury = df_week.drop(["injury", "Athlete ID", "Date"], axis=1)

    # scale selon une loi standard
    scaler = StandardScaler()
    df_normalise = pd.DataFrame(scaler.fit_transform(df_sans_injury))

    X = df_normalise
    y = df_week["injury"]

    ros = RandomOverSampler()
    X_resampled_over, y_resampled_over = ros.fit_resample(X, y)

    # Séparer les données en un jeu d'entraînement et un jeu de test
    return train_test_split(
        X_resampled_over.to_numpy(),
        y_resampled_over.to_numpy(),
        test_size=0.2,
        random_state=42,
    )
