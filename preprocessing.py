import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler


def create_features(df):
    df = df.sort_values(by=['obs_id', 'order_id'])
    df['price_diff'] = df.groupby('obs_id')['price'].transform(
        lambda x: x.diff()).fillna(0)

    # Ajout d'un epsilon pour éviter la division par zero
    epsilon = 1e-9
    df['log_return'] = df.groupby('obs_id')['price'].transform(
        lambda x: np.log((x+epsilon)/(x.shift(1)+epsilon))
    )
    df['log_return'] = df['log_return'].replace(
        [np.inf, -np.inf], np.nan).fillna(0)

    df['liquidité'] = df['ask'] - df['bid']
    df['mid_price'] = (df['ask'] + df['bid']) / 2

    # Suppression de order_id si inutile
    df = df.drop(columns=['order_id'], errors='ignore')

    return df


def pre_processing(df, encoders=None, scaler=None, is_train=True):
    # Variables catégorielles
    categorical_features = ['venue', 'action', 'side', 'trade']
    # On s’assure que ces colonnes existent
    for cat_feature in categorical_features:
        if cat_feature not in df.columns:
            raise ValueError(f"{cat_feature} not found in dataframe")

    # Encodage label pour les variables catégorielles
    if encoders is None:
        encoders = {}
    for cat_feature in categorical_features:
        if is_train:
            encoder = LabelEncoder()
            df[cat_feature] = encoder.fit_transform(df[cat_feature])
            encoders[cat_feature] = encoder
        else:
            encoder = encoders.get(cat_feature)
            if encoder is not None:
                df[cat_feature] = encoder.transform(df[cat_feature])
            else:
                raise ValueError(
                    f"Encoder for {cat_feature} not found during test processing.")

    # Variables numériques
    numerical_features = ['bid', 'ask', 'price', 'bid_size', 'ask_size', 'flux',
                          'liquidité', 'mid_price', 'price_diff', 'log_return']

    # Séparer les features à scaler et à laisser intactes
    numerical_features_to_scale = [
        feature for feature in numerical_features if feature != 'price']

    if is_train:
        scaler = StandardScaler()
        df[numerical_features_to_scale] = scaler.fit_transform(
            df[numerical_features_to_scale])
    else:
        df[numerical_features_to_scale] = scaler.transform(
            df[numerical_features_to_scale])

    # Définir la liste finale des features (incluant 'price' non-scalée)
    features = categorical_features + numerical_features
    return df, features, encoders, scaler, categorical_features, numerical_features
