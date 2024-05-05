import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

def etude_fichier(df):
    print("Nombre de colonnes :", df.shape)
    print()
    print("Le type est : \n", df.dtypes)
    print()
    print('Nombre de valeurs uniques :')
    print(df.nunique())
    print()
    print('Le nombre de valeurs manquantes :\n', df.isnull().sum())

def analyse_statistique(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    stats = df.describe(include='all')

    stats.loc['median'] = df[numeric_cols].median()
    stats.loc['skewness'] = df[numeric_cols].skew()
    stats.loc['kurtosis'] = df[numeric_cols].kurtosis()

    return stats

def traiter_valeurs_manquantes(df, method='mean', columns=None):
    if method == 'mean':
        if columns is None:
            return df.fillna(df.mean())
        else:
            return df.fillna(df.mean()[columns])
    elif method == 'median':
        if columns is None:
            return df.fillna(df.median())
        else:
            return df.fillna(df.median()[columns])
    elif method == 'mode':
        if columns is None:
            return df.fillna(df.mode().iloc[0])
        else:
            return df.fillna(df.mode().iloc[0][columns])
    else:
        return df.fillna(method=method)

def etude_outliers(df, seuil=2.0):
    ''' 
    Fonction pour détecter et afficher le pourcentage d'outliers dans toutes les colonnes numériques d'un DataFrame,
    en utilisant le Z-score.

    Paramètres :
    df (DataFrame) : DataFrame Pandas.
    seuil (float) : Seuil pour le Z-score utilisé pour définir un outlier. Par défaut à 2.0.
    '''

    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in num_cols:
        series = df[col]
        z_scores = np.abs((series - series.mean()) / series.std())
        outliers_mask = z_scores > seuil
        outliers_percentage = (outliers_mask.mean() * 100).round(2)

        print(f"Le pourcentage de valeurs considérées comme des outliers en utilisant le Z-score au seuil {seuil} dans la colonne '{col}' est {outliers_percentage}%")


def correlation_matrix(df):
    corr_matrix = df.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Matrice de Corrélation")
    plt.show()

def visualiser_valeurs_manquantes(df):
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
    plt.title("Visualisation des Valeurs Manquantes")
    plt.show()

def encoder_variables_categorielles(df, columns=None):
    if columns is None:
        columns = df.select_dtypes(include='object').columns
    return pd.get_dummies(df, columns=columns, drop_first=True)

def reduire_dimension_pca(df, n_components=None):
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(df.select_dtypes(include=[np.number]))
    columns = [f"PC{i+1}" for i in range(components.shape[1])]
    df_pca = pd.DataFrame(components, columns=columns)
    return pd.concat([df.drop(columns=df.select_dtypes(include=[np.number]).columns), df_pca], axis=1)

def plot_skewness_kurtosis(data, column):
    skewness = data[column].skew()
    kurtosis = data[column].kurtosis()

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    sns.histplot(data[column], kde=True)
    plt.axvline(x=data[column].mean(), color='r', linestyle='dashed', linewidth=2)
    plt.title(f'Skewness: {skewness:.2f}')

    plt.subplot(1, 2, 2)
    sns.histplot(data[column], kde=True)
    plt.axvline(x=data[column].mean(), color='r', linestyle='dashed', linewidth=2)
    plt.axvline(x=data[column].mean() + data[column].std(), color='g', linestyle='dashed', linewidth=2)
    plt.axvline(x=data[column].mean() - data[column].std(), color='g', linestyle='dashed', linewidth=2)
    plt.axvline(x=data[column].mean() + 2 * data[column].std(), color='b', linestyle='dashed', linewidth=2)
    plt.axvline(x=data[column].mean() - 2 * data[column].std(), color='b', linestyle='dashed', linewidth=2)
    plt.title(f'Kurtosis: {kurtosis:.2f}')

    plt.tight_layout()

    if skewness > 0:
        skewness_analysis = "La distribution est inclinée positivement vers la droite (queue à droite)."
    elif skewness < 0:
        skewness_analysis = "La distribution est inclinée positivement vers la gauche (queue à gauche)."
    else:
        skewness_analysis = "La distribution est parfaitement symétrique."

    if kurtosis > 0:
        kurtosis_analysis = "La distribution est leptokurtique, avec des pics plus fins et des queues plus épaisses."
    elif kurtosis < 0:
        kurtosis_analysis = "La distribution est platykurtique, avec des pics plus larges et des queues plus minces."
    else:
        kurtosis_analysis = "La distribution est mésokurtique, similaire à une distribution normale."

    print("Analyse de la répartition (skewness):", skewness_analysis)
    print("Analyse de l'aplatissement (kurtosis):", kurtosis_analysis)

    plt.show()

def correlation_matrix(df):
    corr_matrix = df.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Matrice de Corrélation")
    plt.show()

def visualiser_valeurs_manquantes(df):
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
    plt.title("Visualisation des Valeurs Manquantes")
    plt.show()

def encoder_variables_categorielles(df, columns=None):
    if columns is None:
        columns = df.select_dtypes(include='object').columns
    return pd.get_dummies(df, columns=columns, drop_first=True)

def reduire_dimension_pca(df, n_components=None):
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(df.select_dtypes(include=[np.number]))
    columns = [f"PC{i+1}" for i in range(components.shape[1])]
    df_pca = pd.DataFrame(components, columns=columns)
    return pd.concat([df.drop(columns=df.select_dtypes(include=[np.number]).columns), df_pca], axis=1)

def plot_pca_correlation_circle(pca, features, x=0, y=1):
    fig, ax = plt.subplots(figsize=(10, 9))
    for i in range(0, pca.components_.shape[1]):
        ax.arrow(0,
                 0,  # Start the arrow at the origin
                 pca.components_[x, i],  # x for PCx
                 pca.components_[y, i],  # y for PCy
                 head_width=0.07,
                 head_length=0.07,
                 width=0.02)

        plt.text(pca.components_[x, i] + 0.05,
                 pca.components_[y, i] + 0.05,
                 features[i])

    # affichage des lignes horizontales et verticales
    plt.plot([-1, 1], [0, 0], color='grey', ls='--')
    plt.plot([0, 0], [-1, 1], color='grey', ls='--')

    # nom des axes, avec le pourcentage d'inertie expliqué
    plt.xlabel('F{} ({}%)'.format(x+1, round(100*pca.explained_variance_ratio_[x],1)))
    plt.ylabel('F{} ({}%)'.format(y+1, round(100*pca.explained_variance_ratio_[y],1)))

    plt.title("Cercle des corrélations (F{} et F{})".format(x+1, y+1))

    an = np.linspace(0, 2 * np.pi, 100)
    plt.plot(np.cos(an), np.sin(an))  # Add a unit circle for scale
    plt.axis('equal')
    plt.show(block=False)
    
def analyze_cluster(data, cluster_column, cluster_value):
    """Analyse un cluster spécifié en appelant toutes les fonctions."""
    cluster_data = get_cluster_data(data, cluster_column, cluster_value)
    print('Boxplot et radarplot pour le cluster', cluster_value)
    plot_subplots([cluster_data], data)
   
def plot_all_pca_correlation_circles(pca, features):
    n_components = pca.n_components_
    fig, axs = plt.subplots(n_components-1, n_components-1, figsize=(15, 15))

    for i in range(n_components):
        for j in range(i+1, n_components):
            ax = axs[i, j-1]  # j-1 car il n'y a pas de subplot pour i=j
            for k in range(0, pca.components_.shape[1]):
                ax.arrow(0, 0, pca.components_[i, k], pca.components_[j, k], head_width=0.07, head_length=0.07, width=0.02)
                ax.text(pca.components_[i, k] + 0.05, pca.components_[j, k] + 0.05, features[k])
            ax.plot([-1, 1], [0, 0], color='grey', ls='--')
            ax.plot([0, 0], [-1, 1], color='grey', ls='--')
            ax.set_xlabel('F{} ({}%)'.format(i+1, round(100*pca.explained_variance_ratio_[i],1)))
            ax.set_ylabel('F{} ({}%)'.format(j+1, round(100*pca.explained_variance_ratio_[j],1)))
            ax.set_title("Cercle des corrélations (F{} et F{})".format(i+1, j+1))
            an = np.linspace(0, 2 * np.pi, 100)
            ax.plot(np.cos(an), np.sin(an))  # Add a unit circle for scale
            ax.axis('equal')

    plt.tight_layout()
    plt.show(block=False)   

def get_cluster_data(data, cluster_column, cluster_value):
    """Renvoie un sous-ensemble de données pour le cluster spécifié."""
    return data[data[cluster_column] == cluster_value]
    
def analyze_cluster(data, cluster_column, cluster_value):
    """Analyse un cluster spécifié en appelant toutes les fonctions."""
    cluster_data = get_cluster_data(data, cluster_column, cluster_value)
    print('Boxplot et radarplot pour le cluster', cluster_value)
    plot_subplots([cluster_data], data)
    # print("Boxplot des colonnes numériques :")
    # plot_boxplot(cluster_data)
    # print("Radar plot des colonnes numériques :")
    # radar_plot(cluster_data)
    # print("Statistiques descriptives des colonnes numériques :")
    # descriptive_statistics(cluster_data)    

import pandas as pd
from sklearn.decomposition import PCA

def apply_pca(X):
    """
    Applique l'Analyse en Composantes Principales (ACP) sur les données X.

    Parameters:
        X (DataFrame): Les données d'entrée.

    Returns:
        pca (PCA): L'objet PCA ajusté.
    """
    # Créer les composantes principales
    pca = PCA()
    X_acp = pca.fit_transform(X)
    # Convertir en dataframe
    noms_composantes = [f"CP{i+1}" for i in range(X_acp.shape[1])]
    X_acp = pd.DataFrame(X_acp, columns=noms_composantes)
    # Créer les chargements
    chargements = pd.DataFrame(
        pca.components_.T,  # transposer la matrice des chargements
        columns=noms_composantes,  # les colonnes sont les composantes principales
        index=X.columns,  # les lignes sont les variables originales
    )
    return pca

def plot_variance(acp, largeur=8, dpi=100):
    """
    Trace les graphiques de la variance expliquée et cumulative de l'ACP.

    Parameters:
        acp (PCA): L'objet PCA ajusté.
        largeur (int): La largeur de la figure.
        dpi (int): La résolution de la figure.

    Returns:
        axs (array): Les axes des graphiques.
    """
    # Créer la figure
    fig, axs = plt.subplots(1, 2)
    n = acp.n_components_
    grille = np.arange(1, n + 1)
    # Variance expliquée
    variance_exp = acp.explained_variance_ratio_
    axs[0].bar(grille, variance_exp)
    axs[0].set(
        xlabel="Composante", title="% Variance Expliquée", ylim=(0.0, 1.0)
    )
    # Variance cumulative
    variance_cumul = np.cumsum(variance_exp)
    axs[1].plot(np.r_[0, grille], np.r_[0, variance_cumul], "o-")
    axs[1].set(
        xlabel="Composante", title="% Variance Cumulative", ylim=(0.0, 1.0)
    )
    # Configurer la figure
    fig.set(figwidth=largeur, dpi=dpi)
    return axs

def plot_all_pca_correlation_circles(pca, features):
    n_components = pca.n_components_
    fig, axs = plt.subplots(n_components-1, n_components-1, figsize=(15, 15))

    for i in range(n_components):
        for j in range(i+1, n_components):
            ax = axs[i, j-1]  # j-1 car il n'y a pas de subplot pour i=j
            for k in range(0, pca.components_.shape[1]):
                ax.arrow(0, 0, pca.components_[i, k], pca.components_[j, k], head_width=0.07, head_length=0.07, width=0.02)
                ax.text(pca.components_[i, k] + 0.05, pca.components_[j, k] + 0.05, features[k])
            ax.plot([-1, 1], [0, 0], color='grey', ls='--')
            ax.plot([0, 0], [-1, 1], color='grey', ls='--')
            ax.set_xlabel('F{} ({}%)'.format(i+1, round(100*pca.explained_variance_ratio_[i],1)))
            ax.set_ylabel('F{} ({}%)'.format(j+1, round(100*pca.explained_variance_ratio_[j],1)))
            ax.set_title("Cercle des corrélations (F{} et F{})".format(i+1, j+1))
            an = np.linspace(0, 2 * np.pi, 100)
            ax.plot(np.cos(an), np.sin(an))  # Add a unit circle for scale
            ax.axis('equal')

    plt.tight_layout()
    plt.show(block=False)

# Feature Scaling
def scale_features(df, scaler):
    """
    Scale numerical features using a specified scaler.

    Parameters:
        df (DataFrame): Input DataFrame with numerical features.
        scaler: Scaler object (e.g., StandardScaler, MinMaxScaler).

    Returns:
        DataFrame: DataFrame with scaled numerical features.
    """
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df

# Model Evaluation
from sklearn.metrics import confusion_matrix, classification_report

def evaluate_model(y_true, y_pred):
    """
    Evaluate the performance of a classification model.

    Parameters:
        y_true: True labels.
        y_pred: Predicted labels.

    Returns:
        dict: Classification report and confusion matrix.
    """
    report = classification_report(y_true, y_pred)
    matrix = confusion_matrix(y_true, y_pred)
    return {'classification_report': report, 'confusion_matrix': matrix}

# Data Splitting
from sklearn.model_selection import train_test_split

def split_data(df, target_column, test_size=0.2, random_state=42):
    """
    Split the dataset into training and testing sets.

    Parameters:
        df (DataFrame): Input DataFrame.
        target_column (str): Name of the target column.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Seed for random number generation.

    Returns:
        dict: Dictionary containing training and testing DataFrames.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop(target_column, axis=1), df[target_column],
        test_size=test_size, random_state=random_state
    )
    return {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test}