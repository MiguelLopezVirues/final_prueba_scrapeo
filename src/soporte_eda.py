# Tratamiento de datos
# -----------------------------------------------------------------------
import pandas as pd
import numpy as np


# Para pruebas estadísticas
# -----------------------------------------------------------------------
from scipy import stats
import math
from statsmodels.stats.proportion import proportions_ztest # para hacer el ztest


# Visualizacion de datos
# -----------------------------------------------------------------------
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.graphics.mosaicplot import mosaic


def exploracion_dataframe(dataframe):
    """
    Realiza un análisis exploratorio básico de un DataFrame, mostrando información sobre duplicados,
    valores nulos, tipos de datos, valores únicos para columnas categóricas y estadísticas descriptivas
    para columnas categóricas y numéricas, agrupadas por la columna de control.

    Params:
    - dataframe (DataFrame): El DataFrame que se va a explorar.
    - columna_control (str): El nombre de la columna que se utilizará como control para dividir el DataFrame.

    Returns: 
    No devuelve nada directamente, pero imprime en la consola la información exploratoria.
    """
    print(f"El número de datos es {dataframe.shape[0]} y el de columnas es {dataframe.shape[1]}")
    print("\n ..................... \n")

    print(f"5 filas aleatorias del dataframe son:")
    display(dataframe.sample(5))
    print("\n ..................... \n")

    print(f"Los tipos de las columnas y sus valores únicos son:")
    datos_conteos = pd.DataFrame(dataframe.dtypes, columns = ["tipo_dato"])
    datos_conteos["conteo"] = dataframe.nunique()
    display(datos_conteos)
    print("\n ..................... \n")

    print(f"Los duplicados que tenemos en el conjunto de datos son: {dataframe.duplicated().sum()}")
    print("\n ..................... \n")
    
    
    # generamos un DataFrame para los valores nulos
    print("Los nulos que tenemos en el conjunto de datos son:")
    df_nulos = pd.DataFrame(dataframe.isnull().sum() / dataframe.shape[0] * 100, columns = ["%_nulos"])
    display(df_nulos[df_nulos["%_nulos"] >0])
    print("\n ..................... \n")



    # controlar valores únicos de las variables
    print("Comprobamos que no haya valores con una sola variable:")
    for feature in dataframe.columns:
        if dataframe[feature].nunique() == 1:
            print(f"● La variable {feature} tiene 1 solo valor único. Se elimina.")
            dataframe.drop(columns=feature, inplace=True)
    print("\n ..................... \n")
    

    # controlar valores únicos de las variables numericas
    print("Comprobamos una representación mínima para valores numéricos:")
    for feature in dataframe.select_dtypes(np.number).columns:
        if dataframe[feature].nunique() <= 15:
            print(f"● La variable {feature} tiene {dataframe[feature].nunique()} < 15 valores únicos. Se convierte a objeto.")
            dataframe[feature] = dataframe[feature].astype("object")

    print("\n ..................... \n")

    print("Estadísticas descriptivas de las columnas numéricas:")
    display(dataframe.describe().T)
    print("\n ..................... \n")


    print("Estadísticas descriptivas de las columnas categóricas:")
    categorical_columns = dataframe.select_dtypes(include=['object']).columns
    if len(categorical_columns) > 0:
        display(dataframe[categorical_columns].describe().T)
    else:
        print("No hay columnas categóricas en el DataFrame.")
    print("\n ..................... \n")
        
    print("Los valores que tenemos para las columnas categóricas son: ")
    dataframe_categoricas = dataframe.select_dtypes(include = "O")
    
    for col in dataframe_categoricas.columns:
        print(f"La columna {col.upper()} tiene {dataframe[col].nunique()} valores únicos, de los cuales los primeros son:")
        display(pd.DataFrame(dataframe[col].value_counts()).assign(pct=lambda x: round(x["count"]/dataframe.shape[0],3)*100).head())    
    



def custom_properties(mapping):
    def properties(key):
        purchase_status = str(key[1])  # Extract PurchaseStatus from key
        return {"color": mapping.get(purchase_status, "gray")}  # Default to gray if not mapped
    return properties

def plot_relationships_categorical_target(df, target,hue=None, cat_type="count", num_type="hist",mapping={}):
    columns = df.drop(columns=target).columns.to_list()

    num_cols = 2
    num_rows = math.ceil(len(columns)/num_cols)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15,num_rows*4))
    axes = axes.flat

    fig.suptitle("Difference in distrubtion by target class.", y=0.93)
    for ax, feature in zip(axes, columns):
        if df[feature].dtype in ["int64","float64"]:
            if num_type == "box":
                sns.boxplot(data=df,
                            x=target,
                            y=feature,
                            ax=ax,
                            hue=None)
            else:
                sns.histplot(data=df,
                                x=feature,
                                hue=target,
                                ax=ax,
                                stat="proportion")

        else:
            # mosaic plots
            if cat_type == "mosaic":
                mosaic(df, [feature,target], properties=custom_properties(mapping), ax=ax)
            else:
                sns.countplot(data=df,
                            x=feature,
                            hue=target,
                            ax=ax)

        
        ax.set_title(feature)

    if len(columns) % 2 != 0:
            fig.delaxes(ax=axes[-1])


    plt.subplots_adjust(hspace=0.6)
    plt.show()


def plot_combined_target_distribution(df, target, feature, bins=25, repl_dict={}, figsize=(10, 6)):
    fig, ax = plt.subplots(figsize=figsize)

    fig.suptitle(f"Proportion of '{target}' by '{feature}' distribution")
    # Plot histogram without automatic legend
    sns.histplot(data=df,
                x=feature,
                bins='auto',
                ax=ax)

    # Create second axis
    ax2 = ax.twinx()

    sns.histplot(data=df,
             x=feature,
             hue=target,  # Differentiates by attrition status
             stat="probability",  # Normalize within bins
             bins='auto',  # Adjust number of bins as needed
             multiple="fill",  # Makes each bin stack to 1 (100%)
             palette={"Yes": "red", "No": "#FFFFFF"},
             ax=ax2,
             alpha=0.3,
             edgecolor=None)


    # Set y-axis limits
    ax2.set_ylim(0, 1)

    # remov automatic ax2 legend
    ax2.get_legend().remove()

    # Add custom legend for both plots
    fig.legend([f"{feature.capitalize()} distribution", f"{target.capitalize()} proportion"], loc="upper right")

    plt.show()


def calculate_rank_biserial(df, target, features):
    """
    Calcula Rank-Biserial y p-valores para variables continuas con un objetivo binario.

    Parameters:
        df (pd.DataFrame): DataFrame con los datos.
        target (str): Nombre de la columna binaria (0/1).
        features (list): Lista de columnas continuas a evaluar.

    Returns:
        pd.DataFrame: DataFrame con Rank-Biserial y p-valores.
    """
    rb_corr_target = {"Rank-biserial": [],
                      "P-value": []}

    tested_features = []
    for feature in features:
        if feature != target:
            x_0 = df.loc[df[target] == 0, feature]
            x_1 = df.loc[df[target] == 1, feature]
            
            stat, p_value = stats.mannwhitneyu(x_0, x_1, alternative='two-sided')
            n_0 = len(x_0)
            n_1 = len(x_1)
            rank_biserial = (2 * stat) / (n_0 * n_1) - 1

            rb_corr_target["Rank-biserial"].append(rank_biserial)
            rb_corr_target["P-value"].append(p_value)
            tested_features.append(feature)

    results_df = pd.DataFrame(rb_corr_target, index=tested_features)
    return results_df