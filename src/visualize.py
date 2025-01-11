import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_feature_importance(model, features, top_n=10):
    """
    Random Forest modelinden elde edilen özellik önemini görselleştirir.
    
    :param model: Eğitilmiş model (RandomForest gibi feature_importances_ özelliği olan)
    :param features: Özellik isimlerinin listesi
    :param top_n: Görselleştirilecek en önemli özellik sayısı (varsayılan: 10)
    """
    # Özellik önemlerini bir DataFrame'e dönüştür
    feature_importances = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    # Görselleştirme
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importances.head(top_n), palette="viridis")
    plt.title(f"En Önemli {top_n} Özellik - Random Forest", fontsize=16)
    plt.xlabel("Özellik Önemi", fontsize=14)
    plt.ylabel("Özellikler", fontsize=14)
    plt.tight_layout()
    plt.show()
