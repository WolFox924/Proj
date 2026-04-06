import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

output_dir = Path(__file__).resolve().parent / 'output'

def plot():
    data = []
    for f in output_dir.glob("results_*.csv"):
        df = pd.read_csv(f)
        # Nettoyage du nom de la stratégie
        strat_name = f.stem.replace("results_", "")
        df['Strategy'] = strat_name
        data.append(df)
    
    if not data: print("Pas de fichiers !"); return
    full_df = pd.concat(data)

    # 1. LATENCE (Boxplot)
    plt.figure(figsize=(10,5))
    resp_df = full_df[full_df['metric'] == 'response_time']
    # Avant la ligne 25 était
    # resp_tot = resp_df 
    resp_tot = resp_df.groupby('Strategy')['duration'].sum().reset_index()
    sns.boxplot(data=resp_tot, x='Strategy', y='duration', palette='Set2')
    plt.title("Comparaison de la Latence (Performance)")
    plt.ylabel("Temps de réponse (secondes)")
    plt.savefig(output_dir / "plot_latency.png")
    print(f"Graphique généré : {output_dir}/plot_latency.png")

    # 2. ENERGIE (Barplot total)
    plt.figure(figsize=(10,5))
    nrg_df = full_df[full_df['metric'] == 'energy_consumption']
    # Somme de l'énergie par stratégie
    total_nrg = nrg_df.groupby('Strategy')['joules'].sum().reset_index()
    sns.barplot(data=total_nrg, x='Strategy', y='joules', palette='RdYlGn_r') # Rouge = Haut, Vert = Bas
    plt.title("Consommation Énergétique Totale")
    plt.ylabel("Joules")
    plt.savefig(output_dir / "plot_energy.png")
    print(f"Graphique généré : {output_dir}/plot_energy.png")

if __name__ == "__main__":
    plot()