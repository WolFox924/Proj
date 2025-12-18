import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Répertoires
data_dir = Path(__file__).parent
output_dir = data_dir / "output"
output_dir.mkdir(exist_ok=True, parents=True)

def plot():
    energy_totals = []
    data_latency = []
    data_fets = []

    # Parcourir chaque sous-dossier de stratégie
    for strategy_dir in data_dir.iterdir():
        if strategy_dir.is_dir():
            strategy_name = strategy_dir.name

            # ---- ENERGIE (somme lignes 180482 à 180601) ----
            energy_file = strategy_dir / "energy_df.csv"
            if energy_file.exists():
                df_energy = pd.read_csv(energy_file)
                if 'energy_joules' in df_energy.columns:
                    df_energy['energy_joules'] = pd.to_numeric(df_energy['energy_joules'], errors='coerce')
                    # Somme exacte du subset
                    subset_sum = df_energy.iloc[180482:180602]['energy_joules'].sum()
                    energy_totals.append({'Strategy': strategy_name, 'total_energy_joules': subset_sum})

            # ---- LATENCE ----
            latency_file = strategy_dir / "scaling_evaluations_df.csv"
            if latency_file.exists():
                df_latency = pd.read_csv(latency_file)
                if 'avg_response_time_ms' in df_latency.columns:
                    df_latency['avg_response_time_ms'] = pd.to_numeric(df_latency['avg_response_time_ms'], errors='coerce')
                    df_latency['Strategy'] = strategy_name
                    data_latency.append(df_latency)

            # ---- FETS ----
            fets_file = strategy_dir / "fets_df.csv"
            if fets_file.exists():
                df_fets = pd.read_csv(fets_file)
                df_fets['fet_duration'] = pd.to_numeric(df_fets['t_fet_end'], errors='coerce') - pd.to_numeric(df_fets['t_fet_start'], errors='coerce')
                df_fets['wait_duration'] = pd.to_numeric(df_fets['t_wait_end'], errors='coerce') - pd.to_numeric(df_fets['t_wait_start'], errors='coerce')
                df_fets['Strategy'] = strategy_name
                data_fets.append(df_fets)

    # ----- PLOT LATENCE -----
    if data_latency:
        full_latency_df = pd.concat(data_latency, ignore_index=True)
        
        plt.figure(figsize=(10,5))
        sns.boxplot(data=full_latency_df, x='Strategy', y='avg_response_time_ms', palette='Set2')
        plt.title("Comparaison de la Latence (Distribution)")
        plt.ylabel("Temps de réponse (ms)")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(output_dir / "plot_latency_boxplot.png")
        plt.close()
        
        latency_stats = full_latency_df.groupby('Strategy', as_index=False)['avg_response_time_ms'].agg(['sum','mean']).reset_index()
        latency_stats.rename(columns={'sum':'total_latency_ms', 'mean':'avg_latency_ms'}, inplace=True)
        
        plt.figure(figsize=(10,5))
        sns.barplot(data=latency_stats, x='Strategy', y='total_latency_ms', palette='Blues_r')
        plt.title("Somme des Temps de Réponse par Stratégie")
        plt.ylabel("Total latence (ms)")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(output_dir / "plot_latency_sum.png")
        plt.close()
        print("Graphique latence généré.")

    # ----- PLOT ÉNERGIE -----
    if energy_totals:
        energy_df = pd.DataFrame(energy_totals)
        plt.figure(figsize=(10,5))
        sns.barplot(data=energy_df, x='Strategy', y='total_energy_joules', palette='RdYlGn_r')
        plt.title("Consommation Énergétique Totale (lignes 180482-180601) par Stratégie")
        plt.ylabel("Énergie (joules)")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(output_dir / "plot_energy_total_subset.png")
        plt.close()
        print("Graphique énergie généré.")

    # ----- PLOT FETS -----
    if data_fets:
        full_fets_df = pd.concat(data_fets, ignore_index=True)
        
        for metric in ['fet_duration','wait_duration']:
            plt.figure(figsize=(12,6))
            sns.boxplot(data=full_fets_df, x='Strategy', y=metric, palette='coolwarm')
            plt.title(f"{metric.replace('_',' ').capitalize()} par Stratégie")
            plt.ylabel("Durée (s)")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(output_dir / f"plot_{metric}.png")
            plt.close()
        print("Graphiques FET générés.")

if __name__ == "__main__":
    plot()

