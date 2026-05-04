import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# ================= CONFIGURATION =================
current_dir = Path.cwd()
data_dir = current_dir / 'output' / 'scripts_data'
if not data_dir.exists():
    data_dir = Path(__file__).resolve().parent / 'output' / 'scripts_data'

# Couleurs consistantes
STRATEGY_COLORS = {
    'HPST_Performance': '#d62728', # Rouge
    'LPLT_Energy': '#2ca02c',      # Vert
    'APE_H_Hybrid': '#1f77b4',     # Bleu
    'Standard_K8s': '#9467bd',     # Violet
    'Standard_First': '#8c564b',    # Marron
    'RL_PPO_30_50_20_DLnew' : '#db00db'
}


def load_all_data():
    data = {'invocations': [], 'power': [], 'scaling': [], 'energy_stats': []}
    
    for folder in data_dir.glob("batch_*"):
        strat = folder.name.replace("batch_", "")
        if strat in STRATEGY_COLORS.keys():
            # 1. Invocations
            try:
                inv = pd.read_csv(folder / "invocations_df.csv")
                inv['Strategy'] = strat
                # Nettoyage latence
                if 't_exec' in inv.columns: inv['duration'] = inv['t_exec']
                elif 'execution_time' in inv.columns: inv['duration'] = inv['execution_time']
                # On garde les succès uniquement pour la latence
                inv = inv[inv['duration'] > 0].dropna(subset=['node'])
                data['invocations'].append(inv)
            except: pass

            # 2. Power (Watts over time)
            try:
                pwr = pd.read_csv(folder / "power_df.csv")
                pwr['Strategy'] = strat
                # Aggrégation par seconde pour alléger le graph
                pwr_grouped = pwr.groupby('timestamp')['power_watts'].sum().reset_index()
                pwr_grouped['Strategy'] = strat
                data['power'].append(pwr_grouped)
                
                # Calcul Energie Totale pour le Pareto
                total_joules = pwr['power_watts'].sum() # Approx si 1 row = 1 sec
                data['energy_stats'].append({'Strategy': strat, 'Total_Joules': total_joules})
            except: pass

            # 3. Scaling Decisions
            try:
                sc = pd.read_csv(folder / "scaling_decisions_df.csv")
                sc['Strategy'] = strat
                # On veut savoir combien de répliques TOTALES il y a à chaque instant
                # C'est un peu complexe à reconstruire parfaitement, mais on peut plotter les "Actions"
                # ou mieux : si ton CSV contient 'replicas_count' global. 
                # Sinon, on compte les actions de scale up/down cumulées.
                # Ici on va supposer qu'on peut tracer les actions brutes ou le count si dispo.
                data['scaling'].append(sc)
            except: pass

    # Concaténation
    dfs = {}
    for k, v in data.items():
        if v: dfs[k] = pd.concat(v, ignore_index=True) if isinstance(v[0], pd.DataFrame) else pd.DataFrame(v)
        else: dfs[k] = pd.DataFrame()
    return dfs

def plot_advanced():
    sns.set_theme(style="whitegrid", context="talk") #talk pour textes plus gros
    dfs = load_all_data()
    
    out_dir = current_dir / 'output' / 'advanced_plots'
    out_dir.mkdir(exist_ok=True, parents=True)

    # Ordre de tri
    strategies = [s for s in STRATEGY_COLORS.keys() if s in dfs['invocations']['Strategy'].unique()]
    print(strategies)
    # ================= 1. PARETO FRONTIER (Energie vs Latence) =================
    print("Génération du Pareto")
    if not dfs['invocations'].empty and not dfs['energy_stats'].empty:
        # Calcul Latence Moyenne
        perf_stats = dfs['invocations'].groupby('Strategy')['duration'].agg(['mean', lambda x: x.quantile(0.95)]).reset_index()
        perf_stats.columns = ['Strategy', 'Mean_Latency', 'P95_Latency']
        
        #Merge
        pareto_df = pd.merge(perf_stats, dfs['energy_stats'], on='Strategy')
        
        plt.figure(figsize=(10, 8))
        sns.scatterplot(data=pareto_df, x='Mean_Latency', y='Total_Joules', hue='Strategy', palette=STRATEGY_COLORS, s=200, edgecolor='black')
        
        for i, row in pareto_df.iterrows():
            plt.text(row['Mean_Latency'], row['Total_Joules'] + (row['Total_Joules']*0.02), 
                     row['Strategy'], 
                     #horizontalalignment='center',
                     fontsize=9, color='black')
        plt.title("Performance vs Energy (Pareto)")
        plt.xlabel("Mean Latency (s)")
        plt.ylabel("Total Energy (J)")
        #Inverser les axes pour que le "Coin optimal" soit en bas à gauche
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(out_dir / "plot_pareto.png", bbox_inches='tight')



    print("Génération du Pareto")
    if not dfs['invocations'].empty and not dfs['energy_stats'].empty:
        # Calcul Latence Moyenne
        perf_stats = dfs['invocations'].groupby('Strategy')['duration'].agg(['mean', lambda x: x.quantile(0.95)]).reset_index()
        perf_stats.columns = ['Strategy', 'Mean_Latency', 'P95_Latency']
        
        # Merge
        pareto_df = pd.merge(perf_stats, dfs['energy_stats'], on='Strategy')
        
        plt.figure(figsize=(10, 8))
        sns.scatterplot(data=pareto_df, x='Mean_Latency', y='Total_Joules', hue='Strategy', palette=STRATEGY_COLORS, s=200, edgecolor='black')
        
        for i, row in pareto_df.iterrows():
            plt.text(row['Mean_Latency'], row['Total_Joules'] + (row['Total_Joules']*0.02), 
                     row['Strategy'], 
                     fontsize=9, color='black')
        
        plt.title("Performance vs Energy (Pareto)")
        plt.xlabel("Mean Latency (s)")
        plt.ylabel("Total Energy (J)")
        
        max_lat = pareto_df['Mean_Latency'].max()
        max_joules = pareto_df['Total_Joules'].max()
        
        plt.xlim(0, max_lat * 1.15)
        plt.ylim(0, max_joules * 1.15)
        # ---------------------------------------------------------

        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(out_dir / "plot_pareto2.png", bbox_inches='tight')
    # ================= 2. CDF LATENCE (Lecture Fine) =================
    print("Génération CDF Latence")
    if not dfs['invocations'].empty:
        plt.figure(figsize=(12, 6))
        sns.ecdfplot(data=dfs['invocations'], x='duration', hue='Strategy', palette=STRATEGY_COLORS, linewidth=3)
        plt.title("Cumulative Latency Distribution")
        plt.xlabel("Response Time (s)")
        plt.ylabel("Density of Requests")
        plt.xlim(0, 50) #On zoome sur les 50 premières secondes pour ignorer les outliers extrêmes
        plt.grid(True, which='both', linestyle='--', alpha=0.5)
        plt.savefig(out_dir / "plot_cdf_latency.png", bbox_inches='tight')

    # ================= 3. SCALING DYNAMICS =================
    print("Génération Scaling")
    if not dfs['scaling'].empty:
        if 'new_replica_count' in dfs['scaling'].columns:
            plt.figure(figsize=(14, 6))
            
            target_strats = ['APE_H_Hybrid', 'HPST_Performance', 'LPLT_Energy','RL_PPO']
            filtered_df = dfs['scaling'][dfs['scaling']['Strategy'].isin(target_strats)].copy()
            
            if not filtered_df.empty:
                # On capture l'objet ax pour manipuler la légende après
                ax = sns.scatterplot(
                    data=filtered_df, 
                    x='timestamp', 
                    y='new_replica_count', 
                    hue='Strategy', 
                    style='action', 
                    palette=STRATEGY_COLORS, 
                    alpha=0.7
                )
                
                plt.title("Scaling Decisions over time")
                plt.ylabel("Number of Replicas")
                plt.xlabel("Time (s)")

                #LÉGENDE
                # On récupère les handles (symboles) et labels (textes) actuels (j'ai pas trop compris mais ça marche)
                h, l = ax.get_legend_handles_labels()
                #On enlève tout depuis action
                if 'action' in l:
                    idx = l.index('action')
                    #On coupe la liste pour ne garder que ce qu'il y a AVANT 'action'
                    ax.legend(h[:idx], l[:idx], title="Strategy", loc='upper left')

                plt.savefig(out_dir / "plot_scaling_dots.png", bbox_inches='tight')


    # ================= 3. SCALING DYNAMICS (CUMULÉ) =================
    print("Génération Scaling Cumulé")
    if not dfs['scaling'].empty:
        if 'new_replica_count' in dfs['scaling'].columns:
            plt.figure(figsize=(14, 6))
            
            # Filtre stratégies
            target_strats = ['APE_H_Hybrid', 'HPST_Performance', 'LPLT_Energy', 'RL_PPO']
            filtered_df = dfs['scaling'][dfs['scaling']['Strategy'].isin(target_strats)].copy()
            cumulative_df = filtered_df.groupby(['timestamp', 'Strategy'])['new_replica_count'].sum().reset_index()
                
            #On trace des lignes (lineplot) au lieu de points pour bien voir l'évolution
            sns.lineplot(data=cumulative_df, x='timestamp', y='new_replica_count', hue='Strategy', palette=STRATEGY_COLORS, linewidth=2.5, alpha=0.8)
            # Ajout de points
            sns.scatterplot(data=cumulative_df,x='timestamp',y='new_replica_count',hue='Strategy',palette=STRATEGY_COLORS,legend=False,s=40)
                
            plt.title("Total Active Replicas over iime")
            plt.ylabel("Total Replicas")
            plt.xlabel("Time (s)")
                
            # Légende propre
            plt.legend(title="Strategy", loc='upper left')
            plt.savefig(out_dir / "plot_scaling_cumulative.png", bbox_inches='tight')

    # ================= 4. POWER PROFILE (Watts over Time) =================
    print("Génération Profil de Puissance")
    plt.figure(figsize=(14, 6))
    # Lineplot avec moyenne lissée si beaucoup de données
    sns.lineplot(data=dfs['power'], x='timestamp', y='power_watts', hue='Strategy', palette=STRATEGY_COLORS, estimator=None, lw=1.5, alpha=0.8)
        
    plt.title("Instant Electrical Consumption")
    plt.ylabel("Power (Watts)")
    plt.xlabel("Time (s)")
    plt.savefig(out_dir / "plot_power_profile.png", bbox_inches='tight')

if __name__ == "__main__":
    plot_advanced()


    