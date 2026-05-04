import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

current_dir = Path.cwd() 
data_dir = current_dir / 'output' / 'scripts_data'
if not data_dir.exists():
    data_dir = Path(__file__).resolve().parent / 'output' / 'scripts_data'
# COULEURS & ORDRE
STRATEGY_COLORS = {'HPST_Performance': '#d62728', 'LPLT_Energy': '#2ca02c', 'APE_H_Hybrid': '#1f77b4', 'Standard_K8s': '#9467bd', 'Standard_First': '#8c564b', 'RL_PPO_30_50_20_DLnew' : '#db00db'}

STRATEGY_ORDER = ['Standard_First', 'Standard_K8s', 'LPLT_Energy', 'APE_H_Hybrid', 'HPST_Performance','RL_PPO_30_50_20_DLnew']

NODE_TYPE_ORDER = ['rpi3', 'rpi4', 'nano', 'coral', 'rockpi', 'tx2', 'nx', 'nuc', 'xeon']

def clean_node_name(node_name):
    #Transforme 'rockpi_142' -> 'rockpi'
    if pd.isna(node_name):
        return "Unknown"
    s_name = str(node_name)
    # On coupe au premier '_' s'il existe
    if '_' in s_name:
        return s_name.split('_')[0]
    return s_name

def load_data_from_folders():
    all_invocations = []
    all_power = []
    batch_folders = list(data_dir.glob("batch_*"))
    for folder in batch_folders:
        strat_name = folder.name.replace("batch_", "")
        if strat_name in STRATEGY_ORDER:
            print(f"🔹Stratégie : {strat_name}")
            
            # --- 1. Lecture Invocations (Latence & Timeline) ---
            inv_file = folder / "invocations_df.csv"
            if inv_file.exists():
                try:
                    df = pd.read_csv(inv_file)
                    # On ne garde que les succès (t_exec > 0 et node défini)
                    df = df.dropna(subset=['node', 't_exec'])
                    df = df[df['t_exec'] > 0].copy()
                    
                    # Nettoyage du nom du noeud (regroupement par type)
                    df['node_type'] = df['node'].apply(clean_node_name)
                    
                    # Colonnes standardisées pour la suite
                    df['Strategy'] = strat_name
                    df['duration'] = df['t_exec']
                    df['start_time'] = df['t_start']
                    # Timestamp de fin pour compatibilité
                    df['timestamp'] = df['t_start'] + df['t_exec'] 
                    
                    all_invocations.append(df)
                except Exception as e:
                    print(f"Erreur lecture invocations: {e}")

            # --- 2. Lecture Power (Energie) ---
            pwr_file = folder / "power_df.csv"
            if pwr_file.exists():
                try:
                    pdf = pd.read_csv(pwr_file)
                    # On suppose que chaque ligne est une mesure à 1 seconde d'intervalle
                    # Donc Energie (Joules) = Somme(Puissance Watts * 1s)
                    if 'power_watts' in pdf.columns:
                        total_joules = pdf['power_watts'].sum()
                        all_power.append({
                            'Strategy': strat_name,
                            'joules': total_joules
                        })
                except Exception as e:
                    print(f"Erreur lecture power: {e}")
    # Concaténation finale
    inv_final = pd.concat(all_invocations, ignore_index=True) if all_invocations else pd.DataFrame()
    pwr_final = pd.DataFrame(all_power) if all_power else pd.DataFrame()
    
    return inv_final, pwr_final

def get_sorted_nodes(found_nodes):
    """Trie les types de noeuds selon l'ordre défini + ceux trouvés en plus"""
    ordered = [n for n in NODE_TYPE_ORDER if n in found_nodes]
    others = sorted([n for n in found_nodes if n not in NODE_TYPE_ORDER])
    return ordered + others

def plot():
    sns.set_theme(style="whitegrid")
    output_img_dir = current_dir / 'output'
    output_img_dir.mkdir(exist_ok=True)
    # Chargement
    inv_df, pwr_df = load_data_from_folders()
    # Filtrer uniquement les stratégies présentes
    active_strategies = [s for s in STRATEGY_ORDER if s in inv_df['Strategy'].unique() or (not pwr_df.empty and s in pwr_df['Strategy'].unique())]

    # ================= GRAPHIQUE 1 : LATENCE (Boxplot) =================
    if not inv_df.empty:
        print("\nGénération Boxplot Latence...")
        plt.figure(figsize=(12, 6))
        plt.yscale('log')
        sns.boxplot(data=inv_df, x='Strategy',y='duration',hue='Strategy',palette=STRATEGY_COLORS, order=active_strategies, showfliers=False)
        sns.stripplot(data=inv_df,x='Strategy',y='duration',color='black',alpha=0.3,jitter=True,order=active_strategies,size=2)
        plt.title("Latency per Strategy (Execution Time)")
        plt.ylabel("Seconds")
        plt.xlabel("")
        plt.tight_layout()
        plt.savefig(output_img_dir / "plot_latency.png", dpi=150)

    # ================= GRAPHIQUE 2 : ÉNERGIE (Barplot) =================
    moy = pwr_df['joules'].mean()
    #pwr_df['joules_centered'] = pwr_df['joules']- moy
    if not pwr_df.empty:
        print("Génération Barplot Énergie")
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(data=pwr_df,x='Strategy',y='joules',hue='Strategy',palette=STRATEGY_COLORS,order=active_strategies)
        ax.set_ylim(80000, 100000)
        # Annotation des valeurs
        for p in ax.patches:
            if p.get_height() > 0:
                ax.annotate(f'{int(p.get_height())} J', 
                           (p.get_x() + p.get_width() / 2., p.get_height()), 
                           ha='center', va='bottom', fontweight='bold')
        
        plt.title("Total Energy Consumption")
        plt.ylabel("Joules (cumulated Watts)")
        plt.xlabel("")
        plt.tight_layout()
        plt.savefig(output_img_dir / "plot_energy.png", dpi=150)

    # ================= GRAPHIQUE 3 : TIMELINE GROUPÉE (Gantt) =================
    if not inv_df.empty:
        print("Génération Timeline (Groupée par type de noeud)")
        
        # Liste des types de noeuds présents pour l'axe Y
        node_types = get_sorted_nodes(inv_df['node_type'].unique())
        
        num_strats = len(active_strategies)
        fig, axes = plt.subplots(num_strats, 1, figsize=(15, 3 * num_strats), sharex=True, sharey=True)
        if num_strats == 1: axes = [axes]

        for ax, strat in zip(axes, active_strategies):
            subset = inv_df[inv_df['Strategy'] == strat]
            color = STRATEGY_COLORS.get(strat, 'gray')
            
            if subset.empty:
                ax.text(0.5, 0.5, "Pas de données", transform=ax.transAxes, ha='center')
                continue

            # On dessine les barres
            ax.barh(y=subset['node_type'],width=subset['duration'],left=subset['start_time'],height=0.6,color=color,alpha=0.6,  # Transparence pour voir les superpositions (densité)
                edgecolor='none')
            ax.set_title(f"Strategy : {strat}", fontweight='bold', loc='left', fontsize=11)
            ax.grid(True, axis='x', linestyle='--', alpha=0.5)
            # On force l'axe Y à afficher tous les types de noeuds connus
            ax.set_yticks(node_types)

        axes[-1].set_xlabel("Simulation time (seconds)")
        plt.tight_layout()
        plt.savefig(output_img_dir / "plot_timeline.png", dpi=150)

if __name__ == "__main__":
    plot()

    