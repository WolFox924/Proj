import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ================= CONFIGURATION =================
# On garde votre gestion des chemins
script_path = Path(__file__).resolve()
base_dir = script_path.parent
output_dir = base_dir / 'output'

# 1. PALETTE DE COULEURS FIXE (Identité visuelle)
STRATEGY_COLORS = {
    'HPST_Performance': '#d62728', # Rouge (Puissance)
    'LPLT_Energy': '#2ca02c',      # Vert (Eco)
    'APE_H_Hybrid': '#1f77b4',     # Bleu (Hybride)
    'Standard_K8s': '#9467bd',     # Violet
    'Standard_First': '#8c564b',    # Marron
    'RL_PPO' : '#db00db'
}

# 2. ORDRE D'AFFICHAGE (Pour comparer logiquement)
STRATEGY_ORDER = ['Standard_First', 'Standard_K8s', 'LPLT_Energy', 'APE_H_Hybrid', 'HPST_Performance' ,'RL_PPO']

# 3. ORDRE DES NŒUDS (Du moins puissant au plus puissant pour le Gantt)
NODE_ORDER = [
    'rpi3-1', 'rpi3-2', 'rpi3-3',
    'rpi4-1', 'rpi4-2',
    'coral-1', 'coral-2',
    'nano-1', 'nano-2',
    'rockpi-1',
    'tx2-1', 'tx2-2',
    'nx-1', 'nx-2',
    'nuc-1',
    'xeoncpu-1', 'xeongpu-1'
]

def load_data():
    data = []
    # On utilise votre boucle glob existante
    for f in output_dir.glob("results_*.csv"):
        try:
            df = pd.read_csv(f)
            # Nettoyage du nom
            strat_name = f.stem.replace("results_", "")
            df['Strategy'] = strat_name
            data.append(df)
        except Exception as e:
            print(f"⚠️ Erreur lecture {f.name}: {e}")
    
    if not data: return pd.DataFrame()
    return pd.concat(data, ignore_index=True)

def plot():
    full_df = load_data()
    if full_df.empty:
        print("❌ Pas de fichiers CSV trouvés dans 'output' !")
        return

    # Filtrage des stratégies présentes dans les données
    active_strategies = [s for s in STRATEGY_ORDER if s in full_df['Strategy'].unique()]
    active_nodes = [n for n in NODE_ORDER if n in full_df['node'].unique()]

    # ================= GRAPHIQUE 1 : LATENCE (Boxplot) =================
    plt.figure(figsize=(12, 6))
    resp_df = full_df[full_df['metric'] == 'response_time']
    
    # Boxplot avec ordre et couleurs forcés
    sns.boxplot(
        data=resp_df, 
        x='Strategy', 
        y='duration', 
        palette=STRATEGY_COLORS, 
        order=active_strategies
    )
    # Ajout des points noirs pour voir la densité (utile pour voir si on sature)
    sns.stripplot(
        data=resp_df, x='Strategy', y='duration', 
        color='black', alpha=0.3, jitter=True, order=active_strategies
    )
    
    plt.title("Comparaison de la Latence (Performance)")
    plt.ylabel("Temps de réponse (secondes)")
    plt.grid(True, axis='y', alpha=0.3)
    plt.savefig(output_dir / "plot_latency.png")
    print(f"✅ Graphique généré : {output_dir}/plot_latency.png")

    # ================= GRAPHIQUE 2 : ENERGIE (Barplot) =================
    plt.figure(figsize=(10, 6))
    nrg_df = full_df[full_df['metric'] == 'energy_consumption']
    total_nrg = nrg_df.groupby('Strategy')['joules'].sum().reset_index()
    
    ax = sns.barplot(
        data=total_nrg, 
        x='Strategy', 
        y='joules', 
        palette=STRATEGY_COLORS, 
        order=active_strategies
    )
    
    # Affichage des valeurs sur les barres
    for p in ax.patches:
        if p.get_height() > 0:
            ax.annotate(f'{int(p.get_height())} J', 
                       (p.get_x() + p.get_width() / 2., p.get_height()), 
                       ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.title("Consommation Énergétique Totale")
    plt.ylabel("Joules")
    plt.savefig(output_dir / "plot_energy.png")
    print(f"✅ Graphique généré : {output_dir}/plot_energy.png")

    # ================= GRAPHIQUE 3 : TIMELINE (Qui fait quoi ?) =================
    # C'est CELUI-LA qui va vous montrer les nœuds utilisés !
    
    timeline_df = full_df[full_df['metric'] == 'response_time'].copy()
    # On recrée le temps de début car on a que la fin et la durée
    timeline_df['start_time'] = timeline_df['timestamp'] - timeline_df['duration']
    
    # On crée un sous-graphique par stratégie
    fig, axes = plt.subplots(len(active_strategies), 1, figsize=(14, 3 * len(active_strategies)), sharex=True)
    if len(active_strategies) == 1: axes = [axes] # Gérer le cas unique

    for ax, strat in zip(axes, active_strategies):
        strat_data = timeline_df[timeline_df['Strategy'] == strat]
        color = STRATEGY_COLORS.get(strat, 'gray')
        
        # Pour chaque requête, on dessine une barre horizontale au niveau du nœud
        ax.barh(
            y=strat_data['node'], 
            width=strat_data['duration'], 
            left=strat_data['start_time'], 
            height=0.6, 
            color=color, 
            alpha=0.8,
            edgecolor='black', linewidth=0.3
        )
        
        ax.set_title(f"Stratégie : {strat}", fontweight='bold', color=color, loc='left')
        ax.grid(True, axis='x', linestyle='--', alpha=0.5)
        
        # Force l'ordre des nœuds sur l'axe Y (du RPi en haut au Xeon en bas, ou l'inverse)
        # On filtre NODE_ORDER pour ne garder que ceux qui existent dans la simulation globale
        ax.set_yticks([n for n in NODE_ORDER if n in full_df['node'].unique()])
        
    axes[-1].set_xlabel("Temps de simulation (secondes)")
    plt.tight_layout()
    plt.savefig(output_dir / "plot_timeline.png")
    print(f"✅ Graphique généré : {output_dir}/plot_timeline.png")

if __name__ == "__main__":
    plot()