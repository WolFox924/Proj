import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sqlite3
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict
import json
from datetime import datetime, timedelta

class AutoscalerVisualizer:
    """
    Système de visualisation pour l'autoscaler learning-based
    
    Génère différents types de graphiques :
    - Comparaison des performances par type de nœud
    - Evolution temporelle des métriques
    - Matrices de corrélation
    - Diagrammes de distribution
    - Analyse énergétique
    """
    
    def __init__(self, db_path: str = "autoscaler_history.db"):
        self.db_path = Path(db_path)
        
        # Configuration style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
        
        # Palette de couleurs par type de nœud
        self.node_colors = {
            'xeongpu': '#FF6B6B',
            'xeoncpu': '#FF8E53',
            'nuc': '#FFA07A',
            'nx': '#4ECDC4',
            'tx2': '#45B7D1',
            'rockpi': '#96CEB4',
            'nano': '#FFEAA7',
            'coral': '#DFE6E9',
            'rpi4': '#74B9FF',
            'rpi3': '#A29BFE'
        }
    
    def load_data(self, deployment_name: Optional[str] = None, 
                  days: int = 7) -> pd.DataFrame:
        """
        Charge les données depuis la base SQLite
        
        Args:
            deployment_name: Filtre sur un déploiement spécifique (None = tous)
            days: Nombre de jours d'historique à charger
        """
        try:
            conn = sqlite3.connect(str(self.db_path))
            
            query = """
                SELECT 
                    timestamp,
                    deployment_name,
                    node_type,
                    node_name,
                    execution_time_ms,
                    response_time_ms,
                    power_consumption_w,
                    cpu_utilization,
                    memory_utilization,
                    gpu_utilization,
                    success,
                    workload_type,
                    load_rps
                FROM execution_history
                WHERE timestamp > datetime('now', '-{} days')
            """.format(days)
            
            if deployment_name:
                query += f" AND deployment_name = '{deployment_name}'"
            
            query += " ORDER BY timestamp"
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            # Conversions
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['energy_j'] = df['power_consumption_w'] * df['execution_time_ms'] / 1000.0
            df['success'] = df['success'].astype(bool)
            
            print(f"✅ Loaded {len(df)} records from database")
            return df
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return pd.DataFrame()
    
    def plot_performance_comparison(self, df: pd.DataFrame, 
                                   deployment_name: Optional[str] = None,
                                   save_path: Optional[str] = None):
        """
        Graphique 1: Comparaison des performances par type de nœud
        
        Affiche: temps d'exécution moyen, énergie, taux de succès
        """
        if df.empty:
            print("⚠️ No data to plot")
            return
        
        # Filtrage
        if deployment_name:
            df = df[df['deployment_name'] == deployment_name]
        
        # Agrégation par type de nœud
        stats = df.groupby('node_type').agg({
            'execution_time_ms': ['mean', 'std'],
            'energy_j': ['mean', 'std'],
            'success': 'mean',
            'response_time_ms': ['mean', 'std']
        }).reset_index()
        
        # Aplatir les colonnes multi-index
        stats.columns = ['_'.join(col).strip('_') for col in stats.columns.values]
        stats = stats.rename(columns={'node_type_': 'node_type'})
        
        # Création de la figure avec sous-graphiques
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Performance Comparison by Node Type', fontsize=16, fontweight='bold')
        
        # 1. Temps d'exécution
        ax1 = axes[0, 0]
        colors = [self.node_colors.get(nt, '#95a5a6') for nt in stats['node_type']]
        bars1 = ax1.bar(stats['node_type'], stats['execution_time_ms_mean'], 
                       yerr=stats['execution_time_ms_std'], color=colors, alpha=0.8)
        ax1.set_xlabel('Node Type')
        ax1.set_ylabel('Execution Time (ms)')
        ax1.set_title('⏱️ Average Execution Time')
        ax1.tick_params(axis='x', rotation=45)
        
        # Annotations
        for i, (bar, val) in enumerate(zip(bars1, stats['execution_time_ms_mean'])):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                    f'{val:.0f}ms', ha='center', va='bottom', fontsize=8)
        
        # 2. Consommation énergétique
        ax2 = axes[0, 1]
        bars2 = ax2.bar(stats['node_type'], stats['energy_j_mean'], 
                       yerr=stats['energy_j_std'], color=colors, alpha=0.8)
        ax2.set_xlabel('Node Type')
        ax2.set_ylabel('Energy (J)')
        ax2.set_title('⚡ Average Energy Consumption')
        ax2.tick_params(axis='x', rotation=45)
        
        for i, (bar, val) in enumerate(zip(bars2, stats['energy_j_mean'])):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                    f'{val:.2f}J', ha='center', va='bottom', fontsize=8)
        
        # 3. Taux de succès
        ax3 = axes[1, 0]
        bars3 = ax3.bar(stats['node_type'], stats['success_mean'] * 100, 
                       color=colors, alpha=0.8)
        ax3.set_xlabel('Node Type')
        ax3.set_ylabel('Success Rate (%)')
        ax3.set_title('✅ Success Rate')
        ax3.set_ylim(0, 105)
        ax3.tick_params(axis='x', rotation=45)
        ax3.axhline(y=95, color='green', linestyle='--', alpha=0.5, label='Target (95%)')
        ax3.legend()
        
        for bar, val in zip(bars3, stats['success_mean']):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                    f'{val*100:.1f}%', ha='center', va='bottom', fontsize=8)
        
        # 4. Temps de réponse
        ax4 = axes[1, 1]
        bars4 = ax4.bar(stats['node_type'], stats['response_time_ms_mean'], 
                       yerr=stats['response_time_ms_std'], color=colors, alpha=0.8)
        ax4.set_xlabel('Node Type')
        ax4.set_ylabel('Response Time (ms)')
        ax4.set_title('📊 Average Response Time')
        ax4.tick_params(axis='x', rotation=45)
        
        for bar, val in zip(bars4, stats['response_time_ms_mean']):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                    f'{val:.0f}ms', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Saved to {save_path}")
        
        plt.show()
    
    def plot_energy_efficiency(self, df: pd.DataFrame, 
                               save_path: Optional[str] = None):
        """
        Graphique 2: Analyse de l'efficacité énergétique
        
        Scatter plot: Énergie vs Performance avec bulle = taux de succès
        """
        if df.empty:
            return
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Agrégation par type de nœud
        stats = df.groupby('node_type').agg({
            'execution_time_ms': 'mean',
            'energy_j': 'mean',
            'success': 'mean'
        }).reset_index()
        
        # Calcul de l'efficacité (moins = mieux)
        stats['efficiency_score'] = stats['energy_j'] / stats['execution_time_ms'] * 1000
        
        # Scatter plot avec tailles proportionnelles au succès
        for _, row in stats.iterrows():
            node_type = row['node_type']
            color = self.node_colors.get(node_type, '#95a5a6')
            size = row['success'] * 1000  # Taille proportionnelle au succès
            
            ax.scatter(row['execution_time_ms'], row['energy_j'], 
                      s=size, c=color, alpha=0.6, edgecolors='black', linewidth=1.5)
            
            # Annotation
            ax.annotate(node_type, 
                       (row['execution_time_ms'], row['energy_j']),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=10, fontweight='bold')
        
        ax.set_xlabel('Execution Time (ms)', fontsize=12)
        ax.set_ylabel('Energy Consumption (J)', fontsize=12)
        ax.set_title('⚡ Energy Efficiency Analysis\n(Bubble size = Success Rate)', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Zone optimale (bas-gauche = rapide et économe)
        ax.axvline(x=df['execution_time_ms'].median(), color='green', 
                  linestyle='--', alpha=0.3, label='Median Exec Time')
        ax.axhline(y=df['energy_j'].median(), color='blue', 
                  linestyle='--', alpha=0.3, label='Median Energy')
        ax.legend()
        
        # Quadrants
        ax.text(0.05, 0.95, '🎯 OPTIMAL\n(Fast & Efficient)', 
               transform=ax.transAxes, fontsize=11, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Saved to {save_path}")
        
        plt.show()
    
    def plot_temporal_evolution(self, df: pd.DataFrame, 
                               metric: str = 'execution_time_ms',
                               save_path: Optional[str] = None):
        """
        Graphique 3: Évolution temporelle des métriques
        
        Ligne par type de nœud montrant l'évolution dans le temps
        """
        if df.empty:
            return
        
        fig, ax = plt.subplots(figsize=(16, 8))
        
        # Grouper par jour et type de nœud
        df['date'] = df['timestamp'].dt.date
        daily_stats = df.groupby(['date', 'node_type'])[metric].mean().reset_index()
        
        # Tracer une ligne par type de nœud
        for node_type in daily_stats['node_type'].unique():
            data = daily_stats[daily_stats['node_type'] == node_type]
            color = self.node_colors.get(node_type, '#95a5a6')
            
            ax.plot(data['date'], data[metric], 
                   marker='o', label=node_type, color=color, linewidth=2, markersize=6)
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
        ax.set_title(f'📈 Temporal Evolution of {metric.replace("_", " ").title()}', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='best', ncol=2)
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Saved to {save_path}")
        
        plt.show()
    
    def plot_workload_distribution(self, df: pd.DataFrame, 
                                   save_path: Optional[str] = None):
        """
        Graphique 4: Distribution des workloads par type de nœud
        
        Heatmap montrant quelle charge chaque type gère
        """
        if df.empty:
            return
        
        # Créer une matrice de distribution
        distribution = df.groupby(['deployment_name', 'node_type']).size().unstack(fill_value=0)
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        sns.heatmap(distribution, annot=True, fmt='d', cmap='YlOrRd', 
                   cbar_kws={'label': 'Number of Executions'}, ax=ax)
        
        ax.set_xlabel('Node Type', fontsize=12)
        ax.set_ylabel('Deployment', fontsize=12)
        ax.set_title('🔥 Workload Distribution Heatmap', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Saved to {save_path}")
        
        plt.show()
    
    def plot_resource_utilization(self, df: pd.DataFrame, 
                                 save_path: Optional[str] = None):
        """
        Graphique 5: Utilisation des ressources (CPU, Memory, GPU)
        
        Box plots comparant l'utilisation par type de nœud
        """
        if df.empty:
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('💻 Resource Utilization by Node Type', fontsize=16, fontweight='bold')
        
        resources = [
            ('cpu_utilization', 'CPU Utilization', '🖥️'),
            ('memory_utilization', 'Memory Utilization', '💾'),
            ('gpu_utilization', 'GPU Utilization', '🎮')
        ]
        
        for ax, (metric, title, emoji) in zip(axes, resources):
            data_to_plot = []
            labels = []
            colors_list = []
            
            for node_type in sorted(df['node_type'].unique()):
                node_data = df[df['node_type'] == node_type][metric].dropna()
                if len(node_data) > 0:
                    data_to_plot.append(node_data)
                    labels.append(node_type)
                    colors_list.append(self.node_colors.get(node_type, '#95a5a6'))
            
            bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
            
            # Coloration des box plots
            for patch, color in zip(bp['boxes'], colors_list):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax.set_ylabel('Utilization (%)', fontsize=11)
            ax.set_title(f'{emoji} {title}', fontsize=12)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim(0, 1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Saved to {save_path}")
        
        plt.show()
    
    def plot_success_rate_analysis(self, df: pd.DataFrame, 
                                   save_path: Optional[str] = None):
        """
        Graphique 6: Analyse détaillée du taux de succès
        
        Pie chart + bar chart combinés
        """
        if df.empty:
            return
        
        fig = plt.figure(figsize=(16, 8))
        gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.5])
        
        # 1. Pie chart global
        ax1 = fig.add_subplot(gs[0])
        success_counts = df['success'].value_counts()
        colors_pie = ['#2ecc71', '#e74c3c']
        explode = (0.1, 0)
        
        ax1.pie(success_counts.values, labels=['Success', 'Failure'], 
               autopct='%1.1f%%', colors=colors_pie, explode=explode,
               shadow=True, startangle=90)
        ax1.set_title('✅ Overall Success Rate', fontsize=14, fontweight='bold')
        
        # 2. Bar chart par type de nœud
        ax2 = fig.add_subplot(gs[1])
        success_by_node = df.groupby('node_type')['success'].agg(['sum', 'count'])
        success_by_node['rate'] = (success_by_node['sum'] / success_by_node['count']) * 100
        success_by_node = success_by_node.sort_values('rate', ascending=False)
        
        colors = [self.node_colors.get(nt, '#95a5a6') for nt in success_by_node.index]
        bars = ax2.barh(success_by_node.index, success_by_node['rate'], color=colors, alpha=0.8)
        
        ax2.set_xlabel('Success Rate (%)', fontsize=12)
        ax2.set_ylabel('Node Type', fontsize=12)
        ax2.set_title('📊 Success Rate by Node Type', fontsize=14, fontweight='bold')
        ax2.set_xlim(0, 105)
        ax2.axvline(x=95, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Target (95%)')
        ax2.legend()
        
        # Annotations avec nombre d'échantillons
        for i, (bar, (idx, row)) in enumerate(zip(bars, success_by_node.iterrows())):
            ax2.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                    f'{row["rate"]:.1f}% (n={int(row["count"])})', 
                    va='center', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Saved to {save_path}")
        
        plt.show()
    
    def generate_full_report(self, deployment_name: Optional[str] = None, 
                           days: int = 7, output_dir: str = "reports"):
        """
        Génère un rapport complet avec tous les graphiques
        
        Args:
            deployment_name: Filtre optionnel sur un déploiement
            days: Nombre de jours d'historique
            output_dir: Dossier de sortie pour les images
        """
        print("📊 Generating full visualization report...")
        
        # Créer le dossier de sortie
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Charger les données
        df = self.load_data(deployment_name, days)
        
        if df.empty:
            print("❌ No data available for visualization")
            return
        
        # Générer tous les graphiques
        print("\n1️⃣ Performance Comparison...")
        self.plot_performance_comparison(df, deployment_name, 
                                        str(output_path / "1_performance_comparison.png"))
        
        print("\n2️⃣ Energy Efficiency...")
        self.plot_energy_efficiency(df, 
                                   str(output_path / "2_energy_efficiency.png"))
        
        print("\n3️⃣ Temporal Evolution...")
        self.plot_temporal_evolution(df, 'execution_time_ms',
                                    str(output_path / "3_temporal_evolution.png"))
        
        print("\n4️⃣ Workload Distribution...")
        self.plot_workload_distribution(df, 
                                       str(output_path / "4_workload_distribution.png"))
        
        print("\n5️⃣ Resource Utilization...")
        self.plot_resource_utilization(df, 
                                      str(output_path / "5_resource_utilization.png"))
        
        print("\n6️⃣ Success Rate Analysis...")
        self.plot_success_rate_analysis(df, 
                                       str(output_path / "6_success_rate_analysis.png"))
        
        # Générer un rapport textuel
        self._generate_text_report(df, output_path)
        
        print(f"\n✅ Full report generated in {output_dir}/")
    
    def _generate_text_report(self, df: pd.DataFrame, output_path: Path):
        """Génère un rapport textuel avec statistiques clés"""
        
        report = []
        report.append("=" * 80)
        report.append("AUTOSCALER PERFORMANCE REPORT")
        report.append("=" * 80)
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Data period: {df['timestamp'].min()} to {df['timestamp'].max()}")
        report.append(f"Total executions: {len(df)}")
        report.append(f"Unique deployments: {df['deployment_name'].nunique()}")
        report.append(f"Unique nodes: {df['node_name'].nunique()}")
        
        report.append("\n" + "=" * 80)
        report.append("PERFORMANCE SUMMARY BY NODE TYPE")
        report.append("=" * 80)
        
        summary = df.groupby('node_type').agg({
            'execution_time_ms': ['mean', 'std', 'min', 'max'],
            'energy_j': ['mean', 'std'],
            'success': ['mean', 'count'],
            'response_time_ms': 'mean'
        }).round(2)
        
        report.append("\n" + summary.to_string())
        
        report.append("\n" + "=" * 80)
        report.append("TOP 5 BEST PERFORMING NODES (by execution time)")
        report.append("=" * 80)
        
        best = df.groupby('node_type')['execution_time_ms'].mean().sort_values().head(5)
        for i, (node, time) in enumerate(best.items(), 1):
            report.append(f"{i}. {node}: {time:.2f} ms")
        
        report.append("\n" + "=" * 80)
        report.append("TOP 5 MOST ENERGY EFFICIENT NODES")
        report.append("=" * 80)
        
        efficient = df.groupby('node_type')['energy_j'].mean().sort_values().head(5)
        for i, (node, energy) in enumerate(efficient.items(), 1):
            report.append(f"{i}. {node}: {energy:.2f} J")
        
        # Écrire le rapport
        report_path = output_path / "performance_report.txt"
        with open(report_path, 'w') as f:
            f.write('\n'.join(report))
        
        print(f"📄 Text report saved to {report_path}")


# ============================================================================
# EXEMPLE D'UTILISATION
# ============================================================================

def example_usage():
    """
    Exemple d'utilisation du système de visualisation
    """
    
    # Initialisation
    viz = AutoscalerVisualizer(db_path="autoscaler_history.db")
    
    # Option 1: Rapport complet
    viz.generate_full_report(
        deployment_name=None,  # Tous les déploiements
        days=7,                # 7 derniers jours
        output_dir="reports"   # Dossier de sortie
    )
    
    # Option 2: Graphiques individuels
    df = viz.load_data(days=7)
    
    if not df.empty:
        # Graphique de comparaison
        viz.plot_performance_comparison(df, save_path="performance.png")
        
        # Analyse énergétique
        viz.plot_energy_efficiency(df, save_path="energy.png")
        
        # Évolution temporelle
        viz.plot_temporal_evolution(df, metric='execution_time_ms')
        
        # Distribution des workloads
        viz.plot_workload_distribution(df)
        
        # Utilisation des ressources
        viz.plot_resource_utilization(df)
        
        # Taux de succès
        viz.plot_success_rate_analysis(df)


if __name__ == "__main__":
    # Exécuter l'exemple
    example_usage()
