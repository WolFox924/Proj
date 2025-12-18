import os
import sys
import importlib
import inspect
from pathlib import Path

# Ajouter la racine au path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Chemin vers vos stratégies
STRATEGIES_DIR = project_root / 'ext' / 'mhfd' / 'scaling' / 'strategies'

def list_classes():
    print(f"🔍 Recherche dans : {STRATEGIES_DIR}\n")
    
    if not STRATEGIES_DIR.exists():
        print(f"❌ LE DOSSIER N'EXISTE PAS : {STRATEGIES_DIR}")
        print("Vérifiez que vous avez bien cloné le projet et que le dossier 'ext' est là.")
        return

    # Vérification des __init__.py
    for parent in [STRATEGIES_DIR.parent.parent.parent, STRATEGIES_DIR.parent.parent, STRATEGIES_DIR.parent, STRATEGIES_DIR]:
        init_file = parent / "__init__.py"
        if not init_file.exists():
            print(f"⚠️  ATTENTION : Manque '__init__.py' dans {parent.name}")
            print(f"   -> Créez-le avec : touch {init_file}")

    # Lister les fichiers .py
    files = [f for f in os.listdir(STRATEGIES_DIR) if f.endswith('.py') and f != '__init__.py']
    
    if not files:
        print("❌ Aucun fichier Python trouvé dans le dossier strategies.")
        return

    print(f"✅ Fichiers trouvés : {len(files)}")
    print("-" * 60)

    for filename in files:
        module_name = filename[:-3] # Enlever .py
        full_import_path = f"ext.mhfd.scaling.strategies.{module_name}"
        
        try:
            # Import dynamique du module
            module = importlib.import_module(full_import_path)
            
            # Trouver les classes définies dans ce module
            classes = []
            for name, obj in inspect.getmembers(module, inspect.isclass):
                # On filtre pour ne garder que les classes définies DANS ce fichier
                # (et pas les imports comme 'pd', 'sys', etc.)
                if obj.__module__ == full_import_path:
                    classes.append(name)
            
            if classes:
                print(f"📄 Fichier : {filename}")
                for cls_name in classes:
                    print(f"   🚀 Classe trouvée : \033[1m{cls_name}\033[0m")
                    print(f"   📋 Import à copier :")
                    print(f"      \033[92mfrom {full_import_path} import {cls_name}\033[0m")
                print("-" * 60)
                
        except Exception as e:
            print(f"❌ Erreur lors de l'analyse de {filename} : {e}")

if __name__ == "__main__":
    list_classes()