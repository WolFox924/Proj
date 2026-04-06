import subprocess
import sys
import os

scripts = [f for f in os.listdir("scripts") if f.endswith('.py')]
print("""
                ,;;:.
       ##########.   $
       ###########.   .
       $$$$$$$####.    $                 ####                                          ###
      ,;;''  $#####.    .               ##
     .;'      $#####.    $            #####   ###.##   ###.##    ####          ####   ###     ###  ###
    ,Y$'       $#####.    .           ##     ##  ##   ##  ##.  ##.     ####  ##.      ##    #    ##   ##
   ,$'    ,Y$$$ $####Y.   :          ##     ##  ##   ##  ##       ##  ####      ##   ##    ##   ##   ##
  $ .     Y$$$b# $###Pa.    .       ##       #. # #.  #. ##. #####        ######   ##    ##   ##   ##
   \Y:. ,YY$$$#   $###\#bou$.    ###                                 
    \Y:,####$P      $#######P$   
       ######        ########b
         ``Y::.,       ,`
               ``Y$bob`  """)
print("Voici les scripts qui peuvent être lancés :\n---------------------------")
i=0
for script in scripts :
    print(f"{i} - {script}")
    i+=1
print("---------------------------")
good = False
while not good :
    rep = input("\nMettez le nombre correspondant au script à lancer : ")
    rep = rep.strip()
    if rep.isnumeric() :
        if 0 <= (int)(rep) and (int)(rep) < i :
            good = True 
        else :
            print("Un nombre valable est requis.")
    else :
        print("Ecrivez un nombre.")
        
# Lancer un script spécifique
print("Lancement du script de Comparaison des Stratégies...")
subprocess.run([sys.executable, "scripts/"+scripts[(int)(rep)]])


scripts = [f for f in os.listdir("../Thao") if f.endswith('.py') and not f=="strat_to_plot.py" and not f=="list_strategies.py"]
print("Voici les scripts qui peuvent être lancés :\n---------------------------")
i=0
for script in scripts :
    print(f"{i} - {script}")
    i+=1
print("---------------------------")
good = False
while not good :
    rep = input("\nMettez le nombre correspondant au script des plot à lancer : ")
    rep = rep.strip()
    if rep.isnumeric() :
        if 0 <= (int)(rep) and (int)(rep) < i :
            good = True 
        else :
            print("Un nombre valable est requis.")
    else :
        print("Ecrivez un nombre.")

print("Lancement du script de création des plots...")
subprocess.run([sys.executable, ""+scripts[(int)(rep)]])

print("Tout est terminé !")