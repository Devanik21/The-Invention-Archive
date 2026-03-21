import os
import time
import urllib.request
import ssl
import json
import shutil

ssl._create_default_https_context = ssl._create_unverified_context
GITHUB_USER = "Devanik21"

INVENTIONS = {
    "constructs/001_HRF": "Harmonic-Resonance-Forest",
    "constructs/002_NECF": "Non-Equilibrium-Cognitive-Field",
    "constructs/003_TSP": "The-Schrodinger-Paradox",
    "constructs/004_AS": "Aether-SPARC",
    "constructs/005_FRAE": "FRAE",
    "constructs/006_TCA": "The-Cytomorphic-Architecture",
    "constructs/007_DDG": "Dreamer-Dark-Genesis",
    "constructs/008_CS": "causa-sui",
    "constructs/009_RHO": "Recursive-Hebbian-Organism",
    "constructs/010_LDD": "Lucid-Dark-Dreamer",
    "constructs/011_BA": "BSHDER-Architecture",
    "constructs/012_GGEO": "GENEVO-GENetic-EVolutionary-Organoid",
    "constructs/013_HM": "HAG-MoE",
    "constructs/Misc/001_LIM": "Latent-Inference-Manifold",
    "constructs/Misc/002_LCM": "Latent-Consensus-Manifold",
    "constructs/Misc/003_XV": "xylia-vision",
    "constructs/Misc/004_LB": "Life-Beyond",
    "constructs/Misc/005_TM": "Thermodynamic-Mind",
    "constructs/Misc/006_DTM": "Dark-Thermodynamic-Mind",
    "constructs/Misc/007_AARGE": "AION-Algorithmic-Reversal-of-Genomic-Entropy",
    "constructs/Misc/008_DU": "Deep-Universe"
}

PENDING = []

def make_perfect_structure(folder, repo_name=""):
    snap_path = os.path.join(folder, "snapshot")
    notes_path = os.path.join(folder, "notes")
    versions_path = os.path.join(folder, "versions")
    
    os.makedirs(snap_path, exist_ok=True)
    os.makedirs(notes_path, exist_ok=True)
    os.makedirs(versions_path, exist_ok=True)
    
    # Generate perfectly formatted internal markdown files
    with open(os.path.join(notes_path, "evolution.md"), "w") as f: f.write("# Evolution\n")
    with open(os.path.join(notes_path, "thoughts.md"), "w") as f: f.write("# Thoughts\n")
    with open(os.path.join(versions_path, "v1.md"), "w") as f: f.write("# Version 1.0\n")
    
    # Generate standard metadata JSON
    meta = {"name": repo_name if repo_name else folder.split("/")[-1], "status": "active"}
    with open(os.path.join(folder, "metadata.json"), "w") as f: json.dump(meta, f, indent=4)
    
    return snap_path

# 1. Process known repos
for folder, repo in INVENTIONS.items():
    snap_path = make_perfect_structure(folder, repo)
    
    for branch in ['main', 'master']:
        try:
            readme_url = f"https://raw.githubusercontent.com/{GITHUB_USER}/{repo}/{branch}/README.md"
            zip_url = f"https://github.com/{GITHUB_USER}/{repo}/archive/refs/heads/{branch}.zip"
            
            # A. Download README directly into the snapshot folder
            snap_readme = os.path.join(snap_path, "README.md")
            urllib.request.urlretrieve(readme_url, snap_readme)
            
            # B. Copy that exact README into the main invention folder as requested
            main_readme = os.path.join(folder, "README.md")
            shutil.copy(snap_readme, main_readme)
            
            # C. Download ZIP strictly into the snapshot folder
            urllib.request.urlretrieve(zip_url, os.path.join(snap_path, f"{repo}.zip"))
            
            print(f"✅ Synced {repo} ({branch})")
            break
        except Exception:
            continue
    time.sleep(1)
    
# 2. Build empty pending structures to maintain standard architecture
for folder in PENDING:
    make_perfect_structure(folder)
    
print("All downloads and structural formatting complete.")
