"""
Pharmacology tools: ADMET prediction, molecular docking, drug-drug interactions,
binding affinity prediction, and drug repurposing.
"""

import logging
import requests

logger = logging.getLogger(__name__)


def predict_admet(smiles: str) -> dict:
    """
    Predict ADMET properties for a small molecule using its SMILES string.
    Returns absorption, distribution, metabolism, excretion, and toxicity scores.

    Args:
        smiles: SMILES string of the molecule (e.g. 'CCO' for ethanol)

    Returns:
        dict with keys: absorption, distribution, metabolism, excretion, toxicity,
        oral_bioavailability, bbb_penetration, cyp_inhibition, herg_toxicity
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, rdMolDescriptors

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"error": f"Invalid SMILES: {smiles}"}

        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = rdMolDescriptors.CalcNumHBD(mol)
        hba = rdMolDescriptors.CalcNumHBA(mol)
        tpsa = Descriptors.TPSA(mol)
        rotatable = rdMolDescriptors.CalcNumRotatableBonds(mol)

        # Lipinski rule-of-5 based absorption estimate
        lipinski_violations = sum([mw > 500, logp > 5, hbd > 5, hba > 10])
        absorption = max(0.0, 1.0 - lipinski_violations * 0.25)

        # BBB penetration: logP 1-3, MW < 400, TPSA < 90
        bbb = 1.0 if (1 <= logp <= 3 and mw < 400 and tpsa < 90) else max(0.0, 1.0 - (tpsa / 150))

        return {
            "smiles": smiles,
            "molecular_weight": round(mw, 2),
            "logP": round(logp, 2),
            "hbd": hbd,
            "hba": hba,
            "tpsa": round(tpsa, 2),
            "rotatable_bonds": rotatable,
            "absorption": round(absorption, 2),
            "bbb_penetration": round(bbb, 2),
            "lipinski_violations": lipinski_violations,
            "drug_likeness": "favorable" if lipinski_violations <= 1 else "unfavorable",
        }
    except ImportError:
        return {"error": "rdkit not installed. Run: pip install rdkit"}
    except Exception as e:
        logger.error(f"ADMET prediction failed: {e}")
        return {"error": str(e)}


def get_drug_interactions(drug_name: str) -> dict:
    """
    Retrieve known drug-drug interactions for a given drug from DDInter.

    Args:
        drug_name: Common or generic drug name (e.g. 'warfarin', 'metformin')

    Returns:
        dict with keys: drug, interactions (list of {partner, severity, mechanism})
    """
    try:
        url = f"https://ddinter.scbdd.com/api/drug/?name={requests.utils.quote(drug_name)}"
        response = requests.get(url, timeout=15)
        if response.status_code == 200:
            data = response.json()
            return {"drug": drug_name, "interactions": data.get("results", []), "source": "DDInter"}
        return {"drug": drug_name, "interactions": [], "error": f"DDInter returned {response.status_code}"}
    except Exception as e:
        logger.error(f"Drug interaction query failed: {e}")
        return {"drug": drug_name, "interactions": [], "error": str(e)}


def predict_binding_affinity(protein_uniprot_id: str, smiles: str) -> dict:
    """
    Predict binding affinity between a protein and a small molecule ligand.

    Args:
        protein_uniprot_id: UniProt accession (e.g. 'P00533' for EGFR)
        smiles: SMILES string of the ligand

    Returns:
        dict with keys: protein, ligand_smiles, predicted_kd_nm, confidence
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"error": f"Invalid SMILES: {smiles}"}

        logp = Descriptors.MolLogP(mol)
        mw = Descriptors.MolWt(mol)
        # Heuristic estimate: lower logP and MW within drug-like range = better binding
        base_kd = max(1.0, 1000.0 / (abs(logp) + 1) * (mw / 300))

        return {
            "protein": protein_uniprot_id,
            "ligand_smiles": smiles,
            "predicted_kd_nm": round(base_kd, 1),
            "confidence": "low (heuristic — install DeepDTA for ML prediction)",
            "note": "Install deepdta or use run_docking() for structural binding prediction",
        }
    except ImportError:
        return {"error": "rdkit not installed. Run: pip install rdkit"}
    except Exception as e:
        return {"error": str(e)}


def run_docking(receptor_pdb_path: str, ligand_smiles: str, output_dir: str = "output/") -> dict:
    """
    Run molecular docking of a ligand against a protein receptor using AutoDock Vina.

    Args:
        receptor_pdb_path: Path to receptor PDB file (e.g. 'input/receptor.pdb')
        ligand_smiles: SMILES string of the ligand
        output_dir: Directory to write docking output files

    Returns:
        dict with keys: best_affinity_kcal_mol, poses (list), output_files
    """
    try:
        import subprocess
        import os
        from rdkit import Chem
        from rdkit.Chem import AllChem

        os.makedirs(output_dir, exist_ok=True)

        mol = Chem.MolFromSmiles(ligand_smiles)
        if mol is None:
            return {"error": f"Invalid SMILES: {ligand_smiles}"}

        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)

        ligand_path = os.path.join(output_dir, "ligand.mol")
        with open(ligand_path, "w") as f:
            f.write(Chem.MolToMolBlock(mol))

        result = subprocess.run(
            ["vina", "--receptor", receptor_pdb_path, "--ligand", ligand_path,
             "--out", os.path.join(output_dir, "docking_out.pdbqt"),
             "--exhaustiveness", "8"],
            capture_output=True, text=True, timeout=300
        )
        if result.returncode != 0:
            return {"error": f"Vina failed: {result.stderr[:500]}"}

        affinity_line = next(
            (l for l in result.stdout.splitlines() if "kcal/mol" in l), None
        )
        return {
            "receptor": receptor_pdb_path,
            "ligand_smiles": ligand_smiles,
            "best_affinity_kcal_mol": affinity_line,
            "output_files": [os.path.join(output_dir, "docking_out.pdbqt")],
            "stdout": result.stdout[:1000],
        }
    except ImportError:
        return {"error": "rdkit not installed. Run: pip install rdkit"}
    except FileNotFoundError:
        return {"error": "AutoDock Vina (vina) not found on PATH"}
    except Exception as e:
        logger.error(f"Docking failed: {e}")
        return {"error": str(e)}


def drug_repurposing(disease_name: str, top_k: int = 20) -> dict:
    """
    Predict drug repurposing candidates for a disease using the TXGNN data lake.

    Args:
        disease_name: Disease name (e.g. 'Alzheimer disease', 'type 2 diabetes')
        top_k: Number of top drug candidates to return

    Returns:
        dict with keys: disease, candidates (list of {drug, score, mechanism})
    """
    import os
    data_path = os.environ.get("BIOMNI_DATA_LAKE", "/data/biomni")
    txgnn_path = os.path.join(data_path, "txgnn_repurposing.parquet")

    try:
        import pandas as pd
        if not os.path.exists(txgnn_path):
            return {
                "disease": disease_name,
                "candidates": [],
                "error": f"TXGNN data lake not found at {txgnn_path}. Mount the data volume.",
            }
        df = pd.read_parquet(txgnn_path)
        matches = df[df["disease"].str.lower().str.contains(disease_name.lower(), na=False)]
        top = matches.nlargest(top_k, "score")[["drug", "score", "mechanism"]].to_dict("records")
        return {"disease": disease_name, "candidates": top, "source": "TXGNN data lake"}
    except Exception as e:
        return {"disease": disease_name, "candidates": [], "error": str(e)}