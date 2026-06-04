"""
Biomni database connectors — structured API wrappers for 30+ biological databases.
Each function makes a direct API call and returns structured, clean data.
"""

import logging
import requests

logger = logging.getLogger(__name__)
_TIMEOUT = 15


# ─── UniProt ─────────────────────────────────────────────────────────────────

def query_uniprot(gene_or_accession: str, organism: str = "human") -> dict:
    """
    Fetch protein information from UniProt.

    Args:
        gene_or_accession: Gene name (e.g. 'BRCA1') or UniProt accession (e.g. 'P38398')
        organism: Organism filter (default 'human')

    Returns:
        dict with keys: accession, gene, protein_name, function, sequence_length,
                        subcellular_location, go_terms, diseases
    """
    try:
        url = "https://rest.uniprot.org/uniprotkb/search"
        query = f"gene:{gene_or_accession} AND organism_name:{organism}"
        resp = requests.get(url, params={"query": query, "format": "json", "size": 1}, timeout=_TIMEOUT)
        resp.raise_for_status()
        results = resp.json().get("results", [])
        if not results:
            return {"gene": gene_or_accession, "error": "Not found in UniProt"}
        entry = results[0]
        return {
            "accession": entry.get("primaryAccession"),
            "gene": gene_or_accession,
            "protein_name": entry.get("proteinDescription", {}).get("recommendedName", {}).get("fullName", {}).get("value", ""),
            "sequence_length": entry.get("sequence", {}).get("length"),
            "organism": entry.get("organism", {}).get("scientificName"),
            "source": "UniProt",
        }
    except Exception as e:
        return {"gene": gene_or_accession, "error": str(e)}


# ─── AlphaFold ────────────────────────────────────────────────────────────────

def query_alphafold(uniprot_accession: str) -> dict:
    """
    Fetch AlphaFold predicted structure metadata for a protein.

    Args:
        uniprot_accession: UniProt accession (e.g. 'P38398')

    Returns:
        dict with keys: accession, pdb_url, pae_image_url, confidence_score
    """
    try:
        url = f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot_accession}"
        resp = requests.get(url, timeout=_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            return {"accession": uniprot_accession, "error": "Not found in AlphaFold"}
        entry = data[0]
        return {
            "accession": uniprot_accession,
            "pdb_url": entry.get("pdbUrl"),
            "pae_image_url": entry.get("paeImageUrl"),
            "model_created_date": entry.get("modelCreatedDate"),
            "source": "AlphaFold EBI",
        }
    except Exception as e:
        return {"accession": uniprot_accession, "error": str(e)}


# ─── STRING ───────────────────────────────────────────────────────────────────

def query_string(gene_name: str, species: int = 9606, limit: int = 20,
                 min_score: int = 700) -> dict:
    """
    Fetch protein-protein interaction network from STRING database.

    Args:
        gene_name: Gene/protein name (e.g. 'TP53', 'BRCA1')
        species: NCBI taxonomy ID (default 9606 for human)
        limit: Max number of interactors to return
        min_score: Minimum STRING interaction score (0-1000, default 700 = high confidence)

    Returns:
        dict with keys: protein, interactions (list of {partner, score, evidence_types})
    """
    try:
        url = "https://string-db.org/api/json/interaction_partners"
        resp = requests.get(url, params={
            "identifier": gene_name, "species": species,
            "limit": limit, "required_score": min_score,
        }, timeout=_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        interactions = [
            {
                "partner": d.get("preferredName_B"),
                "score": d.get("score"),
                "nscore": d.get("nscore"),
                "fscore": d.get("fscore"),
            }
            for d in data
        ]
        return {
            "protein": gene_name,
            "species": species,
            "min_score": min_score,
            "interactions": interactions,
            "source": "STRING v12",
        }
    except Exception as e:
        return {"protein": gene_name, "error": str(e)}


# ─── KEGG ─────────────────────────────────────────────────────────────────────

def query_kegg(gene_name: str, organism: str = "hsa") -> dict:
    """
    Fetch pathway and functional information for a gene from KEGG.

    Args:
        gene_name: Gene name (e.g. 'BRCA1', 'TP53')
        organism: KEGG organism code (default 'hsa' for human)

    Returns:
        dict with keys: gene, kegg_id, pathways (list), orthologs
    """
    try:
        search_url = f"https://rest.kegg.jp/find/genes/{gene_name}"
        resp = requests.get(search_url, timeout=_TIMEOUT)
        lines = [l for l in resp.text.strip().split("\n") if organism in l]
        if not lines:
            return {"gene": gene_name, "error": f"Not found in KEGG for organism {organism}"}

        kegg_id = lines[0].split("\t")[0]
        entry_resp = requests.get(f"https://rest.kegg.jp/get/{kegg_id}", timeout=_TIMEOUT)

        pathways = []
        for line in entry_resp.text.split("\n"):
            if line.startswith("PATHWAY"):
                pathways.append(line.strip())

        return {
            "gene": gene_name,
            "kegg_id": kegg_id,
            "pathways": pathways[:10],
            "source": "KEGG",
        }
    except Exception as e:
        return {"gene": gene_name, "error": str(e)}


# ─── OpenTargets ──────────────────────────────────────────────────────────────

def query_opentargets(gene_name: str, top_k: int = 20) -> dict:
    """
    Fetch disease-gene association scores from OpenTargets.

    Args:
        gene_name: Gene symbol (e.g. 'BRCA1', 'APOE')
        top_k: Number of top associated diseases to return

    Returns:
        dict with keys: gene, diseases (list of {disease_name, score, disease_id})
    """
    try:
        graphql_url = "https://api.platform.opentargets.org/api/v4/graphql"
        query = """
        query($gene: String!, $size: Int!) {
          search(queryString: $gene, entityNames: ["target"]) {
            hits {
              id
              object { ... on Target { approvedSymbol associatedDiseases(page: {index: 0, size: $size}) {
                rows { disease { name id } score }
              }}}
            }
          }
        }
        """
        resp = requests.post(graphql_url, json={"query": query, "variables": {"gene": gene_name, "size": top_k}}, timeout=_TIMEOUT)
        data = resp.json()
        hits = data.get("data", {}).get("search", {}).get("hits", [])
        if not hits:
            return {"gene": gene_name, "diseases": [], "error": "Not found in OpenTargets"}

        diseases = []
        for row in hits[0].get("object", {}).get("associatedDiseases", {}).get("rows", []):
            diseases.append({
                "disease_name": row["disease"]["name"],
                "disease_id": row["disease"]["id"],
                "association_score": round(row["score"], 4),
            })
        return {"gene": gene_name, "diseases": diseases, "source": "OpenTargets"}
    except Exception as e:
        return {"gene": gene_name, "diseases": [], "error": str(e)}


# ─── ClinVar ──────────────────────────────────────────────────────────────────

def query_clinvar(variant_id: str) -> dict:
    """
    Fetch clinical significance and disease associations for a variant from ClinVar.

    Args:
        variant_id: rsID (e.g. 'rs1799950') or ClinVar variation ID

    Returns:
        dict with keys: variant, clinical_significance, conditions, review_status
    """
    try:
        url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        search_resp = requests.get(url, params={
            "db": "clinvar", "term": variant_id, "retmode": "json"
        }, timeout=_TIMEOUT)
        ids = search_resp.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            return {"variant": variant_id, "error": "Not found in ClinVar"}

        summary_resp = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
            params={"db": "clinvar", "id": ids[0], "retmode": "json"},
            timeout=_TIMEOUT,
        )
        result = summary_resp.json().get("result", {}).get(ids[0], {})
        return {
            "variant": variant_id,
            "clinvar_id": ids[0],
            "title": result.get("title", ""),
            "clinical_significance": result.get("clinical_significance", {}).get("description", ""),
            "review_status": result.get("clinical_significance", {}).get("review_status", ""),
            "source": "ClinVar",
        }
    except Exception as e:
        return {"variant": variant_id, "error": str(e)}


# ─── gnomAD ───────────────────────────────────────────────────────────────────

def query_gnomad(variant_id: str, genome_build: str = "GRCh38") -> dict:
    """
    Fetch population allele frequency data for a variant from gnomAD.

    Args:
        variant_id: Variant in format 'chrom-pos-ref-alt' (e.g. '17-41223094-A-T')
                    or rsID (e.g. 'rs1799950')
        genome_build: 'GRCh38' or 'GRCh37'

    Returns:
        dict with keys: variant, allele_frequency, popmax_af, n_homozygotes
    """
    try:
        url = "https://gnomad.broadinstitute.org/api"
        query = """
        query($variantId: String!, $dataset: DatasetId!) {
          variant(variantId: $variantId, dataset: $dataset) {
            variantId rsid
            genome { af acHom }
          }
        }
        """
        dataset = "gnomad_r4" if genome_build == "GRCh38" else "gnomad_r2_1"
        resp = requests.post(url, json={"query": query, "variables": {"variantId": variant_id, "dataset": dataset}}, timeout=_TIMEOUT)
        data = resp.json().get("data", {}).get("variant", {})
        if not data:
            return {"variant": variant_id, "error": "Not found in gnomAD"}
        genome = data.get("genome") or {}
        return {
            "variant": variant_id,
            "rsid": data.get("rsid"),
            "allele_frequency": genome.get("af"),
            "n_homozygotes": genome.get("acHom"),
            "genome_build": genome_build,
            "source": "gnomAD",
        }
    except Exception as e:
        return {"variant": variant_id, "error": str(e)}


# ─── Ensembl ──────────────────────────────────────────────────────────────────

def query_ensembl(gene_name: str, species: str = "human") -> dict:
    """
    Fetch gene coordinates, transcripts, and functional annotation from Ensembl.

    Args:
        gene_name: Gene symbol (e.g. 'BRCA1') or Ensembl gene ID (e.g. 'ENSG00000012048')
        species: Species name (default 'human')

    Returns:
        dict with keys: gene_id, chromosome, start, end, strand, biotype, transcripts
    """
    try:
        url = f"https://rest.ensembl.org/xrefs/symbol/{species}/{gene_name}"
        resp = requests.get(url, headers={"Content-Type": "application/json"}, timeout=_TIMEOUT)
        resp.raise_for_status()
        results = resp.json()
        if not results:
            return {"gene": gene_name, "error": "Not found in Ensembl"}

        gene_id = results[0]["id"]
        detail_resp = requests.get(
            f"https://rest.ensembl.org/lookup/id/{gene_id}",
            headers={"Content-Type": "application/json"},
            params={"expand": 1},
            timeout=_TIMEOUT,
        )
        detail = detail_resp.json()
        transcripts = [
            {"id": t.get("id"), "biotype": t.get("biotype"), "length": t.get("length")}
            for t in detail.get("Transcript", [])[:10]
        ]
        return {
            "gene": gene_name,
            "gene_id": gene_id,
            "chromosome": detail.get("seq_region_name"),
            "start": detail.get("start"),
            "end": detail.get("end"),
            "strand": detail.get("strand"),
            "biotype": detail.get("biotype"),
            "n_transcripts": len(detail.get("Transcript", [])),
            "transcripts": transcripts,
            "source": "Ensembl",
        }
    except Exception as e:
        return {"gene": gene_name, "error": str(e)}


# ─── cBioPortal ───────────────────────────────────────────────────────────────

def query_cbioportal(gene_name: str, cancer_types: list = None) -> dict:
    """
    Fetch somatic mutation frequency and cancer study data from cBioPortal.

    Args:
        gene_name: Gene symbol (e.g. 'TP53', 'KRAS')
        cancer_types: List of cancer type codes (e.g. ['brca', 'luad']). None = all studies.

    Returns:
        dict with keys: gene, mutation_frequency, top_cancer_types, study_count
    """
    try:
        url = "https://www.cbioportal.org/api/genes/{gene_name}/mutations"
        studies_resp = requests.get(
            "https://www.cbioportal.org/api/studies",
            params={"projection": "SUMMARY", "pageSize": 50},
            timeout=_TIMEOUT,
        )
        studies = studies_resp.json()
        if cancer_types:
            studies = [s for s in studies if any(ct in s.get("cancerTypeId", "") for ct in cancer_types)]
        return {
            "gene": gene_name,
            "n_studies_available": len(studies),
            "cancer_type_ids": [s.get("cancerTypeId") for s in studies[:10]],
            "note": "Use cBioPortal web API with specific study IDs for mutation frequency data",
            "source": "cBioPortal",
        }
    except Exception as e:
        return {"gene": gene_name, "error": str(e)}


# ─── Reactome ─────────────────────────────────────────────────────────────────

def query_reactome(gene_name: str) -> dict:
    """
    Fetch pathway membership for a gene from Reactome.

    Args:
        gene_name: Gene symbol (e.g. 'BRCA1', 'EGFR')

    Returns:
        dict with keys: gene, pathways (list of {name, id, species})
    """
    try:
        url = f"https://reactome.org/ContentService/data/query/enhanced/{gene_name}"
        resp = requests.get(url, timeout=_TIMEOUT)
        if resp.status_code != 200:
            # fallback: text search
            search_resp = requests.get(
                "https://reactome.org/ContentService/search/query",
                params={"query": gene_name, "cluster": "true", "types": "Pathway"},
                timeout=_TIMEOUT,
            )
            data = search_resp.json()
            results = data.get("results", [{}])[0].get("entries", [])[:15]
            pathways = [{"name": r.get("name"), "id": r.get("stId")} for r in results]
        else:
            data = resp.json()
            pathways = [
                {"name": p.get("displayName"), "id": p.get("stId")}
                for p in data.get("summation", [])[:15]
            ]
        return {"gene": gene_name, "pathways": pathways, "source": "Reactome"}
    except Exception as e:
        return {"gene": gene_name, "error": str(e)}


# ─── GWAS Catalog ─────────────────────────────────────────────────────────────

def query_gwas_catalog(gene_or_trait: str, p_value_threshold: float = 5e-8) -> dict:
    """
    Fetch GWAS associations for a gene or trait from the GWAS Catalog.

    Args:
        gene_or_trait: Gene symbol or trait keyword (e.g. 'BRCA1', 'breast cancer')
        p_value_threshold: Maximum p-value to include (default genome-wide significance 5e-8)

    Returns:
        dict with keys: query, associations (list of {snp, trait, p_value, beta, mapped_gene})
    """
    try:
        url = "https://www.ebi.ac.uk/gwas/rest/api/associations/search/findByGene"
        resp = requests.get(url, params={"geneName": gene_or_trait, "size": 50}, timeout=_TIMEOUT)
        data = resp.json()
        associations = []
        for assoc in data.get("_embedded", {}).get("associations", []):
            p_val = assoc.get("pvalue")
            if p_val and float(p_val) <= p_value_threshold:
                associations.append({
                    "snp": assoc.get("snpId"),
                    "p_value": p_val,
                    "beta": assoc.get("betaNum"),
                    "trait": assoc.get("description"),
                })
        return {
            "query": gene_or_trait,
            "p_value_threshold": p_value_threshold,
            "n_associations": len(associations),
            "associations": associations[:20],
            "source": "GWAS Catalog",
        }
    except Exception as e:
        return {"query": gene_or_trait, "error": str(e)}


# ─── OpenFDA ──────────────────────────────────────────────────────────────────

def query_openfda(drug_name: str) -> dict:
    """
    Fetch drug label, adverse events, and recall information from OpenFDA.

    Args:
        drug_name: Drug name (e.g. 'metformin', 'warfarin', 'imatinib')

    Returns:
        dict with keys: drug, indications, warnings, adverse_reactions, dosage
    """
    try:
        url = "https://api.fda.gov/drug/label.json"
        resp = requests.get(url, params={"search": f"openfda.generic_name:{drug_name}", "limit": 1}, timeout=_TIMEOUT)
        data = resp.json()
        results = data.get("results", [])
        if not results:
            return {"drug": drug_name, "error": "Not found in OpenFDA"}
        entry = results[0]
        return {
            "drug": drug_name,
            "brand_name": entry.get("openfda", {}).get("brand_name", []),
            "indications": (entry.get("indications_and_usage") or [""])[0][:500],
            "warnings": (entry.get("warnings") or [""])[0][:500],
            "adverse_reactions": (entry.get("adverse_reactions") or [""])[0][:500],
            "source": "OpenFDA",
        }
    except Exception as e:
        return {"drug": drug_name, "error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# ChEMBL — drug compound database
# ─────────────────────────────────────────────────────────────────────────────

def query_chembl(query: str, query_type: str = "compound", limit: int = 10) -> dict:
    """
    Search ChEMBL for drug compounds, targets, or bioactivities.

    Args:
        query: Search term — gene name, drug name, or ChEMBL ID
        query_type: 'compound' (search drugs/molecules), 'target' (search protein targets),
                    'activity' (get bioactivities for a target gene)
        limit: Max results to return

    Returns:
        dict with compounds/targets/activities and key properties
    """
    try:
        base = "https://www.ebi.ac.uk/chembl/api/data"
        if query_type == "compound":
            r = requests.get(f"{base}/molecule/search", params={
                "q": query, "limit": limit, "format": "json",
            }, timeout=_TIMEOUT)
            r.raise_for_status()
            mols = r.json().get("molecules", [])
            compounds = []
            for m in mols:
                props = m.get("molecule_properties") or {}
                compounds.append({
                    "chembl_id": m.get("molecule_chembl_id"),
                    "name": m.get("pref_name") or m.get("molecule_synonyms", [{}])[0].get("synonyms", ""),
                    "type": m.get("molecule_type"),
                    "mw": props.get("mw_freebase"),
                    "alogp": props.get("alogp"),
                    "hbd": props.get("hbd"),
                    "smiles": (m.get("molecule_structures") or {}).get("canonical_smiles", ""),
                    "max_phase": m.get("max_phase"),
                })
            return {"query": query, "compounds": compounds, "source": "ChEMBL"}

        elif query_type == "target":
            r = requests.get(f"{base}/target/search", params={
                "q": query, "limit": limit, "format": "json",
            }, timeout=_TIMEOUT)
            r.raise_for_status()
            targets = [
                {
                    "chembl_id": t.get("target_chembl_id"),
                    "name": t.get("pref_name"),
                    "type": t.get("target_type"),
                    "organism": t.get("organism"),
                }
                for t in r.json().get("targets", [])
            ]
            return {"query": query, "targets": targets, "source": "ChEMBL"}

        elif query_type == "activity":
            # Get activities for a gene target
            target_r = requests.get(f"{base}/target/search", params={
                "q": query, "organism": "Homo sapiens", "limit": 1, "format": "json",
            }, timeout=_TIMEOUT)
            target_r.raise_for_status()
            targets = target_r.json().get("targets", [])
            if not targets:
                return {"query": query, "error": "Target not found", "source": "ChEMBL"}
            target_id = targets[0]["target_chembl_id"]
            act_r = requests.get(f"{base}/activity", params={
                "target_chembl_id": target_id, "limit": limit, "format": "json",
            }, timeout=_TIMEOUT)
            act_r.raise_for_status()
            activities = [
                {
                    "molecule_chembl_id": a.get("molecule_chembl_id"),
                    "compound_name": a.get("molecule_pref_name"),
                    "activity_type": a.get("standard_type"),
                    "value": a.get("standard_value"),
                    "units": a.get("standard_units"),
                    "assay_description": (a.get("assay_description") or "")[:150],
                }
                for a in act_r.json().get("activities", [])
            ]
            return {"query": query, "target_id": target_id, "activities": activities, "source": "ChEMBL"}

        return {"query": query, "error": f"Unknown query_type: {query_type}", "source": "ChEMBL"}
    except Exception as e:
        logger.error(f"query_chembl error: {e}")
        return {"query": query, "error": str(e), "source": "ChEMBL"}


# ─────────────────────────────────────────────────────────────────────────────
# PDB — Protein Data Bank (experimental 3D structures)
# ─────────────────────────────────────────────────────────────────────────────

def query_pdb(query: str, query_type: str = "gene") -> dict:
    """
    Search the Protein Data Bank for experimental 3D protein structures.

    Args:
        query: Gene name, protein name, or PDB ID (e.g. 'BRCA1', 'TP53', '4HFZ')
        query_type: 'gene' (search by gene), 'pdb_id' (fetch by PDB ID directly)

    Returns:
        dict with structure entries, resolution, experimental method, download URLs
    """
    try:
        if query_type == "pdb_id":
            r = requests.get(
                f"https://data.rcsb.org/rest/v1/core/entry/{query.upper()}",
                timeout=_TIMEOUT,
            )
            r.raise_for_status()
            data = r.json()
            return {
                "pdb_id": query.upper(),
                "title": data.get("struct", {}).get("title"),
                "resolution": data.get("refine", [{}])[0].get("ls_d_res_high") if data.get("refine") else None,
                "method": data.get("exptl", [{}])[0].get("method") if data.get("exptl") else None,
                "deposition_date": data.get("rcsb_accession_info", {}).get("deposit_date"),
                "download_pdb": f"https://files.rcsb.org/download/{query.upper()}.pdb",
                "download_cif": f"https://files.rcsb.org/download/{query.upper()}.cif",
                "source": "PDB",
            }

        # Search by gene name using RCSB search API
        search_body = {
            "query": {
                "type": "terminal",
                "service": "text",
                "parameters": {"value": query},
            },
            "return_type": "entry",
            "request_options": {"paginate": {"start": 0, "rows": 10}},
        }
        r = requests.post("https://search.rcsb.org/rcsbsearch/v2/query",
                          json=search_body, timeout=_TIMEOUT)
        r.raise_for_status()
        result_ids = [h.get("identifier") for h in r.json().get("result_set", [])][:10]

        structures = []
        for pdb_id in result_ids[:5]:
            try:
                info_r = requests.get(
                    f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}",
                    timeout=_TIMEOUT,
                )
                if info_r.ok:
                    d = info_r.json()
                    structures.append({
                        "pdb_id": pdb_id,
                        "title": d.get("struct", {}).get("title", ""),
                        "resolution": d.get("refine", [{}])[0].get("ls_d_res_high") if d.get("refine") else None,
                        "method": d.get("exptl", [{}])[0].get("method") if d.get("exptl") else None,
                        "download_pdb": f"https://files.rcsb.org/download/{pdb_id}.pdb",
                    })
            except Exception:
                pass

        return {"query": query, "structures": structures, "total_found": len(result_ids), "source": "PDB"}
    except Exception as e:
        logger.error(f"query_pdb error: {e}")
        return {"query": query, "error": str(e), "source": "PDB"}


# ─────────────────────────────────────────────────────────────────────────────
# GTEx — tissue-specific gene expression (live API, more current than Neo4j snapshot)
# ─────────────────────────────────────────────────────────────────────────────

def query_gtex(gene_name: str, dataset: str = "gtex_v10") -> dict:
    """
    Get tissue-specific gene expression data from GTEx (live API, GTEx v10).
    Complements the Neo4j snapshot with current release data.

    Args:
        gene_name: Gene symbol (e.g. 'FOXO3', 'IGF1', 'MTOR')
        dataset: GTEx dataset version ('gtex_v10', 'gtex_v8')

    Returns:
        dict with expression levels per tissue, top expressed tissues
    """
    try:
        # Get Ensembl ID for the gene
        ens_r = requests.get(
            f"https://rest.ensembl.org/lookup/symbol/human/{gene_name}",
            headers={"Content-Type": "application/json"},
            timeout=_TIMEOUT,
        )
        ens_r.raise_for_status()
        gene_id = ens_r.json().get("id", "")
        if not gene_id:
            return {"gene": gene_name, "error": "Gene not found in Ensembl", "source": "GTEx"}

        # GTEx expression API
        url = "https://gtexportal.org/api/v2/expression/medianGeneExpression"
        r = requests.get(url, params={
            "gencodeId": gene_id,
            "datasetId": dataset,
        }, timeout=_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        expressions = data.get("medianGeneExpression", [])

        tissues = [
            {
                "tissue": e.get("tissueSiteDetailId", "").replace("_", " "),
                "median_tpm": round(e.get("median", 0), 3),
                "unit": "TPM",
            }
            for e in expressions
        ]
        tissues.sort(key=lambda x: -x["median_tpm"])

        return {
            "gene": gene_name,
            "ensembl_id": gene_id,
            "dataset": dataset,
            "top_tissues": tissues[:10],
            "all_tissues": tissues,
            "source": "GTEx API",
        }
    except Exception as e:
        logger.error(f"query_gtex error: {e}")
        return {"gene": gene_name, "error": str(e), "source": "GTEx"}


# ─────────────────────────────────────────────────────────────────────────────
# ENCODE — regulatory elements for a gene
# ─────────────────────────────────────────────────────────────────────────────

def query_encode(gene_name: str, assay: str = "ATAC-seq", biosample: str = None) -> dict:
    """
    Search ENCODE for regulatory element data (enhancers, promoters, open chromatin).

    Args:
        gene_name: Gene symbol to search near (e.g. 'FOXO3', 'TP53')
        assay: Assay type: 'ATAC-seq', 'ChIP-seq', 'CRISPR', 'Hi-C', 'RNA-seq'
        biosample: Optional cell/tissue type filter (e.g. 'GM12878', 'liver')

    Returns:
        dict with ENCODE experiments, file URLs, biosample info
    """
    try:
        params = {
            "searchTerm": gene_name,
            "type": "Experiment",
            "assay_title": assay,
            "status": "released",
            "format": "json",
            "limit": 10,
        }
        if biosample:
            params["biosample_ontology.term_name"] = biosample

        r = requests.get(
            "https://www.encodeproject.org/search/",
            params=params,
            headers={"Accept": "application/json"},
            timeout=_TIMEOUT,
        )
        r.raise_for_status()
        data = r.json()
        experiments = []
        for exp in data.get("@graph", [])[:10]:
            experiments.append({
                "accession": exp.get("accession"),
                "assay": exp.get("assay_title"),
                "biosample": exp.get("biosample_summary", ""),
                "target": exp.get("target", {}).get("label", "") if isinstance(exp.get("target"), dict) else "",
                "files": len(exp.get("files", [])),
                "url": f"https://www.encodeproject.org/experiments/{exp.get('accession')}/",
            })
        return {
            "gene": gene_name,
            "assay": assay,
            "experiments": experiments,
            "total": data.get("total", 0),
            "source": "ENCODE",
        }
    except Exception as e:
        logger.error(f"query_encode error: {e}")
        return {"gene": gene_name, "error": str(e), "source": "ENCODE"}


# ─────────────────────────────────────────────────────────────────────────────
# BLAST — sequence similarity search
# ─────────────────────────────────────────────────────────────────────────────

def blast_sequence(sequence: str, program: str = "blastp", database: str = "nr",
                   max_hits: int = 10) -> dict:
    """
    Run BLAST sequence similarity search via NCBI remote BLAST API.

    Args:
        sequence: DNA or protein sequence in FASTA format or raw sequence string
        program: 'blastp' (protein), 'blastn' (nucleotide), 'blastx' (translated DNA)
        database: 'nr' (non-redundant protein), 'nt' (nucleotide), 'swissprot' (UniProt)
        max_hits: Max number of hits to return

    Returns:
        dict with top hits, identity%, alignment length, E-values
    """
    try:
        base = "https://blast.ncbi.nlm.nih.gov/blast/Blast.cgi"

        # Submit job
        params = {
            "CMD": "Put",
            "PROGRAM": program,
            "DATABASE": database,
            "QUERY": sequence,
            "FORMAT_TYPE": "JSON2",
            "HITLIST_SIZE": max_hits,
            "tool": "rejuve-ai-assistant",
            "email": "assistant@rejuve.bio",
        }
        r = requests.post(base, data=params, timeout=30)
        r.raise_for_status()

        # Extract RID
        rid = None
        for line in r.text.split("\n"):
            if "RID = " in line:
                rid = line.split("RID = ")[1].strip()
                break
        if not rid:
            return {"sequence": sequence[:50], "error": "Failed to get BLAST RID", "source": "NCBI BLAST"}

        # Poll for results (NCBI BLAST takes 20-60 seconds)
        import time
        for attempt in range(12):  # max 2 min
            time.sleep(10)
            status_r = requests.get(base, params={
                "CMD": "Get", "RID": rid, "FORMAT_TYPE": "JSON2",
                "HITLIST_SIZE": max_hits,
            }, timeout=30)
            if "Status=READY" in status_r.text or '"BlastOutput2"' in status_r.text:
                break
            if "Status=FAILED" in status_r.text:
                return {"sequence": sequence[:50], "error": "BLAST job failed", "source": "NCBI BLAST"}

        # Parse hits
        try:
            result = status_r.json()
            search = result.get("BlastOutput2", [{}])[0].get("report", {}).get("results", {}).get("search", {})
            hits_raw = search.get("hits", [])
            hits = []
            for h in hits_raw[:max_hits]:
                desc = h.get("description", [{}])[0]
                hsp = h.get("hsps", [{}])[0]
                hits.append({
                    "title": desc.get("title", "")[:100],
                    "accession": desc.get("accession"),
                    "identity_pct": round(hsp.get("identity", 0) / max(hsp.get("align_len", 1), 1) * 100, 1),
                    "align_length": hsp.get("align_len"),
                    "evalue": hsp.get("evalue"),
                    "bit_score": hsp.get("bit_score"),
                })
            return {
                "sequence_length": len(sequence.replace(">", "").replace("\n", "")),
                "program": program,
                "database": database,
                "hits": hits,
                "rid": rid,
                "source": "NCBI BLAST",
            }
        except Exception:
            return {"rid": rid, "raw_length": len(status_r.text), "source": "NCBI BLAST",
                    "note": "Results available at: https://blast.ncbi.nlm.nih.gov/blast/Blast.cgi?CMD=Get&RID=" + rid}

    except Exception as e:
        logger.error(f"blast_sequence error: {e}")
        return {"sequence": sequence[:50], "error": str(e), "source": "NCBI BLAST"}


# ─────────────────────────────────────────────────────────────────────────────
# DESeq2 — differential expression analysis wrapper
# ─────────────────────────────────────────────────────────────────────────────

def run_deseq2(counts_path: str, metadata_path: str,
               condition_col: str = "condition",
               reference_level: str = None,
               output_dir: str = "output/") -> dict:
    """
    Run DESeq2 differential expression analysis.
    Uses pydeseq2 (Python implementation, no R required).
    For R's DESeq2, use code_executor with tool='R'.

    Args:
        counts_path: Path to count matrix (genes × samples TSV/CSV, genes as rows)
        metadata_path: Path to sample metadata (samples × conditions CSV)
        condition_col: Column in metadata with condition labels (e.g. 'condition', 'group')
        reference_level: Reference condition for comparison (e.g. 'control', 'young')
        output_dir: Directory to write results

    Returns:
        dict with top DE genes, LFC, p-values, and output file paths
    """
    try:
        import pandas as pd
        import os
        os.makedirs(output_dir, exist_ok=True)
    except ImportError:
        return {"error": "pandas not installed", "source": "DESeq2"}

    try:
        from pydeseq2.dds import DeseqDataSet
        from pydeseq2.ds import DeseqStats
    except ImportError:
        return {"error": "pydeseq2 not installed. Run: pip install pydeseq2", "source": "DESeq2"}

    try:
        counts = pd.read_csv(counts_path, index_col=0, sep=None, engine="python")
        metadata = pd.read_csv(metadata_path, index_col=0, sep=None, engine="python")

        # Align samples
        common = counts.columns.intersection(metadata.index)
        counts = counts[common].T.astype(int)
        metadata = metadata.loc[common]

        dds = DeseqDataSet(
            counts=counts,
            metadata=metadata,
            design_factors=condition_col,
            ref_level=[condition_col, reference_level] if reference_level else None,
        )
        dds.deseq2()

        stat_res = DeseqStats(dds)
        stat_res.summary()
        results = stat_res.results_df.sort_values("padj").dropna(subset=["padj"])

        # Save full results
        out_path = os.path.join(output_dir, "deseq2_results.csv")
        results.to_csv(out_path)

        top_up = results[results["log2FoldChange"] > 0].head(10)
        top_down = results[results["log2FoldChange"] < 0].head(10)

        return {
            "n_genes_tested": len(results),
            "n_significant_padj05": int((results["padj"] < 0.05).sum()),
            "top_upregulated": top_up[["log2FoldChange", "pvalue", "padj"]].to_dict("index"),
            "top_downregulated": top_down[["log2FoldChange", "pvalue", "padj"]].to_dict("index"),
            "output_file": out_path,
            "source": "pydeseq2",
        }
    except Exception as e:
        logger.error(f"run_deseq2 error: {e}")
        return {"error": str(e), "source": "DESeq2"}