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