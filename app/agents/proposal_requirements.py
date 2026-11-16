"""Proposal data shapes and attachable-file requirements, per kind.

The Threads API v1 doc's "Proposal data by kind" section documents what
fields a proposal's `data` object needs for each kind. PROPOSAL_DATA_SHAPE
records that shape; FILE_FORMAT_REQUIREMENTS is read by the orchestrator
and the hypothesis propose path to tell users which attached files fit.
"""

PROPOSAL_DATA_SHAPE = {
    "annotation": {
        "status": "built",
        "fields": "nodes, predicates -- today's json_format shape",
    },
    "hypothesis": {
        "status": "built -- propose_only path, app/hypothesis_generation/"
                  "hypothesis.py::propose_hypothesis, matches against "
                  "existing GWAS projects, or proposes a new project from "
                  "an attached file (kind 'hypothesis-project', data.file_id) "
                  "when none match",
        "fields": {
            "project_id": "the matched project",
            "project_name": "added beyond the doc's original 4 fields -- "
                             "needed self-contained for the title builder, "
                             "same reason annotation's data carries full "
                             "node labels rather than just ids",
            "phenotype": "added, same reason as project_name",
            "credible_set_id": "None when the named variant didn't match "
                                "any of the project's credible sets -- "
                                "the UI's 'you choose' fallback",
            "variant": "added -- the rsID named in the query, if any",
            "tissue": "auto-picked: the project's lowest-p_value LDSC tissue",
            "seed": "always 0 for now -- no basis yet to pick otherwise",
        },
    },
    "galaxy": {
        "status": "not built -- doc marks this 'after v1'",
        "fields": {
            "workflow_id": "required",
            "inputs": "keyed by input label, each {'dataset_id': ...} or {'value': ...}",
        },
    },
}


FILE_FORMAT_REQUIREMENTS = {
    "hypothesis": {
        "status": "confirmed with the Hypothesis team directly (Slack, "
                   "2026-11-16): a new project from an uploaded file goes "
                   "through POST /analysis-pipeline as file_id, the id the "
                   "file service returns, not gwas_file -- gwas_file stays "
                   "for matching an existing catalog project",
        "accepted_extensions": [".tsv", ".txt", ".csv", ".gz", ".bgz"],
        "note": "a GWAS summary-statistics file, uploaded as the file_id "
                "field when creating a project from a new upload",
    },
    "galaxy": {
        "status": "unknown -- not yet investigated against a real Galaxy "
                   "service; do not guess a format list for this until "
                   "confirmed the same way hypothesis's was",
        "accepted_extensions": None,
        "note": None,
    },
}
