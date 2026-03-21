import os
import requests
import logging

logger = logging.getLogger(__name__)

class ClinicalTrialMatcherAgent:
    """Agent responsible for querying ClinicalTrials.gov for active trials."""
    
    def __init__(self):
        self.api_url = os.getenv("CLINICAL_TRIALS_API_URL", "https://clinicaltrials.gov/api/v2/studies")
        
    def find_trials(self, query: str) -> str:
        """Execute ClinicalTrials.gov search and return formatted string."""
        try:
            params = {
                "query.cond": query,  # Search by condition/disease/target
                "pageSize": 5,        # Limit to top 5 results
                "fields": "NCTId,Condition,BriefTitle,OverallStatus,Phase"
            }
            
            response = requests.get(self.api_url, params=params, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            studies = data.get("studies", [])
            
            if not studies:
                return f"No clinical trials found for query: '{query}'."
            
            result_str = f"Found {len(studies)} top clinical trials for '{query}':\n\n"
            for study in studies:
                protocol = study.get("protocolSection", {})
                
                ident = protocol.get("identificationModule", {})
                status_mod = protocol.get("statusModule", {})
                design = protocol.get("designModule", {})
                conditions = protocol.get("conditionsModule", {})
                
                nct_id = ident.get("nctId", "Unknown ID")
                title = ident.get("briefTitle", "No Title")
                status = status_mod.get("overallStatus", "Unknown Status")
                
                phases = design.get("phases", [])
                phase_str = ", ".join(phases) if phases else "Phase Not Specified"
                
                conds = conditions.get("conditions", [])
                cond_str = ", ".join(conds) if conds else "No Conditions Listed"

                result_str += f"- Trial ID: {nct_id}\n"
                result_str += f"  Title: {title}\n"
                result_str += f"  Status: {status}\n"
                result_str += f"  Phase: {phase_str}\n"
                result_str += f"  Conditions: {cond_str}\n\n"
            
            return result_str.strip()
            
        except requests.exceptions.Timeout:
            return "Error: ClinicalTrials.gov API request timed out."
        except requests.exceptions.RequestException as e:
            return f"Error connecting to ClinicalTrials.gov API: {str(e)}"
        except Exception as e:
            return f"Error parsing clinical trial data: {str(e)}"
