import os
import time
import logging
from typing import Dict, List
import json

from bioblend.galaxy.objects import GalaxyInstance
from bioblend.galaxy.objects.wrappers import History, Dataset, HistoryDatasetAssociation, HistoryDatasetCollectionAssociation
from dotenv import load_dotenv
import time
from sys import path
path.append('.')

from tool_info import extract_tool_info

load_dotenv()

galaxy_url = os.getenv('GALAXY_URL')
api_key = os.getenv('GALAXY_API')

class GalaxyExecutor:
    """Handle Galaxy operations including tools, workflows, and data management with enhanced features."""
    
    def __init__(self, galaxy_url: str = galaxy_url, api_key: str = api_key):
        self.logger = self._setup_logger()
        self.gi = self._connect_galaxy(galaxy_url, api_key)

    def _setup_logger(self) -> logging.Logger:
        """Configure and return logger instance."""
        logger = logging.getLogger(self.__class__.__name__)
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        return logger

    def _connect_galaxy(self, url: str, key: str) -> GalaxyInstance:
        """Establish connection to Galaxy instance with retry logic."""
        try:
            gi = GalaxyInstance(url=url, api_key=key)
            self.logger.info(f"Connected to Galaxy: {url}")
            return gi
        except Exception as e:
            self.logger.info(f"Connection failed: {e}")


    def _get_history(self, history_id: str = None) -> History:
        """Get or create a Galaxy history."""
        if history_id:
            self.logger.info(f"Using existing history: {history_id}")
            return self.gi.histories.get(history_id)
        new_history = self.gi.histories.create(name=f"GalaxyOps_{time.strftime('%Y%m%d%H%M%S')}")
        self.logger.info(f"Created new history: Name: {new_history.name}, id: {new_history.id}")
        return new_history

    def _upload_file(self, history: History, path: str, file_type: str = "auto") -> Dataset:
        """Upload file to Galaxy history."""
        self.logger.info(f"Uploading file: {path} (type: {file_type})")
        try:
            return history.upload_file(
                path,
                file_type=file_type,
                dbkey="?",  # Auto-detect genome build
                to_posix_lines=True,
                wait=True
            )
        except Exception as e:
            self.logger.error("File upload error")
            raise RuntimeError(f"Upload failed: {str(e)}") from e

    def run_tool(
        self,
        tool_id: str,
        inputs: Dict,
        history_id: str = None,
        keep_history: bool = False
    ) -> Dict:
        """
        Execute a Galaxy tool with flexible inputs.
        
        :param tool_id: Galaxy tool ID
        :param inputs: Preconstructed tool inputs dictionary
        :param history_id: Existing history ID (optional)
        :param polling_interval: Wait time between checks
        :param keep_history: Preserve history after execution
        :return: Execution results metadata
        """
        start_time = time.time()
        history = self._get_history(history_id)
        created_history = history_id is None

        try:
            tool = self.gi.tools.get(tool_id)
            self.logger.info(f"Executing tool: {tool.name}")
            
            job = tool.run(history=history, inputs=inputs)
            outputs = job[0].wait()
            
            return self._prepare_result(history, outputs, start_time, created_history, keep_history)
            
        except Exception as e:
            self.logger.error(f"Tool execution failed: {str(e)}")
            raise
        finally:
            if created_history and not keep_history:
                self._purge_history(history)

    def get_step_output(self, step):
        outputs = {}
        
        if hasattr(step, "get_outputs"):
            outputs.update(step.get_outputs())  # Returns dict[str, HistoryDatasetAssociation]
        
        if hasattr(step, "get_output_collections"):
            outputs.update(step.get_output_collections())  # Returns dict[str, HistoryDatasetCollectionAssociation]

        return outputs  # Combined dict of all outputs


    def invoke_workflow(
        self,
        inputs: Dict,
        workflow_params: Dict,
        history_id: str = None,
        keep_history: bool = False
    ) -> Dict:
        """
        Invoke a Galaxy workflow with specified inputs.
        
        :param workflow_id: Galaxy workflow ID
        :param inputs: Workflow inputs mapping
        :param history_id: Existing history ID (optional)
        :param polling_interval: Wait time between checks
        :param keep_history: Preserve history after execution
        :return: Workflow results metadata
        """
        start_time = time.time()
        history = self._get_history(history_id)
        created_history = history_id is None
        keep_workflow=workflow_params['keep_workflow']

        try:
            if workflow_params['type']== "workflow_id":
                workflow = self.gi.workflows.get(workflow_params['id'])

            elif workflow_params['type']== "workflow_file":
                with open(workflow_params['file'], 'r') as f:
                    wf_read=f.read()

                # initialize the workflow object
                workflow=None
                workflow_json = json.loads(wf_read)
                workflow_list= self.gi.workflows.get()

                for wf in workflow_list:
                    if wf.name==workflow_json["name"]:
                        self.logger.info(f'workflow {wf.name} already exists in the instance, skipping import')
                        workflow=self.gi.workflows.get(wf.id)
                        break
                if workflow is None:
                    workflow=self.gi.workflows.import_new(wf_read)
                    self.logger.info(f"workflow {workflow.name} imported")
                print('''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''')
                print(workflow.inputs)

                ## Input structure Examples ##


                # input of the files should be formated like this for the rnaseq-pe.ga workflow
                # {'0': {'label': 'Collection paired FASTQ files', 'value': '', 'uuid': '020f7580-f647-471f-b210-dbf9dfcf80bf'},
                #  '1': {'label': 'Forward adapter', 'value': '', 'uuid': '6d1de206-97c4-41ba-8702-a980f748a689'},
                #  '2': {'label': 'Reverse adapter', 'value': '', 'uuid': '7aae38b8-39a8-4fba-bc6b-bda8c1ea9162'},
                #  '3': {'label': 'Generate additional QC reports', 'value': '', 'uuid': 'bea331ce-d6fa-4312-8dce-a8b13bc4e77b'},
                #  '4': {'label': 'Reference genome', 'value': '', 'uuid': 'a72b29ae-65a7-4e4d-a996-2943ab5b02aa'}, 
                # '5': {'label': 'GTF file of annotation', 'value': '', 'uuid': '4987fd01-b72c-4bc0-9d49-a840a0d04b8e'}, 
                # '6': {'label': 'Strandedness', 'value': '', 'uuid': '2ccf2242-dddd-45b6-8cea-746d9b5d49bc'}, 
                # '7': {'label': 'Use featureCounts for generating count tables', 'value': '', 'uuid': 'f43d45e6-5692-41cb-8cfc-205dbeab1bd0'}, 
                # '8': {'label': 'Compute Cufflinks FPKM', 'value': '', 'uuid': '78868cdb-baff-4388-a41a-a700259da1ed'},
                #  '9': {'label': 'GTF with regions to exclude from FPKM normalization with Cufflinks', 'value': '', 'uuid': '0d4e1623-adc9-4ff3-8156-7e1322b1e278'},
                #  '10': {'label': 'Compute StringTie FPKM', 'value': '', 'uuid': '056f11f4-bdd9-4361-b037-2c0822320cc2'}}


                # input of the files should be formated like this for the Generic-variation-analysis-on-WGS-PE-data.ga workflow
                # {'0': {'label': 'Paired Collection', 'value': '', 'uuid': '2352cd98-29f2-456b-a807-ede1227f6089'},
                #   '1': {'label': 'GenBank genome', 'value': '', 'uuid': '5e6e8cac-8a75-42e8-9932-6c63bcebb861'}, 
                #  '2': {'label': 'Name for genome database', 'value': '', 'uuid': '7e5e0a5f-4386-44cb-8941-f04ec7d8b725'}}
                print('''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''')
            else:
                raise ValueError("Invalid workflow parameters")
        
            if workflow.is_runnable:
                self.logger.info(f"Invoking workflow: {workflow.name}")
                invocation = workflow.invoke( inputs=inputs, history=history)
            else:
                raise RuntimeError("Tools missing in instance, Workflow is not runnable")
            
            # using invocation.wait() to wait for the workflow to finish will work just fine but to monitor the steps
            #invocation = invocation.wait()

            # Monitoring the steps of the workflows invocation
            # Keep track of previous states
            step_states = {}
            intermediate_outputs=[]
            self.logger.info(f"workflow state: {invocation.state}")
            while invocation.state != "ready":


                # Refresh invocation state, but doesn't seem to do the job???
                # Possibly trying this using the client api will work since it is more direct
                invocation = invocation.refresh()
                print(invocation.state)

                for step in invocation.steps:
                    label = f"step_{step.order_index}"
                    prev_state = step_states.get(label)
                    current_state = step.state

                    # Log only if state changed
                    if current_state != prev_state:
                        step_states[label] = current_state
                        if current_state == "ready":
                            step_output=self.get_step_output(step)
                            intermediate_outputs.append(step_output)
                            self.logger.info(f"Step {label} completed successfully.")

                    # Handle failure immediately
                    if current_state == "failed":
                        self.logger.error(f"Step {label} failed. Cancelling workflow.")
                        invocation.cancel()
                        raise RuntimeError(f"Workflow failed at step: {label}")

            time.sleep(1)

            self.logger.info("Workflow completed successfully.")
            return self._prepare_result(history=history, outputs=invocation, start_time=start_time, created_history=created_history, keep_history=keep_history)
            
        except Exception as e:
            self.logger.error(f"Workflow invocation failed: {str(e)}")
            raise
        finally:
            if created_history and not keep_history:
                # delete history if the history is temporarily created
                self._purge_history(history)
            if not keep_workflow:
                # delete workflow if the workflow is temporarily created
                workflow.delete()

    def _download_outputs(self, outputs: List[Dataset], output_path: str, output_type: str = None) -> None:
        """Download Galaxy datasets to local filesystem."""
        if not outputs:
            raise RuntimeError("No outputs available for download")

        if len(outputs) > 1 and not os.path.isdir(output_path):
            raise ValueError("Multiple outputs require output_path to be a directory")

        for dataset in outputs:
            if output_type and not dataset.name.endswith(output_type):
                filename = f"{dataset.name}.{output_type}"
            else:
                filename = dataset.name

            if os.path.isdir(output_path):
                full_path = os.path.join(output_path, filename)
            else:
                full_path = output_path

            os.makedirs(os.path.dirname(full_path), exist_ok=True)

            self.logger.info(f"Saving output to: {full_path}")
            with open(full_path, "wb") as f:
                dataset.download(f)

    def _prepare_result(
        self,
        history: History,
        outputs: List[Dataset],
        start_time: float,
        created_history: bool,
        keep_history: bool
    ) -> Dict:
        """Prepare standardized result dictionary."""
        elapsed_time = time.time() - start_time
        result = {
            "history_id": history.id,
            "output_ids": [ds.id for ds in outputs],
            "output_names": [ds.name for ds in outputs],
            "execution_time": elapsed_time,
            "history_preserved": keep_history or not created_history,
            "outputs": outputs  # Include actual outputs for downloading
        }
        self.logger.info(f"Operation completed in {result['execution_time']}s")
        return result

    def _purge_history(self, history: History) -> None:
        """Permanently delete a Galaxy history."""
        try:
            history.delete(purge=True)
            self.logger.info(f"Purged history: {history.id}")
        except Exception as e:
            self.logger.warning(f"History purge failed: {str(e)}")




def Execute (
    galaxy: GalaxyExecutor,
    output_path: str,
    input: dict,
    tool_name: Dict=None,
    workflow_params: Dict=None,
    history_id: str = None,
    file_type: str = "auto",
    output_type: str = None,
    keep_history: bool = False
    ) -> Dict:
    """
    Convert files using Galaxy tools with automatic input handling.
    
    :param input_file_path: Local file path to convert
    :param tool_id: Galaxy tool ID
    :param output_path: Output directory or file path
    :param tool_params: Additional tool parameters
    :param history_id: Existing history ID (optional)
    :param file_type: Input file type
    :param output_type: Output file extension
    :param polling_interval: Wait time between checks
    :param keep_history: Preserve history after execution
    :return: Conversion results metadata
    """
    
    history = galaxy._get_history(history_id)
    created_history = history_id is None
    start_time = time.time()

    if input['type']=="file":
        if not os.path.exists(input['file']):
            raise FileNotFoundError(f"Input file not found: {input['file']}")
        dataset = galaxy._upload_file(history, input['file'], file_type)
    elif input['type']=="dataset":
        dataset=galaxy.gi.datasets.get(input['dataset_id'])
    elif input['type']=="other":
        # Handle other input types (e.g., URLs, text)
        other_input=input['other']

    try:

        if tool_name:
            # If tool_params are provided, use them directly 
            inputs = extract_tool_info( 
                tool_name=tool_name, 
                dataset_id=dataset.id if dataset else None,
                other_input=other_input if other_input else None
            )
            
            result = galaxy.run_tool(
                tool_id=inputs['tool_id'],
                inputs=inputs['tool_input'],
                history_id=history.id,
                keep_history=True  # Keep history until final cleanup
            )
        elif workflow_params:
            workflow_input={'0': {'id': dataset.id, 'src': dataset.SRC}}
            # invoke worflows
            result= galaxy.invoke_workflow(inputs=workflow_input, workflow_params=workflow_params, history_id=history.id)
                
        
        galaxy._download_outputs(result['outputs'], output_path, output_type)
        
        # Prepare the final result, considering keep_history
        final_result = galaxy._prepare_result(history, result['outputs'], start_time, created_history, keep_history)
        
        return final_result
        
    except Exception as e:
        galaxy.logger.error(f"File conversion failed: {str(e)}")
        raise
    finally:
        if created_history and not keep_history:
            galaxy._purge_history(history)


if __name__=="__main__":
    # Example usage of the GalaxyExecutor class
    galaxy = GalaxyExecutor()
    input= {
        'type': 'file',
        'file': 'E:/Icog/Code/AI_Assistant_Rejuve/Galaxy/bioblend-tutorial-main/test-data/1.txt'
    }
    workflow_params = {
        'type': 'workflow_file',
        'file': 'E:/Icog/Code/AI_Assistant_Rejuve/Galaxy/bioblend-tutorial-main/test-data/convert_to_tab.ga',
        'keep_workflow': False
    }

    result= Execute (
    galaxy=galaxy,
    output_path= "E:/Icog/Code/AI_Assistant_Rejuve/Galaxy/Experiment_workflow",
    input=input,
    tool_name=None,
    workflow_params=workflow_params,
    history_id = None,
    file_type = "auto",
    output_type= None,
    keep_history = False
    )

    from pprint import pprint
    pprint(result)