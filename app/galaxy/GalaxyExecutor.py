import os
import time
import logging
from typing import Dict, List
import json

from bioblend.galaxy.toolshed import ToolShedClient
from bioblend.galaxy.objects import GalaxyInstance
from bioblend.galaxy.objects.wrappers import History, Dataset
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
        self.gi, self.gi_client= self._connect_galaxy(galaxy_url, api_key)
        self.toolshed = ToolShedClient(self.gi_client)

    def _setup_logger(self) -> logging.Logger:
        """Configure and return logger instance."""
        logger = logging.getLogger(self.__class__.__name__)
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        return logger

    def _connect_galaxy(self, url: str, key: str):
        """Establish connection to Galaxy instance."""
        try:
            gi = GalaxyInstance(url=url, api_key=key)
            gi_client=gi.gi
            self.logger.info(f"Connected to Galaxy: {url}")
            return gi, gi_client
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

    # Funciton to check if a version compatible tool exists within the galaxy instance for workflow invocation
    def tool_exists(self, step) -> bool:
        tool_id = step.get('tool_id')
        if not tool_id:
            return True

        try:
            tool = self.gi_client.tools.show_tool(tool_id)
        except Exception:
            return False
        if not tool:
            return False

        # Grab repository info (None if local tool)
        step_repo = step.get('tool_shed_repository')
        tool_repo = tool.get('tool_shed_repository')
        # If the step was defined to come from a Tool Shed, enforce that
        if step_repo:
            # tool must also be from a Tool Shed
            if not tool_repo:
                return False
            # revisions must match exactly
            if tool_repo.get('changeset_revision') != step_repo.get('changeset_revision'):
                return False

        # If step_repo is None, we’re happy with any existing tool (shed or local)
        return True

    # Fucntion that installs tools missing in the galaxy instance for the workflow invocation        
    def tool_check_install(self, steps):
        if steps['tool_id'] is None:  
                    self.logger.info(f"skipping step {steps['id']}")
        else:
            tool_check=self.tool_exists(steps)
            if not tool_check:
                self.logger.info(f'installing tool for step {steps["id"]} ')
                toolshed_info=steps['tool_shed_repository']
                try:
                    install_result = self.gi_client.toolshed.install_repository_revision(
                                                                            tool_shed_url=f'https://{toolshed_info["tool_shed"]}',
                                                                            name=toolshed_info["name"],
                                                                            owner=toolshed_info["owner"],
                                                                            changeset_revision=toolshed_info["changeset_revision"],
                                                                            install_tool_dependencies=True,          # use Tool Shed recipes
                                                                            install_repository_dependencies=True,    # install any declared repo-level deps
                                                                            install_resolver_dependencies=True,      # invoke Conda to fetch packages
                                                                            tool_panel_section_id=None,             # or set to your desired panel section
                                                                            new_tool_panel_section_label= None
                                                                        )
                    for repo_info in install_result:
                        self.logger.info(f"  - Name: {repo_info.get('name')}, Owner: {repo_info.get('owner')}, Status: {repo_info.get('status')}, Error: {repo_info.get('error_message', 'None') if repo_info.get('status')!='installed' else 'None'}")
                    
                except Exception:
                    self.logger.info(f'failed to install tool with name: {toolshed_info["name"]}')
            else:
                self.logger.info(f'tool found for step {steps["id"]}, skipping installation')

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
                workflow_list= self.gi_client.workflows.get_workflows(published=True)
                self.logger.info(f'workflow list: {len(workflow_list)}')
                for wf in workflow_list:
                    if wf['name']==workflow_json["name"]:
                        self.logger.info(f"workflow {wf['name']} already exists in the instance, skipping import")
                        workflow=self.gi.workflows.get(wf['id'])
                        break
                if workflow is None:
                    workflow=self.gi.workflows.import_new(wf_read)
                    self.logger.info(f"workflow {workflow.name} imported")
                self.logger.info(workflow.inputs)
            else:
                raise ValueError("Invalid workflow parameters")
            
            # Checking and installing tools from the toolshed missing from the galaxy instance to run the workflow
            self.logger.info('checking if the tools in the workflow exist in the galaxy instance')
            workflow_steps = workflow_json['steps']
            
            for steps in workflow_steps.values():
                self.tool_check_install(steps)     
                if steps['type']=='subworkflow':
                    self.logger.info(f"installing subworkflow step tools for step {steps['id']}")
                    for sub_steps in steps['subworkflow']['steps'].values():
                        self.tool_check_install(sub_steps)
        
            if workflow.is_runnable:
                self.logger.info(f"Invoking workflow: {workflow.name}")
                # Attempting to combine the clients and the objects api to get optimal implementation.
                # invocation = workflow.invoke( inputs=inputs, history=history) # is also possible to use but for more specific 
                invoke=self.gi_client.workflows.invoke_workflow(workflow_id=workflow.id, inputs=inputs, history_id=history.id,parameters_normalized=True, require_exact_tool_versions=False)
                invocation=self.gi.invocations.get(invoke['id'])
            else:
                raise RuntimeError("Tools missing in instance, Workflow is not runnable")
            
            invocation_outputs=[]
            while True:
                step_jobs = self.gi_client.invocations.get_invocation_step_jobs_summary(invocation_id=invocation.id)
                all_ok = True
                ## Tracking the workflow invocation and also getting the intermediate outputs when job state is ok.
                previous_states = {}

                step_no=1 # step number counter for logging
                for step in step_jobs:
                    step_id = step['id']
                    states = step['states']
                    current_state = 'ok' if states.get('ok') == 1 else 'error' if states.get('error') == 1 else str(states)

                    # Check if state has changed
                    if previous_states.get(step_id) != current_state:
                        self.logger.info(f"Step: {step_no} ... job id: {step_id} ... state: {states}")
                        previous_states[step_id] = current_state

                    if current_state == 'ok':
                        job = self.gi_client.jobs.show_job(step_id)
                        # Adjusting output key handling if the name is uncertain
                        outputs = job.get('outputs', {})
                        if outputs:
                            first_output = next(iter(outputs.values())) # since the name of the outputs keeps changing.
                            step_output_id = first_output['id']
                            output = self.gi.datasets.get(step_output_id)
                            invocation_outputs.append(output)
                    elif current_state != 'ok':
                        all_ok = False
                    step_no +=1
                
                if all_ok:
                    self.logger.info("All jobs are ok! Workflow invocation has completed successfully.")
                    break
                elif any(step['states'].get('error') == 1 for step in step_jobs):
                    self.logger.info("One or more jobs failed. Workflow invocation has errors.")
                    invocation.cancel()
                    break
                time.sleep(3)

            return self._prepare_result(history=history, outputs=invocation_outputs, start_time=start_time, created_history=created_history, keep_history=keep_history)
            
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
            "output_ids": [ds.id for ds in outputs] ,
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
        
    except Exception:
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
        'file': 'E:/Icog/Code/AI_Assistant_Rejuve/Galaxy/Experiment_workflow/rnaseq-pe.ga',
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