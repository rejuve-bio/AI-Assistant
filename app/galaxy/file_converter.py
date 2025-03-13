import os
import time
import logging

from bioblend.galaxy.objects import GalaxyInstance
from bioblend.galaxy.objects.wrappers import History, Dataset

import os
from dotenv import load_dotenv

from tool_info import extract_tool_info


load_dotenv()

galaxy_url= os.getenv('GALAXY_URL') 
api_key=os.getenv('GALAXY_API')

class GalaxyFileConverter:
    """Handle file conversions through Galaxy's API with enhanced error handling, logging, and timing."""

    def __init__(self, galaxy_url= galaxy_url, api_key= api_key):
        self.logger = logging.getLogger(self.__class__.__name__)
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

        try:
            self.gi = GalaxyInstance(url=galaxy_url, api_key=api_key)
            self.logger.info(f"Connected to Galaxy: {galaxy_url}")
        except Exception as e:
            self.logger.error("Galaxy connection failed")
            raise ConnectionError(f"Failed to connect to Galaxy: {str(e)}") from e

    def _get_history(self, history_id=None):
        if history_id:
            self.logger.info(f"Using history: {history_id}")
            return self.gi.histories.get(history_id)
        new_history = self.gi.histories.create(name=f"Conversion_{time.strftime('%Y%m%d%H%M%S')}")
        self.logger.info(f"Created history: {new_history.id}")
        return new_history

    def _upload_file(self, history, path, file_type):
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

    def _run_tool(
        self,
        history,
        tool_id,
        dataset,
        tool_input,
        polling_interval,
        src="hda"
    ):
        #  adding dataset id into the tool inputs
        inputs=extract_tool_info(tool_name = tool_input, dataset_id=dataset.id, src=src)["tool_input"]

        try:
            tool = self.gi.tools.get(tool_id)
            self.logger.info(f"Running tool: {tool.name}")
            tool_run = tool.run(history=history, inputs=inputs)
        except Exception as e:
            self.logger.error(f"Conversion failed: {str(e)}")
          

        try:
            output_dataset = tool_run[0]
            output_dataset.wait(polling_interval=polling_interval)
            self.logger.info(f"Tool run complete: {output_dataset.name}")
        except Exception as e:
            self.logger.error("Tool execution error")
            raise RuntimeError(f"Tool execution failed: {str(e)}") from e

        return tool_run

    def _handle_outputs(
        self,
        outputs,
        output_path,
        output_type
    ):
        if not outputs:
            self.logger.error("No outputs produced")
            raise RuntimeError("No output datasets generated")

        main_output = outputs[0]
        self.logger.info(f"Downloading output to: {output_path}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(f'{output_path}/converted.{output_type}', "wb") as f:
            main_output.download(f)
        return main_output

    def _purge_history(self, history) -> None:
        self.logger.info(f"Purging history: {history.id}")
        try:
            history.delete(purge= True)
        except Exception as e:
            self.logger.warning(f"History purge failed: {str(e)}")

    def convert_file(
        self,
        input_file_path,
        tool_id,
        output_file_path,
        tool_input,
        history_id= None,
        file_type= "auto",
        polling_interval= 5,
        keep_history= False,
        output_file_type = None
    ):
        if not os.path.exists(input_file_path):
            self.logger.error("Input file not found")
            raise FileNotFoundError(f"Input file not found: {input_file_path}")

        start_time = time.time()
        self.logger.info("Conversion started")

        history = self._get_history(history_id)
        created_history = history_id is None
        delete_history = False

        try:
            dataset = self._upload_file(history, input_file_path, file_type)
            tool_outputs = self._run_tool(
                history=history,
                tool_id=tool_id,
                dataset=dataset,
                tool_input=tool_input,
                polling_interval=polling_interval
            )

            self._handle_outputs(tool_outputs, output_file_path, output_file_type)

            elapsed_time = time.time() - start_time
            self.logger.info(f"Conversion succeeded in {elapsed_time:.2f} seconds")
            return {
                "history id": history.id,
                "output id": [ds.id for ds in tool_outputs],
                "output path": output_file_path,
                "tool outputs": [ds.misc_info for ds in tool_outputs],
                "execution time": elapsed_time
            }
           
        except Exception as e:
            self.logger.error(f"Conversion failed: {str(e)}")
            raise
        finally:
            delete_history = created_history and not keep_history
            if delete_history:
                self._purge_history(history)
                self.logger.info(f'History purge complete')

# Defining a file converter function
def file_converter(selected_tool, converter: GalaxyFileConverter,input_path, output_path, tools_mapping):

    tool_info = tools_mapping(selected_tool)

    
    result = converter.convert_file(
        input_file_path = input_path,
        tool_id = tool_info["tool_id"],
        output_file_path = output_path,
        tool_input= selected_tool,
        file_type = tool_info["input_format"],
        output_file_type = tool_info["output_format"]
    )

    print(f"Converted file: {result['output path']}")
    print(f"History ID: {result['history id']}")
    print(f"Execution time: {result['execution time']:.2f} seconds")


if __name__ == "__main__":
    input_path = "app/galaxy/test_data/converted.gff"
    output_path = "app/galaxy/test_data/"
    selected_tool="gff to bed"

        
    # calling the file converter with a simple usecase
    file_converter(
                   selected_tool = selected_tool, 
                   converter = GalaxyFileConverter(), 
                   input_path = input_path,
                   output_path = output_path, 
                   tools_mapping = extract_tool_info
                    )





