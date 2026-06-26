import os
import requests
import tarfile
import tempfile
from pathlib import Path
import subprocess
from pyUDLF.utils import readData, outputType, evaluation, parser
import sys
import zipfile
from pyUDLF.utils.logger import get_logger
import shutil  # adiciona se não tiver

UDLF_REPO = "https://github.com/UDLF/UDLF.git"


# ---------- Logger configuration ----------
# logger = logging.getLogger(__name__)
# if not logger.hasHandlers(): 
#     handler = logging.StreamHandler()
#     formatter = logging.Formatter("[%(levelname)s] %(message)s")
#     handler.setFormatter(formatter)
#     logger.addHandler(handler)
#     logger.setLevel(logging.INFO)
logger = get_logger(__name__)

__all__ = ["setBinaryPath", "getBinaryPath", "setConfigPath", "getConfigPath", "runWithConfig", "run", "verify_bin", "build_udlf_from_source"]

# ---------- Paths ----------
user_home = Path.home()
pyudlf_dir = user_home / ".pyudlf"
udlf_install_path = pyudlf_dir / "bin"

bin_path = str(udlf_install_path / "udlf")
config_path = str(udlf_install_path / "config.ini")

original_bin_path = bin_path
original_config_path = config_path
compressed_binary_path = str(pyudlf_dir / "udlf_bin.tar.gz")

# ---------- Detect OS ----------
if sys.platform.startswith("linux"):
    operating_system = "linux"
elif sys.platform.startswith("win"):
    operating_system = "windows"
elif sys.platform == "darwin":
    operating_system = "macos"
else:
    operating_system = "unsupported"
    logger.warning("Unsupported operating system detected: %s", sys.platform)
    
# ---------- Binary URLs ----------
udlf_urls = {"linux": "http://udlf_linux.lucasvalem.com",
             "windows": "http://udlf_windows.lucasvalem.com"}

#---------
def setBinaryPath(path: str) -> None:
    """
    Update the binary path if the file exists, otherwise revert to the original.
    """
    global bin_path
    if os.path.isfile(path):
        bin_path = path
        logger.info(f"Binary path set to: {bin_path}")
    else:
        logger.warning(f"Binary not found at: {path}. Reverting to default.")
        bin_path = original_bin_path
        logger.info(f"Binary path set to: {bin_path}")


def getBinaryPath() -> str:
    """
    Return the current binary path.
    """
    global bin_path
    # logger.info(f"Current binary path: {bin_path}")
    return bin_path


def setConfigPath(path: str) -> None:
    """
    Update the config path if the file exists, otherwise revert to the original.
    """
    global config_path
    if os.path.isfile(path):
        config_path = path
        logger.info(f"Config path set to: {config_path}")
    else:
        logger.warning(f"Config not found at: {path}. Reverting to default.")
        config_path = original_config_path
        logger.info(f"Config path set to: {config_path}")


def getConfigPath() -> str:
    """
    Return the current config path.
    """
    global config_path
    # logger.info(f"Current config path: {config_path}")
    return config_path


def download_url(url: str, save_path: str, chunk_size: int = 128) -> bool:
    """
    Download a file from a given URL and save it locally.

    Args:
        url (str): URL to download the file from.
        save_path (str): Path to save the downloaded file.
        chunk_size (int): Size of chunks to stream the download. Default is 128.

    Returns:
        bool: True if download succeeded, False otherwise.
    """
    logger.info(f"Starting download from {url}")
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        total_size = int(response.headers.get("content-length", 0))
        downloaded = 0

        with open(save_path, "wb") as fd:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    fd.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        logger.debug(f"Downloaded {percent:.2f}%")

        logger.info(f"Download complete. File saved at: {save_path}")
        return True

    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download UDLF binary from {url}: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error while saving binary: {e}")
        return False


def verify_bin(config_path: str, bin_path: str) -> None:
    """
    Verify if UDLF binary and config exist. If not, download and extract them.

    Args:
        config_path (str): Path to the config file.
        bin_path (str): Path to the UDLF binary.
    """
    global operating_system
    global compressed_binary_path

    # Check if binary and config already exist
    if os.path.isfile(bin_path) and os.path.isfile(config_path):
        logger.info("UDLF binary and config found successfully.")
        return

    logger.warning("UDLF binary or config is missing...")

    # Ensure ~/.pyudlf directory exists
    logger.debug(f"Ensuring installation directory exists at {pyudlf_dir}")
    try:
        os.makedirs(pyudlf_dir, exist_ok=True)
    except Exception as e:
        logger.error(f"Could not create directory {pyudlf_dir}: {e}")
        return

    # Get download URL for the current OS
    url = udlf_urls.get(operating_system)
    if not url:
        logger.error(f"No download URL available for OS: {operating_system}")
        return

    # Download the binary archive
    logger.info(f"Attempting to download UDLF binary from {url}")
    logger.debug(f"File will be saved to {compressed_binary_path}")
    try:
        download_success = download_url(url, compressed_binary_path)
    except Exception as e:
        logger.error(f"Could not download file due to exception: {e}")
        return

    if not download_success:
        logger.error(f"Could not download file! Invalid URL {url}")
        return

    # Extract the binary according to the operating system
    try:
        if operating_system == "linux":
            with tarfile.open(compressed_binary_path, "r:gz") as archive:
                archive.extractall(pyudlf_dir)
            logger.info(f"UDLF binary extracted to {pyudlf_dir} (tar.gz)")

        elif operating_system == "windows":
            with zipfile.ZipFile(compressed_binary_path, "r") as archive:
                archive.extractall(pyudlf_dir)
            logger.info(f"UDLF binary extracted to {pyudlf_dir} (zip)")

        else:
            logger.error(f"Unsupported operating system: {operating_system}")
            return

    except Exception as e:
        logger.error(f"Failed to extract binary from {compressed_binary_path}: {e}")
        return
    logger.debug(f"Extraction complete, checking for binary at {bin_path}")

    if not os.path.isfile(bin_path):
        logger.error("Extraction completed but binary not found at: %s", bin_path)
        return
    if operating_system == "linux":
        os.chmod(bin_path, 0o755)  # garante permissão de execução no Linux
    logger.info("Binary verified at: %s", bin_path)


def run_platform(config_file: str, bin_path: str):
    """
    Run the UDLF binary with the given config file and verify execution.

    Args:
        config_file (str): Path to the configuration file.
        bin_path (str): Path to the UDLF binary.

    Returns:
        tuple:
            bool: True if run completed without detected errors, False otherwise.
            str: Path to the generated log file.
    """
    global operating_system

    # Ensure binary and config exist (download/extract if missing)
    verify_bin(config_file, bin_path)

    # Create a unique temporary log file
    tmp_log = tempfile.NamedTemporaryFile(suffix="_udlf_log.txt", delete=False)
    path_log_out = tmp_log.name
    tmp_log.close()

    # Build command
    cmd = [bin_path, config_file]
    if operating_system == "windows":
        cmd = ["cmd", "/c"] + cmd

    logger.info(f"Running UDLF framework with config: {config_file}")
    logger.debug(f"Command: {' '.join(cmd)}")
    logger.debug(f"Logs will be written to: {path_log_out}")

    try:
        with open(path_log_out, "w") as log_file:
            subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT, check=False)
    except Exception as e:
        logger.error(f"Failed to run UDLF binary: {e}")
        return False, path_log_out

    # Verify run completed successfully
    has_errors = verify_running(path_log_out)
    return has_errors, path_log_out 

def verify_running(path: str) -> bool:
    """
    Check a log file for error keywords or warning messages.

    Returns:
        bool: True if any error (or aborting warning) is found, False otherwise.
    """
    error_flag = False
    error_keywords = [
        "invalid",
        "error",
        "can't",
        "failed",
        "failure",
        "exception",
        "traceback",
        "not found",
        "critical",
        "unable",
        "denied",
        
    ]

    try:
        with open(path, "r", encoding="utf-8") as file:
            for line in file:
                lowercase_line = line.lower()

                # Check for errors
                if any(keyword in lowercase_line for keyword in error_keywords):
                    logger.error(f"[LOG ERROR] {line.strip()}")
                    error_flag = True

                # Check for warnings
                if "warning" in lowercase_line:
                    logger.warning(f"[LOG WARNING] {line.strip()}")
                    # Special case: warnings that abort execution
                    if "abort" in lowercase_line:
                        logger.error(f"[LOG ABORT] Execution stopped due to warning: {line.strip()}")
                        error_flag = True

    except FileNotFoundError:
        logger.error(f"Log file not found: {path}")
        return True
    except Exception as e:
        logger.error(f"Unexpected error while verifying log file {path}: {e}")
        return True

    return error_flag

def individual_gain_config_running(config_file: str, depth: int = -1):
    """
    Compute individual gain using parameters defined in a UDLF config file.

    Args:
        config_file (str): Path to configuration file.
        depth (int, optional): Depth to compute gain. If -1, uses dataset size.

    Returns:
        list or None: List of individual gains if computation is possible,
                      None otherwise.
    """
    try:
        params = parser.parse_config(config_file)
    except Exception as e:
        logger.error(f"Failed to parse config file {config_file}: {e}")
        return None

    # --- Validations ---
    if params["task"] != "UDL":
        logger.error("Task must be UDL. Running without calculating individual gain!")
        return None

    if params["out_file"] != "TRUE":
        logger.error("OUTPUT_FILE must be TRUE to calculate individual gain.")
        return None

    if params["in_file_format"] == "MATRIX":
        logger.error("Input file must be ranked list type, not MATRIX.")
        return None

    if params["in_file_format"] == "AUTO":
        logger.error("Input format set to AUTO. Expected ranked list type.")
        return None

    if params["in_rk_format"] != "NUM" or params["out_rk_format"] != "NUM":
        logger.error("Input and output must be numerical (NUM) format.")
        return None

    if params["out_file_format"] != "RK":
        logger.error("Output file must be ranked list type (RK).")
        return None

    if not params.get("before_path"):
        logger.error("Could not resolve input ranked list path from config.")
        return None
    if not params.get("after_path"):
        logger.error("Could not resolve output ranked list path from config.")
        return None

    # --- Handle depth ---
    if depth == -1:
        logger.warning("Depth not set, using dataset size instead.")
        try:
            with open(params["list_path"], "r") as f:
                depth = len([line.strip() for line in f])
        except Exception as e:
            logger.error(f"Failed to read list file {params['list_path']} to determine depth: {e}")
            return None

    # --- Read data ---
    try:
        classes_list = readData.read_classes(params["list_path"], params["classes_path"])
        rks_before = readData.read_ranked_lists_file_numeric(params["before_path"], top_k=depth)
        rks_after = readData.read_ranked_lists_file_numeric(params["after_path"], top_k=depth)
    except Exception as e:
        logger.error(f"Failed to read input/output ranked lists: {e}")
        return None

    if len(rks_before) < depth:
        logger.warning("Depth larger than ranked list size. Adjusting depth to max size.")
        depth = len(rks_before)

    # --- Compute gain ---
    try:
        individual_gain_list = evaluation.compute_gain(
            rks_before, rks_after, classes_list, depth, measure="MAP", verbose=True)
        logger.info("Individual gain computation completed successfully.")
        #print(individual_gain_list)
        return individual_gain_list
    except Exception as e:
        logger.error(f"Error computing individual gain: {e}")
        return None


def validate_config_and_binary(config_file: str, bin_path: str) -> bool:
    """
    Validate that config file and binary exist. If the binary is missing,
    attempt to download and install it.

    Args:
        config_file (str): Path to the configuration file.
        bin_path (str): Path to the UDLF binary.

    Returns:
        bool: True if validation passes, False otherwise.
    """
    if not os.path.isfile(config_file):
        logger.error("Config file is missing! Unable to run.")
        return False

    if not os.path.isfile(bin_path):
        logger.warning("Binary is missing! Attempting to download...")
        verify_bin(config_file, bin_path)

    return True

def prepare_visualization(params: dict, output: "OutputType") -> bool:
    """
    Prepare visualization data for RK/NUM outputs.
    Updates the output object in place.

    Args:
        params (dict): Parsed config parameters.
        output (OutputType): Output object to update.

    Returns:
        bool: True if visualization paths were set successfully, False otherwise.
    """
    out_format = params.get("out_file_format", "")
    out_rk_format = params.get("out_rk_format", "")
    img_path = params.get("images_path", "")
    list_path = params.get("list_path", "")
    classes_path = params.get("classes_path","")

    if out_format != "RK":
        logger.error("The output file must be of type 'RK'.")
        return False
    if out_rk_format != "NUM":
        logger.error("The output format of the ranked lists must be 'NUM'.")
        return False
    if not os.path.isdir(img_path):
        logger.warning(f"Images directory does not exist: {img_path}")
        return False
    
    if not os.path.isfile(list_path):
        logger.warning("List file does not exist: %s", list_path)
        return False

    if not os.path.isfile(classes_path):
        logger.warning("Classes file does not exist: %s", classes_path)
        return False

    # If everything is valid, update output
    output.images_path = img_path
    output.list_path = params.get("list_path", "")
    output.classes_path = params.get("classes_path","")
    logger.info("Visualization paths set successfully.")
    return True

def runWithConfig(
    config_file: str = None,
    get_output: bool = False,
    compute_individual_gain: bool = False,
    depth: int = -1,
    visualization: bool = False
):
    """
    Run UDLF framework with an existing configuration file.
    """
    global bin_path
    output = outputType.OutputType()

    # Step 1: validate config and binary
    if not validate_config_and_binary(config_file, bin_path):
        return False

    # Step 2: run platform
    has_errors, log_out_path = run_platform(config_file, bin_path)
    if has_errors:
        logger.error("UDLF execution failed. Check log: %s", log_out_path)
        return False
    logger.info("pyUDLF execution complete!")

    # Step 3: parse config + log if requested
    params = {}
    if get_output:
        try:
            params = parser.parse_config(config_file)
            output.rk_path = params["rk_path"]
            output.matrix_path = params["matrix_path"]
            output.log_path = params["log_path"]
            output.classes_path = params["classes_path"]
            output.list_path = params["list_path"]
            output.images_path = params["images_path"]
            output.log_dict = parser.parse_log_and_cleanup(log_out_path)
        except Exception as e:
            logger.error(f"Error parsing config file {config_file}: {e}")
            return False

    # Step 4: compute individual gain if requested
    if compute_individual_gain:
        ig_list = individual_gain_config_running(config_file, depth)
        if ig_list is None:
            logger.warning("Individual gain could not be computed. Continuing without it.")
        else:
            output.individual_gain_list = ig_list

    # Step 5: visualization
    if visualization:
        if get_output:
            success = prepare_visualization(params, output)
            if not success:
                logger.error("Visualization requested but could not be prepared.")
        else:
            logger.error("Visualization requested but output parsing was not enabled.")

    return output

def run(
    input_type,
    get_output: bool = False,
    compute_individual_gain: bool = False,
    depth: int = -1,
    visualization: bool = False
):
    """
    Run UDLF with a generated configuration file.

    Args:
        input_type: InputType object, must contain a valid config_path.
        get_output (bool, optional): If True, parse and return output paths.
        compute_individual_gain (bool, optional): If True, compute individual gain list.
        depth (int, optional): Depth for gain computation. Default is -1.
        visualization (bool, optional): If True, prepare visualization info.

    Returns:
        OutputType or False: OutputType object with parsed results, or False if execution failed.
    """
    if not os.path.isfile(input_type.config_path):
        logger.error("Unable to run: input_type was not initialized correctly (missing config).")
        return False

    global bin_path

    # Create a unique temporary config file
    tmp_file = tempfile.NamedTemporaryFile(suffix=".ini", delete=False)
    input_path = tmp_file.name
    tmp_file.close()  # close so input_type can write into it

    try:
        # Write config and run
        input_type.write_config(input_path)
        logger.debug(f"Temporary config written: {input_path}")

        output = runWithConfig(
            config_file=input_path,
            get_output=get_output,
            compute_individual_gain=compute_individual_gain,
            depth=depth,
            visualization=visualization
        )

        return output

    except Exception as e:
        logger.error(f"Error during run execution: {e}")
        return False

    finally:
        # Always clean up temp config
        if os.path.exists(input_path):
            os.remove(input_path)
            logger.debug(f"Temporary config removed: {input_path}")

def build_udlf_from_source(install_path: str, branch: str = "master") -> bool:
    """
    Clona o repositório UDLF, compila via make e instala o binário.

    Args:
        install_path: diretório onde o binário será instalado (em install_path/bin/).
        branch: branch do repositório ("master" ou "openmp").
                No macOS, apenas "master" é suportado.

    Returns:
        True se instalação bem-sucedida, False caso contrário.
    """
    global operating_system

    # macOS não suporta OpenMP
    if operating_system == "macos" and branch == "openmp":
        logger.error(
            "A branch 'openmp' não é suportada no macOS. "
            "O clang da Apple não suporta -fopenmp nativamente. "
            "Use branch='master' ou compile manualmente com libomp via Homebrew."
        )
        return False

    if operating_system == "windows":
        logger.error("build_udlf_from_source não é suportado no Windows. O binário pré-compilado é baixado automaticamente via verify_bin().")
        return False

    if operating_system == "unsupported":
        logger.error("Sistema operacional não suportado.")
        return False

    # Checa se g++ está disponível
    try:
        subprocess.run(["g++", "--version"], check=True, capture_output=True)
    except FileNotFoundError:
        if operating_system == "macos":
            logger.error("g++ não encontrado. Instale com: xcode-select --install")
        else:
            logger.error("g++ não encontrado. Instale com: sudo apt install g++")
        return False

    install_path = Path(install_path)
    repo_path = install_path / "UDLF"

    try:
        logger.info("Clonando UDLF branch '%s'...", branch)
        subprocess.run(
            ["git", "clone", "--branch", branch, "--depth", "1",
             UDLF_REPO, str(repo_path)],
            check=True
        )

        logger.info("Compilando...")
        subprocess.run(["make"], cwd=repo_path, check=True)

        compiled_bin = repo_path / "bin" / "udlf"
        compiled_config = repo_path / "bin" / "config.ini"

        if not compiled_bin.exists():
            logger.error("Compilação concluída mas binário não encontrado em: %s", compiled_bin)
            return False

        dest = install_path / "bin"
        dest.mkdir(exist_ok=True)
        shutil.copy(compiled_bin, dest / "udlf")
        shutil.copy(compiled_config, dest / "config.ini")
        os.chmod(str(dest / "udlf"), 0o755)

        logger.info("Binário instalado em: %s", dest / "udlf")
        return True

    except subprocess.CalledProcessError as e:
        logger.error("Falha durante clone ou compilação: %s", e)
        return False
    except Exception as e:
        logger.error("Erro inesperado em build_udlf_from_source: %s", e)
        return False
    finally:
        if repo_path.exists():
            shutil.rmtree(repo_path)
            logger.info("Repositório removido: %s", repo_path)
