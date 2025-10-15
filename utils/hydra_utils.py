from hydra import compose, core, initialize
from hydra.core.global_hydra import GlobalHydra
from pathlib import Path
import os

def setup_config(config_path):
    """
    Setup Hydra configuration from the given path.
    
    Args:
        config_path (str): Path to the directory containing config files
        
    Returns:
        OmegaConf: Composed configuration object
    """
    # Reset Hydra to avoid conflicts if already initialized
    GlobalHydra.instance().clear()
    
    # Convert to absolute path if relative
    config_path = str(Path(config_path).resolve())
    
    # Initialize Hydra and load config manually
    initialize(config_path=config_path, version_base=None)
    
    # Load the configuration
    cfg = compose(config_name="config")
    return cfg

def compose_config_from_path(config_path, config_name="config"):
    """
    Compose a Hydra-compatible configuration from the specified path.
    
    Args:
        config_path (str): Path to the directory containing config files
        config_name (str, optional): Name of the main config file (without .yaml extension). 
                                   Defaults to "config".
    
    Returns:
        OmegaConf: Composed configuration object
    
    Raises:
        FileNotFoundError: If the config directory or config file doesn't exist
        hydra.errors.HydraException: If there are issues with Hydra configuration
    """
    # Reset Hydra to avoid conflicts
    GlobalHydra.instance().clear()
    
    # Convert to absolute path and validate
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config directory not found: {config_path}")
    
    if not config_path.is_dir():
        raise ValueError(f"Config path must be a directory: {config_path}")
    
    # Get absolute path and parent directory
    abs_config_path = config_path.resolve()
    parent_dir = abs_config_path.parent
    relative_config_path = abs_config_path.name
    
    # Save current working directory
    original_cwd = os.getcwd()
    
    try:
        # Change to parent directory and use relative path for Hydra
        os.chdir(parent_dir)
        
        # Initialize Hydra with the relative config path
        initialize(config_path=relative_config_path, version_base=None)
        
        # Compose the configuration
        cfg = compose(config_name=config_name)
        return cfg
        
    except Exception as e:
        # Clean up Hydra instance on error
        GlobalHydra.instance().clear()
        raise e
    finally:
        # Always restore original working directory
        os.chdir(original_cwd)

# For backward compatibility
setup_config = compose_config_from_path