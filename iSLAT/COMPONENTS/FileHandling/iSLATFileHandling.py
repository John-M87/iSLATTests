import os
import csv
import json
import pandas as pd
import numpy as np
from ...Constants import MOLECULES_DATA

save_folder_path = "DATAFILES/SAVES"
user_configuration_file_path = config_file_path = "DATAFILES/CONFIG"
theme_file_path = "DATAFILES/CONFIG/GUIThemes"
user_configuration_file_name = "UserSettings.json"

molsave_file_name = "molsave.csv"
defaults_file_name = "default.csv"
molecule_list_file_name = "molecules_list.csv"

default_molecule_parameters_file_name = "DefaultMoleculeParameters.json"
default_initial_parameters_file_name = "DefaultMoleculeParameters.json"

line_saves_file_name = "saved_lines.csv"
atomic_lines_file_name = "DATAFILES/LINELISTS/Atomic_lines.csv"
models_folder_path = "DATAFILES/MODELS"

def load_user_settings(file_path=user_configuration_file_path, file_name=user_configuration_file_name, theme_file_path=theme_file_path):
    """ load_user_settings() loads the user settings from the UserSettings.json file."""
    file = os.path.join(file_path, file_name)
    if os.path.exists(file):
        with open(file, 'r') as f:
            user_settings = json.load(f)
    else:
        # If the file does not exist, return default settings and save them as a new json file
        default_settings = {
            "first_startup": True,
            "reload_default_files": True,
            "theme": "LightTheme"
        }
        with open(file, 'w') as f:
            json.dump(default_settings, f, indent=4)
        user_settings = default_settings
    
    # append theme information to the user settings dictonary
    theme_file = f"{theme_file_path}/{user_settings['theme']}.json"
    if os.path.exists(theme_file):
        with open(theme_file, 'r') as f:
            theme_settings = json.load(f)
        user_settings["theme"] = theme_settings
    return user_settings

def read_from_csv(file_path=save_folder_path, file_name=molsave_file_name):
    file = os.path.join(file_path, file_name)
    if os.path.exists(file):
        try:
            with open(file, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                return [row for row in reader]
        except FileNotFoundError:
            pass
    return MOLECULES_DATA

def read_default_csv(file_path=save_folder_path, file_name=defaults_file_name):
    file = os.path.join(file_path, file_name)
    if os.path.exists(file):
        try:
            with open(file, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                return {row['Molecule Name']: row for row in reader if 'Molecule Name' in row}
        except FileNotFoundError:
            pass
    return MOLECULES_DATA

def read_from_user_csv(file_path=save_folder_path, file_name=molecule_list_file_name):
    file = os.path.join(file_path, file_name)
    if os.path.exists(file):
        try:
            with open(file, 'r') as csvfile:
                #print("hey hye hye")
                reader = csv.DictReader(csvfile)
                return {row['Molecule Name']: row for row in reader if 'Molecule Name' in row}
        except FileNotFoundError:
            pass
    return MOLECULES_DATA

def read_default_molecule_parameters(file_path=config_file_path, file_name=default_molecule_parameters_file_name):
    """
    read_default_molecule_parameters() updates the default molecule parameters from the DefaultMoleculeParameters.json file.
    """
    file = os.path.join(file_path, file_name)
    with open(file, 'r') as f:
        default_molecule_parameters = json.load(f)["default_initial_params"]
    return default_molecule_parameters

def read_initial_molecule_parameters(file_path=config_file_path, file_name=default_initial_parameters_file_name):
    """
    read_initial_molecule_parameters() updates the initial molecule parameters from the DefaultMoleculeParameters.json file.
    """
    file = os.path.join(file_path, file_name)
    with open(file, 'r') as f:
        initial_molecule_parameters = json.load(f)["initial_parameters"]
    return initial_molecule_parameters

def read_save_data(file_path = save_folder_path, file_name=molecule_list_file_name):
    """
    read_save_data() loads the save data from the SAVES folder.
    It returns a list of dictionaries with the save data.
    """
    #save_file = os.path.join("SAVES", "molecules_list.csv")
    file = os.path.join(file_path, file_name)
    if os.path.exists(file):
        try:
            df = pd.read_csv(file)
            savedata = {row['Molecule Name']: {col: row[col] for col in df.columns if col != 'Molecule Name'} for _, row in df.iterrows()}
            return savedata
        except Exception as e:
            print(f"Error reading save file: {e}")
            savedata = {}
            return savedata
    else:
        print("No save file found.")
        savedata = {}
        return savedata

def read_HITRAN_data(file_path):
    """
    read_HITRAN_data(file_path) reads the HITRAN .par file at the given path.
    Returns the contents as a list of lines (or processes to DataFrame if needed).
    """
    if not os.path.exists(file_path):
        #print(f"HITRAN file '{file_path}' does not exist.")
        return []

    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        #print(f"Successfully read HITRAN data from {file_path}")
        return lines
    except Exception as e:
        #print(f"Failed to read HITRAN file '{file_path}': {e}")
        return []

def read_line_saves(file_path=save_folder_path, file_name=line_saves_file_name):
    filename = os.path.join(file_path, file_name)
    if os.path.exists(file_path):
        try:
            with open(filename, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                return [row for row in reader]
        except FileNotFoundError:
            pass
    return []

def save_line(line_info, file_path=save_folder_path, file_name=line_saves_file_name):
    """Save a line to the line saves file."""
    filename = os.path.join(file_path, file_name)
    #print(f"Saving line to {filename}")
    df = pd.DataFrame([line_info])
    
    # Ensure the directory exists
    os.makedirs(file_path, exist_ok=True)
    
    # Save the line to the CSV file
    df.to_csv(filename, mode='a', header=not os.path.exists(filename), index=False)
    print(f"Saved line at ~{line_info['lam']:.4f} μm to {filename}")

def read_spectral_data(file_path : str):
    """
    read_spectral_data() reads the spectral data from the provided file path.
    Returns a DataFrame with the spectral data.
    """
    if os.path.exists(file_path):
        try:
            if (file_path.endswith('.csv') or file_path.endswith('.txt')) and os.path.isfile(file_path):
                df = pd.read_csv(file_path)
                return df
            elif file_path.endswith('.dat') and os.path.isfile(file_path):
                columns_to_load = ['wave', 'flux']
                df = pd.DataFrame(np.loadtxt(file_path), columns=columns_to_load)
                return df
        except Exception as e:
            print(f"Error reading spectral data: {e}")
            return pd.DataFrame()
    else:
        print("Spectral data file does not exist.")
        return pd.DataFrame()

def write_molecules_to_csv(molecules_dict, file_path=save_folder_path, file_name=molsave_file_name, loaded_spectrum_name="unknown"):
    """
    Write molecule parameters to CSV file.
    
    Parameters:
    -----------
    molecules_dict : MoleculeDict
        Dictionary containing molecule objects
    file_path : str
        Path to save folder
    file_name : str
        Name of the CSV file (will be prefixed with spectrum name)
    loaded_spectrum_name : str
        Name of the currently loaded spectrum file
    """
    # Create filename based on loaded spectrum
    spectrum_base_name = os.path.splitext(loaded_spectrum_name)[0] if loaded_spectrum_name != "unknown" else "default"
    csv_filename = os.path.join(file_path, f"{spectrum_base_name}-{file_name}")
    
    # Ensure the directory exists
    os.makedirs(file_path, exist_ok=True)
    
    try:
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Use field names compatible with Molecule class expectations
            header = ['Molecule Name', 'File Path', 'Molecule Label', 'Temp', 'Rad', 'N_Mol', 'Color', 'Vis', 'Dist', 'StellarRV', 'FWHM', 'Broad']
            writer.writerow(header)
            
            for mol_name, mol_obj in molecules_dict.items():
                row = [
                    mol_name,
                    getattr(mol_obj, 'filepath', getattr(mol_obj, 'hitran_data', '')),
                    getattr(mol_obj, 'name', mol_name),
                    getattr(mol_obj, 'temp', 600),
                    getattr(mol_obj, 'radius', 0.5),
                    getattr(mol_obj, 'n_mol', 1e17),
                    getattr(mol_obj, 'color', '#FF0000'),
                    getattr(mol_obj, 'is_visible', True),
                    getattr(mol_obj, 'distance', 140),  # Default distance in pc
                    getattr(mol_obj, 'stellar_rv', 0),   # Default stellar RV
                    getattr(mol_obj, 'fwhm', 200),       # Default FWHM in km/s
                    getattr(mol_obj, 'broad', 2.5)       # Default broadening
                ]
                writer.writerow(row)
        
        return csv_filename
        
    except Exception as e:
        print(f"Error saving molecule parameters: {e}")
        return None

def write_molecules_list_csv(molecules_dict, file_path=save_folder_path, file_name=molecule_list_file_name):
    """
    Write complete molecule list to CSV file for user session persistence.
    
    Parameters:
    -----------
    molecules_dict : MoleculeDict
        Dictionary containing molecule objects
    file_path : str
        Path to save folder
    file_name : str
        Name of the CSV file
    """
    csv_filename = os.path.join(file_path, file_name)
    
    # Ensure the directory exists
    os.makedirs(file_path, exist_ok=True)
    
    try:
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Use field names compatible with Molecule class expectations
            header = ['Molecule Name', 'File Path', 'Molecule Label', 'Temp', 'Rad', 'N_Mol', 'Color', 'Vis', 'Dist', 'StellarRV', 'FWHM', 'Broad']
            writer.writerow(header)
            
            for mol_name, mol_obj in molecules_dict.items():
                row = [
                    mol_name,
                    getattr(mol_obj, 'filepath', getattr(mol_obj, 'hitran_data', '')),
                    getattr(mol_obj, 'name', mol_name),
                    getattr(mol_obj, 'temp', 600),
                    getattr(mol_obj, 'radius', 0.5),
                    getattr(mol_obj, 'n_mol', 1e17),
                    getattr(mol_obj, 'color', '#FF0000'),
                    getattr(mol_obj, 'is_visible', True),
                    getattr(mol_obj, 'distance', 140),  # Default distance in pc
                    getattr(mol_obj, 'stellar_rv', 0),   # Default stellar RV
                    getattr(mol_obj, 'fwhm', 200),       # Default FWHM in km/s
                    getattr(mol_obj, 'broad', 2.5)       # Default broadening
                ]
                writer.writerow(row)
        
        return csv_filename
        
    except Exception as e:
        print(f"Error saving molecules list: {e}")
        return None

def load_atomic_lines(file_path=atomic_lines_file_name):
    """
    Load atomic line database from CSV file.
    
    Parameters
    ----------
    file_path : str
        Path to the atomic lines CSV file
        
    Returns
    -------
    pandas.DataFrame
        DataFrame containing atomic line data with columns: wave, species, line
    """
    try:
        # Try to find the atomic lines file relative to the current working directory
        if not os.path.exists(file_path):
            # Try alternative paths
            alt_paths = [
                os.path.join('iSLAT', file_path),
                os.path.join('..', file_path),
                os.path.join(os.path.dirname(__file__), '..', file_path)
            ]
            
            for alt_path in alt_paths:
                if os.path.exists(alt_path):
                    file_path = alt_path
                    break
            else:
                print(f"Warning: Atomic lines file not found at {file_path} or alternative paths")
                return pd.DataFrame()
        
        atomic_lines = pd.read_csv(file_path)
        print(f"Loaded {len(atomic_lines)} atomic lines from {file_path}")
        return atomic_lines
        
    except Exception as e:
        print(f"Error loading atomic lines: {str(e)}")
        return pd.DataFrame()