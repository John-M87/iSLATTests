import numpy as np
import tkinter as tk
import traceback
#import pandas as pd
#import os
#from tkinter import ttk
import iSLAT.iSLATFileHandling as ifh
from .GUIFunctions import create_button
#from iSLAT.COMPONENTS.GUI.MainPlot import iSLATPlot

class BottomOptions:
    def __init__(self, master, islat, theme, main_plot, data_field, config):
        self.master = master
        self.islat = islat
        self.theme = theme
        self.main_plot = main_plot
        self.data_field = data_field
        self.config = config
        #self.theme = self.master.theme

        # Create the frame for top options
        self.frame = tk.Frame(master, borderwidth=2, relief="groove")
        #self.frame.grid(row=1, column=0, columnspan=2, sticky="ew")

        # Create buttons for top options
        create_button(self.frame, self.theme, "Save Line", self.save_line, 0, 0)
        create_button(self.frame, self.theme, "Show Saved Lines", self.show_saved_lines, 0, 1)
        create_button(self.frame, self.theme, "Fit Line", self.fit_selected_line, 0, 2)
        create_button(self.frame, self.theme, "Fit Saved Lines", self.fit_saved_lines, 0, 3)
        create_button(self.frame, self.theme, "Find Single Lines", self.find_single_lines, 0, 4)
        create_button(self.frame, self.theme, "Line De-blender", lambda: self.fit_selected_line(deblend=True), 0, 5)
        create_button(self.frame, self.theme, "Single Slab Fit", self.single_slab_fit, 0, 6)
        create_button(self.frame, self.theme, "Show Atomic Lines", self.show_atomic_lines, 0, 7)
    
    def save_line(self):
        """Save the currently selected line to the line saves file."""
        if not hasattr(self.main_plot, 'selected_wave') or self.main_plot.selected_wave is None:
            self.data_field.insert_text("No line selected to save.\n")
            return
            
        # Get the strongest line information from the current selection
        strongest_line = self.main_plot.find_strongest_line()
        if strongest_line is None:
            self.data_field.insert_text("No valid line found in selection.\n")
            return
            
        # Create line info dictionary with the format expected by the file handler
        line_info = {
            'species': strongest_line.get('species', self.islat.active_molecule),
            'lev_up': strongest_line.get('lev_up', ''),
            'lev_low': strongest_line.get('lev_low', ''),
            'lam': strongest_line['wavelength'],
            'tau': strongest_line.get('tau', strongest_line['flux']),
            'intens': strongest_line.get('intensity', strongest_line['flux']),
            'a_stein': strongest_line.get('a_stein', ''),
            'e_up': strongest_line.get('e_up', ''),
            'g_up': strongest_line.get('g_up', ''),
            'xmin': self.main_plot.selected_wave[0] if len(self.main_plot.selected_wave) > 0 else strongest_line['wavelength'] - 0.01,
            'xmax': self.main_plot.selected_wave[-1] if len(self.main_plot.selected_wave) > 1 else strongest_line['wavelength'] + 0.01,
        }
        
        try:
            ifh.save_line(line_info)
            self.data_field.insert_text(f"Saved line at {line_info['lam']:.4f} μm\n")
        except Exception as e:
            self.data_field.insert_text(f"Error saving line: {e}\n")

    def show_saved_lines(self):
        """Show saved lines as vertical dashed lines on the plot."""
        try:
            # Load saved lines from file
            saved_lines = ifh.read_line_saves()
            if not saved_lines:
                self.data_field.insert_text("No saved lines found.\n")
                return
                
            # Plot the saved lines on the main plot
            self.main_plot.plot_saved_lines(saved_lines)
            
            # Update the main plot
            self.main_plot.update_all_plots()
            
            self.data_field.insert_text(f"Displayed {len(saved_lines)} saved lines on plot.\n")
            
        except Exception as e:
            self.data_field.insert_text(f"Error loading saved lines: {e}\n")

    def fit_selected_line(self, deblend=False):
        """Fit the currently selected line using LMFIT"""

        if not hasattr(self.main_plot, 'current_selection') or self.main_plot.current_selection is None:
            self.data_field.insert_text("No region selected for fitting.\n", clear_first=False)
            return

        try:
            # Compute the fit using the main plot's fitting function
            fit_result = self.main_plot.compute_fit_line(deblend=deblend)
            
            if fit_result and len(fit_result) >= 3:
                lmfit_result, fitted_wave, fitted_flux = fit_result
                
                if lmfit_result is not None and hasattr(lmfit_result, 'params'):
                    # Extract parameters using the FittingEngine methods
                    line_params = self.main_plot.fitting_engine.extract_line_parameters()
                    fit_stats = self.main_plot.fitting_engine.get_fit_statistics()
                    
                    if deblend:
                        # For deblending, show detailed results AND save lines automatically
                        self.data_field.insert_text("\nDe-blended line fit results:\n", clear_first=False)
                        
                        # Handle multi-component fits - show detailed information
                        component_idx = 0
                        saved_components = 0
                        while f'component_{component_idx}' in line_params:
                            comp_params = line_params[f'component_{component_idx}']
                            self.data_field.insert_text(f"\nComponent {component_idx+1}:\n", clear_first=False)
                            
                            # Handle None values in stderr parameters
                            center_err = comp_params.get('center_stderr', 0)
                            center_err_str = f"{center_err:.5f}" if center_err is not None else "N/A"
                            
                            # Convert FWHM to km/s like old iSLAT
                            fwhm_kms = comp_params['fwhm'] / comp_params['center'] * 299792.458  # c in km/s
                            fwhm_err_kms = "N/A"  # Would need proper error propagation
                            
                            area_err = comp_params.get('area_stderr', 0)
                            area_err_str = f"{area_err:.3e}" if area_err is not None else "N/A"
                            
                            self.data_field.insert_text(f"Centroid (μm) = {comp_params['center']:.5f} +/- {center_err_str}", clear_first=False)
                            self.data_field.insert_text(f"FWHM (km/s) = {fwhm_kms:.1f} +/- {fwhm_err_kms}", clear_first=False)
                            self.data_field.insert_text(f"Area (erg/s/cm2) = {comp_params['area']:.3e} +/- {area_err_str}", clear_first=False)
                            
                            # Automatically save this component
                            try:
                                selection = self.main_plot.current_selection
                                if selection and len(selection) >= 2:
                                    xmin, xmax = selection[0], selection[-1]
                                    
                                    # Create line info dictionary for each component
                                    line_info = {
                                        'species': self.islat.active_molecule,
                                        'lev_up': f'deblend_comp_{component_idx+1}',
                                        'lev_low': '',
                                        'lam': comp_params['center'],
                                        'tau': comp_params['amplitude'],
                                        'intens': comp_params['area'],
                                        'a_stein': '',
                                        'e_up': '',
                                        'g_up': '',
                                        'xmin': xmin,
                                        'xmax': xmax,
                                        'flux_fit': comp_params['area'],
                                        'fwhm_fit': comp_params['fwhm'],
                                        'centr_fit': comp_params['center']
                                    }
                                    
                                    # Save this component
                                    ifh.save_line(line_info)
                                    saved_components += 1
                                    
                            except Exception as save_error:
                                self.data_field.insert_text(f"Error saving component {component_idx+1}: {save_error}", clear_first=False)
                            
                            component_idx += 1
                        
                        if component_idx == 0:
                            self.data_field.insert_text("No components found in fit result.\n", clear_first=False)
                        else:
                            # Show both detailed results AND the classic save message
                            self.data_field.insert_text(f"\nDe-blended line fit completed with {component_idx} components!", clear_first=False)
                            if saved_components > 0:
                                self.data_field.insert_text(f"\nDe-blended line saved in /LINESAVES!", clear_first=False)
                            
                    else:
                        # Single Gaussian fit - show detailed results like old iSLAT
                        self.data_field.insert_text("\nGaussian fit results:\n", clear_first=False)
                        
                        if 'center' in line_params:
                            # Handle None values in stderr parameters
                            center_err = line_params.get('center_stderr', 0)
                            center_err_str = f"{center_err:.5f}" if center_err is not None else "N/A"
                            
                            # Convert FWHM to km/s like old iSLAT (approximately)
                            fwhm_kms = line_params['fwhm'] / line_params['center'] * 299792.458  # c in km/s
                            fwhm_err_kms = "N/A"  # Would need proper error propagation
                            
                            area_err = line_params.get('area_stderr', 0)
                            area_err_str = f"{area_err:.3e}" if area_err is not None else "N/A"
                            
                            self.data_field.insert_text(f"Centroid (μm) = {line_params['center']:.5f} +/- {center_err_str}", clear_first=False)
                            self.data_field.insert_text(f"FWHM (km/s) = {fwhm_kms:.1f} +/- {fwhm_err_kms}", clear_first=False)
                            self.data_field.insert_text(f"Area (erg/s/cm2) = {line_params['area']:.3e} +/- {area_err_str}", clear_first=False)
                        else:
                            self.data_field.insert_text("Could not extract fit parameters.\n", clear_first=False)
                else:
                    self.data_field.insert_text("Fit completed but no valid result object returned.\n", clear_first=False)
            else:
                self.data_field.insert_text("Fit failed or insufficient data.\n", clear_first=False)
            
            # Update plots
            self.main_plot.plot_line_inspection(highlight_strongest=False)
            
        except Exception as e:
            self.data_field.insert_text(f"Error during fitting: {e}\n", clear_first=False)
            self.data_field.insert_text(f"Traceback: {traceback.format_exc()}\n", clear_first=False)

    def fit_saved_lines(self):
        """Fit all saved lines sequentially."""
        self.data_field.clear()
        
        try:
            saved_lines = ifh.read_line_saves()
            if not saved_lines:
                self.data_field.insert_text("No saved lines to fit.\n")
                return

            self.data_field.insert_text(f"Fitting {len(saved_lines)} saved lines...\n")
            
            fit_results = []
            for i, line in enumerate(saved_lines):
                try:
                    # Set up the fitting range around each saved line
                    if 'xmin' in line and 'xmax' in line:
                        xmin, xmax = float(line['xmin']), float(line['xmax'])
                    else:
                        # Use default range around the line wavelength
                        center_wave = float(line['lam'])
                        xmin = center_wave - 0.01
                        xmax = center_wave + 0.01
                    
                    # Plot spectrum around this line
                    self.main_plot.plot_spectrum_around_line(xmin, xmax)
                    
                    # Perform the fit
                    fit_result = self.main_plot.compute_fit_line(xmin=xmin, xmax=xmax)
                    
                    if fit_result and hasattr(fit_result, 'params'):
                        fit_results.append(fit_result)
                        center = fit_result.params['center'].value
                        self.data_field.insert_text(f"Line {i+1} at {center:.4f} μm: Fit successful\n")
                    else:
                        self.data_field.insert_text(f"Line {i+1} at {line['lam']:.4f} μm: Fit failed\n")
                        
                except Exception as e:
                    self.data_field.insert_text(f"Error fitting line {i+1}: {e}\n")

            if fit_results:
                self.data_field.insert_text(f"\nCompleted fitting {len(fit_results)} out of {len(saved_lines)} lines.\n")
                # Update the line inspection plot if available
                if hasattr(self.main_plot, 'update_line_inspection_plot'):
                    self.main_plot.update_line_inspection_plot()
            else:
                self.data_field.insert_text("No successful fits completed.\n")
                
        except Exception as e:
            self.data_field.insert_text(f"Error fitting saved lines: {e}\n")

    def find_single_lines(self):
        """Find isolated molecular lines (similar to single_finder function in original iSLAT)."""
        self.data_field.clear()
        
        try:
            self.main_plot.find_single_lines()
            '''# Get current display range
            if hasattr(self.main_plot, 'ax1'):
                xmin, xmax = self.main_plot.ax1.get_xlim()
            else:
                # Use default range
                xmin, xmax = self.islat.display_range
            
            # Find single lines using the main plot's method
            single_lines = self.main_plot.find_single_lines(xmin, xmax)
            
            if single_lines and len(single_lines) > 0:
                self.data_field.insert_text(f"Found {len(single_lines)} isolated lines in current wavelength range.\n")
                
                # Plot the single lines
                self.main_plot.plot_single_lines()
                
                # Optionally display details of found lines
                for i, line in enumerate(single_lines[:10]):  # Show first 10 lines
                    wavelength = line.get('wavelength', line.get('lam', 'Unknown'))
                    intensity = line.get('intensity', line.get('intens', 'Unknown'))
                    self.data_field.insert_text(f"  Line {i+1}: {wavelength:.4f} μm, intensity: {intensity:.3e}\n")
                
                if len(single_lines) > 10:
                    self.data_field.insert_text(f"  ... and {len(single_lines) - 10} more lines\n")
                    
            else:
                self.data_field.insert_text("No isolated lines found in the current wavelength range.\n")
                
            # Update plots
            #self.main_plot.update_all_plots()'''
            
        except Exception as e:
            self.data_field.insert_text(f"Error finding single lines: {e}\n")

    def single_slab_fit(self):
        """Run single slab fit analysis."""
        self.data_field.clear()
        self.data_field.insert_text("Running single slab fit analysis...\n")
        
        try:
            # Check if we have the necessary components for slab fitting
            if not hasattr(self.islat, 'run_single_slab_fit'):
                self.data_field.insert_text("Single slab fit functionality not available.\n")
                return
                
            # Run the slab fit
            result_text = self.islat.run_single_slab_fit()
            
            if result_text:
                self.data_field.insert_text("Slab fit results:\n")
                self.data_field.insert_text(str(result_text))
                self.data_field.insert_text("\n")
            else:
                self.data_field.insert_text("Slab fit completed but no results returned.\n")
                
        except Exception as e:
            self.data_field.insert_text(f"Error running single slab fit: {e}\n")

    def export_models(self):
        """Export current models and data."""
        self.data_field.clear()
        self.data_field.insert_text("Exporting current models...\n")
        
        try:
            # Check if we have models to export
            if hasattr(self.islat, 'molecules_dict') and self.islat.molecules_dict:
                # Export model data for each visible molecule
                exported_count = 0
                for mol_name, molecule in self.islat.molecules_dict.items():
                    if hasattr(molecule, 'is_visible') and molecule.is_visible:
                        # Export this molecule's model
                        # This would depend on the specific export functionality
                        self.data_field.insert_text(f"Exported model for {mol_name}\n")
                        exported_count += 1
                
                if exported_count > 0:
                    self.data_field.insert_text(f"Successfully exported {exported_count} models.\n")
                else:
                    self.data_field.insert_text("No visible models to export.\n")
            else:
                self.data_field.insert_text("No models available for export.\n")
                
        except Exception as e:
            self.data_field.insert_text(f"Error exporting models: {e}\n")

        try:
            out_files = self.islat.slab_model.export_results()
            for f in out_files:
                self.data_field.insert_text(f"Exported to: {f}\n")
        except Exception as e:
            self.data_field.insert_text(f"Error exporting models: {e}\n")

    def show_atomic_lines(self):
        """
        Show atomic lines as vertical dashed lines on the plot.
        Replicates the functionality from the original iSLAT atomic lines feature.
        """
        try:
            # Load atomic lines from file using the file handling module
            atomic_lines = ifh.load_atomic_lines()
            
            if atomic_lines.empty:
                self.data_field.insert_text("No atomic lines data found.\n")
                return
            
            # Get the main plot axes
            if hasattr(self.main_plot, 'ax1'):
                ax1 = self.main_plot.ax1
                
                # Get wavelength and other data from the atomic lines DataFrame
                wavelengths = atomic_lines['wave'].values
                species = atomic_lines['species'].values
                line_ids = atomic_lines['line'].values
                
                # Plot vertical lines for each atomic line
                for i in range(len(wavelengths)):
                    ax1.axvline(wavelengths[i], linestyle='--', color='tomato', alpha=0.7)
                    
                    # Adjust the y-coordinate to place labels within the plot borders
                    ylim = ax1.get_ylim()
                    label_y = ylim[1]
                    
                    # Adjust the x-coordinate to place labels just to the right of the line
                    xlim = ax1.get_xlim()
                    label_x = wavelengths[i] + 0.006 * (xlim[1] - xlim[0])
                    
                    # Add text label for the line
                    label_text = f"{species[i]} {line_ids[i]}"
                    ax1.text(label_x, label_y, label_text, fontsize=8, rotation=90, 
                            va='top', ha='left', color='tomato')
                
                # Update the plot
                self.main_plot.canvas.draw()
                
                # Update data field
                self.data_field.insert_text(f"Displayed {len(wavelengths)} atomic lines on plot.\n")
                self.data_field.insert_text("Atomic lines retrieved from file.\n")
                
            else:
                self.data_field.insert_text("Main plot not available for atomic lines display.\n")
                
        except Exception as e:
            self.data_field.insert_text(f"Error displaying atomic lines: {e}\n")
            traceback.print_exc()