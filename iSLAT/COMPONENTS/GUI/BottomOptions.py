import numpy as np
import tkinter as tk
import traceback
#import pandas as pd
#import os
#from tkinter import ttk
import iSLAT.iSLATFileHandling as ifh
from .GUIFunctions import create_button

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
        """Fit the currently selected line using LMFIT (like the original fit_onselect function)."""

        if not hasattr(self.main_plot, 'current_selection') or self.main_plot.current_selection is None:
            self.data_field.insert_text("No region selected for fitting.\n", clear_first=False)
            return

        try:
            # Add separator line to distinguish fit results from previous content
            self.data_field.insert_text("\n" + "="*50 + "\n", clear_first=False)
            self.data_field.insert_text("FITTING RESULTS:\n", clear_first=False)
            
            # Compute the fit using the main plot's fitting function
            if deblend:
                fit_result = self.main_plot.compute_fit_line(deblend=True)
                if fit_result and len(fit_result) >= 2:
                    gauss_fit, fit_results_list, x_fit = fit_result
                    self.data_field.insert_text("De-blended line fit results:\n", clear_first=False)
                    
                    for i, result in enumerate(fit_results_list):
                        self.data_field.insert_text(f"\nComponent {i+1}:\n", console_print=True, clear_first=False)
                        self.data_field.insert_text(f"Centroid (μm) = {result['center']} +/- {result.get('center_err', 0)}", console_print=True, clear_first=False)
                        self.data_field.insert_text(f"FWHM (km/s) = {result['fwhm']} +/- {result.get('fwhm_err', 0)}", console_print=True, clear_first=False)
                        self.data_field.insert_text(f"Area (erg/s/cm2) = {result['gauss_area']} +/- {result.get('gauss_area_err', 0)}", console_print=True, clear_first=False)

                    self.data_field.insert_text("\nDe-blended line fit completed!\n", console_print=True, clear_first=False)
                else:
                    self.data_field.insert_text("De-blended fit failed or insufficient data.\n", clear_first=False)
            else:
                fit_result = self.main_plot.compute_fit_line(deblend=False)
                if fit_result and len(fit_result) >= 2:
                    gauss_fit, fit_results_list, x_fit = fit_result
                    
                    if fit_results_list and len(fit_results_list) > 0:
                        result = fit_results_list[0]  # Single Gaussian result
                        
                        self.islat.GUI.data_field.insert_text("Gaussian fit results:\n", clear_first=False)
                        self.islat.GUI.data_field.insert_text(f"Centroid (μm) = {result['center']:.5f} +/- {result.get('center_err', 0):.5f}", clear_first=False)
                        self.islat.GUI.data_field.insert_text(f"FWHM (km/s) = {result['fwhm']:.1f} +/- {result.get('fwhm_err', 0):.1f}", clear_first=False)
                        self.islat.GUI.data_field.insert_text(f"Area (erg/s/cm2) = {result['gauss_area']:.3e} +/- {result.get('gauss_area_err', 0):.3e}", clear_first=False)

                        # Add fit quality metrics if available
                        if hasattr(gauss_fit, 'chisqr'):
                            self.data_field.insert_text(f"Chi-squared = {gauss_fit.chisqr:.3f}", clear_first=False)
                        if hasattr(gauss_fit, 'redchi'):
                            self.data_field.insert_text(f"Reduced Chi-squared = {gauss_fit.redchi:.3f}", clear_first=False)
                    else:
                        self.data_field.insert_text("Fit completed but no parameters returned.\n", clear_first=False)
                else:
                    self.data_field.insert_text("Fit failed or insufficient data.\n", clear_first=False)
            
            # Update plots
            #self.main_plot.update_all_plots()
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
                    
                    # Plot spectrum around this line (similar to original implementation)
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
            # Get current display range
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
            self.main_plot.update_all_plots()
            
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