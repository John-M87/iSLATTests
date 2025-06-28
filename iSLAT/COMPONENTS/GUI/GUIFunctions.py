import numpy as np
import pandas as pd

class GUIHandlers:
    def __init__(self, plot, data_field, config, islat):
        self.plot = plot
        self.data_field = data_field
        self.config = config
        self.islat = islat

    def save_line(self):
        if self.islat.fit_result is None:
            print("No fit result to save.")
            return

        line_info = {
            'center': self.islat.fit_result.params['center'].value,
            'amplitude': self.islat.fit_result.params['amplitude'].value,
            'sigma': self.islat.fit_result.params['sigma'].value
        }
        self.islat.save_line(line_info)
        print("Line info saved:", line_info)

    def fit_selected_line(self, deblend=False):
        if self.islat.xp1 is None or self.islat.xp2 is None:
            print("No region selected.")
            return

        self.islat.fit_selected_line(self.islat.xp1, self.islat.xp2, deblend=deblend)
        self.plot.onselect(self.islat.xp1, self.islat.xp2)  # redraw plots with fit

        report = self.islat.fit_result.fit_report()
        self.data_field.update_text(report)

    def find_single_lines(self):
        found = self.islat.find_single_lines()
        self.data_field.update_text(f"Found {len(found)} lines.")

    def single_slab_fit(self):
        summary = self.islat.run_single_slab_fit()
        self.data_field.update_text(summary)