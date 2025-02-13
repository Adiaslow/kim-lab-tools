"""GUI for object counting."""

import tkinter as tk
from tkinter import ttk, filedialog
from pathlib import Path
from typing import Dict, Any, Optional

from ..processors.count_processor import CountProcessor


class FileLocationsFrame(ttk.Frame):
    """Frame for file location inputs."""

    def __init__(self, parent: tk.Widget, controller: Any):
        """Initialize the frame.

        Args:
            parent: Parent widget.
            controller: GUI controller.
        """
        super().__init__(parent)
        self.controller = controller
        self.setup_ui()

    def setup_ui(self) -> None:
        """Setup the user interface."""
        # Objects file
        ttk.Label(self, text="Objects File").grid(row=0, column=0, padx=10, pady=10)
        self.objects_entry = ttk.Entry(self)
        self.objects_entry.grid(row=0, column=1, padx=10, pady=10)
        ttk.Button(self, text="Browse", command=lambda: self._browse_objects()).grid(
            row=0, column=2, padx=10, pady=10
        )

        # Output file
        ttk.Label(self, text="Output File").grid(row=1, column=0, padx=10, pady=10)
        self.output_entry = ttk.Entry(self)
        self.output_entry.grid(row=1, column=1, padx=10, pady=10)
        ttk.Button(self, text="Save As", command=lambda: self._save_output()).grid(
            row=1, column=2
        )

        # Structures file
        ttk.Label(self, text="Structures File").grid(row=2, column=0, padx=10, pady=10)
        self.structures_entry = ttk.Entry(self)
        self.structures_entry.grid(row=2, column=1, padx=10, pady=10)
        ttk.Button(self, text="Browse", command=lambda: self._browse_structures()).grid(
            row=2, column=2, padx=10, pady=10
        )

        # Process button
        ttk.Button(self, text="Process", command=self._process_counts).grid(
            row=3, column=2, padx=10, pady=20
        )

    def _browse_objects(self) -> None:
        """Browse for objects file."""
        filename = filedialog.askopenfilename(
            title="Select Objects File",
            filetypes=(("CSV Files", "*.csv"), ("All Files", "*.*")),
        )
        if filename:
            self.objects_entry.delete(0, tk.END)
            self.objects_entry.insert(0, filename)
            self.controller.objects_file = filename

    def _browse_structures(self) -> None:
        """Browse for structures file."""
        filename = filedialog.askopenfilename(
            title="Select Structures File",
            filetypes=(("CSV Files", "*.csv"), ("All Files", "*.*")),
        )
        if filename:
            self.structures_entry.delete(0, tk.END)
            self.structures_entry.insert(0, filename)
            self.controller.structures_file = filename

    def _save_output(self) -> None:
        """Choose output file location."""
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            title="Save Results As",
            filetypes=(("CSV Files", "*.csv"), ("All Files", "*.*")),
        )
        if filename:
            self.output_entry.delete(0, tk.END)
            self.output_entry.insert(0, filename)
            self.controller.output_file = filename

    def _process_counts(self) -> None:
        """Process the counts."""
        processor = CountProcessor()
        try:
            counts = processor.process(
                (
                    Path(self.controller.objects_file),
                    Path(self.controller.structures_file),
                )
            )
            self._save_results(counts)
        except Exception as e:
            tk.messagebox.showerror("Error", str(e))

    def _save_results(self, counts: Dict[int, int]) -> None:
        """Save the counting results.

        Args:
            counts: Dict mapping region IDs to counts.
        """
        output_path = Path(self.controller.output_file)
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Region", "Acronym", "Count"])
            for region_id, count in counts.items():
                region_info = self.controller.structures[region_id]
                writer.writerow([region_info["name"], region_info["acronym"], count])
