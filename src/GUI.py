

import customtkinter as ctk
import json
import os
from typing import Dict, Any


class ConfigurationUI:
    def __init__(self):
        self.window = ctk.CTk()
        self.window.title("Configure Transformer model")
        self.window.geometry("1200x500")

        self.config = self.load_config()

        self.debug_levels = ["INFO", "MINIMAL DEBUG", "FULL"]
        self.tokenizer_options = ["Byte Pair Encoding", "Mel_spec"]

        self.create_ui()


    def load_config(self):
        config = {
            "tokenizer" : "",
            "example_data" : "",
            "training_data" : "",
            "test_data"     : "",
            "validation_data" : ""
        }

        return config
    
    def save_config(self):
        print("saved!")

    def create_ui(self):
        main_frame = ctk.CTkFrame(self.window)
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)

        left_frame = ctk.CTkFrame(main_frame)
        left_frame.pack(side="left", fill="both", expand=True, padx=10)


        row_frame = ctk.CTkFrame(left_frame)
        row_frame.pack(fill="x", pady=5)

        ctk.CTkLabel(row_frame, text="Select a Tokenizer").pack(side="left", pady=5)
        self.tokenizer_dropdown = ctk.CTkOptionMenu(
            row_frame,
            values=self.tokenizer_options,
            command=self.save_config()
        )
        self.tokenizer_dropdown.pack(side="right", pady=5, fill='x')

        data_fields = [
            ("Example data", "example_data"),
            ("Training data", "training_data"),
            ("Test data", "test_data"),
            ("Validation data", "validation_data"),
        ]
        for label, key in data_fields:
            row_frame = ctk.CTkFrame(left_frame)
            row_frame.pack(fill="x", pady=5)
            ctk.CTkLabel(row_frame, text=label, width=80).pack(side="left", padx=5)

            entry = ctk.CTkEntry(row_frame)
            entry.pack(side="left", expand=True, pady=5, fill="x")
            entry.insert(0, self.config.get(key, ""))
            entry.configure(placeholder_text="Select a file/folder")
            browse_btn = ctk.CTkButton(
                row_frame, 
                text="Browse files",
                width=70,
                command=lambda e=entry: self.create_file_dialog(e)
            )
            browse_folder_btn = ctk.CTkButton(
                row_frame, 
                text="Browse folders",
                width=70,
                command=lambda e=entry: self.create_folder_dialog(e)
            )
            browse_folder_btn.pack(side="right", padx=5)
            browse_btn.pack(side="right", padx=5)


        middle_frame = ctk.CTkFrame(main_frame)
        middle_frame.pack(side="left", fill="both", expand=True, padx=10)

        row_frame = ctk.CTkFrame(middle_frame)
        row_frame.pack(fill="x", pady=5)

        ctk.CTkLabel(row_frame, text="Selected model", width=80).pack(padx=5, side="left")
        self.model_entry = ctk.CTkEntry(row_frame)
        self.model_entry.pack(pady=5,fill="x", side="left", expand="true")
        self.model_entry.configure(placeholder_text = "Select a model")
        browse_btn = ctk.CTkButton(
                row_frame, 
                text="Browse files",
                width=70,
                command=lambda e=entry: self.create_file_dialog(e)
            )
        browse_btn.pack(side="right", padx=5)

        row_frame = ctk.CTkFrame(middle_frame)
        row_frame.pack(fill="x", pady=5)

        ctk.CTkLabel(row_frame, text="Select debug_level").pack(side="left", pady=5)
        self.debug_dropdown = ctk.CTkOptionMenu(
            row_frame,
            values=self.debug_levels,
            command=self.save_config()
        )
        self.debug_dropdown.pack(side="right", pady=5, fill='x')

        




        


    def create_file_dialog(self, entry):
        """Create and handle file dialog for the given entry widget"""
        file_path = ctk.filedialog.askopenfilename()
        if file_path:
            entry.delete(0, 'end')
            entry.insert(0, file_path)

    def create_folder_dialog(self, entry):
        """Create and handle file dialog for the given entry widget"""
        folder_path = ctk.filedialog.askdirectory()
        if folder_path:
            entry.delete(0, 'end')
            entry.insert(0, folder_path)

    def run(self):
        self.window.mainloop()

if __name__ == "__main__":
    app = ConfigurationUI()
    app.run()