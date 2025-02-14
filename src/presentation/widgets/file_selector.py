"""File selector widget implementation."""

from qtpy.QtWidgets import QMainWindow, QLineEdit, QListWidget


class FileSelector(QMainWindow):
    """List of loaded files with search functionality."""

    def __init__(self, files: list[str]):
        """Initialize the selector.

        Args:
            files: List of file paths.
        """
        super().__init__()
        self.files = files
        self.selected_file = None
        self.selected_file_index = None
        self.initUI()

    def initUI(self) -> None:
        """Initialize the user interface."""
        self.setWindowTitle("Select a file")
        self.search_bar = QLineEdit(self)
        self.search_bar.textChanged.connect(self.search)

        self.file_list = QListWidget(self)
        self.file_list.addItems(self.files)
        self.file_list.itemClicked.connect(self.file_selected)
        self.file_list.itemDoubleClicked.connect(self.file_selected)
        self.file_list.setSortingEnabled(True)

        self.setCentralWidget(self.file_list)

    def search(self) -> None:
        """Filter files based on search text."""
        search_text = self.search_bar.text()
        self.file_list.clear()
        if search_text:
            matches = [f for f in self.files if search_text in f]
        else:
            matches = self.files
        self.file_list.addItems(matches)

    def file_selected(self, item) -> None:
        """Handle file selection.

        Args:
            item: Selected list item.
        """
        self.selected_file = item.text()
        self.selected_file_index = self.file_list.index(item)
        self.close()
