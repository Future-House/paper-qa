class DOINotFoundError(Exception):
    def __init__(self, message="DOI not found"):
        self.message = message
        super().__init__(self.message)

class CitationConversionError(Exception):
    """Exception to throw when we can't process a citation from a BibTeX."""