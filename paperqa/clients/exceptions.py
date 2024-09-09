class DOINotFoundError(Exception):
    def __init__(self, message="DOI not found") -> None:
        self.message = message
        super().__init__(self.message)
