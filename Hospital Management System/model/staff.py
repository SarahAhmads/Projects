from .person import Person


class Staff(Person):
    """Class for hospital staff, inheriting from Person."""

    def __init__(self, name: str, age: int, position: str):
        """
        Initializes the Staff class with name, age, and position.
        """
        super().__init__(name, age)
        self.position = position

    def view_info(self) -> str:
        """View staff information."""
        return f"Staff Name: {self.name}, Age: {self.age}, Position: {self.position}"