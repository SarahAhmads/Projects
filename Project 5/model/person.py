class Person:
    """
    Base class representing a person in the Hospital System.
    """
    
    def __init__(self, name: str, age: int) -> None:
        """
        Initializes the Person class with name and age.
        """
        # Input validation
        if not isinstance(name, str) or not isinstance(age, int):
            raise TypeError("Name must be a string and age must be an integer!")
        
        self.name = name
        self.age = age

    def view_info(self) -> str:
        """
        Returns person information as a string.
        """
        return f"Person Name: {self.name}, Age: {self.age}"