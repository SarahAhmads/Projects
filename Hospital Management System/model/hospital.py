from .department import Department

class Hospital:
    """
    This class for hospital, managing departments
    """

    def __init__(self, name: str, location: str):
        """Initializes the hospital with name and location."""
        # Input validation
        if not isinstance(name, str) or not isinstance(location, str):
            raise TypeError("Name and location must be strings!")
            
        self.name: str = name
        self.location: str = location
        self.departments: list[Department] = [] # List of Department objects

    def add_department(self, department: Department) -> None:
        """Adds a department to the hospital."""
        # Input validation
        if not isinstance(department, Department):
            raise TypeError("Department must be a Department object!")
            
        self.departments.append(department)