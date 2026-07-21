from .person import Person


class Patient(Person):
    """
    Represents a patient in the Hospital System.
    Inherits from Person class and adds medical record functionality.
    """

    def __init__(self, name: str, age: int, medical_record: str) -> None:
        """
        Initializes the Patient class with name, age, and medical record.

        Args:
            name: The patient's name.
            age: The patient's age.
            medical_record: The patient's medical record information.
        """
        # Call the parent class constructor
        super().__init__(name, age)
        self.medical_record = medical_record

    def view_record(self) -> str:
        """
        Returns the patient's medical record.

        Returns:
            A string containing the patient's medical record.
        """
        return f"Medical Record for {self.name}: {self.medical_record}"

    def view_info(self) -> str:
        """
        Returns complete patient information including medical record.
        Overrides the parent class method.

        Returns:
            A string containing all patient information.
        """
        return f"Patient Name: {self.name}, Age: {self.age}, Medical Record: {self.medical_record}"
