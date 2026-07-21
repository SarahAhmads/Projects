from .patient import Patient
from .staff import Staff

class Department:
    """
    This class for hospital department, managing patients and staff
    """
    def __init__(self, name):
        """
        Initializes the Department class with name.
        """
        # Input validation
        if not isinstance(name, str):
            raise TypeError("Department name must be a string!")
        
        self.name = name
        self.patients = []
        self.staff = []

    def add_patient(self, patient: Patient) -> None:
        """
        Adds a patient to the department
        """
        # Input validation
        if not isinstance(patient, Patient):
            raise TypeError("Patient must be a Patient object!")
            
        self.patients.append(patient)
        print(f"Patient '{patient.name}' added to {self.name} department.")

    def add_staff_member(self, staff_member: Staff) -> None:
        """
        Adds a staff member to the department
        """
        # Input validation
        if not isinstance(staff_member, Staff):
            raise TypeError("Staff member must be a Staff object!")
            
        self.staff.append(staff_member)
        print(f"Staff '{staff_member.name}' added to {self.name} department.") 