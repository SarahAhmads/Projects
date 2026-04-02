from model.hospital import Hospital


class SystemManager:
    """
    This class for system manager, managing hospital system
    """
    def __init__(self, hospital: Hospital) -> None:
        """Initializes the System Manager with a Hospital instance."""
        # Input validation
        if not isinstance(hospital, Hospital):
            raise TypeError("Hospital must be a Hospital object!")
            
        self.hospital: Hospital = hospital

    def display_all_patients(self) -> None:
        """Displays info for all patients in every department."""
        print(f"\n--- Patient List for {self.hospital.name} ---")
        for dept in self.hospital.departments:
            for patient in dept.patients:
                print(patient.view_info())

    def display_all_staff(self) -> None:
        """Displays info for all staff members in every department."""
        print(f"\n--- Staff List for {self.hospital.name} ---")
        for dept in self.hospital.departments:
            for member in dept.staff:
                print(member.view_info())
