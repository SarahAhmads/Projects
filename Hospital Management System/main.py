"""
Hospital Management System - Main Application
A CLI interface for managing hospital patients, staff, and departments.
With JSON data persistence.
"""
# Import required modules
import json
import os
from model import Patient, Staff, Department, Hospital, Person
from core import SystemManager


# Path to the data file
DATA_FILE = os.path.join(os.path.dirname(__file__), "data", "hospital_data.json")


def clear_screen():
    """Clears the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def print_header(title: str):
    """Prints a formatted header."""
    print("\n" + "=" * 60)
    print(f"{'🏥 ' + title + ' 🏥':^60}")
    print("=" * 60)


def print_menu(hospital_name: str):
    """Prints the main menu."""

    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║            🏥 HOSPITAL MANAGEMENT SYSTEM                 ║")
    print(f"║                  {hospital_name}                          ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print("║                                                          ║")
    print("║   [1] 👥 View All Patients                               ║")
    print("║   [2] 👨‍⚕️ View All Staff                                 ║")
    print("║   [3] 🏢 View All Departments                            ║")
    print("║   [4] ➕ Add New Patient                                 ║")
    print("║   [5] ➕ Add New Staff Member                            ║")
    print("║   [6] ➕ Add New Department                              ║")
    print("║   [7] 📊 Hospital Statistics                             ║")
    print("║   [8] 💾 Save Data                                       ║")
    print("║   [9]  🔍 Search Patient                                 ║")
    print("║   [10] 🔍 Search Staff                                   ║")
    print("║   [11] 🗑️  Delete Patient                                 ║")
    print("║   [12] 🗑️  Delete Staff                                   ║")
    print("║   [0] 🚪 Exit                                            ║")
    print("║                                                          ║")
    print("╚══════════════════════════════════════════════════════════╝")


def load_data() -> Hospital:
    """Loads hospital data from JSON file."""
    try:
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Create hospital
        hospital = Hospital(
            data['hospital']['name'],
            data['hospital']['location']
        )
        
        # Load departments
        for dept_data in data['departments']:
            department = Department(dept_data['name'])
            
            # Load patients
            for patient_data in dept_data['patients']:
                patient = Patient(
                    patient_data['name'],
                    patient_data['age'],
                    patient_data['medical_record']
                )
                department.patients.append(patient)
            
            # Load staff
            for staff_data in dept_data['staff']:
                staff_member = Staff(
                    staff_data['name'],
                    staff_data['age'],
                    staff_data['position']
                )
                department.staff.append(staff_member)
            
            hospital.departments.append(department)
        
        return hospital
    # Exception handling    
    except FileNotFoundError:
        print("⚠️  Data file not found. Creating new hospital...")
        return Hospital("Cairo Hospital", "Cairo, Egypt")
    except json.JSONDecodeError:
        print("⚠️  Error reading data file. Creating new hospital...")
        return Hospital("Cairo Hospital", "Cairo, Egypt")


def save_data(hospital: Hospital):
    """Saves hospital data to JSON file."""
    data = {
        'hospital': {
            'name': hospital.name,
            'location': hospital.location
        },
        'departments': []
    }
    
    for dept in hospital.departments:
        dept_data = {
            'name': dept.name,
            'patients': [
                {
                    'name': p.name,
                    'age': p.age,
                    'medical_record': p.medical_record
                }
                for p in dept.patients
            ],
            'staff': [
                {
                    'name': s.name,
                    'age': s.age,
                    'position': s.position
                }
                for s in dept.staff
            ]
        }
        data['departments'].append(dept_data)
    
    # Ensure data directory exists
    os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
    
    with open(DATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    
    print("\n✅ Data saved successfully!")


def view_all_patients(manager: SystemManager):
    """Displays all patients in the hospital."""
    print_header("ALL PATIENTS")
    manager.display_all_patients()
    input("\nPress Enter to return to main menu...")


def view_all_staff(manager: SystemManager):
    """Displays all staff members in the hospital."""
    print_header("ALL STAFF")
    manager.display_all_staff()
    input("\nPress Enter to return to main menu...")


def view_all_departments(hospital: Hospital):
    """Displays all departments in the hospital."""
    print_header("ALL DEPARTMENTS")
    if not hospital.departments:
        print("\n⚠️  No departments found!")
    else:
        print(f"\n{'#':<5}{'Department Name':<30}{'Patients':<12}{'Staff':<12}")
        print("-" * 59)
        for i, dept in enumerate(hospital.departments, 1):
            print(f"{i:<5}{dept.name:<30}{len(dept.patients):<12}{len(dept.staff):<12}")
    input("\nPress Enter to return to main menu...")


def add_patient(hospital: Hospital):
    """Adds a new patient to a department."""
    print_header("ADD NEW PATIENT")
    
    if not hospital.departments:
        print("\n⚠️  No departments available! Please add a department first.")
        input("\nPress Enter to return to main menu...")
        return
    
    try:
        name = input("\nEnter Patient Name: ").strip()
        if not name:
            print("❌ Name cannot be empty!")
            return
            
        age = int(input("Enter Patient Age: "))
        medical_record = input("Enter Medical Record: ").strip()
        
        print("\nSelect Department:")
        for i, dept in enumerate(hospital.departments, 1):
            print(f"  [{i}] {dept.name}")
        
        dept_choice = int(input("\nChoice: ")) - 1
        
        if 0 <= dept_choice < len(hospital.departments):
            patient = Patient(name, age, medical_record)
            hospital.departments[dept_choice].add_patient(patient)
            print(f"\n✅ Patient '{name}' added successfully!")
            
            # Auto-save
            save_data(hospital)
        else:
            print("❌ Invalid department selection!")
            
    except ValueError as e:
        print(f"❌ Invalid input: {e}")
    
    input("\nPress Enter to return to main menu...")


def add_staff(hospital: Hospital):
    """Adds a new staff member to a department."""
    print_header("ADD NEW STAFF MEMBER")
    
    if not hospital.departments:
        print("\n⚠️  No departments available! Please add a department first.")
        input("\nPress Enter to return to main menu...")
        return
    
    try:
        name = input("\nEnter Staff Name: ").strip()
        if not name:
            print("❌ Name cannot be empty!")
            return
            
        age = int(input("Enter Staff Age: "))
        position = input("Enter Position (e.g., Doctor, Nurse): ").strip()
        
        print("\nSelect Department:")
        for i, dept in enumerate(hospital.departments, 1):
            print(f"  [{i}] {dept.name}")
        
        dept_choice = int(input("\nChoice: ")) - 1
        
        if 0 <= dept_choice < len(hospital.departments):
            staff_member = Staff(name, age, position)
            hospital.departments[dept_choice].add_staff_member(staff_member)
            print(f"\n✅ Staff '{name}' added successfully!")
            
            # Auto-save
            save_data(hospital)
        else:
            print("❌ Invalid department selection!")
            
    except ValueError as e:
        print(f"❌ Invalid input: {e}")
    
    input("\nPress Enter to return to main menu...")


def add_department(hospital: Hospital):
    """Adds a new department to the hospital."""
    print_header("ADD NEW DEPARTMENT")
    
    name = input("\nEnter Department Name: ").strip()
    
    if not name:
        print("❌ Department name cannot be empty!")
    else:
        department = Department(name)
        hospital.add_department(department)
        print(f"\n✅ Department '{name}' added successfully!")
        
        # Auto-save
        save_data(hospital)
    
    input("\nPress Enter to return to main menu...")


def show_statistics(hospital: Hospital):
    """Displays hospital statistics."""
    print_header("HOSPITAL STATISTICS")
    
    total_patients = sum(len(dept.patients) for dept in hospital.departments)
    total_staff = sum(len(dept.staff) for dept in hospital.departments)
    
    print(f"\n   🏥 Hospital: {hospital.name}")
    print(f"   📍 Location: {hospital.location}")
    print("\n   ┌─────────────────────────────────────┐")
    print(f"   │ 🏢 Departments:     {len(hospital.departments):<15}│")
    print(f"   │ 👥 Total Patients:  {total_patients:<15}│")
    print(f"   │ 👨‍⚕️ Total Staff:     {total_staff:<15}│")
    print("   └─────────────────────────────────────┘")
    
    if hospital.departments:
        print("\n   Department Breakdown:")
        for dept in hospital.departments:
            print(f"   ├── {dept.name}: {len(dept.patients)} patients, {len(dept.staff)} staff")
    
    input("\nPress Enter to return to main menu...")


def search_patient(hospital: Hospital):
    """Searches for a patient by name."""
    print_header("SEARCH PATIENT")
    
    name = input("\nEnter Patient Name to search: ").strip().lower()
    if not name:
        print("❌ Name cannot be empty!")
        input("\nPress Enter to return to main menu...")
        return
    
    found = False
    for dept in hospital.departments:
        for patient in dept.patients:
            if name in patient.name.lower():
                found = True
                print(f"\n✅ Found in {dept.name} department:")
                print(f"   Name: {patient.name}")
                print(f"   Age: {patient.age}")
                print(f"   Medical Record: {patient.medical_record}")
    
    if not found:
        print(f"\n❌ No patient found with name containing '{name}'")
    
    input("\nPress Enter to return to main menu...")


def search_staff(hospital: Hospital):
    """Searches for a staff member by name."""
    print_header("SEARCH STAFF")
    
    name = input("\nEnter Staff Name to search: ").strip().lower()
    if not name:
        print("❌ Name cannot be empty!")
        input("\nPress Enter to return to main menu...")
        return
    
    found = False
    for dept in hospital.departments:
        for staff in dept.staff:
            if name in staff.name.lower():
                found = True
                print(f"\n✅ Found in {dept.name} department:")
                print(f"   Name: {staff.name}")
                print(f"   Age: {staff.age}")
                print(f"   Position: {staff.position}")
    
    if not found:
        print(f"\n❌ No staff found with name containing '{name}'")
    
    input("\nPress Enter to return to main menu...")


def delete_patient(hospital: Hospital):
    """Deletes a patient by name."""
    print_header("DELETE PATIENT")
    
    name = input("\nEnter Patient Name to delete: ").strip().lower()
    if not name:
        print("❌ Name cannot be empty!")
        input("\nPress Enter to return to main menu...")
        return
    
    # Find patient
    for dept in hospital.departments:
        for i, patient in enumerate(dept.patients):
            if name in patient.name.lower():
                print(f"\n⚠️  Found: {patient.name} (Age: {patient.age})")
                print(f"   Department: {dept.name}")
                confirm = input("\n   Are you sure you want to delete? (y/n): ").strip().lower()
                
                if confirm == 'y':
                    dept.patients.pop(i)
                    save_data(hospital)
                    print(f"\n✅ Patient '{patient.name}' deleted successfully!")
                else:
                    print("\n❌ Deletion cancelled.")
                
                input("\nPress Enter to return to main menu...")
                return
    
    print(f"\n❌ No patient found with name containing '{name}'")
    input("\nPress Enter to return to main menu...")


def delete_staff(hospital: Hospital):
    """Deletes a staff member by name."""
    print_header("DELETE STAFF")
    
    name = input("\nEnter Staff Name to delete: ").strip().lower()
    if not name:
        print("❌ Name cannot be empty!")
        input("\nPress Enter to return to main menu...")
        return
    
    # Find staff member
    for dept in hospital.departments:
        for i, staff in enumerate(dept.staff):
            if name in staff.name.lower():
                print(f"\n⚠️  Found: {staff.name} (Position: {staff.position})")
                print(f"   Department: {dept.name}")
                confirm = input("\n   Are you sure you want to delete? (y/n): ").strip().lower()
                
                if confirm == 'y':
                    dept.staff.pop(i)
                    save_data(hospital)
                    print(f"\n✅ Staff '{staff.name}' deleted successfully!")
                else:
                    print("\n❌ Deletion cancelled.")
                
                input("\nPress Enter to return to main menu...")
                return
    
    print(f"\n❌ No staff found with name containing '{name}'")
    input("\nPress Enter to return to main menu...")


def main():
    """Main application entry point."""
    # Load data from JSON
    print("🔄 Loading data from file...")
    hospital = load_data()
    manager = SystemManager(hospital)
    print(f"✅ Loaded {len(hospital.departments)} departments!\n")
    
    while True:
        clear_screen()
        print_menu(hospital.name)
        
        try:
            choice = input("\nEnter your choice: ").strip()
            
            if choice == "1":
                view_all_patients(manager)
            elif choice == "2":
                view_all_staff(manager)
            elif choice == "3":
                view_all_departments(hospital)
            elif choice == "4":
                add_patient(hospital)
            elif choice == "5":
                add_staff(hospital)
            elif choice == "6":
                add_department(hospital)
            elif choice == "7":
                show_statistics(hospital)
            elif choice == "8":
                save_data(hospital)
                input("\nPress Enter to continue...")
            elif choice == "9":
                search_patient(hospital)
            elif choice == "10":
                search_staff(hospital)
            elif choice == "11":
                delete_patient(hospital)
            elif choice == "12":
                delete_staff(hospital)
            elif choice == "0":
                # Save before exit
                save_data(hospital)
                print("\n👋 Thank you for using Hospital Management System!")
                print("   Goodbye!\n")
                break
            else:
                print("❌ Invalid choice! Please try again.")
                input("\nPress Enter to continue...")
                
        except KeyboardInterrupt:
            save_data(hospital)
            print("\n\n👋 Goodbye!")
            break


if __name__ == "__main__":
    main()
