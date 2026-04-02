
## Team Members & Assignments

### Ziad Tarek (Leader)

| Item | Details |
|------|---------|
| **Files** | `main.py` + `data/hospital_data.json` |
| **Responsibility** | Application Entry Point + Integration |
| **Branch** | `feature/main-application` |
| **Dependencies** | All other tasks must be completed first |

**Tasks:**
- Write the main application that integrates all classes
- Create sample data to test the system
- Perform final integration testing
- Review all Pull Requests before merging

-------------------------------------------------------------------------------

### Ahmed Abdelnaby

| Item | Details |
|------|---------|
| **File** | `model/person.py` |
| **Class** | `Person` (Base Class) |
| **Branch** | `feature/person-class` |
| **Dependencies** | None (Start First!) |

**Class Specifications:**
```python
class Person:
    # Attributes
    name: str
    age: int
    
    # Methods
    def view_info(self) -> str:
        """Returns person information as a string"""
```

**Tasks:**
- Create `Person` class with `__init__` method
- Implement `name` and `age` attributes with type hints
- Implement `view_info()` method
- Add docstrings and comments
- Open Pull Request for review

-------------------------------------------------------------------------------

### Mohamed Mashhhour

| Item | Details |
|------|---------|
| **File** | `model/patient.py` |
| **Class** | `Patient` (inherits from `Person`) |
| **Branch** | `feature/patient-class` |
| **Dependencies** | Wait for `Person` class to be merged |

**Class Specifications:**
```python
class Patient(Person):
    # Attributes (in addition to inherited ones)
    medical_record: str
    
    # Methods
    def view_record(self) -> str:
        """Returns the patient's medical record"""
```


**Tasks:**
- Import `Person` from `model.person`
- Create `Patient` class that inherits from `Person`
- Implement `medical_record` attribute
- Implement `view_record()` method
- Call `super().__init__()` properly
- Add docstrings and comments
- Open Pull Request for review

-------------------------------------------------------------------------------

### Mahmoud Youssef

| Item | Details |
|------|---------|
| **File** | `model/staff.py` |
| **Class** | `Staff` (inherits from `Person`) |
| **Branch** | `feature/staff-class` |
| **Dependencies** | Wait for `Person` class to be merged |

**Class Specifications:**
```python
class Staff(Person):
    # Attributes (in addition to inherited ones)
    position: str
    
    # Methods
    def view_info(self) -> str:
        """Override: Returns staff info including position"""
```

**Tasks:**
- Import `Person` from `model.person`
- Create `Staff` class that inherits from `Person`
- Implement `position` attribute
- Override `view_info()` method to include position
- Call `super().__init__()` properly
- Add docstrings and comments
- Open Pull Request for review

-------------------------------------------------------------------------------

### Sarah Ahmed

| Item | Details |
|------|---------|
| **File** | `model/department.py` |
| **Class** | `Department` |
| **Branch** | `feature/department-class` |
| **Dependencies** | Wait for `Patient` and `Staff` classes to be merged |

**Class Specifications:**
```python
class Department:
    # Attributes
    name: str
    patients: list  # List of Patient objects
    staff: list     # List of Staff objects
    
    # Methods
    def add_patient(self, patient: Patient) -> None:
        """Adds a patient to the department"""
    
    def add_staff(self, staff_member: Staff) -> None:
        """Adds a staff member to the department"""
```

**Tasks:**
- Import `Patient` from `model.patient`
- Import `Staff` from `model.staff`
- Create `Department` class
- Implement `name`, `patients`, and `staff` attributes
- Implement `add_patient()` method
- Implement `add_staff()` method
- Add docstrings and comments
- Open Pull Request for review

-------------------------------------------------------------------------------

### Yasmeen

| Item | Details |
|------|---------|
| **Files** | `model/hospital.py` + `core/system_manager.py` |
| **Classes** | `Hospital` + `SystemManager` |
| **Branch** | `feature/hospital-system` |
| **Dependencies** | Wait for `Department` class to be merged |

**Class Specifications:**
```python
# model/hospital.py
class Hospital:
    # Attributes
    name: str
    location: str
    departments: list  # List of Department objects
    
    # Methods
    def add_department(self, department: Department) -> None:
        """Adds a department to the hospital"""

# core/system_manager.py
class SystemManager:
    # This class manages the hospital system
    # Add methods for:
    # - Display all departments
    # - Display all patients
    # - Display all staff
```

**Tasks:**
- Import `Department` from `model.department`
- Create `Hospital` class in `model/hospital.py`
- Implement `add_department()` method
- Create `SystemManager` class in `core/system_manager.py`
- Add docstrings and comments
- Open Pull Request for review

-------------------------------------------------------------------------------

## 📊 Task Dependencies (Workflow)

```
┌───────────────────┐
│  Ahmed Abdelnaby  │
│   Person Class    │
└─────────┬─────────┘
          │
     ┌────┴────┐
     ▼         ▼
┌──────────────────┐ ┌───────────────┐
│Mohamed Mashhhour │ │Mahmoud Youssef│
│     Patient      │ │    Staff      │
└────────┬─────────┘ └───────┬───────┘
         │                   │
         └─────────┬─────────┘
                   ▼
         ┌─────────────────┐
         │   Sarah Ahmed   │
         │   Department    │
         └────────┬────────┘
                  ▼
         ┌─────────────────┐
         │     Yasmeen     │
         │    Hospital     │
         └────────┬────────┘
                  ▼
         ┌─────────────────┐
         │   Ziad Tarek    │
         │    main.py      │
         └─────────────────┘
```
