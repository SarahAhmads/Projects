# Hospital Management System

A Python-based hospital management system with CLI interface for managing patients, staff, and departments.

## Team Members

| Name | Role | Task |
|------|------|------|
| **Ziad Tarek** | Team Leader | `main.py` + `data/hospital_data.json` |
| **Ahmed Abdelnaby** | Developer | `model/person.py` - Person Base Class |
| **Mohamed Mashhhour** | Developer | `model/patient.py` - Patient Class |
| **Mahmoud Youssef** | Developer | `model/staff.py` - Staff Class |
| **Sarah Ahmed** | Developer | `model/department.py` - Department Class |
| **Yasmeen** | Developer | `model/hospital.py` + `core/system_manager.py` |

---

## How to Run

```bash
# Navigate to project directory
cd "Hospital System"

# Run the application
python main.py
```

---

## Project Structure

```
Hospital System/
├── main.py                  # Main application with CLI interface
├── data/
│   └── hospital_data.json   # Persistent data storage
├── model/
│   ├── __init__.py
│   ├── person.py            # Base class for Person
│   ├── patient.py           # Patient class (inherits Person)
│   ├── staff.py             # Staff class (inherits Person)
│   ├── department.py        # Department class
│   └── hospital.py          # Hospital class
├── core/
│   ├── __init__.py
│   └── system_manager.py    # System display manager
├── TASKS.md                 # Team task assignments
├── CONTRIBUTING.md          # Contribution guidelines
└── README.md                # This file
```

---

## Features

- ✅ **View** all patients, staff, and departments
- ✅ **Add** new patients, staff members, and departments
- ✅ **Hospital Statistics** - Overview of all data
- ✅ **JSON Persistence** - Data saved automatically
- ✅ **Input Validation** - Prevents invalid data entry
- ✅ **Beautiful CLI** - User-friendly interface


---

## Technologies Used

- **Python 3.10+**
- **JSON** for data persistence
- **OOP** concepts (Inheritance, Encapsulation)

