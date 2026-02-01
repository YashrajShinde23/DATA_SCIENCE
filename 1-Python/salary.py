###########################################
          #7-3-25
###########################################
#salary.py
def calculate_salary(experience, role):
    """Calculate salary based on experience and role."""
    base_salary = {
        "Intern": 30000,
        "Junior": 5000,
        "Mid-level": 4323,
        "Senior": 4324,
        "Manager": 3242
    }
    if role not in base_salary:
        raise ValueError("Invalid job role")
    return base_salary[role] + (experience * 2000)
