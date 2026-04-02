

## TEAM WORKFLOW

1. **Clone the Project**: `git clone [REPO_URL]`.
2. **Daily Update**: Always run `git pull origin main` before starting.
3. **Create a Branch**: Never work on main. Use `git checkout -b feature/task-name`.
4. **Save Work**: Use `git add .` and `git commit -m "clear description"`.
5. **Upload & Review**: Use `git push origin feature/task-name` then open a "Pull Request" on GitHub.

---

## CONTRIBUTING GUIDE

### Project Contribution Rules
To maintain code quality, follow these strict rules:

#### A. Branching Strategy
* **Naming**: Use `feature/` for new code and `bugfix/` for repairs.
* **Isolation**: One branch per task. Keep it clean.

#### B. Development Standards
* **Type Hinting**: Use Python type hints (e.g., `data: list`) for clarity inputs and outputs of functions and classes.
* **Comments**: Add comments to explain the purpose of the code.

#### C. The Merge Process
* **Pipeline Success**: Your code will NOT merge if the Pipeline fails.
* **Code Review**: The Leader will review changes in the "Files changed" tab on GitHub.
* **Approval**: You must receive an "Approve" from the Lead to merge your code to main branch.

#### D. Prohibited Actions
* **NO MAIN PUSH**: Direct pushes to main are blocked.