# nova-financial-insights data analysis challenge

# Git bash
mkdir -p notebooks tests src scripts
touch .gitignore README.md notebooks/README.md tests/__init__.py scripts/{init.py,README.md}

# powershell bash
python -m venv env
.\env\Scripts\activate
pip install pandas numpy matplotlib seaborn scipy streamlit windrose
pip freeze > requirements.txt

# Git bash
git add .
git commit -m "Initial commit: project setup and folder structure"
git push origin main