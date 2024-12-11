mkdir -p notebooks tests src scripts
touch .gitignore README.md notebooks/README.md tests/__init__.py scripts/{init.py,README.md}

python -m venv env
.\env\Scripts\activate
pip install pandas numpy matplotlib seaborn streamlit windrose
pip freeze > requirements.txt

git add .
git commit -m "Skeleton folder structure created"
git push origin main