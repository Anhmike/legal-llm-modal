# legal-llm-modal

LLM model modal for Legal 

echo "# legal-llm" >> README.md
git init
git add README.md
git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/Anhmike/legal-llm-modal.git

pip install modal-client
modal token new
modal run app.py


modal run app_2.py

modal run app_3.py::train
modal serve app_3.py

modal run app_4.py

/workspaces/legal-llm-modal/tmp
