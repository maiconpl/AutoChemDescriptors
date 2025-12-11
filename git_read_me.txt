# 04/12/25

git init

cd examples; touch .gitignore

git add .

git commit -m "Initial commit"

git config --global user.email maiconpl01@gmail.com
git config --global user.name maiconpl

# Send a project to github from terminal:

git remote add origin YOUR_REPOSITORY_URL
git remote add origin https://github.com/maiconpl/AutoChemDescriptors
git remote add origin maiconpl/AutoChemDescriptors
git push -u origin main
