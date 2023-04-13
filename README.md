# Github Template for Calico's Python Library

Create internal Python packages using this template. 


## Update the following parameters in the `setup.cfg` file.
Change the package name, and urls to reflect the name of your package.
```
name = calicolabs-{replace-with-package-name}
url = https://github.com/calico/replace-with-repo-name
project_urls =
    Bug Tracker = https://github.com/calico/replace-with-repo-name/issues
```

Uncomment the following lines and add your package dependencies.
Where possible, please add `~=` instead of `==`  
```
;install_requires =
;    package~=3.17.0
;    package2~=3.15.1
```

## Update the following parameters in the `pyproject.toml` file.

```text
[project]
name = "calicolabs-{python-library-name}" # Replace with your library name
```

## Updating Github permissions for CODEOWNERS  

1. After a new repository is created using this template, go to the Github repository Settings > Collaborators and teams. 
2. Click on "Add teams" to add "sweng-dev" with <b>Maintain</b> permissions for the repository.
