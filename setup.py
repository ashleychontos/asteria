import re
import setuptools

def get_property(prop, project):
    result = re.search(r'{}\s*=\s*[\'"]([^\'"]*)[\'"]'.format(prop),
                       open(project+'/__init__.py').read())
    return result.group(1)

with open("README.md", "r", encoding="utf-8") as file:
    long_description = file.read()

reqs = []
for line in open('requirements.txt', 'r').readlines():
    reqs.append(line)

setuptools.setup(
    name="prot",
    version=get_property('__version__', 'rotate'),
    license="MIT",
    author="Ashley Chontos",
    author_email="achontos@hawaii.edu",
    description="A quick package for estimating stellar rotation periods given their teff and logg",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ashleychontos/rotate",
    project_urls={
        "Source": "https://github.com/ashleychontos/rotate",
        "Bug Tracker": "https://github.com/ashleychontos/rotate/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=reqs,
    packages=setuptools.find_packages(),
    entry_points={'console_scripts':['rotate=rotate.cli:main']},
    python_requires=">=3.6",
)
