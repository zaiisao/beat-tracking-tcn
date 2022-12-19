#MJ:https://caremad.io/posts/2013/07/setup-vs-requirement/
# https://edykim.com/ko/post/how-does-setup.py-differ-from-requirements.txt-and-how-to-use-it/
#https://stackoverflow.com/questions/35064426/when-would-the-e-editable-option-be-useful-with-pip-install
#MH: about find_packages(): https://stackoverflow.com/questions/54430694/python-setup-py-how-to-get-find-packages-to-identify-packages-in-subdirectori

from setuptools import setup, find_packages

setup(name='beat_tracking_tcn', version='1.0', packages=find_packages())